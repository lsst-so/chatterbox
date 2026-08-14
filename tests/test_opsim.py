"""Tests for the cached visit history the simulation starts from.

The real fetch talks to ConsDB, so `chatterbox.sim.opsim.fetch_opsim` is
monkeypatched throughout: what matters here is the refresh policy, not ConsDB.
"""

import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from chatterbox.sim import opsim as opsim_mod
from chatterbox.sim.opsim import OpsimCache, default_day_obs, ensure_opsim

DAY_OBS = 20260814


def visits(n=3, day_obs=DAY_OBS - 1):
    """A minimal opsim-shaped visit table."""
    return pd.DataFrame(
        {
            "observationId": range(n),
            "day_obs": [day_obs] * n,
            "fieldRA": [10.0] * n,
            "fieldDec": [-30.0] * n,
            "band": ["r"] * n,
        }
    )


@pytest.fixture
def fake_fetch(monkeypatch):
    """Replace the ConsDB fetch, recording how often it was called."""
    calls = []

    def _fetch(day_obs, tokenfile, site="usdf", lsst_survey_sim=None):
        calls.append({"day_obs": day_obs, "tokenfile": tokenfile, "site": site})
        return visits()

    monkeypatch.setattr(opsim_mod, "fetch_opsim", _fetch)
    return calls


@pytest.fixture
def clean_import_state():
    """Isolate sys.path and cached lsst_survey_sim modules.

    `_import_fetch_previous_visits` deliberately inserts a checkout into
    sys.path, which would otherwise leak into later tests.
    """
    import sys

    original_path = list(sys.path)
    cached = {k: v for k, v in sys.modules.items() if k.split(".")[0] == "lsst_survey_sim"}
    for name in cached:
        del sys.modules[name]
    yield
    sys.path[:] = original_path
    for name in [k for k in sys.modules if k.split(".")[0] == "lsst_survey_sim"]:
        del sys.modules[name]
    sys.modules.update(cached)


@pytest.fixture
def no_lsst_survey_sim(clean_import_state):
    """Make ``import lsst_survey_sim`` fail, as on a bare interpreter."""
    import sys

    class _Blocker:
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] == "lsst_survey_sim":
                raise ImportError(f"No module named {name!r}")
            return None

    blocker = _Blocker()
    sys.meta_path.insert(0, blocker)
    yield
    sys.meta_path.remove(blocker)


def age_cache(cache_path, hours):
    """Backdate a cache's recorded fetch time."""
    meta_path = cache_path.with_suffix(cache_path.suffix + ".meta.json")
    meta = json.loads(meta_path.read_text())
    meta["fetched_at"] = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat(timespec="seconds")
    meta_path.write_text(json.dumps(meta))


# ------------------------------------------------------------------- day_obs


def test_default_day_obs_is_tomorrow():
    """fetch_previous_visits returns visits *before* the day it is given."""
    when = datetime(2026, 8, 14, 3, 0, tzinfo=timezone.utc)
    assert default_day_obs(when) == 20260815


# -------------------------------------------------------------------- policy


def test_first_call_fetches_and_caches(tmp_path, fake_fetch):
    path = tmp_path / "opsim.parquet"
    table, cache = ensure_opsim(path, day_obs=DAY_OBS, tokenfile=None)

    assert len(fake_fetch) == 1
    assert len(table) == 3
    assert cache.refreshed is True
    assert cache.n_visits == 3
    assert cache.day_obs == DAY_OBS
    assert path.is_file()


def test_fresh_cache_is_reused(tmp_path, fake_fetch):
    path = tmp_path / "opsim.parquet"
    ensure_opsim(path, day_obs=DAY_OBS, tokenfile=None)
    _, cache = ensure_opsim(path, day_obs=DAY_OBS, tokenfile=None)

    assert len(fake_fetch) == 1, "a current cache must not re-fetch"
    assert cache.refreshed is False
    assert cache.stale_reason == ""


def test_cache_older_than_max_age_is_refreshed(tmp_path, fake_fetch):
    """This is the 'once a night' behaviour."""
    path = tmp_path / "opsim.parquet"
    ensure_opsim(path, day_obs=DAY_OBS, tokenfile=None, max_age_hours=24.0)
    age_cache(path, hours=30.0)

    _, cache = ensure_opsim(path, day_obs=DAY_OBS, tokenfile=None, max_age_hours=24.0)
    assert len(fake_fetch) == 2
    assert cache.refreshed is True


def test_cache_within_max_age_is_not_refreshed(tmp_path, fake_fetch):
    path = tmp_path / "opsim.parquet"
    ensure_opsim(path, day_obs=DAY_OBS, tokenfile=None, max_age_hours=24.0)
    age_cache(path, hours=6.0)

    _, cache = ensure_opsim(path, day_obs=DAY_OBS, tokenfile=None, max_age_hours=24.0)
    assert len(fake_fetch) == 1
    assert cache.refreshed is False


def test_cache_for_an_earlier_night_is_refreshed(tmp_path, fake_fetch):
    """A cache built for an earlier day_obs is missing visits since then.

    This is a correctness issue rather than staleness, so age does not save it.
    """
    path = tmp_path / "opsim.parquet"
    ensure_opsim(path, day_obs=DAY_OBS, tokenfile=None)
    assert len(fake_fetch) == 1

    _, cache = ensure_opsim(path, day_obs=DAY_OBS + 1, tokenfile=None)
    assert len(fake_fetch) == 2, "a cache covering an earlier night must refresh"
    assert cache.day_obs == DAY_OBS + 1


def test_cache_for_a_later_night_is_reused(tmp_path, fake_fetch):
    """A superset cache is fine; the driver filters by day_obs anyway."""
    path = tmp_path / "opsim.parquet"
    ensure_opsim(path, day_obs=DAY_OBS + 5, tokenfile=None)
    ensure_opsim(path, day_obs=DAY_OBS, tokenfile=None)
    assert len(fake_fetch) == 1


def test_max_age_zero_always_refreshes(tmp_path, fake_fetch):
    path = tmp_path / "opsim.parquet"
    ensure_opsim(path, day_obs=DAY_OBS, tokenfile=None, max_age_hours=0)
    ensure_opsim(path, day_obs=DAY_OBS, tokenfile=None, max_age_hours=0)
    assert len(fake_fetch) == 2


def test_force_refreshes_a_current_cache(tmp_path, fake_fetch):
    path = tmp_path / "opsim.parquet"
    ensure_opsim(path, day_obs=DAY_OBS, tokenfile=None)
    ensure_opsim(path, day_obs=DAY_OBS, tokenfile=None, force=True)
    assert len(fake_fetch) == 2


def test_token_and_site_are_passed_through(tmp_path, fake_fetch):
    path = tmp_path / "opsim.parquet"
    ensure_opsim(path, day_obs=DAY_OBS, tokenfile="~/.lsst/usdf_rsp", site="summit")
    assert fake_fetch[0]["site"] == "summit"
    assert fake_fetch[0]["tokenfile"] == "~/.lsst/usdf_rsp"


# ------------------------------------------------------------------ fallbacks


def test_failed_refresh_falls_back_to_the_cache(tmp_path, fake_fetch, monkeypatch, caplog):
    """A stale survey state beats no coverage estimate at all."""
    path = tmp_path / "opsim.parquet"
    ensure_opsim(path, day_obs=DAY_OBS, tokenfile=None)
    age_cache(path, hours=99.0)

    def _boom(*args, **kwargs):
        raise RuntimeError("ConsDB unreachable")

    monkeypatch.setattr(opsim_mod, "fetch_opsim", _boom)
    with caplog.at_level("WARNING"):
        table, cache = ensure_opsim(path, day_obs=DAY_OBS, tokenfile=None)

    assert len(table) == 3
    assert cache.refreshed is False
    assert "ConsDB unreachable" in cache.stale_reason
    assert "stale visit history" in caplog.text


def test_empty_fetch_falls_back_to_the_cache(tmp_path, fake_fetch, monkeypatch):
    path = tmp_path / "opsim.parquet"
    ensure_opsim(path, day_obs=DAY_OBS, tokenfile=None)
    monkeypatch.setattr(opsim_mod, "fetch_opsim", lambda *a, **k: None)

    _, cache = ensure_opsim(path, day_obs=DAY_OBS, tokenfile=None, force=True)
    assert "no visits" in cache.stale_reason


def test_failure_with_no_cache_raises_with_guidance(tmp_path, monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("no token")

    monkeypatch.setattr(opsim_mod, "fetch_opsim", _boom)
    with pytest.raises(RuntimeError, match="sim.opsim_tokenfile"):
        ensure_opsim(tmp_path / "opsim.parquet", day_obs=DAY_OBS, tokenfile=None)


# -------------------------------------------------- importing the fetch tool


def test_checkout_is_added_to_sys_path(tmp_path, clean_import_state):
    """lsst_survey_sim is often a checkout, not an installed package.

    The simulation subprocess gets it via PYTHONPATH; refresh-opsim runs
    in-process, so the path must be added here too.
    """
    import sys

    from chatterbox.sim.opsim import _import_fetch_previous_visits

    checkout = tmp_path / "lsst_survey_sim_checkout"
    pkg = checkout / "lsst_survey_sim"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "simulate_lsst.py").write_text(
        "def fetch_previous_visits(day_obs, tokenfile, site='usdf'):\n" "    return 'called'\n"
    )
    fetch = _import_fetch_previous_visits(checkout)
    assert str(checkout) in sys.path
    assert fetch(0, None) == "called"


def test_missing_tool_names_the_setting(tmp_path, no_lsst_survey_sim):
    """The old message blamed the token; the real cause is the environment."""
    from chatterbox.sim.opsim import OpsimToolUnavailableError, _import_fetch_previous_visits

    with pytest.raises(OpsimToolUnavailableError) as excinfo:
        _import_fetch_previous_visits(tmp_path / "not-a-checkout")

    message = str(excinfo.value)
    assert "sim.lsst_survey_sim" in message
    assert "not a directory" in message
    # It must not send the reader after the token or ConsDB.
    assert "opsim_tokenfile" not in message


def test_missing_tool_with_no_cache_raises_the_right_error(tmp_path, monkeypatch):
    from chatterbox.sim.opsim import OpsimToolUnavailableError

    def _no_module(*args, **kwargs):
        raise OpsimToolUnavailableError("Cannot import lsst_survey_sim (No module named ...). Set it.")

    monkeypatch.setattr(opsim_mod, "fetch_opsim", _no_module)
    with pytest.raises(OpsimToolUnavailableError) as excinfo:
        ensure_opsim(tmp_path / "opsim.parquet", day_obs=DAY_OBS)

    message = str(excinfo.value)
    assert "Cannot import lsst_survey_sim" in message
    assert "No cache exists" in message
    assert "opsim_tokenfile" not in message, "must not misdiagnose an import failure"


def test_missing_tool_still_falls_back_to_a_cache(tmp_path, fake_fetch, monkeypatch):
    """A broken environment must not throw away a cache we already have."""
    from chatterbox.sim.opsim import OpsimToolUnavailableError

    path = tmp_path / "opsim.parquet"
    ensure_opsim(path, day_obs=DAY_OBS)

    monkeypatch.setattr(
        opsim_mod,
        "fetch_opsim",
        lambda *a, **k: (_ for _ in ()).throw(OpsimToolUnavailableError("Cannot import lsst_survey_sim")),
    )
    table, cache = ensure_opsim(path, day_obs=DAY_OBS, force=True)
    assert len(table) == 3
    assert "Cannot import lsst_survey_sim" in cache.stale_reason


def test_network_failure_still_blames_the_token(tmp_path, monkeypatch):
    """A genuine ConsDB failure should still point at the token and site."""

    def _boom(*args, **kwargs):
        raise RuntimeError("401 Unauthorized")

    monkeypatch.setattr(opsim_mod, "fetch_opsim", _boom)
    with pytest.raises(RuntimeError) as excinfo:
        ensure_opsim(tmp_path / "opsim.parquet", day_obs=DAY_OBS)
    assert "sim.opsim_tokenfile" in str(excinfo.value)
    assert "401 Unauthorized" in str(excinfo.value)


def test_empty_consdb_with_no_cache_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(opsim_mod, "fetch_opsim", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="returned no visits"):
        ensure_opsim(tmp_path / "opsim.parquet", day_obs=DAY_OBS)


def test_checkout_is_forwarded_to_the_fetch(tmp_path, monkeypatch):
    """ensure_opsim must pass the checkout down, or the fix does nothing."""
    seen = {}

    def _fetch(day_obs, tokenfile, site="usdf", lsst_survey_sim=None):
        seen["lsst_survey_sim"] = lsst_survey_sim
        return visits()

    monkeypatch.setattr(opsim_mod, "fetch_opsim", _fetch)
    ensure_opsim(tmp_path / "opsim.parquet", day_obs=DAY_OBS, lsst_survey_sim="/opt/lss")
    assert seen["lsst_survey_sim"] == "/opt/lss"


def test_cli_forwards_the_configured_checkout(tmp_path, monkeypatch):
    from chatterbox.cli import main

    seen = {}

    def _fetch(day_obs, tokenfile, site="usdf", lsst_survey_sim=None):
        seen["lsst_survey_sim"] = lsst_survey_sim
        return visits()

    monkeypatch.setattr(opsim_mod, "fetch_opsim", _fetch)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        json.dumps(
            {
                "sim": {
                    "opsim_cache": str(tmp_path / "opsim.parquet"),
                    "lsst_survey_sim": "/sdf/home/s/seanmacb/lsst_survey_sim",
                }
            }
        )
    )
    assert main(["-c", str(config_path), "refresh-opsim"]) == 0
    assert seen["lsst_survey_sim"] == "/sdf/home/s/seanmacb/lsst_survey_sim"


def test_corrupt_metadata_triggers_a_refresh(tmp_path, fake_fetch):
    path = tmp_path / "opsim.parquet"
    ensure_opsim(path, day_obs=DAY_OBS, tokenfile=None)
    path.with_suffix(path.suffix + ".meta.json").write_text("{not json")

    ensure_opsim(path, day_obs=DAY_OBS, tokenfile=None)
    assert len(fake_fetch) == 2


# ----------------------------------------------------------------- provenance


def test_describe_reports_freshness():
    cache = OpsimCache(
        path=None,
        day_obs=DAY_OBS,
        fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        n_visits=1234,
        refreshed=True,
    )
    text = cache.describe()
    assert "1,234 visits" in text
    assert "refreshed" in text
    assert str(DAY_OBS) in text


def test_describe_flags_a_stale_cache():
    cache = OpsimCache(
        path=None,
        day_obs=DAY_OBS,
        fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        n_visits=10,
        stale_reason="could not refresh (boom), using the cached copy",
    )
    assert "could not refresh" in cache.describe()


def test_age_hours_handles_a_missing_timestamp():
    cache = OpsimCache(path=None, day_obs=DAY_OBS, fetched_at="", n_visits=0)
    assert cache.age_hours != cache.age_hours  # NaN
    assert "unknown age" in cache.describe()


# ------------------------------------------------------------------------ CLI


def test_cli_refresh_opsim(tmp_path, fake_fetch, capsys):
    from chatterbox.cli import main

    config_path = tmp_path / "config.yaml"
    config_path.write_text(json.dumps({"sim": {"opsim_cache": str(tmp_path / "opsim.parquet")}}))

    assert main(["-c", str(config_path), "refresh-opsim"]) == 0
    assert len(fake_fetch) == 1
    assert "3 visits" in capsys.readouterr().out


def test_cli_refresh_opsim_force(tmp_path, fake_fetch):
    from chatterbox.cli import main

    config_path = tmp_path / "config.yaml"
    config_path.write_text(json.dumps({"sim": {"opsim_cache": str(tmp_path / "opsim.parquet")}}))

    main(["-c", str(config_path), "refresh-opsim"])
    main(["-c", str(config_path), "refresh-opsim", "--force"])
    assert len(fake_fetch) == 2


def test_cli_refresh_opsim_reports_failure(tmp_path, monkeypatch, capsys):
    from chatterbox.cli import main

    def _boom(*args, **kwargs):
        raise RuntimeError("ConsDB unreachable")

    monkeypatch.setattr(opsim_mod, "fetch_opsim", _boom)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(json.dumps({"sim": {"opsim_cache": str(tmp_path / "opsim.parquet")}}))

    assert main(["-c", str(config_path), "refresh-opsim"]) == 1
    assert "ConsDB unreachable" in capsys.readouterr().err
