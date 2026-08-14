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

    def _fetch(day_obs, tokenfile, site="usdf"):
        calls.append({"day_obs": day_obs, "tokenfile": tokenfile, "site": site})
        return visits()

    monkeypatch.setattr(opsim_mod, "fetch_opsim", _fetch)
    return calls


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

    def _boom(day_obs, tokenfile, site="usdf"):
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
    def _boom(day_obs, tokenfile, site="usdf"):
        raise RuntimeError("no token")

    monkeypatch.setattr(opsim_mod, "fetch_opsim", _boom)
    with pytest.raises(RuntimeError, match="sim.opsim_tokenfile"):
        ensure_opsim(tmp_path / "opsim.parquet", day_obs=DAY_OBS, tokenfile=None)


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

    def _boom(day_obs, tokenfile, site="usdf"):
        raise RuntimeError("ConsDB unreachable")

    monkeypatch.setattr(opsim_mod, "fetch_opsim", _boom)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(json.dumps({"sim": {"opsim_cache": str(tmp_path / "opsim.parquet")}}))

    assert main(["-c", str(config_path), "refresh-opsim"]) == 1
    assert "ConsDB unreachable" in capsys.readouterr().err
