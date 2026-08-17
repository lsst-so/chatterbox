"""End-to-end tests for the alert path, including its latency budget."""

import json

import pytest

from chatterbox.app import process_trigger
from chatterbox.ingest.decode import load_record_file
from chatterbox.slackbot.client import SlackPoster

#: The first post is the whole point of the service, so its cost is asserted.
#: Generous relative to the measured ~7 s so this does not flake on a busy
#: host.
STAGE1_BUDGET_S = 25.0


@pytest.fixture
def offline_poster(config, tmp_path):
    return SlackPoster(config, dry_run=True, output_dir=tmp_path / "posts")


@pytest.fixture
def template_cache(config):
    """A small template cache so the template panel is exercised."""
    import healpy as hp
    import numpy as np

    from chatterbox.astro.templates import TemplateCoverage

    nside = 64
    maps = {}
    for band in ("r", "i"):
        m = np.zeros(hp.nside2npix(nside))
        m[hp.query_disc(nside, hp.ang2vec(60.0, -35.0, lonlat=True), np.radians(6.0), inclusive=True)] = 1.0
        maps[band] = m
    coverage = TemplateCoverage(
        maps=maps, nside=nside, built_at="2026-08-13T00:00:00+00:00", source="integration test"
    )
    coverage.save(config.templates.cache_dir)
    return coverage


def test_full_alert_path(record, config, offline_poster, template_cache, rubin_scheduler, tmp_path):
    report = process_trigger(
        record,
        config,
        poster=offline_poster,
        post=True,
        run_sim=False,
        out_dir=tmp_path / "plots",
    )

    # Both plots rendered.
    assert len(report.plots) == 2
    for plot in report.plots:
        assert plot.is_file()
        assert plot.stat().st_size > 10_000, "plot looks empty"

    # The post was produced and written offline.
    assert report.posted is not None
    assert report.posted.offline is True
    payload = json.loads((tmp_path / "posts" / f"{report.trigger.source}_post.json").read_text())
    assert payload["blocks"]
    assert len(payload["files"]) == 2

    # Statistics were computed.
    assert report.events.night_length_hours > 0
    assert 0.0 <= report.dark_stats["fraction_accessible"] <= 1.0
    assert report.template_coverage is not None
    assert "no template coverage cache" not in " ".join(report.warnings)


def test_stage1_meets_its_latency_budget(
    record, config, offline_poster, template_cache, rubin_scheduler, tmp_path
):
    report = process_trigger(
        record, config, poster=offline_poster, post=True, run_sim=False, out_dir=tmp_path / "plots"
    )
    assert (
        report.elapsed_s < STAGE1_BUDGET_S
    ), f"stage 1 took {report.elapsed_s:.1f} s, over the {STAGE1_BUDGET_S:.0f} s budget"


def test_missing_template_cache_still_posts(record, config, offline_poster, rubin_scheduler, tmp_path):
    """A missing cache must cost the template panel, not the whole alert."""
    report = process_trigger(
        record, config, poster=offline_poster, post=True, run_sim=False, out_dir=tmp_path / "plots"
    )
    assert report.posted is not None
    assert report.template_coverage is None
    assert len(report.plots) == 1  # dark hours only
    assert any("template coverage cache" in w for w in report.warnings)


@pytest.mark.parametrize("fixture_name", ["gw_case_b", "sn_galactic", "neutrino", "gw_case_d_test"])
def test_every_fixture_runs_end_to_end(
    fixture_name, data_dir, config, offline_poster, rubin_scheduler, tmp_path
):
    record = load_record_file(data_dir / f"{fixture_name}.json")
    report = process_trigger(
        record,
        config,
        poster=offline_poster,
        post=True,
        run_sim=False,
        out_dir=tmp_path / fixture_name,
    )
    assert report.posted is not None
    assert report.plots, f"{fixture_name} produced no plots"
    json.dumps(report.blocks)


def test_simulation_is_skipped_when_disabled(record, config, offline_poster, rubin_scheduler, tmp_path):
    config.sim.enabled = False
    report = process_trigger(
        record, config, poster=offline_poster, post=True, run_sim=True, out_dir=tmp_path / "plots"
    )
    assert report.posted is not None
    assert not (tmp_path / "sim").exists()


def test_missing_sim_interpreter_is_reported_not_raised(
    record, config, offline_poster, rubin_scheduler, tmp_path
):
    """A misconfigured interpreter must not take the alert path down."""
    from chatterbox.ingest.decode import decode_record
    from chatterbox.sim.runner import launch_simulation, load_sim_result

    config.sim.enabled = True
    config.sim.python = str(tmp_path / "no-such-python")
    trigger = decode_record(record)

    job = launch_simulation(trigger, config)
    assert job is not None
    result = load_sim_result(job.job_dir)
    assert result is not None
    assert result.ok is False
    assert "interpreter not found" in result.error


def test_run_service_over_replay_paths(config, data_dir, tmp_path, rubin_scheduler, monkeypatch):
    from chatterbox.app import run_service

    monkeypatch.delenv(config.slack.bot_token_env, raising=False)
    config.paths.work_dir = str(tmp_path / "work")
    paths = [data_dir / "neutrino.json", data_dir / "sn_galactic.json"]
    assert run_service(config, paths=paths, run_sim=False) == 2


def test_run_service_continues_past_a_bad_record(config, data_dir, tmp_path, rubin_scheduler, caplog):
    from chatterbox.app import run_service

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"source": "S1"}))  # missing required fields
    paths = [bad, data_dir / "neutrino.json"]
    with caplog.at_level("ERROR"):
        handled = run_service(config, paths=paths, run_sim=False)
    assert handled == 1
    assert "handling a ToO record" in caplog.text


def test_a_dropped_record_is_announced_in_the_channel(config, data_dir, tmp_path, rubin_scheduler):
    """A ToO nobody hears about is indistinguishable from no ToO at all."""
    from chatterbox.app import run_service

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"source": "S251112cm", "is_test": True}))
    assert run_service(config, paths=[bad], run_sim=False) == 0

    # No token in the test environment, so the poster is offline and writes the
    # payload it would have sent.
    payload = json.loads((tmp_path / "work" / "posts" / "S251112cm_failure.json").read_text())
    text = json.dumps(payload)
    assert "chatterbox failed while handling a ToO record" in text
    assert "S251112cm" in text
    assert "missing required field" in text, "the reason has to be in the message"


class BrokenSource:
    """A monitoring source that fails the way an EFD outage would."""

    def __init__(self):
        self.closed = False

    def __iter__(self):
        raise RuntimeError("5 consecutive EFD queries failed; last error: TimeoutError: ")
        yield  # pragma: no cover - unreachable, but makes this a generator

    def mark_done(self, metadata):
        pass

    def close(self):
        self.closed = True


def test_a_broken_stream_is_announced_before_the_service_stops(config, tmp_path, monkeypatch):
    """Nothing else would notice: the service just stops seeing alerts."""
    from chatterbox import app
    from chatterbox.ingest import source as source_module

    broken = BrokenSource()
    monkeypatch.setattr(source_module, "make_source", lambda *args, **kwargs: broken)
    config.ingest.kind = "efd"

    with pytest.raises(RuntimeError, match="consecutive EFD queries"):
        app.run_service(config, run_sim=False)

    assert broken.closed, "the source is closed even on the way out"
    payload = json.loads((tmp_path / "work" / "posts" / "chatterbox_failure.json").read_text())
    text = json.dumps(payload)
    assert "monitoring for ToO alerts (efd)" in text
    assert "consecutive EFD queries failed" in text


def test_serve_exits_non_zero_instead_of_tracebacking(config, tmp_path, monkeypatch):
    from chatterbox import app, cli

    def explode(*args, **kwargs):
        raise RuntimeError("influx is unreachable")

    monkeypatch.setattr(app, "run_service", explode)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"paths: {{work_dir: {tmp_path / 'work'}}}\n")
    assert cli.main(["-c", str(config_path), "serve"]) == 1


def test_serve_lookback_and_since_resolve_to_seconds():
    """`--since`/`--lookback` become the EFD first-poll lookback; None keeps the
    configured value. This is the knob that catches an already-injected alert."""
    from chatterbox import cli

    parser = cli.build_parser()

    assert cli._serve_lookback(parser.parse_args(["serve", "--lookback", "600"])) == 600.0
    # A past --since is a positive reach-back; the exact value tracks "now", so
    # only its sign and rough scale are asserted.
    since = cli._serve_lookback(parser.parse_args(["serve", "--since", "2000-01-01T00:00:00"]))
    assert since > 0
    assert cli._serve_lookback(parser.parse_args(["serve"])) is None


def test_serve_rejects_bad_lookback_options():
    from chatterbox import cli

    parser = cli.build_parser()

    with pytest.raises(ValueError, match="only one of"):
        cli._serve_lookback(parser.parse_args(["serve", "--since", "2000-01-01T00:00:00", "--lookback", "5"]))
    with pytest.raises(ValueError, match="future"):
        cli._serve_lookback(parser.parse_args(["serve", "--since", "2999-01-01T00:00:00"]))
    with pytest.raises(ValueError, match="negative"):
        cli._serve_lookback(parser.parse_args(["serve", "--lookback", "-5"]))


def test_serve_hands_the_lookback_override_to_run_service(tmp_path, monkeypatch):
    """A bad option must exit before the override ever reaches run_service."""
    from chatterbox import app, cli

    captured = {}

    def capture(config, run_sim=True, lookback_s=None, **kwargs):
        captured["lookback_s"] = lookback_s
        return 0

    monkeypatch.setattr(app, "run_service", capture)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"paths: {{work_dir: {tmp_path / 'work'}}}\n")

    assert cli.main(["-c", str(config_path), "serve", "--lookback", "1800"]) == 0
    assert captured["lookback_s"] == 1800.0

    assert cli.main(["-c", str(config_path), "serve"]) == 0
    assert captured["lookback_s"] is None

    # A future --since is caught in the CLI and never reaches run_service.
    captured.clear()
    assert cli.main(["-c", str(config_path), "serve", "--since", "2999-01-01T00:00:00"]) == 2
    assert captured == {}


def test_sim_job_spec_is_self_describing(record, config, tmp_path):
    """The job directory is the contract between the bot and the sim driver."""
    from chatterbox.ingest.decode import decode_record
    from chatterbox.sim.runner import write_job_spec

    trigger = decode_record(record)
    job_dir = tmp_path / "job"
    write_job_spec(trigger, config, job_dir, nights=20)

    spec = json.loads((job_dir / "job.json").read_text())
    assert spec["source"] == trigger.source
    assert spec["alert_type"] == trigger.alert_type
    assert spec["nights"] == 20
    assert spec["reward_map_nside"] == trigger.reward_map_nside
    assert spec["event_mjd"] == pytest.approx(trigger.event_time.mjd)
    # Paths the driver needs must be present rather than hardcoded there.
    for key in ("lsst_survey_sim", "ts_config_scheduler", "opsim_cache", "rubin_sim_data"):
        assert spec["sim"][key]
    assert (job_dir / "reward_map.npy").is_file()


def test_cli_replay_dry_run(data_dir, tmp_path, rubin_scheduler, monkeypatch, capsys):
    """The primary development loop: render everything, post nothing."""
    from chatterbox.cli import main

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        json.dumps(
            {
                "paths": {"work_dir": str(tmp_path / "work")},
                "templates": {"cache_dir": str(tmp_path / "templates")},
                "enrich": {"gracedb": False},
                "sim": {"enabled": False},
            }
        )
    )
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)

    code = main(
        [
            "-c",
            str(config_path),
            "replay",
            str(data_dir / "neutrino.json"),
            "--dry-run",
            "--no-sim",
            "--out-dir",
            str(tmp_path / "plots"),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "IceCube-260814A" in out
    assert "Accessible dark hours" in out
    assert "Stage 1 completed in" in out


# ------------------------------------------------- anchoring nights on the ToO


def test_day_obs_is_the_observing_night_not_the_utc_date(rubin_scheduler):
    """An alert after midnight UTC must not be pushed to the next evening.

    A Rubin observing day runs 12:00 UTC to 12:00 UTC. Using the trigger's UTC
    calendar date instead splits the Chilean night in half, so anything
    arriving between 00:00 and 12:00 UTC -- the back half of the night, when a
    fast response matters most -- named the following evening and delayed the
    whole simulated follow-up by a day.
    """
    from astropy.time import Time

    from chatterbox.sim.driver import day_obs_for

    # 02:00 UTC is 23:00 the previous evening in Chile: the same night.
    assert day_obs_for(Time("2025-11-12T02:00:00").mjd) == 20251111
    assert day_obs_for(Time("2025-11-12T11:59:00").mjd) == 20251111
    # ...and the boundary is 12:00 UTC, not local midnight.
    assert day_obs_for(Time("2025-11-12T12:01:00").mjd) == 20251112
    # S251112cm fired at 15:18 UTC, which was already correct.
    assert day_obs_for(Time("2025-11-12T15:18:45.362").mjd) == 20251112


def test_day_obs_matches_rubin_nights(rubin_scheduler):
    """The convention is upstream's; this must not drift from it."""
    from astropy.time import Time
    from rubin_nights import dayobs_utils

    from chatterbox.sim.driver import day_obs_for

    for iso in ("2025-11-12T02:00:00", "2025-11-12T15:18:45", "2026-01-01T00:30:00"):
        when = Time(iso)
        assert day_obs_for(when.mjd) == int(dayobs_utils.time_to_day_obs_int(when))


def test_the_trigger_night_uses_the_observatory_night_counter(rubin_scheduler):
    """The only definition that can agree with the ``night`` column.

    ``ModelObservatory`` sets ``observation["night"]`` to
    ``floor(mjd - mjd_start)`` -- a day count from a noon-UTC epoch, not an
    almanac sunset lookup. Anything else is off by one for part of the night.
    """
    from types import SimpleNamespace

    import numpy as np
    from astropy.time import Time
    from rubin_scheduler.utils import SURVEY_START_MJD

    from chatterbox.sim.driver import _trigger_night

    observatory = SimpleNamespace(mjd_start=SURVEY_START_MJD)
    event = Time("2025-11-12T15:18:45.362").mjd
    assert _trigger_night(observatory, event) == int(np.floor(event - SURVEY_START_MJD))
    assert _trigger_night(observatory, event) == 11

    # The whole Chilean night shares one number: 02:00 UTC on the 12th is the
    # back half of the night that began on the 11th.
    assert _trigger_night(observatory, Time("2025-11-12T02:00:00").mjd) == 10
    assert _trigger_night(observatory, Time("2025-11-12T23:30:00").mjd) == 11


def test_the_run_starts_on_the_trigger_night(rubin_scheduler):
    """day_obs and the night counter share the 12:00 UTC boundary.

    That is what makes "night +0" the trigger night: the observing day the
    simulation is told to start on is the one the trigger falls in.
    """
    from types import SimpleNamespace

    import numpy as np
    from astropy.time import Time
    from rubin_scheduler.utils import SURVEY_START_MJD

    from chatterbox.sim.driver import _trigger_night, day_obs_for

    observatory = SimpleNamespace(mjd_start=SURVEY_START_MJD)
    for iso in ("2025-11-12T15:18:45.362", "2025-11-12T02:00:00", "2026-01-01T00:30:00"):
        event = Time(iso).mjd
        # The night the run's own day_obs falls in, taken at its 12:00 UTC
        # start so it cannot land on the boundary.
        day_obs = str(day_obs_for(event))
        start = Time(f"{day_obs[:4]}-{day_obs[4:6]}-{day_obs[6:]}T12:00:00").mjd
        first_night = int(np.floor(start - SURVEY_START_MJD))
        assert first_night == _trigger_night(observatory, event), iso


def test_an_unreadable_observatory_leaves_nights_unanchored(caplog):
    """Better bare survey nights than a wrong anchor."""
    from types import SimpleNamespace

    from chatterbox.sim.driver import _trigger_night

    with caplog.at_level("WARNING"):
        assert _trigger_night(SimpleNamespace(), 60991.0) is None
    assert "which night the trigger fell on" in caplog.text
