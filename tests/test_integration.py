"""End-to-end tests for the alert path, including its latency budget."""

import json

import pandas as pd
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
    from chatterbox.astro.templates import build_template_maps

    visits = pd.DataFrame(
        [
            {"s_ra": 60.0, "s_dec": -35.0, "band": "i"},
            {"s_ra": 61.0, "s_dec": -35.5, "band": "i"},
            {"s_ra": 60.5, "s_dec": -34.5, "band": "r"},
        ]
    )
    coverage = build_template_maps(visits, nside=64, source="integration test")
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
    assert "Failed to handle a record" in caplog.text


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
    for key in ("lsst_survey_sim", "ts_config_scheduler", "opsim_h5", "rubin_sim_data"):
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
