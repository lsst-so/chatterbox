"""Tests for configuration loading and the ingest transports."""

import json

import pytest
import yaml
from conftest import make_record

from chatterbox.config import Config, load_config
from chatterbox.ingest.source import FileTooAlertSource, KafkaTooAlertSource, ReplaySource, make_source

# ---------------------------------------------------------------------- config


def test_defaults_are_usable():
    config = Config()
    assert config.slack.bot_token_env == "SLACK_BOT_TOKEN"
    assert config.dark_hours.airmass_limit == 2.0
    assert config.dark_hours.sun_alt_limit_deg == -12.0
    assert config.templates.bands == ["u", "g", "r", "i", "z", "y"]
    # No Kafka topic may be assumed: topics are deployment configuration.
    assert config.ingest.kafka_url == ""


def test_load_yaml_overrides(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "slack": {"channel": "#alerts", "mention": ["!here"]},
                "dark_hours": {"nside": 128, "airmass_limit": 1.5},
                "links": {"weather": "https://example.invalid/weather", "extra": {"Wiki": "https://w"}},
            }
        )
    )
    config = load_config(path)
    assert config.slack.channel == "#alerts"
    assert config.slack.mention == ["!here"]
    assert config.dark_hours.nside == 128
    assert config.dark_hours.airmass_limit == 1.5
    assert config.links.extra == {"Wiki": "https://w"}
    # Untouched settings keep their defaults.
    assert config.dark_hours.sun_alt_limit_deg == -12.0


def test_dashed_keys_are_accepted(tmp_path):
    """forward_alerts.py's own config uses dashes, so accept that spelling."""
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"dark-hours": {"step-minutes": 2.0}}))
    assert load_config(path).dark_hours.step_minutes == 2.0


def test_unknown_keys_are_ignored_with_a_warning(tmp_path, caplog):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"slack": {"chanel": "#typo"}}))
    with caplog.at_level("WARNING"):
        config = load_config(path)
    assert config.slack.channel == Config().slack.channel
    assert "chanel" in caplog.text


def test_missing_explicit_config_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "absent.yaml")


def test_empty_config_file_yields_defaults(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("")
    assert load_config(path).slack.channel == Config().slack.channel


def test_slack_token_comes_from_the_environment(monkeypatch):
    config = Config()
    monkeypatch.delenv(config.slack.bot_token_env, raising=False)
    assert config.slack_token is None
    monkeypatch.setenv(config.slack.bot_token_env, "xoxb-test")
    assert config.slack_token == "xoxb-test"


def test_empty_token_is_treated_as_absent(monkeypatch):
    config = Config()
    monkeypatch.setenv(config.slack.bot_token_env, "")
    assert config.slack_token is None


def test_work_path_creates_parents(tmp_path):
    config = Config()
    config.paths.work_dir = str(tmp_path / "work")
    path = config.work_path("plots", "S1", "figure.png")
    assert path.parent.is_dir()


# ---------------------------------------------------------------------- ingest


def test_replay_source_yields_records(tmp_path):
    paths = []
    for n in range(2):
        path = tmp_path / f"rec{n}.json"
        path.write_text(json.dumps(make_record(source=f"S{n}")))
        paths.append(path)

    got = list(ReplaySource(paths))
    assert [record["source"] for record, _ in got] == ["S0", "S1"]
    assert all(meta["transport"] == "replay" for _, meta in got)


def test_file_source_reads_existing_files(tmp_path):
    (tmp_path / "S1.json").write_text(json.dumps(make_record(source="S1")))
    source = FileTooAlertSource(str(tmp_path), once=True)
    records = [record for record, _ in source]
    assert [r["source"] for r in records] == ["S1"]


def test_file_source_does_not_reprocess_unchanged_files(tmp_path):
    (tmp_path / "S1.json").write_text(json.dumps(make_record(source="S1")))
    source = FileTooAlertSource(str(tmp_path), once=True)
    assert len(list(source)) == 1
    # Iterating again must not re-yield the same file.
    assert len(list(source)) == 0


def test_file_source_reprocesses_a_rewritten_file(tmp_path):
    """The producer's FileSender overwrites {source}.json on an update."""
    path = tmp_path / "S1.json"
    path.write_text(json.dumps(make_record(source="S1")))
    source = FileTooAlertSource(str(tmp_path), once=True)
    assert len(list(source)) == 1

    # A rewrite with different content changes size and mtime.
    path.write_text(json.dumps(make_record(source="S1", alert_type="GW_case_D")) + " ")
    records = [record for record, _ in source]
    assert len(records) == 1
    assert records[0]["alert_type"] == "GW_case_D"


def test_file_source_ignores_other_suffixes(tmp_path):
    (tmp_path / "notes.txt").write_text("hello")
    (tmp_path / "map.fits").write_bytes(b"")
    assert len(list(FileTooAlertSource(str(tmp_path), once=True))) == 0


def test_file_source_skips_unreadable_records(tmp_path, caplog):
    (tmp_path / "broken.json").write_text("{not json")
    (tmp_path / "good.json").write_text(json.dumps(make_record(source="S2")))
    with caplog.at_level("ERROR"):
        records = [record for record, _ in FileTooAlertSource(str(tmp_path), once=True)]
    assert [r["source"] for r in records] == ["S2"]
    assert "broken.json" in caplog.text


def test_file_source_tolerates_a_missing_directory(tmp_path):
    source = FileTooAlertSource(str(tmp_path / "not-yet"), once=True)
    assert list(source) == []


def test_kafka_source_requires_a_url():
    with pytest.raises(ValueError, match="No Kafka URL"):
        KafkaTooAlertSource(url="")


def test_make_source_selects_by_kind(tmp_path):
    config = Config()
    config.ingest.kind = "files"
    config.ingest.watch_dir = str(tmp_path)
    assert isinstance(make_source(config), FileTooAlertSource)

    config.ingest.kind = "kafka"
    config.ingest.kafka_url = "kafka://broker/topic"
    assert isinstance(make_source(config), KafkaTooAlertSource)

    config.ingest.kind = "replay"
    with pytest.raises(ValueError, match="explicit paths"):
        make_source(config)

    config.ingest.kind = "carrier-pigeon"
    with pytest.raises(ValueError, match="Unknown ingest.kind"):
        make_source(config)


def test_explicit_paths_force_replay(tmp_path):
    config = Config()
    config.ingest.kind = "kafka"
    config.ingest.kafka_url = "kafka://broker/topic"
    path = tmp_path / "r.json"
    path.write_text(json.dumps(make_record()))
    assert isinstance(make_source(config, paths=[path]), ReplaySource)


def test_unwrap_handles_hop_message_shapes():
    from chatterbox.ingest.source import _unwrap

    record = make_record()

    class Blob:
        def __init__(self, content):
            self.content = content

    assert _unwrap(Blob(record))[0]["source"] == record["source"]
    assert _unwrap(Blob([record, record]))[0]["source"] == record["source"]
    assert _unwrap(Blob(json.dumps(record).encode()))[0]["source"] == record["source"]
    assert _unwrap(Blob(json.dumps(record)))[0]["source"] == record["source"]
    assert _unwrap(Blob(12345)) == []
