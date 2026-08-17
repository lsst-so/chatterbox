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


# ------------------------------------------------------- RUBIN_SIM_DATA_DIR


@pytest.fixture(autouse=True)
def _reset_data_dir_memo():
    """Clear the once-per-path diagnostic memo between tests."""
    import chatterbox.config as config_mod

    config_mod._validated_data_dir = None
    yield
    config_mod._validated_data_dir = None


def test_apply_environment_exports_the_configured_data_dir(tmp_path, monkeypatch):
    """rubin_scheduler silently falls back to $HOME/rubin_sim_data otherwise.

    That fallback is what made a configured sim.rubin_sim_data look ignored on
    the alert path, which runs in-process rather than in the sim subprocess.
    """
    from chatterbox.config import apply_environment

    tree = tmp_path / "rubin_sim_data"
    (tree / "site_models").mkdir(parents=True)
    monkeypatch.delenv("RUBIN_SIM_DATA_DIR", raising=False)

    config = Config()
    config.sim.rubin_sim_data = str(tree)
    apply_environment(config)

    import os

    assert os.environ["RUBIN_SIM_DATA_DIR"] == str(tree)


def test_apply_environment_expands_user(tmp_path, monkeypatch):
    from chatterbox.config import apply_environment

    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "data" / "site_models").mkdir(parents=True)
    monkeypatch.delenv("RUBIN_SIM_DATA_DIR", raising=False)

    config = Config()
    config.sim.rubin_sim_data = "~/data"
    apply_environment(config)

    import os

    assert os.environ["RUBIN_SIM_DATA_DIR"] == str(tmp_path / "data")


def test_config_wins_over_an_inherited_env_var(tmp_path, monkeypatch, caplog):
    """A stale shell variable must not quietly override the config file."""
    from chatterbox.config import apply_environment

    tree = tmp_path / "configured"
    (tree / "site_models").mkdir(parents=True)
    monkeypatch.setenv("RUBIN_SIM_DATA_DIR", "/somewhere/stale")

    config = Config()
    config.sim.rubin_sim_data = str(tree)
    with caplog.at_level("INFO"):
        apply_environment(config)

    import os

    assert os.environ["RUBIN_SIM_DATA_DIR"] == str(tree)
    assert "Overriding inherited" in caplog.text


def test_empty_setting_leaves_the_environment_alone(monkeypatch):
    from chatterbox.config import apply_environment

    monkeypatch.setenv("RUBIN_SIM_DATA_DIR", "/inherited")
    config = Config()
    config.sim.rubin_sim_data = ""
    apply_environment(config)

    import os

    assert os.environ["RUBIN_SIM_DATA_DIR"] == "/inherited"


def test_missing_tree_is_reported_up_front(tmp_path, monkeypatch, caplog):
    """Better to say so at startup than several seconds into an alert."""
    from chatterbox.config import apply_environment

    monkeypatch.delenv("RUBIN_SIM_DATA_DIR", raising=False)
    config = Config()
    config.sim.rubin_sim_data = str(tmp_path / "absent")
    with caplog.at_level("ERROR"):
        apply_environment(config)
    assert "does not exist" in caplog.text
    assert "scheduler_download_data" in caplog.text


def test_missing_site_models_is_reported(tmp_path, monkeypatch, caplog):
    from chatterbox.config import apply_environment

    tree = tmp_path / "tree"
    tree.mkdir()
    monkeypatch.delenv("RUBIN_SIM_DATA_DIR", raising=False)
    config = Config()
    config.sim.rubin_sim_data = str(tree)
    with caplog.at_level("ERROR"):
        apply_environment(config)
    assert "site_models" in caplog.text


def test_diagnostics_are_not_repeated_per_call(tmp_path, monkeypatch, caplog):
    """apply_environment runs per alert; a bad path must not spam the log."""
    from chatterbox.config import apply_environment

    monkeypatch.delenv("RUBIN_SIM_DATA_DIR", raising=False)
    config = Config()
    config.sim.rubin_sim_data = str(tmp_path / "absent")
    with caplog.at_level("ERROR"):
        apply_environment(config)
        apply_environment(config)
    assert caplog.text.count("does not exist") == 1


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


def test_the_default_monitoring_path_is_the_efd():
    """Changed deliberately: watching a directory needs a file sender."""
    from chatterbox.ingest.efd import EFD_DATABASE, EFD_TOPIC, EfdTooAlertSource

    config = Config()
    assert config.ingest.kind == "efd"
    source = make_source(config)
    assert isinstance(source, EfdTooAlertSource)
    assert source.topic == EFD_TOPIC
    assert source.database == EFD_DATABASE
    assert source.lookback_s == 0.0, "a restart must not re-post recent alerts"


def test_efd_settings_reach_the_source():
    from chatterbox.ingest.efd import EfdTooAlertSource

    config = Config()
    config.ingest.efd_name = "usdf_efd"
    config.ingest.efd_topic = "lsst.scimma.too_alert_test"
    config.ingest.efd_poll_interval_s = 42.0
    config.ingest.efd_lookback_s = 300.0
    config.ingest.efd_revisit_s = 600.0
    source = make_source(config)
    assert isinstance(source, EfdTooAlertSource)
    assert source.efd_name == "usdf_efd"
    assert source.topic == "lsst.scimma.too_alert_test"
    assert source.poll_interval_s == 42.0
    assert source.lookback_s == 300.0
    assert source.revisit_s == 600.0, "the late-alert re-read window reaches the source"


def test_a_lookback_override_reaches_the_efd_source():
    """`serve --since/--lookback` must beat the configured lookback.

    With ingest.efd_lookback_s at its safe 0.0 default, serve only sees alerts
    that arrive after it starts; an alert injected just before is missed. The
    override is how an operator reaches back to catch it.
    """
    from chatterbox.ingest.efd import EfdTooAlertSource

    config = Config()
    assert config.ingest.efd_lookback_s == 0.0
    source = make_source(config, lookback_s=3600.0)
    assert isinstance(source, EfdTooAlertSource)
    assert source.lookback_s == 3600.0, "the override wins over the configured value"

    # None leaves the configured value untouched.
    assert make_source(config).lookback_s == 0.0


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


# ------------------------------------------------------------------ site


def test_site_fills_every_setting_that_names_a_site(tmp_path):
    """One line instead of three, which is three chances to disagree."""
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"site": "summit"}))
    config = load_config(path)

    assert config.site == "summit"
    assert config.ingest.efd_name == "summit_efd"
    assert config.sim.opsim_site == "summit"
    assert config.sim.opsim_tokenfile == "~/.lsst/summit_rsp"


def test_an_explicit_setting_beats_the_site(tmp_path, caplog):
    """Overriding one piece of a site is normal; it must not be reverted."""
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"site": "summit", "ingest": {"efd_name": "usdf_efd"}}))
    with caplog.at_level("INFO"):
        config = load_config(path)

    assert config.ingest.efd_name == "usdf_efd", "the file is the more deliberate statement"
    # ...and the rest of the site still applies.
    assert config.sim.opsim_site == "summit"
    assert "overridden by the config file" in caplog.text


def test_the_dashed_spelling_counts_as_explicit(tmp_path):
    """The loader accepts dashes, so the override check has to as well."""
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"site": "usdf", "sim": {"opsim-tokenfile": "~/token"}}))
    config = load_config(path)
    assert config.sim.opsim_tokenfile == "~/token"
    assert config.ingest.efd_name == "usdf_efd"


def test_an_unknown_site_is_an_error_not_a_silent_no_op(tmp_path):
    """A typo would otherwise leave the instance pointed somewhere else."""
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"site": "sumit"}))
    with pytest.raises(ValueError, match="Unknown site 'sumit'"):
        load_config(path)


def test_the_error_lists_the_sites_and_the_way_out(tmp_path):
    from chatterbox.config import apply_site_defaults

    config = Config()
    config.site = "idf"
    with pytest.raises(ValueError) as caught:
        apply_site_defaults(config)
    message = str(caught.value)
    for site in ("summit", "base", "usdf", "usdf-dev"):
        assert site in message
    assert "ingest.efd_name" in message, "idf_efd is still reachable, just not as a site"


def test_no_site_changes_nothing(tmp_path):
    """An existing config file must behave exactly as it did before."""
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"ingest": {"kind": "efd"}}))
    config = load_config(path)

    assert config.site == ""
    assert config.ingest.efd_name == Config().ingest.efd_name
    assert config.sim.opsim_site == Config().sim.opsim_site


def test_site_is_case_and_space_insensitive(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"site": " USDF-Dev "}))
    config = load_config(path)
    assert config.site == "usdf-dev"
    assert config.sim.opsim_site == "usdf-dev"
    assert config.ingest.efd_name == "usdf_efd"


def test_site_reaches_the_efd_source(tmp_path):
    """The point of the setting: the poller ends up at the right instance."""
    from chatterbox.ingest.efd import EfdTooAlertSource

    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"site": "base"}))
    source = make_source(load_config(path))
    assert isinstance(source, EfdTooAlertSource)
    assert source.efd_name == "base_efd"
    assert "base_efd" in source.describe()


def test_every_site_names_all_three_settings():
    """A partial entry would leave one service pointed at another site."""
    from chatterbox.config import _SITE_TARGETS, SITES

    wanted = {key for _, key in _SITE_TARGETS}
    for name, values in SITES.items():
        assert set(values) == wanted, f"{name} is missing {wanted - set(values)}"
