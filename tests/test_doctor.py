"""Tests for dependency reporting and the sim-interpreter pre-flight.

These exist because the two most common deployment failures -- a package
installed into a different interpreter, and ``sim.python`` pointing at the
wrong environment -- both surfaced as cryptic errors far from their cause.
"""

import sys

import pytest

from chatterbox.config import Config
from chatterbox.deps import CAPABILITIES, probe, require
from chatterbox.doctor import diagnose, format_report
from chatterbox.sim.runner import check_sim_python, resolve_sim_python

# ------------------------------------------------------------------ deps


def test_require_returns_the_module():
    assert require("json").dumps({"a": 1}) == '{"a": 1}'


def test_require_names_the_interpreter():
    """A package installed under a different python is the usual cause."""
    with pytest.raises(ImportError) as excinfo:
        require("definitely_not_installed_xyz")

    message = str(excinfo.value)
    assert sys.executable in message
    assert "pip install" in message
    assert "different python" in message


def test_require_explains_the_ligo_namespace():
    """'No module named ligo' is cryptic; the message must do better."""
    capability = next(c for c in CAPABILITIES if c.name == "plots")
    assert capability.modules == ("ligo.skymap.plot",)
    assert capability.install == "ligo.skymap"
    # The consequence is stated so a reader can judge urgency.
    assert "plots" in capability.consequence


def test_every_capability_declares_what_breaks():
    for capability in CAPABILITIES:
        assert capability.purpose
        assert capability.consequence
        assert capability.install
        assert capability.modules


def test_probe_reports_the_failing_module():
    from chatterbox.deps import Capability

    ok, detail = probe(
        Capability(
            name="x",
            modules=("json", "definitely_not_installed_xyz"),
            install="x",
            purpose="p",
            consequence="c",
        )
    )
    assert ok is False
    assert "definitely_not_installed_xyz" in detail


def test_probe_succeeds_for_stdlib():
    from chatterbox.deps import Capability

    ok, detail = probe(
        Capability(name="x", modules=("json", "os"), install="x", purpose="p", consequence="c")
    )
    assert ok is True
    assert detail == ""


# ------------------------------------------------- sim interpreter resolution


def test_empty_sim_python_uses_the_running_interpreter():
    """The old default was a hardcoded path that was easy to get wrong."""
    assert resolve_sim_python("") == pytest.approx(resolve_sim_python(""))
    assert str(resolve_sim_python("")) == sys.executable
    assert str(resolve_sim_python("   ")) == sys.executable


def test_explicit_sim_python_is_honoured():
    assert str(resolve_sim_python("/opt/py/bin/python")) == "/opt/py/bin/python"


def test_sim_python_expands_user(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert resolve_sim_python("~/bin/python") == tmp_path / "bin" / "python"


def test_check_sim_python_passes_for_this_interpreter():
    """This interpreter runs the tests, so it has the driver's requirements."""
    ok, detail = check_sim_python(resolve_sim_python(""))
    assert ok is True, detail


def test_check_sim_python_rejects_an_interpreter_without_healpy(tmp_path):
    """The exact USDF failure: a bare conda python with no healpy."""
    stub = tmp_path / "python"
    stub.write_text(
        "#!/bin/sh\n"
        # Pretend to be a Python that has none of the requirements.
        'exec /usr/bin/env python3 -c "'
        "import sys;"
        "print('MISSING numpy (ModuleNotFoundError), healpy (ModuleNotFoundError)')"
        '"\n'
    )
    stub.chmod(0o755)

    ok, detail = check_sim_python(stub)
    assert ok is False
    assert "healpy" in detail


def test_check_sim_python_reports_a_nonexistent_interpreter(tmp_path):
    ok, detail = check_sim_python(tmp_path / "no-such-python")
    assert ok is False
    assert "could not run" in detail or "exited" in detail


# ------------------------------------------------------------------- doctor


@pytest.fixture
def doctor_config(tmp_path):
    config = Config()
    config.paths.work_dir = str(tmp_path / "work")
    config.templates.cache_dir = str(tmp_path / "templates")
    config.sim.opsim_cache = str(tmp_path / "opsim.parquet")
    config.sim.python = ""
    return config


def test_diagnose_covers_every_capability(doctor_config):
    checks = diagnose(doctor_config)
    names = " ".join(c.name for c in checks)
    for capability in CAPABILITIES:
        assert capability.name in names
    for expected in (
        "interpreter",
        "rubin_sim_data tree",
        "template coverage cache",
        "visit history cache",
        "simulation interpreter",
        "Slack token",
    ):
        assert expected in names


def test_diagnose_flags_a_missing_template_cache(doctor_config):
    checks = diagnose(doctor_config)
    check = next(c for c in checks if c.name == "template coverage cache")
    assert check.ok is False
    assert "refresh-templates" in check.fix


def test_diagnose_flags_a_missing_opsim_cache(doctor_config):
    checks = diagnose(doctor_config)
    check = next(c for c in checks if c.name == "visit history cache")
    assert check.ok is False
    assert "refresh-opsim" in check.fix


def test_diagnose_flags_a_bad_sim_interpreter(doctor_config, tmp_path):
    doctor_config.sim.python = str(tmp_path / "absent-python")
    checks = diagnose(doctor_config)
    check = next(c for c in checks if c.name == "simulation interpreter")
    assert check.ok is False
    assert "does not exist" in check.detail
    assert "leave it empty" in check.fix


def test_diagnose_reports_the_data_dir_it_would_use(doctor_config, monkeypatch, tmp_path):
    tree = tmp_path / "rubin_sim_data"
    (tree / "site_models").mkdir(parents=True)
    monkeypatch.setenv("RUBIN_SIM_DATA_DIR", str(tree))

    checks = diagnose(doctor_config)
    check = next(c for c in checks if c.name == "rubin_sim_data tree")
    assert check.ok is True
    assert str(tree) in check.detail


def test_diagnose_flags_a_missing_site_models(doctor_config, monkeypatch, tmp_path):
    tree = tmp_path / "tree"
    tree.mkdir()
    monkeypatch.setenv("RUBIN_SIM_DATA_DIR", str(tree))

    checks = diagnose(doctor_config)
    check = next(c for c in checks if c.name == "rubin_sim_data tree")
    assert check.ok is False
    assert check.fatal is True
    assert "site_models" in check.fix


def test_report_shows_fixes_only_for_failures(doctor_config):
    checks = diagnose(doctor_config)
    report = format_report(checks)
    for line in report.splitlines():
        if "fix:" in line:
            assert "[  ok]" not in line
    assert "chatterbox environment" in report


def test_report_summarises(doctor_config):
    report = format_report(diagnose(doctor_config))
    assert "degraded" in report or "Everything checks out" in report


def test_cli_doctor_exit_code(tmp_path, monkeypatch, capsys):
    """Degraded capabilities are not a failure; a broken almanac is."""
    import json

    from chatterbox.cli import main

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        json.dumps(
            {
                "paths": {"work_dir": str(tmp_path / "work")},
                "templates": {"cache_dir": str(tmp_path / "templates")},
                "sim": {"opsim_cache": str(tmp_path / "opsim.parquet"), "python": ""},
            }
        )
    )
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)

    code = main(["-c", str(config_path), "doctor"])
    out = capsys.readouterr().out
    assert "chatterbox environment" in out
    # Missing caches and no Slack token are warnings, not blocking problems.
    assert code == 0
