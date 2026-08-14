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


# --------------------------------------------------------------- checkouts


def test_import_root_prefers_the_lsst_python_subdirectory(tmp_path):
    """ts_fbs_utils imports from <checkout>/python, not <checkout>."""
    from chatterbox.deps import import_root

    checkout = tmp_path / "ts_fbs_utils"
    (checkout / "python" / "lsst" / "ts" / "fbs" / "utils").mkdir(parents=True)
    assert import_root(checkout) == checkout / "python"


def test_import_root_uses_the_checkout_for_a_flat_layout(tmp_path):
    """lsst_survey_sim imports from <checkout> itself."""
    from chatterbox.deps import import_root

    checkout = tmp_path / "lsst_survey_sim"
    (checkout / "lsst_survey_sim").mkdir(parents=True)
    assert import_root(checkout) == checkout


def test_import_root_ignores_empty_and_missing(tmp_path):
    from chatterbox.deps import import_root

    assert import_root("") is None
    assert import_root(None) is None
    assert import_root("   ") is None
    assert import_root(tmp_path / "absent") is None


def test_import_root_expands_user(tmp_path, monkeypatch):
    from chatterbox.deps import import_root

    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "Rubin-Observatory" / "pkg").mkdir(parents=True)
    assert import_root("~/Rubin-Observatory/pkg") == tmp_path / "Rubin-Observatory" / "pkg"


def test_add_checkout_makes_a_namespaced_package_importable(tmp_path, monkeypatch):
    """The whole point: a clone on sys.path must satisfy a deep import.

    The real checkout nests the package under ``python/`` with no
    ``__init__.py`` at the ``lsst``, ``lsst/ts`` or ``lsst/ts/fbs`` levels --
    native namespace packages, which is what lets them merge with an existing
    ``lsst``. A unique top-level name is used here so the test exercises that
    mechanism without colliding with a real installed copy.
    """
    import importlib
    import sys

    from chatterbox.deps import add_checkout

    monkeypatch.setattr(sys, "path", list(sys.path))
    root = tmp_path / "clone" / "python"
    pkg = root / "cbxns" / "ts" / "fbs" / "utils" / "maintel"
    pkg.mkdir(parents=True)
    # Only the leaf packages get __init__.py, mirroring the real layout.
    (root / "cbxns" / "ts" / "fbs" / "utils" / "__init__.py").write_text("")
    (pkg / "__init__.py").write_text("")
    (pkg / "too_surveys.py").write_text("MARKER = 'from the checkout'\n")

    added = add_checkout(tmp_path / "clone", "sim.ts_fbs_utils")
    assert added == root

    importlib.invalidate_caches()
    module = importlib.import_module("cbxns.ts.fbs.utils.maintel.too_surveys")
    assert module.MARKER == "from the checkout"

    for name in [n for n in list(sys.modules) if n.split(".")[0] == "cbxns"]:
        del sys.modules[name]


def test_add_checkout_does_not_shadow_an_imported_copy(tmp_path, monkeypatch):
    """A regular package already imported keeps winning; doctor says which."""
    import sys

    from chatterbox.deps import add_checkout

    monkeypatch.setattr(sys, "path", list(sys.path))
    checkout = tmp_path / "clone"
    (checkout / "json").mkdir(parents=True)
    (checkout / "json" / "__init__.py").write_text("MARKER = 'shadowed'\n")

    add_checkout(checkout)
    import json

    # stdlib json was imported long ago, so the checkout cannot displace it.
    assert not hasattr(json, "MARKER")


def test_add_checkout_is_idempotent(tmp_path, monkeypatch):
    import sys

    from chatterbox.deps import add_checkout

    monkeypatch.setattr(sys, "path", list(sys.path))
    checkout = tmp_path / "pkg"
    (checkout / "pkg").mkdir(parents=True)

    add_checkout(checkout)
    add_checkout(checkout)
    assert sys.path.count(str(checkout)) == 1


def test_add_checkout_warns_about_a_bad_path(tmp_path, caplog):
    from chatterbox.deps import add_checkout

    with caplog.at_level("WARNING"):
        assert add_checkout(tmp_path / "absent", "sim.ts_fbs_utils") is None
    assert "not a directory" in caplog.text


def test_apply_environment_adds_both_checkouts(tmp_path, monkeypatch):
    """Configured clones must reach every entry point, not just the sim."""
    import sys

    from chatterbox.config import apply_environment

    monkeypatch.setattr(sys, "path", list(sys.path))
    survey = tmp_path / "lsst_survey_sim"
    (survey / "lsst_survey_sim").mkdir(parents=True)
    fbs = tmp_path / "ts_fbs_utils"
    (fbs / "python").mkdir(parents=True)

    config = Config()
    config.sim.lsst_survey_sim = str(survey)
    config.sim.ts_fbs_utils = str(fbs)
    config.sim.rubin_sim_data = ""
    apply_environment(config)

    assert str(survey) in sys.path
    assert str(fbs / "python") in sys.path


def test_doctor_advises_the_setting_not_pip_for_checkouts(tmp_path):
    """ "pip install a checkout" was useless advice."""
    from chatterbox.deps import CAPABILITIES

    for capability in CAPABILITIES:
        if capability.name in ("simulation", "strategy"):
            assert capability.checkout_setting.startswith("sim.")
        else:
            assert capability.checkout_setting == ""


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
