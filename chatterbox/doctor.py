"""Environment diagnosis: what works, what does not, and how to fix it.

chatterbox degrades instead of dying when an optional package is missing, which
is right for a service but means a broken deployment shows up as a scatter of
warnings across a live alert. ``chatterbox doctor`` collects the same
information up front, in one place, naming the interpreter each check ran in --
because a package installed into a different python is the usual cause.
"""

import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .config import Config
from .deps import CAPABILITIES, probe

__all__ = ["Check", "diagnose", "format_report"]

logger = logging.getLogger(__name__)

#: Modules the simulation driver imports before it can do anything.
DRIVER_REQUIREMENTS = ("numpy", "pandas", "healpy", "astropy", "rubin_scheduler")

#: chatterbox uses ``X | Y`` annotations evaluated at runtime, so this is a
#: hard floor rather than a style preference.
MIN_PYTHON = (3, 11)


@dataclass
class Check:
    """One diagnostic result."""

    name: str
    ok: bool
    detail: str
    #: What to do about it. Empty when ok.
    fix: str = ""
    #: False when a failure is tolerable, so the summary can rank it.
    fatal: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        """``ok``, ``FAIL`` or ``warn``."""
        if self.ok:
            return "ok"
        return "FAIL" if self.fatal else "warn"


def _probe_interpreter(python: Path, modules) -> tuple[bool, str]:
    """Ask another interpreter whether it can import a set of modules.

    The simulation runs as a subprocess under ``sim.python``, which is often a
    different environment from the bot's, so its imports have to be checked
    there rather than here.
    """
    script = (
        "import sys, importlib\n"
        f"v = sys.version_info\n"
        f"if v[:2] < {MIN_PYTHON}:\n"
        "    print(f'PYTHON_TOO_OLD {v.major}.{v.minor}')\n"
        "missing = []\n"
        f"for m in {list(modules)!r}:\n"
        "    try:\n"
        "        importlib.import_module(m)\n"
        "    except Exception as exc:\n"
        "        missing.append(f'{m} ({type(exc).__name__})')\n"
        "print('MISSING ' + ', '.join(missing) if missing else 'OK')\n"
    )
    try:
        completed = subprocess.run(
            [str(python), "-c", script],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"could not run {python}: {exc}"

    out = (completed.stdout or "").strip()
    if completed.returncode != 0:
        return False, f"{python} exited {completed.returncode}: {(completed.stderr or out).strip()[:300]}"

    problems = [line for line in out.splitlines() if line.startswith(("PYTHON_TOO_OLD", "MISSING"))]
    if not problems:
        return True, ""
    detail = []
    for line in problems:
        if line.startswith("PYTHON_TOO_OLD"):
            version = line.split(maxsplit=1)[-1]
            detail.append(f"Python {version}, but chatterbox needs >= {'.'.join(map(str, MIN_PYTHON))}")
        else:
            detail.append("cannot import " + line.removeprefix("MISSING ").strip())
    return False, "; ".join(detail)


def _capability_checks(config: Config) -> list[Check]:
    """Probe every optional capability in this interpreter.

    Runs after `chatterbox.config.apply_environment`, so configured checkouts
    are already on ``sys.path`` and a capability supplied by one reports as
    available rather than missing.
    """
    checks = []
    for capability in CAPABILITIES:
        ok, detail = probe(capability)
        fix = ""
        notes = []
        if not ok:
            notes.append(f"Without it: {capability.consequence}.")
            if capability.checkout_setting:
                # These are normally used from a clone, so "pip install" is the
                # wrong first suggestion.
                section, _, key = capability.checkout_setting.partition(".")
                current = getattr(getattr(config, section), key, "")
                where = f" (currently {current!r})" if current else " (currently unset)"
                fix = (
                    f"Set {capability.checkout_setting} to your {capability.install} "
                    f"checkout{where}, or pip install it into {sys.executable}"
                )
            else:
                fix = f"{sys.executable} -m pip install {capability.install}"
        elif capability.checkout_setting:
            root = _capability_source(capability)
            if root:
                notes.append(f"Loaded from {root}.")
        checks.append(
            Check(
                name=f"{capability.name}: {capability.purpose}",
                ok=ok,
                detail=detail or "importable",
                fix=fix,
                fatal=capability.name == "almanac",
                notes=notes,
            )
        )
    return checks


def _capability_source(capability) -> str:
    """Where a capability's module was imported from, for the report.

    Worth showing: it distinguishes "picked up my checkout" from "found some
    other copy", which is otherwise invisible.
    """
    import importlib

    try:
        module = importlib.import_module(capability.modules[0])
    except Exception:
        return ""
    return getattr(module, "__file__", "") or ""


def _data_dir_check(config: Config) -> Check:
    """Is rubin_scheduler's data where the almanac needs it?"""
    resolved = os.environ.get("RUBIN_SIM_DATA_DIR")
    if resolved:
        where = f"RUBIN_SIM_DATA_DIR={resolved}"
    else:
        resolved = str(Path.home() / "rubin_sim_data")
        where = f"{resolved} (default, RUBIN_SIM_DATA_DIR unset)"
    path = Path(resolved).expanduser()

    if not path.is_dir():
        return Check(
            name="rubin_sim_data tree",
            ok=False,
            detail=f"{where} does not exist",
            fix="Set sim.rubin_sim_data, or run 'scheduler_download_data'",
            fatal=True,
        )
    if not (path / "site_models").is_dir():
        return Check(
            name="rubin_sim_data tree",
            ok=False,
            detail=f"{where} has no site_models/",
            fix="scheduler_download_data --dirs site_models",
            fatal=True,
        )
    return Check(name="rubin_sim_data tree", ok=True, detail=where)


def _template_cache_check(config: Config) -> Check:
    """Is the per-band template coverage cache present and readable?"""
    from .astro.templates import load_template_maps

    cache_dir = Path(config.templates.cache_dir).expanduser()
    coverage = load_template_maps(cache_dir)
    if coverage is None:
        return Check(
            name="template coverage cache",
            ok=False,
            detail=f"absent or unreadable at {cache_dir}",
            fix="chatterbox refresh-templates",
            notes=["Without it: posts omit the template comparison."],
        )
    detail = f"{','.join(coverage.bands)} at nside {coverage.nside}, refreshed {coverage.built_at}"
    notes = []
    if coverage.missing_bands:
        notes.append(f"No map published for: {', '.join(coverage.missing_bands)}.")
    return Check(name="template coverage cache", ok=True, detail=detail, notes=notes)


def _opsim_cache_check(config: Config) -> Check:
    """Is the visit history cached, and how old is it?"""
    import json

    cache = Path(config.sim.opsim_cache).expanduser()
    meta = cache.with_suffix(cache.suffix + ".meta.json")
    if not (cache.is_file() and meta.is_file()):
        return Check(
            name="visit history cache",
            ok=False,
            detail=f"absent at {cache}",
            fix="chatterbox refresh-opsim",
            notes=["It is also fetched on demand when a simulation runs."],
        )
    try:
        info = json.loads(meta.read_text())
        from .sim.opsim import OpsimCache

        described = OpsimCache(
            path=cache,
            day_obs=int(info.get("day_obs", 0)),
            fetched_at=info.get("fetched_at", ""),
            n_visits=int(info.get("n_visits", 0)),
        ).describe()
    except Exception as exc:
        return Check(
            name="visit history cache",
            ok=False,
            detail=f"unreadable metadata: {exc}",
            fix="chatterbox refresh-opsim --force",
        )
    return Check(name="visit history cache", ok=True, detail=described)


def _sim_interpreter_check(config: Config) -> Check:
    """Can ``sim.python`` actually run the driver?

    This is the check that turns "traceback in a log file you have to go find"
    into something visible before an alert arrives.
    """
    from .sim.runner import resolve_sim_python

    python = resolve_sim_python(config.sim.python)
    if not python.is_file():
        return Check(
            name="simulation interpreter",
            ok=False,
            detail=f"sim.python={python} does not exist",
            fix="Point sim.python at an interpreter with healpy and rubin_scheduler, "
            "or leave it empty to use the one running chatterbox",
            notes=["Without it: no per-band coverage reply."],
        )

    modules = list(DRIVER_REQUIREMENTS)
    ok, detail = _probe_interpreter(python, modules)
    if ok:
        return Check(name="simulation interpreter", ok=True, detail=str(python))
    return Check(
        name="simulation interpreter",
        ok=False,
        detail=f"{python}: {detail}",
        fix=f"Install the missing packages into {python}, or set sim.python to an "
        "interpreter that already has them (leaving it empty uses "
        f"{sys.executable})",
        notes=["Without it: the simulation subprocess dies before it starts."],
    )


def _slack_token_check(config: Config) -> Check:
    """Is a bot token available?"""
    if config.slack_token:
        return Check(
            name="Slack token",
            ok=True,
            detail=f"{config.slack.bot_token_env} is set; posting to {config.slack.channel}",
        )
    return Check(
        name="Slack token",
        ok=False,
        detail=f"{config.slack.bot_token_env} is not set",
        fix=f"export {config.slack.bot_token_env}=xoxb-...",
        notes=["Without it: payloads and plots are written locally instead of posted."],
    )


def _site_check(config: Config) -> Check:
    """Which site's services this instance is pointed at.

    Three settings name the same site in three vocabularies, so they are shown
    together: a config that polls one site's EFD while pulling visit history
    from another works perfectly well and is still wrong.
    """
    where = [
        f"EFD {config.ingest.efd_name or 'host default'}",
        f"ConsDB {config.sim.opsim_site}",
        f"token {config.sim.opsim_tokenfile}",
    ]
    if config.site:
        return Check(name="site", ok=True, detail=f"{config.site}: " + ", ".join(where))
    return Check(
        name="site",
        ok=True,
        detail="not set; " + ", ".join(where),
        notes=[
            "Set the top-level 'site' (summit, base, usdf, usdf-dev) to fill all " "three from one place.",
        ],
    )


def diagnose(config: Config) -> list[Check]:
    """Run every diagnostic and return the results in report order."""
    checks: list[Check] = [
        Check(
            name="interpreter",
            ok=sys.version_info[:2] >= MIN_PYTHON,
            detail=f"{sys.executable} (Python {'.'.join(map(str, sys.version_info[:3]))})",
            fix=f"chatterbox needs Python >= {'.'.join(map(str, MIN_PYTHON))}",
            fatal=True,
        )
    ]
    checks.append(_site_check(config))
    checks += _capability_checks(config)
    checks.append(_data_dir_check(config))
    checks.append(_template_cache_check(config))
    checks.append(_opsim_cache_check(config))
    checks.append(_sim_interpreter_check(config))
    checks.append(_slack_token_check(config))
    return checks


def format_report(checks: list[Check]) -> str:
    """Render checks as a readable report."""
    width = max(len(c.name) for c in checks) + 2
    lines = ["", "chatterbox environment", "=" * 22, ""]
    for check in checks:
        lines.append(f"[{check.status:>4}] {check.name:<{width}} {check.detail}")
        for note in check.notes:
            lines.append(f"{'':>7} {'':<{width}} {note}")
        if check.fix and not check.ok:
            lines.append(f"{'':>7} {'':<{width}} fix: {check.fix}")

    failures = [c for c in checks if not c.ok and c.fatal]
    warnings = [c for c in checks if not c.ok and not c.fatal]
    lines.append("")
    if failures:
        lines.append(f"{len(failures)} blocking problem(s): " + ", ".join(c.name for c in failures))
    if warnings:
        lines.append(f"{len(warnings)} degraded capability/ies: " + ", ".join(c.name for c in warnings))
    if not failures and not warnings:
        lines.append("Everything checks out.")
    lines.append("")
    return "\n".join(lines)
