"""Optional-dependency handling with actionable error messages.

chatterbox degrades rather than dies when an optional package is missing: no
``ligo.skymap`` costs the plots, no ``slack_sdk`` costs the posting. That is
right, but a bare ``No module named 'ligo'`` does not say *which* interpreter
looked, and that is the one fact needed to fix it -- an install landing in a
different Python is by far the most common cause.

`require` produces the message; `CAPABILITIES` is the same information in a
form `chatterbox.doctor` can report on before anything goes wrong.
"""

import importlib
import logging
import sys
from dataclasses import dataclass

__all__ = ["Capability", "CAPABILITIES", "require", "probe"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Capability:
    """One thing chatterbox can do, and what it needs to do it.

    Attributes
    ----------
    name : `str`
        Short label, e.g. ``"plots"``.
    modules : `tuple` [`str`]
        Import names that must all be importable.
    install : `str`
        Distribution names to install, as passed to pip.
    purpose : `str`
        What the capability provides, for the error message.
    consequence : `str`
        What is lost without it, so a reader can judge urgency.
    """

    name: str
    modules: tuple[str, ...]
    install: str
    purpose: str
    consequence: str
    extra: str = ""


#: Everything optional, in the order a reader most likely cares about.
CAPABILITIES: tuple[Capability, ...] = (
    Capability(
        name="almanac",
        modules=("rubin_scheduler",),
        install="rubin-scheduler",
        purpose="sun/moon times and the accessible-dark-hours map",
        consequence="alerts cannot be processed at all",
        extra="rubin",
    ),
    Capability(
        name="plots",
        # ligo.skymap is a namespace package: the failure surfaces as
        # "No module named 'ligo'", which is why it reads so cryptically.
        modules=("ligo.skymap.plot",),
        install="ligo.skymap",
        purpose="the sky projections used by both maps",
        consequence="no dark-hours or template plots are attached to the post",
        extra="rubin",
    ),
    Capability(
        name="gracedb",
        modules=("ligo.gracedb.rest",),
        install="ligo-gracedb",
        purpose="fetching the real GW skymap, distance, FAR and classification",
        consequence="GW figures stay area-based instead of probability-based",
        extra="rubin",
    ),
    Capability(
        name="slack",
        modules=("slack_sdk",),
        install="slack-sdk",
        purpose="posting messages and uploading plots",
        consequence="output is written locally instead of posted",
        extra="slack",
    ),
    Capability(
        name="kafka",
        modules=("hop",),
        install="hop-client",
        purpose="the Kafka ingest source",
        consequence="only the file and replay ingest sources work",
        extra="kafka",
    ),
    Capability(
        name="simulation",
        modules=("lsst_survey_sim.simulate_lsst",),
        install="a checkout of lsst_survey_sim (see sim.lsst_survey_sim)",
        purpose="the scheduler simulation and the ConsDB visit fetch",
        consequence="no per-band coverage and no visit history",
        extra="",
    ),
    Capability(
        name="strategy",
        modules=("lsst.ts.fbs.utils.maintel.too_surveys",),
        install="ts_fbs_utils",
        purpose="reading the live ToO follow-up strategy",
        consequence="the vendored strategy snapshot is used instead, which can drift",
        extra="",
    ),
)

_BY_MODULE = {c.modules[0]: c for c in CAPABILITIES}


def _hint(module: str, install: str) -> str:
    """Build the install hint, naming the interpreter that actually looked."""
    return (
        f"{sys.executable} cannot import {module}. Install it into *that* "
        f"interpreter: '{sys.executable} -m pip install {install}'. "
        "A package installed under a different python will not be found."
    )


def require(module: str, purpose: str = "", install: str = ""):
    """Import a module, or raise an ImportError that says how to fix it.

    Parameters
    ----------
    module : `str`
        Import name, e.g. ``"ligo.skymap.plot"``.
    purpose : `str`
        What it is needed for. Looked up from `CAPABILITIES` when omitted.
    install : `str`
        What to pip install. Looked up from `CAPABILITIES` when omitted.

    Returns
    -------
    module : module
        The imported module.

    Raises
    ------
    ImportError
        Naming ``sys.executable``, since the usual cause is an install that
        landed in a different interpreter.
    """
    known = _BY_MODULE.get(module)
    purpose = purpose or (known.purpose if known else "")
    install = install or (known.install if known else module)

    try:
        return importlib.import_module(module)
    except ImportError as exc:
        what = f" ({purpose})" if purpose else ""
        raise ImportError(f"{module} is required{what}. {_hint(module, install)} [{exc}]") from exc


def probe(capability: Capability) -> tuple[bool, str]:
    """Check whether a capability's modules are importable.

    Returns
    -------
    ok : `bool`
        True when every module imports.
    detail : `str`
        Empty when ok, otherwise the first failure.
    """
    for module in capability.modules:
        try:
            importlib.import_module(module)
        except Exception as exc:  # ImportError, but a broken install can raise anything
            return False, f"{module}: {type(exc).__name__}: {exc}"
    return True, ""
