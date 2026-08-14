"""Scheduler simulation and per-band localization coverage."""

from .coverage import (
    BANDS,
    CoverageResult,
    band_coverage,
    band_coverage_by_epoch,
    coverage_curve,
    nightly_visit_maps,
    too_visits,
    visits_overlapping,
)
from .opsim import (
    OpsimCache,
    OpsimToolUnavailableError,
    default_day_obs,
    ensure_opsim,
    fetch_opsim,
)
from .runner import SimJob, SimResult, launch_simulation, load_sim_result

__all__ = [
    "BANDS",
    "CoverageResult",
    "too_visits",
    "band_coverage",
    "band_coverage_by_epoch",
    "coverage_curve",
    "visits_overlapping",
    "nightly_visit_maps",
    "OpsimCache",
    "OpsimToolUnavailableError",
    "ensure_opsim",
    "fetch_opsim",
    "default_day_obs",
    "SimJob",
    "SimResult",
    "launch_simulation",
    "load_sim_result",
]
