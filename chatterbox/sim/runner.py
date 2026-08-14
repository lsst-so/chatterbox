"""Launch and collect scheduler simulations as background jobs.

A simulation takes minutes to tens of minutes, so it never runs on the alert
path. `launch_simulation` writes a self-describing job directory and starts
`chatterbox.sim.driver` in a separate process, which lets the simulation run in
an interpreter that has ``rubin_scheduler`` and ``lsst_survey_sim`` available
even if the bot itself does not.

The job directory is the interface between the two processes:

===================  =========================================================
``job.json``         Everything the driver needs: alert type, nights, paths.
``reward_map.npy``   The producer's boolean reward map, NESTED at its nside.
``sim.log``          Driver stdout and stderr.
``visits.db``        Opsim database: the visit history plus the simulated
                     visits (written by the driver).
``coverage.png``     Cumulative coverage against time, per band.
``coverage_by_night  Coverage gained on each night, per band, with the running
.png``               any-band total over it.
``*_night*.png``     Per-night, per-band simulated visit counts.
``result.json``      Coverage results and status (written by the driver).
===================  =========================================================
"""

import json
import logging
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..alerts import sim_nights_for
from ..config import Config
from ..deps import import_root
from ..models import Trigger

__all__ = [
    "SimJob",
    "SimResult",
    "launch_simulation",
    "load_sim_result",
    "write_job_spec",
    "read_job_spec",
    "resolve_sim_python",
    "check_sim_python",
]

logger = logging.getLogger(__name__)


def resolve_sim_python(configured: str) -> Path:
    """Interpreter to run the simulation subprocess with.

    An empty ``sim.python`` means "the one running chatterbox", which is the
    right default: an environment able to compute the almanac can usually run
    the simulation too, and it removes a setting that is easy to point at the
    wrong conda base.

    Parameters
    ----------
    configured : `str`
        The ``sim.python`` setting.

    Returns
    -------
    python : `pathlib.Path`
    """
    configured = (configured or "").strip()
    if not configured:
        return Path(sys.executable)
    return Path(configured).expanduser()


def check_sim_python(python: Path) -> tuple[bool, str]:
    """Can this interpreter import what the simulation driver needs?

    Parameters
    ----------
    python : `pathlib.Path`
        Interpreter to probe.

    Returns
    -------
    ok : `bool`
        True when the driver's imports would succeed.
    detail : `str`
        Empty when ok, otherwise what is wrong.
    """
    from ..doctor import DRIVER_REQUIREMENTS, _probe_interpreter

    # Skip the subprocess when it is this very interpreter: we already know.
    if python.resolve() == Path(sys.executable).resolve():
        missing = []
        for module in DRIVER_REQUIREMENTS:
            try:
                __import__(module)
            except Exception as exc:
                missing.append(f"{module} ({type(exc).__name__})")
        return (not missing), ("cannot import " + ", ".join(missing) if missing else "")

    return _probe_interpreter(python, DRIVER_REQUIREMENTS)


@dataclass
class SimJob:
    """A launched simulation."""

    job_dir: Path
    source: str
    alert_type: str
    nights: int
    process: subprocess.Popen | None = None
    log_path: Path | None = None

    def wait(self, timeout: float | None = None) -> int | None:
        """Block until the driver exits.

        Returns
        -------
        returncode : `int` or None
            None when the job was never started as a subprocess.
        """
        if self.process is None:
            return None
        try:
            return self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            logger.error("Simulation for %s exceeded its timeout; terminating", self.source)
            self.process.kill()
            return self.process.wait()

    @property
    def result_path(self) -> Path:
        """Path the driver writes its result to."""
        return self.job_dir / "result.json"

    def tail_log(self, lines: int = 20) -> str:
        """Last few lines of the driver log, for error reporting."""
        if self.log_path is None or not self.log_path.is_file():
            return ""
        try:
            return "\n".join(self.log_path.read_text(errors="replace").splitlines()[-lines:])
        except OSError:
            return ""


@dataclass
class SimResult:
    """Outcome of a simulation, as read back from ``result.json``."""

    status: str
    source: str
    alert_type: str
    nights: int
    #: Fraction of localization weight covered per band, 0-1.
    coverage: dict[str, float] = field(default_factory=dict)
    n_visits: dict[str, int] = field(default_factory=dict)
    coverage_by_epoch: dict[str, dict[str, float]] = field(default_factory=dict)
    any_band: float = 0.0
    #: "probability" or "area" -- whether coverage is a real probability.
    quantity: str = "area"
    #: Visits the simulation itself scheduled.
    total_visits: int = 0
    #: Rows in the saved database, which also contains the ConsDB history the
    #: simulation started from.
    archive_visits: int = 0
    too_visits: int = 0
    visits_path: str | None = None
    curve_plot: str | None = None
    #: Which band carousel the simulation used; see the note in the driver.
    band_scheduler: str = ""
    #: Provenance of the visit history the simulation started from, including
    #: whether a stale cache had to be reused.
    opsim: str = ""
    #: Directory the simulation ran in; every artifact lives here.
    job_dir: str = ""
    #: Survey night the trigger fell on, which anchors every other night
    #: number. None for a result written before this was recorded, in which
    #: case nights are reported as bare survey nights.
    trigger_night: int | None = None
    #: First night the simulation itself covered, as a survey night. Usually
    #: `trigger_night`, but one earlier when the alert arrived after 00:00 UTC
    #: and before that evening's sunset.
    first_sim_night: int | None = None
    #: Coverage gained on each night, ``nights since trigger -> band ->
    #: fraction``. Add `trigger_night` to recover the survey night.
    coverage_by_night: dict[str, dict[str, float]] = field(default_factory=dict)
    #: Total fraction covered through each night, any band, keyed the same way.
    cumulative_by_night: dict[str, float] = field(default_factory=dict)
    #: Figure of the above; one image however many nights were simulated.
    nightly_coverage_plot: str | None = None
    #: Per-night visit-count figures, and the nights they correspond to.
    nightly_plots: list[str] = field(default_factory=list)
    nightly_plot_nights: list[int] = field(default_factory=list)
    #: Nights whose visits overlap the localization, before the plot cap.
    nights_with_overlap: int = 0
    #: Visits overlapping the localization, ToO-tagged or not.
    overlap_visits: int = 0
    #: How many of `overlap_visits` are tagged ToO follow-up. Counted within
    #: the overlapping set, so it is never larger than `overlap_visits`.
    overlap_too_visits: int = 0
    runtime_s: float = 0.0
    error: str | None = None

    @property
    def ok(self) -> bool:
        """True when the simulation completed and produced coverage."""
        return self.status == "complete" and self.error is None

    def is_probability_result(self) -> bool:
        """True when `coverage` holds probabilities, not area fractions."""
        return self.quantity == "probability"


def write_job_spec(trigger: Trigger, config: Config, job_dir: Path, nights: int) -> Path:
    """Write ``job.json`` and the reward map into a job directory.

    Parameters
    ----------
    trigger : `Trigger`
        Trigger to simulate.
    config : `Config`
        Configuration supplying paths and overrides.
    job_dir : `pathlib.Path`
        Directory to populate.
    nights : `int`
        Nights to simulate.

    Returns
    -------
    path : `pathlib.Path`
        The ``job.json`` written.
    """
    job_dir.mkdir(parents=True, exist_ok=True)
    np.save(job_dir / "reward_map.npy", trigger.reward_map)

    spec: dict[str, Any] = {
        "source": trigger.source,
        "alert_type": trigger.alert_type,
        "nights": nights,
        # Not used by the driver, but 'post-sim' needs it to route a test alert
        # to the test channel long after the trigger itself is gone.
        "is_test": bool(trigger.is_test),
        "event_trigger_timestamp": trigger.event_trigger_timestamp,
        "event_mjd": float(trigger.event_time.mjd) if trigger.event_time is not None else None,
        "reward_map_nside": trigger.reward_map_nside,
        "reward_map": "reward_map.npy",
        # When enrichment succeeded, coverage is measured against the real
        # probability map. The ToO *footprint* handed to the scheduler is still
        # the reward map, because that is what the summit actually receives.
        "skymap_path": trigger.enrichment.skymap_path if trigger.enrichment else None,
        "centroid_ra_deg": trigger.geometry.centroid_ra_deg,
        "centroid_dec_deg": trigger.geometry.centroid_dec_deg,
        "sim": asdict(config.sim),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path = job_dir / "job.json"
    path.write_text(json.dumps(spec, indent=2))
    return path


def launch_simulation(
    trigger: Trigger,
    config: Config,
    nights: int | None = None,
    wait: bool = False,
) -> SimJob | None:
    """Start a scheduler simulation for a trigger.

    Parameters
    ----------
    trigger : `Trigger`
        Trigger to simulate.
    config : `Config`
        Configuration. ``sim.enabled`` gates this entirely.
    nights : `int`, optional
        Override the per-class night count.
    wait : `bool`
        Block until the driver finishes. Used by the CLI; the service does not.

    Returns
    -------
    job : `SimJob` or None
        None when simulation is disabled.
    """
    if not config.sim.enabled:
        logger.info("Simulation disabled; skipping for %s", trigger.source)
        return None

    if nights is None:
        nights = config.sim.nights_override or sim_nights_for(trigger.alert_type)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    job_dir = Path(config.paths.work_dir).expanduser() / "sim" / f"{trigger.source}_{stamp}"
    write_job_spec(trigger, config, job_dir, nights)

    def give_up(message: str) -> SimJob:
        """Record a launch failure as a result, not a buried traceback."""
        logger.error("%s", message)
        (job_dir / "result.json").write_text(
            json.dumps(
                {
                    "status": "failed",
                    "source": trigger.source,
                    "alert_type": trigger.alert_type,
                    "nights": nights,
                    "error": message,
                }
            )
        )
        return SimJob(job_dir=job_dir, source=trigger.source, alert_type=trigger.alert_type, nights=nights)

    python = resolve_sim_python(config.sim.python)
    if not python.is_file():
        return give_up(
            f"sim interpreter not found: {python}. Set sim.python to an interpreter "
            "with healpy, rubin_scheduler and lsst_survey_sim, or leave it empty to "
            f"use the one running chatterbox ({sys.executable}). "
            "Run 'chatterbox doctor' to check."
        )

    # Probe before launching. Otherwise a wrong interpreter surfaces as a
    # traceback inside sim.log, which nobody sees until they go looking.
    ok, detail = check_sim_python(python)
    if not ok:
        return give_up(
            f"sim interpreter {python} cannot run the simulation: {detail}. "
            "Set sim.python to an interpreter that has these, or leave it empty to "
            f"use the one running chatterbox ({sys.executable}). "
            "Run 'chatterbox doctor' to check."
        )

    env = dict(os.environ)
    env["RUBIN_SIM_DATA_DIR"] = str(Path(config.sim.rubin_sim_data).expanduser())
    # The driver needs chatterbox itself plus lsst_survey_sim, which is used
    # from a checkout rather than being pip-installed.
    repo_root = str(Path(__file__).resolve().parents[2])
    # ts_fbs_utils is needed as well: the scheduler config script imports
    # lsst.ts.fbs.utils to build the ToO surveys. import_root() picks the
    # python/ subdirectory for LSST-layout checkouts.
    extra_paths = [repo_root]
    for checkout in (config.sim.lsst_survey_sim, config.sim.ts_fbs_utils):
        root = import_root(checkout)
        if root is not None:
            extra_paths.append(str(root))
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join([p for p in extra_paths + [existing] if p])
    # Matplotlib must not try to open a display from a background process.
    env.setdefault("MPLBACKEND", "Agg")

    log_path = job_dir / "sim.log"
    logger.info(
        "Launching %d-night simulation for %s (%s) in %s",
        nights,
        trigger.source,
        trigger.alert_type,
        job_dir,
    )
    with open(log_path, "w") as log:
        process = subprocess.Popen(
            [python, "-m", "chatterbox.sim.driver", str(job_dir)],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=repo_root,
            start_new_session=True,
        )

    job = SimJob(
        job_dir=job_dir,
        source=trigger.source,
        alert_type=trigger.alert_type,
        nights=nights,
        process=process,
        log_path=log_path,
    )
    if wait:
        job.wait(timeout=config.sim.timeout_s)
    return job


def read_job_spec(job_dir: str | Path) -> dict[str, Any]:
    """Read back a job directory's ``job.json``.

    Returns
    -------
    spec : `dict`
        Empty when there is no readable spec, so a job directory written by an
        older version still posts.
    """
    path = Path(job_dir).expanduser() / "job.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        logger.warning("Could not parse %s: %s", path, exc)
        return {}


def load_sim_result(job_dir: str | Path) -> SimResult | None:
    """Read a simulation result from a job directory.

    Returns
    -------
    result : `SimResult` or None
        None when the driver has not written a result yet.
    """
    path = Path(job_dir).expanduser() / "result.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        logger.error("Could not parse %s: %s", path, exc)
        return None
    known = {f for f in SimResult.__dataclass_fields__}
    return SimResult(**{k: v for k, v in data.items() if k in known})


if __name__ == "__main__":  # pragma: no cover - convenience for manual runs
    print("Run the driver instead: python -m chatterbox.sim.driver <job_dir>", file=sys.stderr)
