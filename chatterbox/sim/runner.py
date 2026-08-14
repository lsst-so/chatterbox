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
``visits.h5``        Simulated observations (written by the driver).
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
from ..models import Trigger

__all__ = ["SimJob", "SimResult", "launch_simulation", "load_sim_result", "write_job_spec"]

logger = logging.getLogger(__name__)


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
    total_visits: int = 0
    too_visits: int = 0
    visits_path: str | None = None
    curve_plot: str | None = None
    #: Which band carousel the simulation used; see the note in the driver.
    band_scheduler: str = ""
    #: Provenance of the visit history the simulation started from, including
    #: whether a stale cache had to be reused.
    opsim: str = ""
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

    python = str(Path(config.sim.python).expanduser())
    if not Path(python).is_file():
        logger.error(
            "Configured sim interpreter %s does not exist; set sim.python to an "
            "interpreter with rubin_scheduler and lsst_survey_sim installed",
            python,
        )
        (job_dir / "result.json").write_text(
            json.dumps(
                {
                    "status": "failed",
                    "source": trigger.source,
                    "alert_type": trigger.alert_type,
                    "nights": nights,
                    "error": f"sim interpreter not found: {python}",
                }
            )
        )
        return SimJob(job_dir=job_dir, source=trigger.source, alert_type=trigger.alert_type, nights=nights)

    env = dict(os.environ)
    env["RUBIN_SIM_DATA_DIR"] = str(Path(config.sim.rubin_sim_data).expanduser())
    # The driver needs chatterbox itself plus lsst_survey_sim, which is used
    # from a checkout rather than being pip-installed.
    repo_root = str(Path(__file__).resolve().parents[2])
    extra_paths = [repo_root, str(Path(config.sim.lsst_survey_sim).expanduser())]
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
