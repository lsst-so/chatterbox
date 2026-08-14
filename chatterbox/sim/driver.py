"""Standalone scheduler-simulation driver, run as a subprocess.

Invoked as ``python -m chatterbox.sim.driver <job_dir>`` by
`chatterbox.sim.runner.launch_simulation`. It runs in an interpreter that has
``rubin_scheduler`` and ``lsst_survey_sim`` available, reads ``job.json``,
simulates Rubin's response to the ToO on top of the nominal cadence, and writes
``result.json``.

Notes
-----
The ToO *footprint* handed to the scheduler is the producer's binary reward
map,
because that is exactly what the summit receives -- reproducing the real
scheduling decision matters more here than using a finer map. Coverage is then
measured against the best localization available: the real probability map when
GraceDB enrichment succeeded, otherwise the same reward map, in which case the
numbers are area fractions and ``result.json`` says so.
"""

import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("chatterbox.sim.driver")

__all__ = ["main", "run_job"]


def _band_scheduler(spec: dict[str, Any]) -> tuple[Any, str]:
    """Build the band (filter carousel) scheduler.

    Returns
    -------
    scheduler : `object`
        A band scheduler.
    description : `str`
        Which schedule was used, recorded in the result.

    Notes
    -----
    ``lsst_survey_sim.simulate_lsst.setup_band_scheduler`` hardcodes a swap
    schedule that ends 2026-03-25 with an end date of 2026-03-30. Past that it
    silently falls back to ``SimpleBandSched``, which picks the carousel from
    Moon illumination -- so a requested ``u`` visit may be dropped by
    ``ScriptedSurvey``'s mounted-band check. That is a real effect worth
    reporting rather than hiding, so the choice is recorded either way, and a
    schedule can be supplied through ``sim.band_swap_schedule`` to override it.
    """
    from astropy.time import Time
    from rubin_scheduler.scheduler.schedulers import DateSwapBandScheduler, SimpleBandSched

    sim_cfg = spec.get("sim", {})
    schedule = sim_cfg.get("band_swap_schedule") or {}
    end_date = sim_cfg.get("band_swap_end_date") or ""

    if schedule:
        scheduler = DateSwapBandScheduler(
            swap_schedule=dict(schedule),
            end_date=Time(end_date) if end_date else None,
            backup_band_scheduler=SimpleBandSched(illum_limit=40),
        )
        return scheduler, f"DateSwapBandScheduler from config ({len(schedule)} swaps)"

    from lsst_survey_sim import simulate_lsst

    scheduler = simulate_lsst.setup_band_scheduler()
    last = max(getattr(scheduler, "swap_schedule", {}) or {"unknown": None})
    return (
        scheduler,
        f"lsst_survey_sim default DateSwapBandScheduler (schedule ends {last}; "
        "SimpleBandSched backup applies after that)",
    )


def run_job(job_dir: Path) -> dict[str, Any]:
    """Run the simulation described by ``job.json`` in `job_dir`.

    Parameters
    ----------
    job_dir : `pathlib.Path`
        Job directory prepared by `chatterbox.sim.runner.write_job_spec`.

    Returns
    -------
    result : `dict`
        The contents written to ``result.json``.
    """
    started = time.monotonic()
    spec = json.loads((job_dir / "job.json").read_text())
    sim_cfg = spec.get("sim", {})

    source = spec["source"]
    alert_type = spec["alert_type"]
    nights = int(spec["nights"])

    result: dict[str, Any] = {
        "status": "failed",
        "source": source,
        "alert_type": alert_type,
        "nights": nights,
    }

    import healpy as hp
    import pandas as pd
    from astropy.time import Time
    from rubin_scheduler.scheduler.utils import SimTargetooServer, TargetoO

    sys.path.insert(0, str(Path(sim_cfg["lsst_survey_sim"]).expanduser()))
    from lsst_survey_sim import lsst_support, simulate_lsst

    from ..astro.skymap import localization_from_probability, localization_from_reward_map
    from .coverage import band_coverage_by_epoch, coverage_curve, too_visits

    # ---------------------------------------------------------------- inputs

    reward_nside = int(spec["reward_map_nside"])
    reward_map = np.load(job_dir / spec["reward_map"])
    # The feature-based scheduler expects RING ordering; the record is NESTED.
    footprint = hp.reorder(reward_map.astype(float), n2r=True)
    footprint = np.where(footprint > 0, 1.0, 0.0)
    if not footprint.any():
        result["error"] = "reward map is empty; nothing to follow up"
        return result

    event_mjd = spec.get("event_mjd")
    if event_mjd is None:
        logger.warning("No event MJD in job spec; using now as the trigger time")
        event_mjd = float(Time.now().mjd)
    event_mjd = float(event_mjd)

    # Centre of the follow-up, used by the scheduler's dithering.
    ring_pixels = np.flatnonzero(footprint > 0)
    ra_deg, dec_deg = hp.pix2ang(reward_nside, ring_pixels, lonlat=True)
    centre_idx = len(ring_pixels) // 2
    ra_rad_center = float(np.radians(ra_deg[centre_idx]))
    dec_rad_center = float(np.radians(dec_deg[centre_idx]))

    too = TargetoO(
        0,
        footprint,
        event_mjd,
        float(nights),
        ra_rad_center=ra_rad_center,
        dec_rad_center=dec_rad_center,
        too_type=alert_type,
    )
    too_server = SimTargetooServer([too])

    day_obs = int(Time(event_mjd, format="mjd").strftime("%Y%m%d"))
    logger.info(
        "Simulating %s (%s) for %d nights from day_obs=%d, footprint %d pixels",
        source,
        alert_type,
        nights,
        day_obs,
        ring_pixels.size,
    )

    # ------------------------------------------------------------- scheduler

    band_scheduler, band_description = _band_scheduler(spec)
    result["band_scheduler"] = band_description
    logger.info("Band scheduler: %s", band_description)

    config_root = Path(sim_cfg["ts_config_scheduler"]).expanduser()
    config_script = config_root / "Scheduler/feature_scheduler/maintel/fbs_config_lsst_survey.py"
    config_ddf = config_root / "Scheduler/ddf_gen/lsst_ddf_gen_block_407.py"
    for path in (config_script, config_ddf):
        if not path.is_file():
            result["error"] = f"scheduler config not found: {path}"
            return result

    opsim_path = Path(sim_cfg["opsim_h5"]).expanduser()
    if not opsim_path.is_file():
        result["error"] = (
            f"visit history not found: {opsim_path}. Set sim.opsim_h5 to a pre-fetched "
            "opsim.h5 (see rubin_nights/scripts/make_opsim.py)."
        )
        return result
    initial_opsim = pd.read_hdf(opsim_path)
    if "day_obs" in initial_opsim.columns:
        initial_opsim = initial_opsim.query("day_obs < @day_obs")

    scheduler, _, nside = simulate_lsst.setup_scheduler(
        config_script_path=str(config_script),
        config_ddf_script_path=str(config_ddf),
        day_obs=day_obs,
        band_scheduler=band_scheduler,
        too_server=too_server,
        initial_opsim=initial_opsim,
    )

    # A single-night run skips the weather and downtime machinery, which is a
    # large part of the setup cost and irrelevant over one night.
    single_night = nights <= 1
    observatory, survey_info = simulate_lsst.setup_observatory(
        day_obs=day_obs,
        nside=nside,
        add_downtime=not single_night,
        add_clouds=not single_night,
        real_downtime=False,
        initial_opsim=initial_opsim,
        too_server=too_server,
    )

    # ------------------------------------------------------------------- run

    observations, scheduler, observatory, _, _, _ = simulate_lsst.run_sim(
        scheduler=scheduler,
        band_scheduler=band_scheduler,
        observatory=observatory,
        survey_info=survey_info,
        day_obs=day_obs,
        sim_nights=nights,
        keep_rewards=False,
    )

    visits_path = job_dir / "visits.h5"
    # save_opsim is used rather than rubin_sim's sim_archive helpers, which
    # require rubin_sim >= 2.6 and are absent in the pinned environment.
    try:
        visits = lsst_support.save_opsim(observatory, observations, initial_opsim, str(visits_path))
    except Exception as exc:
        logger.warning("save_opsim failed (%s); falling back to a plain DataFrame", exc)
        visits = pd.DataFrame(observations)
        visits.to_hdf(visits_path, key="visits")
    result["visits_path"] = str(visits_path)
    result["total_visits"] = int(len(visits))

    # -------------------------------------------------------------- coverage

    follow_up = too_visits(visits)
    result["too_visits"] = int(len(follow_up))
    if follow_up.empty:
        result["status"] = "complete"
        result["error"] = (
            "the simulation scheduled no ToO visits; the localization may be "
            "unobservable in this window or the alert type may have no matching survey"
        )
        result["runtime_s"] = time.monotonic() - started
        return result

    skymap_path = spec.get("skymap_path")
    if skymap_path and Path(skymap_path).is_file():
        from ligo.skymap.io import read_sky_map

        prob, _ = read_sky_map(str(skymap_path), moc=False, nest=True)
        prob = hp.reorder(np.asarray(prob, dtype=float), n2r=True)
        localization = localization_from_probability(
            prob, provenance=f"GraceDB skymap for {source}", credible_level=0.9
        )
    else:
        localization = localization_from_reward_map(reward_map, reward_nside)
        logger.info("No probability skymap available; coverage will be an area fraction")

    coverage = band_coverage_by_epoch(follow_up, localization)
    result["coverage"] = coverage.fractions
    result["n_visits"] = coverage.n_visits
    result["coverage_by_epoch"] = {str(k): v for k, v in coverage.epochs.items()}
    result["any_band"] = coverage.any_band
    result["quantity"] = coverage.quantity

    # ------------------------------------------------------------ curve plot

    try:
        from ..plots.coverage import plot_coverage_curve

        curves = coverage_curve(follow_up, localization)
        plot_path = plot_coverage_curve(
            curves,
            localization,
            job_dir / "coverage.png",
            source=source,
            alert_type=alert_type,
            event_mjd=event_mjd,
        )
        if plot_path is not None:
            result["curve_plot"] = str(plot_path)
    except Exception as exc:
        logger.warning("Could not render the coverage curve: %s", exc)

    result["status"] = "complete"
    result["runtime_s"] = time.monotonic() - started
    logger.info(
        "Done in %.1f s: %d ToO visits, coverage (%s) %s",
        result["runtime_s"],
        result["too_visits"],
        coverage.quantity,
        coverage.summary_line(),
    )
    return result


def main(argv: list[str] | None = None) -> int:
    """Entry point: ``python -m chatterbox.sim.driver <job_dir>``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python -m chatterbox.sim.driver <job_dir>", file=sys.stderr)
        return 2

    job_dir = Path(argv[0]).expanduser()
    if not (job_dir / "job.json").is_file():
        print(f"No job.json in {job_dir}", file=sys.stderr)
        return 2

    try:
        result = run_job(job_dir)
    except Exception as exc:
        logger.error("Simulation failed: %s", exc)
        traceback.print_exc()
        result = {
            "status": "failed",
            "source": "unknown",
            "alert_type": "unknown",
            "nights": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }
        try:
            spec = json.loads((job_dir / "job.json").read_text())
            result.update(
                {
                    "source": spec.get("source", "unknown"),
                    "alert_type": spec.get("alert_type", "unknown"),
                    "nights": int(spec.get("nights", 0)),
                }
            )
        except Exception:
            pass

    (job_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))
    return 0 if result.get("status") == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
