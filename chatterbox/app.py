"""The chatterbox service: turn a ToO record into Slack posts.

Two stages, so a low-latency post and a full-length simulation can coexist:

1. Everything computable locally -- alert statistics, observability, both maps
--
   is posted immediately. GraceDB enrichment runs concurrently with the almanac
   and dark-hours work so it does not simply add to the critical path.
2. The scheduler simulation runs as a background process and its per-band
   coverage is posted as a threaded reply when it finishes.

Failures are contained per stage. If a plot cannot be rendered, the text still
goes out; if enrichment fails, the post says the figures are area-based; if the
simulation fails, the thread reply says why.
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from astropy.time import Time

from .alerts import get_alert_class
from .astro.almanac import NightEvents, night_events
from .astro.darkhours import DarkHoursMap, dark_hours_map
from .astro.templates import TemplateCoverage, load_template_maps
from .config import Config
from .ingest.decode import decode_record
from .ingest.enrich_gracedb import enrich_gravitational_wave
from .models import Trigger
from .plots.darkhours import plot_dark_hours, region_hours_summary
from .plots.templates import plot_template_coverage
from .sim.runner import launch_simulation, load_sim_result
from .slackbot.blocks import build_sim_reply_blocks, build_trigger_blocks, plain_text_summary
from .slackbot.client import PostedMessage, SlackPoster

__all__ = ["TriggerReport", "process_trigger", "run_service"]

logger = logging.getLogger(__name__)


@dataclass
class TriggerReport:
    """Everything computed for one trigger, for inspection and tests."""

    trigger: Trigger
    events: NightEvents
    dark_hours: DarkHoursMap
    dark_stats: dict[str, float]
    template_coverage: TemplateCoverage | None
    blocks: list[dict[str, Any]]
    text: str
    plots: list[Path] = field(default_factory=list)
    posted: PostedMessage | None = None
    elapsed_s: float = 0.0
    #: Non-fatal problems encountered, for reporting in the CLI.
    warnings: list[str] = field(default_factory=list)


def _build_plots(
    trigger: Trigger,
    dark_hours: DarkHoursMap,
    template_coverage: TemplateCoverage | None,
    out_dir: Path,
    warnings: list[str],
) -> list[Path]:
    """Render both maps, tolerating failure of either."""
    plots: list[Path] = []
    centroid = (trigger.geometry.centroid_ra_deg, trigger.geometry.centroid_dec_deg)

    try:
        plots.append(
            plot_dark_hours(
                dark_hours,
                trigger.localization,
                out_dir / f"{trigger.source}_darkhours.png",
                centroid=centroid,
            )
        )
    except Exception as exc:
        logger.error("Could not render the dark-hours plot: %s", exc)
        warnings.append(f"dark-hours plot failed: {exc}")

    if template_coverage is not None:
        try:
            path = plot_template_coverage(
                template_coverage,
                trigger.localization,
                out_dir / f"{trigger.source}_templates.png",
            )
            if path is not None:
                plots.append(path)
        except Exception as exc:
            logger.error("Could not render the template plot: %s", exc)
            warnings.append(f"template plot failed: {exc}")

    return plots


def process_trigger(
    record: dict[str, Any],
    config: Config,
    poster: SlackPoster | None = None,
    post: bool = True,
    run_sim: bool = True,
    sim_wait: bool = False,
    out_dir: Path | None = None,
) -> TriggerReport:
    """Handle one ToO record end to end.

    Parameters
    ----------
    record : `dict`
        Decoded ``too_alert`` record.
    config : `Config`
        Configuration.
    poster : `SlackPoster`, optional
        Delivery mechanism. Built from `config` when omitted.
    post : `bool`
        Actually post. When False the blocks are built and returned only.
    run_sim : `bool`
        Launch the scheduler simulation.
    sim_wait : `bool`
        Block until the simulation finishes and post the reply inline, rather
        than in a background thread. Used by the CLI.
    out_dir : `pathlib.Path`, optional
        Where plots are written. Defaults to a per-source work directory.

    Returns
    -------
    report : `TriggerReport`
    """
    started = time.monotonic()
    warnings: list[str] = []

    trigger = decode_record(record)
    alert_class = get_alert_class(trigger.alert_type)

    if trigger.is_test and not config.ingest.allow_tests:
        logger.info("Dropping test alert %s (ingest.allow_tests is false)", trigger.source)

    if out_dir is None:
        out_dir = Path(config.paths.work_dir).expanduser() / "plots" / trigger.source
    out_dir.mkdir(parents=True, exist_ok=True)

    # Enrichment is network-bound and the almanac work is CPU-bound, so overlap
    # them rather than paying for both in series.
    with ThreadPoolExecutor(max_workers=2) as pool:
        enrich_future = pool.submit(enrich_gravitational_wave, trigger, config.enrich)
        template_future = pool.submit(load_template_maps, config.templates.cache_dir)

        when = trigger.event_time if trigger.event_time is not None else Time.now()
        try:
            events = night_events(when=when)
        except Exception as exc:
            logger.error("Almanac lookup failed: %s", exc)
            raise

        try:
            dark_hours = dark_hours_map(
                events,
                nside=config.dark_hours.nside,
                step_minutes=config.dark_hours.step_minutes,
                airmass_limit=config.dark_hours.airmass_limit,
                sun_alt_limit_deg=config.dark_hours.sun_alt_limit_deg,
                moon_avoidance_deg=config.dark_hours.moon_avoidance_deg,
            )
        except Exception as exc:
            logger.error("Dark-hours computation failed: %s", exc)
            raise

        try:
            enrichment = enrich_future.result()
            if enrichment is not None and enrichment.error:
                warnings.append(f"GraceDB enrichment: {enrichment.error}")
            # enrich_gravitational_wave only attaches on success; make a failed
            # attempt visible in the post too.
            if enrichment is not None and trigger.enrichment is None:
                trigger.enrichment = enrichment
        except Exception as exc:
            logger.error("Enrichment raised: %s", exc)
            warnings.append(f"GraceDB enrichment raised: {exc}")

        template_coverage = template_future.result()
        if template_coverage is None:
            warnings.append("no template coverage cache; run 'chatterbox refresh-templates'")

    dark_stats = region_hours_summary(dark_hours, trigger.localization)
    plots = _build_plots(trigger, dark_hours, template_coverage, out_dir, warnings)

    blocks = build_trigger_blocks(trigger, events, dark_hours, dark_stats, template_coverage, config)
    text = plain_text_summary(trigger, alert_class)

    report = TriggerReport(
        trigger=trigger,
        events=events,
        dark_hours=dark_hours,
        dark_stats=dark_stats,
        template_coverage=template_coverage,
        blocks=blocks,
        text=text,
        plots=plots,
        warnings=warnings,
    )

    if post:
        poster = poster or SlackPoster(config)
        try:
            report.posted = poster.post(
                blocks,
                text,
                is_test=trigger.is_test,
                files=plots,
                label=f"{trigger.source}_post",
            )
        except Exception as exc:
            logger.error("Posting to Slack failed: %s", exc)
            warnings.append(f"Slack post failed: {exc}")

    report.elapsed_s = time.monotonic() - started
    logger.info(
        "Stage 1 for %s complete in %.2f s (%d plots, %d warnings)",
        trigger.source,
        report.elapsed_s,
        len(plots),
        len(warnings),
    )

    if run_sim:
        _start_simulation(trigger, config, poster, report, sim_wait=sim_wait)

    return report


def _start_simulation(
    trigger: Trigger,
    config: Config,
    poster: SlackPoster | None,
    report: TriggerReport,
    sim_wait: bool = False,
) -> None:
    """Launch the simulation and arrange for its result to be posted."""
    job = launch_simulation(trigger, config)
    if job is None:
        return

    def finish() -> None:
        job.wait(timeout=config.sim.timeout_s)
        result = load_sim_result(job.job_dir)
        if result is None:
            logger.error(
                "Simulation for %s produced no result.json. Log tail:\n%s",
                trigger.source,
                job.tail_log(),
            )
            return
        logger.info("Simulation for %s finished with status %s", trigger.source, result.status)
        if poster is None or report.posted is None:
            return
        blocks = build_sim_reply_blocks(result, config)
        files = [result.curve_plot] if result.curve_plot else None
        summary = f"Simulation for {result.source}: " + (
            ", ".join(f"{b} {100 * v:.0f}%" for b, v in result.coverage.items() if v > 0)
            or (result.error or "no coverage")
        )
        try:
            poster.reply(report.posted, blocks, summary, files=files, label=f"{result.source}_sim")
        except Exception as exc:
            logger.error("Could not post the simulation reply: %s", exc)

    if sim_wait:
        finish()
    else:
        thread = threading.Thread(target=finish, name=f"sim-{trigger.source}", daemon=True)
        thread.start()


def run_service(config: Config, paths=None, run_sim: bool = True) -> int:
    """Consume the configured source and post about every record.

    Parameters
    ----------
    config : `Config`
        Configuration.
    paths : sequence, optional
        Explicit record paths, forcing replay mode.
    run_sim : `bool`
        Launch simulations.

    Returns
    -------
    count : `int`
        Records handled.
    """
    from .ingest.source import make_source

    source = make_source(config, paths=paths)
    poster = SlackPoster(config)
    handled = 0
    try:
        for record, metadata in source:
            try:
                process_trigger(record, config, poster=poster, run_sim=run_sim)
                handled += 1
            except Exception as exc:
                logger.error(
                    "Failed to handle a record from %s: %s",
                    metadata.get("origin") or metadata.get("topic"),
                    exc,
                    exc_info=True,
                )
            finally:
                # Acknowledge either way: a record that cannot be processed
                # will
                # not become processable on redelivery, and blocking the stream
                # on it would stall every later alert.
                source.mark_done(metadata)
    except KeyboardInterrupt:
        logger.info("Interrupted; shutting down after %d record(s)", handled)
    finally:
        source.close()
    return handled
