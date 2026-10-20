"""The chatterbox service: turn a ToO record into Slack posts.

Two stages, so a low-latency post and a full-length simulation can coexist:

1. Everything computable locally -- alert statistics, observability, both maps
--
   is posted immediately. GraceDB enrichment runs concurrently with the almanac
   and dark-hours work so it does not simply add to the critical path.
2. The scheduler simulation runs as a background process and its per-band
   coverage is posted as a threaded reply when it finishes. The per-night visit
   figures from the same run go out as a message of their own, since a long run
   attaches many of them.

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
from .astro.visibility import field_visibility
from .config import Config, apply_environment
from .ingest.decode import decode_record
from .ingest.enrich_gracedb import enrich_gravitational_wave
from .models import Trigger
from .plots.darkhours import plot_dark_hours, region_hours_summary
from .plots.templates import plot_template_coverage
from .sim.runner import SimResult, launch_simulation, load_sim_result
from .slackbot.blocks import (
    build_failure_blocks,
    build_sim_figures_blocks,
    build_sim_reply_blocks,
    build_trigger_blocks,
    plain_text_summary,
    sim_figures,
)
from .slackbot.client import PostedMessage, SlackPoster

__all__ = [
    "TriggerReport",
    "process_trigger",
    "post_sim_results",
    "post_failure",
    "drain_simulations",
    "run_service",
]

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
    #: The simulation that was launched, and the thread waiting to post it.
    sim_job: Any = None
    sim_thread: threading.Thread | None = None
    #: The thread rendering and posting the airmass animation, and the file it
    #: wrote. Like the simulation's, the thread is a daemon: it has to be
    #: joined before the process exits or the post never happens.
    animation_thread: threading.Thread | None = None
    animation: Path | None = None

    @property
    def has_background_work(self) -> bool:
        """True when a simulation or an animation is still to be posted."""
        return self.sim_thread is not None or self.animation_thread is not None

    def wait_for_simulation(self, timeout: float | None = None) -> bool:
        """Block until the launched simulation has been posted.

        Callers that are about to exit *must* do this. The simulation itself
        runs in a detached process and survives, but the thread that waits for
        it and posts the result is a daemon: if the process exits first, the
        figures are written to the job directory and never posted.

        Parameters
        ----------
        timeout : `float`, optional
            Seconds to wait. None waits indefinitely.

        Returns
        -------
        done : `bool`
            False when the simulation is still running, so the caller can say
            where to find it.
        """
        threads = [t for t in (self.animation_thread, self.sim_thread) if t is not None]
        for thread in threads:
            thread.join(timeout=timeout)
        return not any(t.is_alive() for t in threads)


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
    run_animation: bool = True,
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
    run_animation : `bool`
        Render the airmass animation and post it in the alert's thread, when
        ``airmass_animation.enabled`` and the alert is actually being posted.

    Returns
    -------
    report : `TriggerReport`
    """
    started = time.monotonic()
    warnings: list[str] = []

    # Idempotent; the CLI already did this, but a library caller may not have.
    apply_environment(config)

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

    # The poster is needed before the blocks: tagging the shift takes Slack's
    # user group IDs, which only it can look up. Skipped for a test alert,
    # which tags nobody, so a replay does not spend a call to learn that.
    if post:
        poster = poster or SlackPoster(config)
    crew_ids = poster.crew_ids() if poster is not None and not trigger.is_test else None

    # When the field is up, not just for how long. A failure here costs that
    # one section, not the alert.
    visibility = None
    try:
        visibility = field_visibility(
            events,
            trigger.localization,
            dark_hours,
            trigger.geometry.centroid_ra_deg,
            trigger.geometry.centroid_dec_deg,
            step_minutes=config.dark_hours.step_minutes,
        )
    except Exception as exc:
        logger.error("Could not work out when the field is visible: %s", exc)
        warnings.append(f"visibility windows failed: {exc}")

    blocks = build_trigger_blocks(
        trigger, events, dark_hours, dark_stats, template_coverage, config, crew_ids, visibility
    )
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

    if run_animation and post and config.airmass_animation.enabled:
        _start_animation(trigger, config, events, poster, report, out_dir)

    if run_sim:
        _start_simulation(trigger, config, poster, report, sim_wait=sim_wait)

    return report


def _start_animation(
    trigger: Trigger,
    config: Config,
    events: NightEvents,
    poster: SlackPoster,
    report: TriggerReport,
    out_dir: Path,
) -> None:
    """Render the airmass animation off the main thread and post it in-thread.

    A night is dozens of frames and takes tens of seconds, so it cannot be part
    of the first post. It is posted as a reply to it instead -- or standalone,
    when the first post could not be threaded onto.
    """
    from .plots.airmass_animation import plot_airmass_animation
    from .slackbot.blocks import build_airmass_animation_blocks

    settings = config.airmass_animation

    def render_and_post() -> None:
        # A background thread's exceptions are invisible, so everything is
        # caught here and logged: a missing animation must never take the alert
        # path down, and it has no right to take a worker thread down silently.
        try:
            path = plot_airmass_animation(
                events,
                trigger.localization,
                out_dir / f"{trigger.source}_airmass.gif",
                mask_above=settings.mask_above,
                step_minutes=settings.step_minutes,
                sun_alt_limit_deg=config.dark_hours.sun_alt_limit_deg,
                centroid=(trigger.geometry.centroid_ra_deg, trigger.geometry.centroid_dec_deg),
                fps=settings.fps,
                dpi=settings.dpi,
            )
            report.animation = path
        except Exception as exc:
            logger.error("Could not render the airmass animation for %s: %s", trigger.source, exc)
            report.warnings.append(f"airmass animation failed: {exc}")
            return

        try:
            blocks = build_airmass_animation_blocks(
                trigger, events, settings.mask_above, config.dark_hours.sun_alt_limit_deg
            )
            text = f"Airmass through the first night for {trigger.source}"
            label = f"{trigger.source}_airmass"
            if report.posted is not None:
                poster.reply(report.posted, blocks, text, files=[path], label=label)
            else:
                poster.post(blocks, text, is_test=trigger.is_test, files=[path], label=label)
        except Exception as exc:
            logger.error("Could not post the airmass animation for %s: %s", trigger.source, exc)
            report.warnings.append(f"airmass animation post failed: {exc}")

    thread = threading.Thread(target=render_and_post, name=f"airmass-{trigger.source}", daemon=True)
    thread.start()
    report.animation_thread = thread


def post_sim_results(
    result: SimResult,
    config: Config,
    poster: SlackPoster,
    parent: PostedMessage | None = None,
    is_test: bool = False,
) -> list[PostedMessage]:
    """Post a finished simulation: coverage first, then the nightly figures.

    Two separate messages, because they are read at different times and one
    would drown the other: a 20-night run attaches many figures.

    Parameters
    ----------
    result : `SimResult`
        Completed simulation.
    config : `Config`
        Configuration.
    poster : `SlackPoster`
        Delivery mechanism. An offline or dry-run poster writes both payloads
        to disk instead, which is how a run is previewed.
    parent : `PostedMessage`, optional
        The alert's own message, to thread the coverage onto. Without one -- a
        dry run, a `simulate` invocation, or a stage-1 post that failed -- the
        coverage is posted standalone rather than dropped.
    is_test : `bool`
        Route to the test channel when one is configured.

    Returns
    -------
    posted : `list` [`PostedMessage`]
        What actually went out, for reporting. Empty when both posts failed.
    """
    sent: list[PostedMessage] = []

    blocks = build_sim_reply_blocks(result, config)
    summary = f"Simulation for {result.source}: " + (
        ", ".join(f"{b} {100 * v:.0f}%" for b, v in result.coverage.items() if v > 0)
        or (result.error or "no coverage")
    )
    try:
        if parent is not None:
            posted = poster.reply(parent, blocks, summary, label=f"{result.source}_sim")
        else:
            posted = poster.post(blocks, summary, is_test=is_test, label=f"{result.source}_sim")
        if posted is not None:
            sent.append(posted)
    except Exception as exc:
        logger.error("Could not post the simulation coverage: %s", exc)

    # Every figure together, in a message of its own, and caught separately:
    # losing the numbers must not also lose the figures.
    figures = sim_figures(result)
    if figures:
        try:
            sent.append(
                poster.post(
                    build_sim_figures_blocks(result, config),
                    f"Simulated coverage of {result.source}: {len(figures)} figure(s), "
                    f"run in {result.job_dir}",
                    is_test=is_test,
                    files=figures,
                    label=f"{result.source}_figures",
                )
            )
        except Exception as exc:
            logger.error("Could not post the simulation figures: %s", exc)

    return sent


def post_failure(
    what: str,
    error: BaseException | str,
    poster: SlackPoster | None,
    source: str = "",
    origin: str = "",
    log_tail: str = "",
    is_test: bool = False,
) -> PostedMessage | None:
    """Tell the channel that chatterbox failed. Never raises.

    The whole point is to be reached from an ``except`` block, so it cannot add
    a second failure on top of the first: everything here, including building
    the blocks, is inside its own guard.

    Parameters
    ----------
    what : `str`
        What was being attempted, as a phrase.
    error : `BaseException` or `str`
        The failure to report.
    poster : `SlackPoster` or None
        Delivery. None -- a dry run with no poster -- logs and returns.
    source : `str`
        Event identifier, when known.
    origin : `str`
        Where the record came from.
    log_tail : `str`
        Last lines of a relevant log.
    is_test : `bool`
        Route to the test channel, when one is configured.

    Returns
    -------
    posted : `PostedMessage` or None
        None when there was no poster, or when posting the failure also failed.
    """
    logger.error("Failure while %s (%s): %s", what, source or "no event", error, exc_info=True)
    if poster is None:
        return None
    label = f"{source or 'chatterbox'}_failure"
    try:
        blocks = build_failure_blocks(what, error, source=source, origin=origin, log_tail=log_tail)
        summary = f"chatterbox failed while {what}"
        if source:
            summary += f" ({source})"
        return poster.post(blocks, summary, is_test=is_test, label=label)
    except Exception as exc:
        # Nothing left to do but say so in the log: the channel is exactly what
        # is not working.
        logger.error("Could not post the failure notice either: %s", exc)
        return None


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
    report.sim_job = job

    def finish() -> None:
        # This runs in a background thread, where an uncaught exception is
        # invisible: it kills the thread, the interpreter says nothing, and the
        # symptom is a simulation that never posts.
        try:
            job.wait(timeout=config.sim.timeout_s)
            result = load_sim_result(job.job_dir)
            if result is None:
                # The driver died before writing a result, so there is nothing
                # to build a coverage post from. Say so in the channel: the
                # alert's thread otherwise just stops.
                post_failure(
                    f"simulating {trigger.source}",
                    f"the simulation produced no result.json in {job.job_dir}",
                    poster,
                    source=trigger.source,
                    origin=str(job.job_dir),
                    log_tail=job.tail_log(),
                    is_test=trigger.is_test,
                )
                return
            logger.info("Simulation for %s finished with status %s", trigger.source, result.status)
            if poster is None:
                return
            # `report.posted` is None in a dry run, and when stage 1 could not
            # be posted. Neither is a reason to throw the simulation away, so
            # the results still go out -- standalone rather than threaded.
            post_sim_results(
                result,
                config,
                poster,
                parent=report.posted,
                is_test=trigger.is_test,
            )
        except Exception as exc:
            post_failure(
                f"posting the simulation of {trigger.source}",
                exc,
                poster,
                source=trigger.source,
                origin=str(job.job_dir),
                log_tail=job.tail_log(),
                is_test=trigger.is_test,
            )

    if sim_wait:
        finish()
    else:
        thread = threading.Thread(target=finish, name=f"sim-{trigger.source}", daemon=True)
        thread.start()
        # The caller has to keep the process alive until this finishes; see
        # `TriggerReport.wait_for_simulation`.
        report.sim_thread = thread


def run_service(
    config: Config,
    paths=None,
    run_sim: bool = True,
    lookback_s: float | None = None,
    run_animation: bool = True,
) -> int:
    """Consume the configured source and post about every record.

    Parameters
    ----------
    config : `Config`
        Configuration.
    paths : sequence, optional
        Explicit record paths, forcing replay mode.
    run_sim : `bool`
        Launch simulations.
    lookback_s : `float`, optional
        Override for the EFD first-poll lookback (from ``serve
        --since/--lookback``). `None` keeps ``ingest.efd_lookback_s``.
    run_animation : `bool`
        Render and post the airmass animation under each alert.

    Returns
    -------
    count : `int`
        Records handled.
    """
    from .ingest.source import make_source

    source = make_source(config, paths=paths, lookback_s=lookback_s)
    poster = SlackPoster(config)
    handled = 0
    pending: list[TriggerReport] = []
    try:
        for record, metadata in source:
            origin = metadata.get("origin") or metadata.get("topic") or metadata.get("transport") or ""
            try:
                report = process_trigger(
                    record, config, poster=poster, run_sim=run_sim, run_animation=run_animation
                )
                pending = [r for r in pending if not r.wait_for_simulation(timeout=0)]
                if report.has_background_work:
                    pending.append(report)
                handled += 1
            except Exception as exc:
                # A dropped alert must never be silent. `record` may not have
                # decoded, so its fields are read defensively -- the identifier
                # and the test flag are the two worth trying for.
                identifier = record.get("source") if isinstance(record, dict) else None
                is_test = bool(record.get("is_test")) if isinstance(record, dict) else False
                post_failure(
                    "handling a ToO record",
                    exc,
                    poster,
                    source=str(identifier or ""),
                    origin=str(origin),
                    is_test=is_test,
                )
            finally:
                # Acknowledge either way: a record that cannot be processed
                # will
                # not become processable on redelivery, and blocking the stream
                # on it would stall every later alert.
                source.mark_done(metadata)
    except KeyboardInterrupt:
        logger.info("Interrupted; shutting down after %d record(s)", handled)
    except Exception as exc:
        # The stream itself failed -- the EFD went away, credentials expired,
        # the broker dropped us. The service is about to stop consuming alerts,
        # which is precisely the thing nobody would otherwise notice. Name what
        # stopped as specifically as the source can: "the summit_efd EFD" is
        # actionable in a way that "efd" is not.
        describe = getattr(source, "describe", None)
        origin = describe() if callable(describe) else config.ingest.kind
        post_failure(
            f"monitoring for ToO alerts ({config.ingest.kind})",
            exc,
            poster,
            origin=str(origin),
        )
        raise
    finally:
        source.close()
        drain_simulations(pending)
    return handled


def drain_simulations(pending: list[TriggerReport], grace_s: float = 10.0) -> None:
    """Give running simulations a moment to post, then say where they are.

    A simulation outlives the service that started it, so shutting down while
    one is in flight would otherwise lose its post silently. Waiting the full
    run out is not an option during a shutdown, so anything still going is
    named along with the command that posts it afterwards.
    """
    for report in pending:
        if report.wait_for_simulation(timeout=grace_s):
            continue
        job_dir = getattr(report.sim_job, "job_dir", "the job directory")
        logger.warning(
            "Simulation for %s is still running; its results were not posted. "
            "Post them when it finishes with: chatterbox post-sim %s",
            report.trigger.source,
            job_dir,
        )
