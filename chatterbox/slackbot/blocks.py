"""Slack Block Kit message construction.

The first post is deliberately dense: an observer reading it on a phone should
be able to decide whether to care without opening anything. Every number that
depends on provenance is labelled -- in particular, coverage and containment
are
described as *area* rather than *probability* whenever the only localization
available was the producer's binary reward map.
"""

import logging
from typing import Any

import numpy as np

from ..alerts import AlertClass, FollowupStrategy, get_alert_class, get_strategy
from ..astro.almanac import CHILE_TZ, NightEvents, format_time, moon_separation_deg
from ..astro.darkhours import DarkHoursMap
from ..astro.templates import TemplateCoverage
from ..config import Config
from ..links import format_link_list, gracedb_link, site_links
from ..models import Trigger
from ..sim.runner import SimResult

__all__ = [
    "build_trigger_blocks",
    "build_sim_reply_blocks",
    "plain_text_summary",
    "MAX_SECTION_CHARS",
]

logger = logging.getLogger(__name__)

#: Slack rejects section text longer than this.
MAX_SECTION_CHARS = 2900


def _section(text: str) -> dict[str, Any]:
    """A mrkdwn section block, truncated to Slack's limit."""
    if len(text) > MAX_SECTION_CHARS:
        text = text[: MAX_SECTION_CHARS - 3] + "..."
    return {"type": "section", "text": {"type": "mrkdwn", "text": text}}


def _context(text: str) -> dict[str, Any]:
    return {"type": "context", "elements": [{"type": "mrkdwn", "text": text[:2900]}]}


def _divider() -> dict[str, Any]:
    return {"type": "divider"}


def _fmt(value: float | None, spec: str = ".1f", suffix: str = "") -> str:
    """Format a possibly-missing number."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "unknown"
    return f"{value:{spec}}{suffix}"


def _duration(seconds: float | None) -> str:
    """Format a duration with a unit that suits its magnitude.

    Alert ages span seconds (a live trigger) to months (a replay of an archived
    event), so a fixed unit is unreadable at one end or the other.
    """
    if seconds is None or not np.isfinite(seconds):
        return "unknown"
    seconds = float(seconds)
    sign = "-" if seconds < 0 else ""
    seconds = abs(seconds)
    if seconds < 90:
        return f"{sign}{seconds:.0f} s"
    if seconds < 5400:
        return f"{sign}{seconds / 60:.1f} min"
    if seconds < 172800:
        return f"{sign}{seconds / 3600:.1f} h"
    return f"{sign}{seconds / 86400:.1f} d"


# --------------------------------------------------------------------- header


def _header_blocks(trigger: Trigger, alert_class: AlertClass) -> list[dict[str, Any]]:
    test_marker = " :test_tube: *TEST ALERT*" if trigger.is_test else ""
    update_marker = " :arrows_counterclockwise: *UPDATE*" if trigger.is_update else ""
    header = (
        f"{alert_class.messenger.emoji} *{alert_class.messenger.value}: {trigger.source}*"
        f"{test_marker}{update_marker}"
    )
    subtitle = f"`{trigger.alert_type}` -- {alert_class.display_name}"
    instruments = ", ".join(trigger.clean_instruments) or "unknown"

    timing = []
    if trigger.event_time is not None:
        timing.append(f"Event: {format_time(trigger.event_time)}")
    if trigger.age_s is not None:
        timing.append(f"age {_duration(trigger.age_s)}")
    if trigger.latency_s is not None:
        timing.append(f"producer latency {_duration(trigger.latency_s)}")

    lines = [subtitle, f"*Instruments:* {instruments}"]
    if timing:
        lines.append(" | ".join(timing))
    if not alert_class.produced:
        lines.append(
            ":warning: No current producer emits this alert type, so this rendering is untested "
            "against real data."
        )
    if alert_class.notes:
        lines.append(f"_{alert_class.notes}_")

    return [
        {"type": "header", "text": {"type": "plain_text", "text": _plain_header(trigger, alert_class)}},
        _section(header + "\n" + "\n".join(lines)),
    ]


def _plain_header(trigger: Trigger, alert_class: AlertClass) -> str:
    """Plain-text header, which Slack requires to be <= 150 chars and
    emoji-free.
    """
    prefix = "[TEST] " if trigger.is_test else ""
    text = f"{prefix}{trigger.source}: {alert_class.display_name}"
    return text[:150]


# ------------------------------------------------------------ alert specifics


def _alert_blocks(trigger: Trigger, alert_class: AlertClass) -> list[dict[str, Any]]:
    geo = trigger.geometry
    loc = trigger.localization
    quantity = loc.quantity_name

    lines = [f"*Localization* ({loc.credible_level:.0%} credible region, {quantity}-weighted)"]
    lines.append(
        f"Area {_fmt(geo.area_deg2, ',.0f', ' deg^2')}  |  "
        f"centroid RA {_fmt(geo.centroid_ra_deg, '.2f')}, Dec {_fmt(geo.centroid_dec_deg, '.2f')}"
    )
    lines.append(
        f"Dec range {_fmt(geo.dec_min_deg, '.1f')} to {_fmt(geo.dec_max_deg, '.1f')} deg  |  "
        f"|b| {_fmt(geo.gal_b_abs_min_deg, '.1f')} to {_fmt(geo.gal_b_abs_max_deg, '.1f')} deg"
    )
    if np.isfinite(geo.dec_min_deg) and geo.dec_min_deg > 30.0:
        lines.append(":warning: Entire region is north of Dec +30; not reachable from Cerro Pachon.")

    blocks = [_section("\n".join(lines))]

    enrichment = trigger.enrichment
    if enrichment is not None and enrichment.error is None:
        detail = []
        if enrichment.area_90_deg2 is not None:
            detail.append(
                f"90% area {enrichment.area_90_deg2:,.0f} deg^2, "
                f"50% area {_fmt(enrichment.area_50_deg2, ',.0f', ' deg^2')}"
            )
        if enrichment.distance_mean_mpc is not None:
            detail.append(
                f"distance {enrichment.distance_mean_mpc:,.0f} +/- "
                f"{_fmt(enrichment.distance_std_mpc, ',.0f')} Mpc"
            )
        if enrichment.inverse_far_years is not None:
            detail.append(f"FAR 1 per {enrichment.inverse_far_years:,.1f} yr")
        most_likely = enrichment.most_likely_class
        if most_likely is not None:
            detail.append(f"most likely {most_likely[0]} ({most_likely[1]:.0%})")
        if enrichment.classification:
            detail.append(
                "classification: "
                + ", ".join(f"{k} {v:.0%}" for k, v in sorted(enrichment.classification.items()))
            )
        if enrichment.properties:
            detail.append(
                "properties: " + ", ".join(f"{k} {v:.2f}" for k, v in sorted(enrichment.properties.items()))
            )
        if enrichment.chirp_mass_median is not None:
            chirp = f"median chirp mass {enrichment.chirp_mass_median:.2f} Msun"
            if enrichment.chirp_mass_prob_above_44 is not None:
                chirp += f", P(>44 Msun) {enrichment.chirp_mass_prob_above_44:.0%}"
            detail.append(chirp)
        pipeline = " / ".join(x for x in (enrichment.group, enrichment.pipeline, enrichment.search) if x)
        if pipeline:
            detail.append(f"pipeline {pipeline}")
        if enrichment.significant is not None:
            detail.append("significant" if enrichment.significant else "*not* significant")
        if detail:
            blocks.append(_section("*Event properties* (from GraceDB)\n" + "\n".join(detail)))
    elif enrichment is not None and enrichment.error:
        blocks.append(
            _context(
                f":warning: GraceDB enrichment failed ({enrichment.error}); event properties and "
                "the true probability map are unavailable, so figures below are area-based."
            )
        )

    blocks.append(_section(f"*Why it triggered*\n{alert_class.criteria}"))
    return blocks


# ------------------------------------------------------------- observability


def _observability_blocks(
    trigger: Trigger,
    events: NightEvents,
    dark_hours: DarkHoursMap,
    dark_stats: dict[str, float],
    config: Config,
) -> list[dict[str, Any]]:
    geo = trigger.geometry
    moon_sep = moon_separation_deg(geo.centroid_ra_deg, geo.centroid_dec_deg, events)

    def span(start, end, hours: float, label: str) -> str:
        """One 'start -> end (duration)' line, UTC with local in brackets."""
        return (
            f"{label}: {format_time(start)} [{format_time(start, CHILE_TZ)}]"
            f"  ->  {format_time(end)} [{format_time(end, CHILE_TZ)}]"
            f"  ({hours:.2f} h)"
        )

    sun_lines = [
        f"*Night of {events.day_obs}* (times UTC, local Chile in brackets)",
        f"Sunset {format_time(events.sunset)} [{format_time(events.sunset, CHILE_TZ)}]  ->  "
        f"sunrise {format_time(events.sunrise)} [{format_time(events.sunrise, CHILE_TZ)}]",
        span(
            events.sun_n12_setting,
            events.sun_n12_rising,
            events.night_length_hours,
            "Sun < -12 deg",
        ),
        span(
            events.sun_n18_setting,
            events.sun_n18_rising,
            events.dark_length_hours,
            "Sun < -18 deg",
        ),
    ]

    moon_lines = [
        f"*Moon:* {events.moon_phase:.0f}% illuminated",
        f"Moonrise {format_time(events.moonrise)} [{format_time(events.moonrise, CHILE_TZ)}]  |  "
        f"moonset {format_time(events.moonset)} [{format_time(events.moonset, CHILE_TZ)}]",
        f"Separation from localization centroid: {_fmt(moon_sep, '.1f', ' deg')}",
    ]
    if events.moonrise is None and events.moonset is None:
        moon_lines.append("_The Moon neither rises nor sets during this night._")

    within = (
        f"Within the localization: max {_fmt(dark_stats.get('max_hours'), '.2f', ' h')}, "
        f"mean {_fmt(dark_stats.get('mean_hours'), '.2f', ' h')}"
    )
    weighted = quantity_weighted(dark_stats, trigger)
    if weighted:
        within += f", {weighted}"
    dark_lines = [
        f"*Accessible dark hours* (airmass < {dark_hours.airmass_limit:g}, i.e. altitude > "
        f"{dark_hours.altitude_limit_deg:.0f} deg, while Sun < {dark_hours.sun_alt_limit_deg:g} deg)",
        within,
        f"{dark_stats.get('fraction_accessible', 0.0):.0%} of the region reaches airmass < "
        f"{dark_hours.airmass_limit:g} at some point during this night "
        f"({dark_hours.window_hours:.2f} h available)",
    ]
    if dark_stats.get("fraction_accessible", 0.0) == 0.0:
        dark_lines.append(
            ":warning: No part of the localization reaches the airmass limit during this night."
        )

    links = site_links(config.links)
    blocks = [
        _section("\n".join(sun_lines)),
        _section("\n".join(moon_lines)),
        _section("\n".join(dark_lines)),
    ]
    if links:
        blocks.append(_section("*Site conditions*\n" + format_link_list(links)))
    return blocks


def quantity_weighted(dark_stats: dict[str, float], trigger: Trigger) -> str:
    """Phrase the weighted dark-hours mean, naming what it is weighted by."""
    value = dark_stats.get("weighted_mean_hours")
    if value is None or not np.isfinite(value):
        return ""
    return f"{trigger.localization.quantity_name}-weighted mean {value:.2f} h"


# ---------------------------------------------------------------- templates


def _template_blocks(
    trigger: Trigger,
    coverage: TemplateCoverage | None,
) -> list[dict[str, Any]]:
    if coverage is None:
        return [
            _context(
                ":warning: No template coverage cache found, so the template comparison is missing. "
                "Run `chatterbox refresh-templates` to build it."
            )
        ]
    fractions = coverage.coverage_in_region(trigger.localization.prob_map, nside=trigger.localization.nside)
    quantity = trigger.localization.quantity_name
    per_band = "  |  ".join(f"{band} {fractions[band]:.0%}" for band in ("u", "g", "r", "i", "z", "y"))
    lines = [
        f"*Existing template coverage* (fraction of localization {quantity} with >= "
        f"{coverage.min_visits} prior visit)",
        per_band,
    ]
    if coverage.built_at:
        lines.append(f"_Cache built {coverage.built_at} from {coverage.source or 'unknown source'}._")
    return [_section("\n".join(lines))]


# ----------------------------------------------------------------- strategy


def _strategy_blocks(strategy: FollowupStrategy, alert_class: AlertClass) -> list[dict[str, Any]]:
    if not strategy.epochs:
        return [
            _context(
                f":warning: No follow-up strategy is defined for `{strategy.alert_type}`, so no "
                "simulation will be run."
            )
        ]
    per_band = strategy.visits_per_band()
    epoch_text = ", ".join(f"+{e.t_hours:g} h {e.bands}x{e.nvis}" for e in strategy.epochs[:8])
    if len(strategy.epochs) > 8:
        epoch_text += f", ... ({len(strategy.epochs)} epochs total)"
    source_note = "ts_fbs_utils" if strategy.from_ts_fbs_utils else "vendored snapshot"
    lines = [
        "*Planned Rubin follow-up*",
        "Requested visits: " + ", ".join(f"{b} x{n}" for b, n in per_band.items()),
        f"Epochs: {epoch_text}",
        f"Spans {strategy.span_hours:g} h; simulation will cover {alert_class.sim_nights} nights.",
        f"_Strategy read from {source_note}._",
    ]
    return [_section("\n".join(lines))]


# -------------------------------------------------------------------- public


def build_trigger_blocks(
    trigger: Trigger,
    events: NightEvents,
    dark_hours: DarkHoursMap,
    dark_stats: dict[str, float],
    template_coverage: TemplateCoverage | None,
    config: Config,
) -> list[dict[str, Any]]:
    """Build the Block Kit payload for the first post about a trigger.

    Parameters
    ----------
    trigger : `Trigger`
        The decoded (and possibly enriched) trigger.
    events : `NightEvents`
        Sun and Moon events for the relevant night.
    dark_hours : `DarkHoursMap`
        Accessible dark hours map.
    dark_stats : `dict`
        Output of `chatterbox.plots.darkhours.region_hours_summary`.
    template_coverage : `TemplateCoverage` or None
        Cached template maps, or None when unavailable.
    config : `Config`
        Configuration, for links.

    Returns
    -------
    blocks : `list` [`dict`]
    """
    alert_class = get_alert_class(trigger.alert_type)
    strategy = get_strategy(trigger.alert_type)

    blocks: list[dict[str, Any]] = []
    blocks += _header_blocks(trigger, alert_class)
    blocks.append(_divider())
    blocks += _alert_blocks(trigger, alert_class)
    blocks.append(_divider())
    blocks += _observability_blocks(trigger, events, dark_hours, dark_stats, config)
    blocks.append(_divider())
    blocks += _template_blocks(trigger, template_coverage)
    blocks += _strategy_blocks(strategy, alert_class)

    footer = [f"Localization source: {trigger.localization.provenance}."]
    if not trigger.localization.is_probability:
        footer.append(
            "Percentages are *area* fractions: the producer's record carries only a binary "
            "credible region, not a probability density."
        )
    links = [("GraceDB", gracedb_link(trigger.source))] if trigger.enrichment else []
    if links:
        footer.append(format_link_list(links))
    blocks.append(_context(" ".join(footer)))

    # Slack caps a message at 50 blocks.
    if len(blocks) > 50:
        logger.warning("Trimming %d blocks to Slack's limit of 50", len(blocks))
        blocks = blocks[:49] + [_context("_Message truncated._")]
    return blocks


def build_sim_reply_blocks(result: SimResult, config: Config) -> list[dict[str, Any]]:
    """Build the threaded reply carrying simulation results.

    Parameters
    ----------
    result : `SimResult`
        Outcome from the simulation driver.
    config : `Config`
        Configuration, for the artifact base URL.

    Returns
    -------
    blocks : `list` [`dict`]
    """
    if not result.ok:
        detail = result.error or "no further detail"
        return [
            _section(
                f":x: *Simulation for {result.source} did not produce coverage*\n"
                f"Status `{result.status}`: {detail}"
            )
        ]

    quantity = result.quantity
    per_band = "  |  ".join(
        f"{band} *{100 * result.coverage.get(band, 0.0):.1f}%*" for band in ("u", "g", "r", "i", "z", "y")
    )
    lines = [
        f":telescope: *Simulated {result.nights}-night follow-up of {result.source}* "
        f"(`{result.alert_type}`)",
        f"Expected fraction of localization {quantity} covered, per band:",
        per_band,
        f"Any band: *{100 * result.any_band:.1f}%*",
        f"{result.too_visits:,} ToO visits out of {result.total_visits:,} total; "
        f"ran in {result.runtime_s / 60.0:.1f} min.",
    ]
    if not result.is_probability_result():
        lines.append(
            ":warning: These are *area* fractions, not probabilities: no probability skymap was "
            "available for this event."
        )
    blocks = [_section("\n".join(lines))]

    if result.coverage_by_epoch:
        epoch_lines = []
        for epoch in sorted(result.coverage_by_epoch, key=lambda k: int(k)):
            fractions = result.coverage_by_epoch[epoch]
            covered = ", ".join(f"{b} {100 * v:.0f}%" for b, v in fractions.items() if v > 0)
            epoch_lines.append(f"epoch {epoch}: {covered or 'nothing scheduled'}")
        blocks.append(_section("*Per epoch*\n" + "\n".join(epoch_lines)))

    notes = []
    if result.band_scheduler:
        notes.append(f"Filter carousel: {result.band_scheduler}")
    if config.sim.artifact_base_url and result.visits_path:
        name = result.visits_path.rsplit("/", 1)[-1]
        notes.append(f"<{config.sim.artifact_base_url.rstrip('/')}/{name}|simulation output>")
    elif result.visits_path:
        notes.append(f"Output: `{result.visits_path}`")
    if notes:
        blocks.append(_context(" -- ".join(notes)))
    return blocks


def plain_text_summary(trigger: Trigger, alert_class: AlertClass | None = None) -> str:
    """Short fallback text, for notifications and refused blocks."""
    alert_class = alert_class or get_alert_class(trigger.alert_type)
    prefix = "[TEST] " if trigger.is_test else ""
    return (
        f"{prefix}{alert_class.messenger.value} {trigger.source} ({trigger.alert_type}): "
        f"{trigger.geometry.area_deg2:,.0f} deg^2, "
        f"Dec {trigger.geometry.dec_min_deg:.0f} to {trigger.geometry.dec_max_deg:.0f}"
    )
