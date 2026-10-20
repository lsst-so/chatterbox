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
    "crew_blocks",
    "build_sim_reply_blocks",
    "build_sim_figures_blocks",
    "build_failure_blocks",
    "sim_figures",
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


# ----------------------------------------------------------------------- crew


def _mention(handle: str, ids: dict[str, str]) -> str:
    """A Slack mention of a user group, which only pings when it has an ID.

    ``@handle`` typed as text is displayed but notifies nobody, so a group
    whose ID is unknown renders as that plain text and `crew_blocks` says so.
    """
    group_id = ids.get(handle, "")
    return f"<!subteam^{group_id}>" if group_id else f"@{handle}"


def crew_blocks(
    trigger: Trigger,
    config: Config,
    ids: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Tell the people on shift that they have a job to do.

    The summit support scientist is named as the decision-maker; the day and
    night observing specialists are told to be aware, contact them as soon as
    possible, and wait for instructions rather than act on their own.

    Parameters
    ----------
    trigger : `Trigger`
        The alert, for its test flag and age.
    config : `Config`
        Configuration naming the groups; see `chatterbox.config.CrewConfig`.
    ids : `dict` [`str`, `str`], optional
        User group IDs by handle, from `SlackPoster.crew_ids`. A group missing
        from it is written as plain text, which does not notify anyone, and
        the post says so. None means the IDs could not be looked up at all --
        a dry run -- and is rendered without that warning, which would
        otherwise be a claim about a post that was never sent.

    Returns
    -------
    blocks : `list` [`dict`]
        Empty when the crew is switched off. When the crew is deliberately
        *not* being tagged -- a test alert, or an event too old to act on --
        a single context line says so, so the absence cannot be mistaken for
        the feature being broken.
    """
    crew = config.slack.crew
    handles = crew.handles()
    if not crew.enabled or not handles:
        return []

    looked_up = ids is not None
    ids = {**crew.ids, **(ids or {})}
    lead = crew.lead.strip().lstrip("@")
    notify = [h for h in (n.strip().lstrip("@") for n in crew.notify) if h and h != lead]
    names = ", ".join(f"@{h}" for h in handles)

    if trigger.is_test:
        return [_context(f":test_tube: Test alert, so the shift was not tagged (a real one tags {names}).")]

    age_s = trigger.age_s
    if crew.max_age_hours > 0 and age_s is not None and age_s > crew.max_age_hours * 3600.0:
        return [
            _context(
                f":hourglass: The event is {_duration(age_s)} old, so the shift was not tagged "
                f"(`slack.crew.max_age_hours` is {crew.max_age_hours:g})."
            )
        ]

    lines = []
    if lead:
        lines.append(
            f":rotating_light: {_mention(lead, ids)} -- you are the summit support scientist on shift for "
            "this event, and you have the authority to make decisions about it."
        )
    if notify:
        lines.append(
            f"{' '.join(_mention(h, ids) for h in notify)} -- please be aware of this event. Get in "
            "contact with your summit support scientist as soon as possible, and await further "
            "instructions from them."
        )

    blocks = [_section("\n".join(lines))]
    unresolved = [h for h in handles if not ids.get(h)]
    if unresolved and looked_up:
        # Loud, because a post that looks tagged and notifies nobody is worse
        # than one that is plainly untagged.
        blocks.append(
            _context(
                ":warning: Not notified: no Slack user group ID for "
                + ", ".join(f"@{h}" for h in unresolved)
                + ". Set `slack.crew.ids` or give the bot the `usergroups:read` scope."
            )
        )
    return blocks


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
        f"*Existing template coverage* (fraction of localization {quantity} with a template)",
        per_band,
    ]
    if coverage.missing_bands:
        lines.append(f"_No coverage map published for: {', '.join(coverage.missing_bands)}._")
    if coverage.built_at:
        lines.append(f"_Cache refreshed {coverage.built_at} from {coverage.source or 'unknown source'}._")
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
    crew_ids: dict[str, str] | None = None,
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
        Configuration, for links and the shift crew.
    crew_ids : `dict` [`str`, `str`], optional
        Slack user group IDs by handle; see `crew_blocks`.

    Returns
    -------
    blocks : `list` [`dict`]
    """
    alert_class = get_alert_class(trigger.alert_type)
    strategy = get_strategy(trigger.alert_type)

    blocks: list[dict[str, Any]] = []
    blocks += _header_blocks(trigger, alert_class)
    # Straight after the header: this is the part of the post somebody has to
    # act on, and it should not be below the fold on a phone.
    blocks += crew_blocks(trigger, config, crew_ids)
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


def _band_row(fractions: dict[str, float]) -> str:
    """One row of per-band percentages, in canonical band order."""
    bands = ("u", "g", "r", "i", "z", "y")
    return "  ".join(f"{band} {100 * fractions.get(band, 0.0):.1f}%" for band in bands)


def _any(fraction: float) -> str:
    """The any-band total, formatted to close a band row."""
    return f"  |  any band *{100 * fraction:.1f}%*"


def _campaign_line(result: SimResult) -> str:
    """What the follow-up cost the survey, or "" when it was not measured.

    The other half of judging a strategy: coverage bought, against the
    observing time it took from the nominal cadence on the same nights.
    """
    timing = result.campaign_time or {}
    seconds = float(timing.get("too_seconds", 0.0))
    if seconds <= 0:
        return ""
    nights = int(timing.get("nights", 0))
    line = (
        f":stopwatch: *Campaign cost* {seconds / 60.0:.0f} min of observing time "
        f"over {nights} night{'s' if nights != 1 else ''}"
    )
    fraction = float(timing.get("fraction", 0.0))
    if fraction > 0:
        line += f", *{100 * fraction:.1f}%* of everything observed on those nights"
    span = float(timing.get("span_hours", 0.0))
    if span > 0:
        line += f"; the follow-up spanned {span:.1f} h"
    measure = str(timing.get("measure", ""))
    if measure and "no visittime" in measure:
        # Say so rather than let an undercount read as a cheap campaign.
        line += f"\n_Time from `{measure}`: slew and readout are not included, so this is a floor._"
    return line


def _region_lines(result: SimResult) -> list[str]:
    """Coverage of the credible region itself: probability and area.

    This is issue #1's question, and it is a different one from the rows
    above. Those are fractions of the whole map -- on a real skymap the tails
    outside the contour are included, which is why an any-band figure can
    exceed the credible level. These are fractions of the 90% region the
    follow-up is actually planned against, and probability and area are
    reported separately because neither implies the other: covering the
    densest part of a region buys most of its probability while leaving most
    of its sky unobserved.
    """
    region = result.region_coverage or {}
    if not result.has_region_coverage():
        return []
    level = 100 * float(region.get("credible_level", 0.9))
    area = float(region.get("region_area_deg2", 0.0))
    is_probability = bool(region.get("is_probability", False))

    def row(label: str, prob, frac) -> str:
        covered = float(frac or 0.0)
        cells = []
        if is_probability and prob is not None:
            cells.append(f"probability *{100 * float(prob):.1f}%*")
        cells.append(f"area {100 * covered:.1f}% ({covered * area:,.0f} deg^2)")
        return f"*{label}*  " + "  |  ".join(cells)

    lines = [f"*Within the {level:.0f}% localization* ({area:,.0f} deg^2)"]
    lines.append(row("ToO follow-up only", region.get("too_probability"), region.get("too_area")))
    if region.get("combined_area") is not None:
        lines.append(
            row(
                "+ nominal survey",
                region.get("combined_probability"),
                region.get("combined_area"),
            )
        )
    if not is_probability:
        lines.append(
            f"_No probability skymap, so only the area is measurable; the region is the "
            f"producer's {level:.0f}% reward map._"
        )
    return lines


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
    lines = [
        f":telescope: *Simulated {result.nights}-night follow-up of {result.source}* "
        f"(`{result.alert_type}`)",
        f"Expected fraction of total localization {quantity} covered, per band "
        f"(the whole map, tails included):",
    ]
    # Two rows, labelled: the ToO strategy on its own, and what the
    # localization actually receives once nominal-cadence visits landing on it
    # are counted. Without the labels a single row invites both readings.
    lines.append("*ToO follow-up only*  " + _band_row(result.coverage) + _any(result.any_band))
    if result.has_combined_coverage():
        gain = result.survey_gain.get("any", result.any_band_combined - result.any_band)
        lines.append(
            "*+ nominal survey*  "
            + _band_row(result.coverage_combined)
            + _any(result.any_band_combined)
            + f"  ({100 * gain:+.1f} points from survey visits)"
        )
        lines.append(
            f"_The survey re-covers what the follow-up already reached, so only that "
            f"difference is new {quantity}; {result.survey_visits_on_localization:,} "
            f"non-ToO visits touched the localization._"
        )
    lines.append(
        f"{result.too_visits:,} ToO visits out of {result.total_visits:,} total; "
        f"ran in {result.runtime_s / 60.0:.1f} min."
    )
    lines.extend(_region_lines(result))
    campaign = _campaign_line(result)
    if campaign:
        lines.append(campaign)
    if not result.is_probability_result():
        lines.append(
            ":warning: These are *area* fractions, not probabilities: no probability skymap was "
            "available for this event."
        )
    if sim_figures(result):
        # Say where they went, so their absence here does not read as "none".
        lines.append("_Every figure from this run is posted in a thread of its own._")
    blocks = [_section("\n".join(lines))]

    if result.coverage_by_epoch:
        epoch_lines = []
        for epoch in sorted(result.coverage_by_epoch, key=lambda k: int(k)):
            fractions = result.coverage_by_epoch[epoch]
            covered = ", ".join(f"{b} {100 * v:.0f}%" for b, v in fractions.items() if v > 0)
            epoch_lines.append(f"epoch {epoch}: {covered or 'nothing scheduled'}")
        blocks.append(_section("*Per epoch*\n" + "\n".join(epoch_lines)))

    if result.cumulative_by_night:
        # The same progression the per-night figure draws, for readers who are
        # scanning the thread on a phone and will not open an image.
        heading = "*Per night* (cumulative, any band)"
        if result.trigger_night is not None:
            heading += ", counted from the trigger night"
        section = heading + "\n" + _night_progression(result)
        time_line = _night_time_line(result)
        if time_line:
            section += "\n*Follow-up time per night*\n" + time_line
        blocks.append(_section(section))

    notes = []
    if result.opsim:
        notes.append(f"Visit history: {result.opsim}")
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


def _night_progression(result: SimResult, max_nights: int = 12) -> str:
    """Cumulative any-band coverage through each night, on one line.

    A long run is elided in the middle rather than truncated, so the number
    that matters -- where it ended up -- is always the last one shown.

    Parameters
    ----------
    result : `SimResult`
        Completed simulation with `SimResult.cumulative_by_night` populated.
    max_nights : `int`
        Most nights to name before eliding.

    Returns
    -------
    line : `str`
    """
    nights = sorted(result.cumulative_by_night, key=lambda k: int(k))
    parts = []
    for night in nights:
        # Already relative to the trigger, so keep the sign: "+0" is the
        # trigger night and "-1" a nominal-cadence pass before it.
        label = f"{int(night):+d}" if result.trigger_night is not None else str(night)
        parts.append(f"n{label} {100 * result.cumulative_by_night[night]:.0f}%")
    if len(parts) > max_nights:
        parts = parts[: max_nights - 1] + ["..."] + parts[-1:]
    return " → ".join(parts)


def _night_time_line(result: SimResult, max_nights: int = 12) -> str:
    """Follow-up minutes per night and the share of that night, on one line.

    Elided in the middle like `_night_progression`, and keyed the same way, so
    the two lines can be read against each other night for night.

    Parameters
    ----------
    result : `SimResult`
        Completed simulation with ``campaign_time["by_night"]`` populated.
    max_nights : `int`
        Most nights to name before eliding.

    Returns
    -------
    line : `str`
        Empty when the campaign time was not measured.
    """
    by_night = (result.campaign_time or {}).get("by_night") or {}
    if not by_night:
        return ""
    parts = []
    for night in sorted(by_night, key=lambda k: int(k)):
        entry = by_night[night]
        minutes = float(entry.get("too_s", 0.0)) / 60.0
        if minutes <= 0:
            continue
        label = f"{int(night):+d}" if result.trigger_night is not None else str(night)
        parts.append(f"n{label} {minutes:.0f} min ({100 * float(entry.get('fraction', 0.0)):.0f}%)")
    if not parts:
        return ""
    if len(parts) > max_nights:
        parts = parts[: max_nights - 1] + ["..."] + parts[-1:]
    return " → ".join(parts)


def sim_figures(result: SimResult) -> list[str]:
    """Every figure a finished simulation produced, in reading order.

    Widest view first: the cumulative curve answers "how much in total?", the
    per-night coverage figure breaks that total down by night, and the
    per-night maps then show where on the sky each of those nights went. They
    travel together so a reader has the whole picture in one place.

    Parameters
    ----------
    result : `SimResult`
        Completed simulation.

    Returns
    -------
    paths : `list` [`str`]
        Possibly empty, when every figure failed to render.
    """
    figures = [result.curve_plot] if result.curve_plot else []
    if result.nightly_coverage_plot:
        figures.append(result.nightly_coverage_plot)
    return figures + list(result.nightly_plots)


def build_sim_figures_blocks(result: SimResult, config: Config) -> list[dict[str, Any]]:
    """Blocks for the message carrying the simulation's figures.

    This starts its own Slack thread rather than replying in the alert's: a
    20-night run attaches many figures, and burying them under the alert would
    make both harder to read.

    Parameters
    ----------
    result : `SimResult`
        Completed simulation, with `SimResult.nightly_plots` or
        `SimResult.curve_plot` populated.
    config : `Config`
        Configuration, for the artifact base URL.

    Returns
    -------
    blocks : `list` [`dict`]
    """
    nights = result.nightly_plot_nights
    shown = len(result.nightly_plots)
    lines = [
        f":milky_way: *Simulated coverage of {result.source}* (`{result.alert_type}`)",
    ]
    # One line per figure, in upload order, so a reader can tell which image is
    # which without opening them. Numbered as they are added, since any of the
    # three can be absent.
    described = []
    if result.curve_plot:
        described.append(
            f"Cumulative coverage of the localization {result.quantity} against time, "
            f"per band, over the {result.nights}-night simulation. The black lines are "
            f"the any-band total: solid for the ToO follow-up alone, dashed once the "
            f"nominal-cadence visits that fell on the localization are added."
        )
    if result.nightly_coverage_plot:
        nightly_text = (
            "The same coverage night by night: what each band added on each night, "
            "with the running any-band total over it. Every night of the run is on "
            "it, from the same ToO follow-up visits the coverage numbers come from."
        )
        if (result.campaign_time or {}).get("by_night"):
            nightly_text += (
                " The lower panel is the observing time each night: the follow-up's "
                "share in red against everything observed in grey."
            )
        described.append(nightly_text)
    if result.nightly_plots:
        described.append(
            f"Per-band visit counts for every night whose visits overlap the "
            f"localization contour, with the contour drawn on each panel. "
            f"{result.nights_with_overlap} such night(s)."
        )
    lines.extend(f"{n}. {text}" for n, text in enumerate(described, start=1))
    if result.overlap_visits:
        # Both numbers are counted within the overlapping set, so the split is
        # a real split; an all-sky ToO count could exceed the overlap total.
        other = max(result.overlap_visits - result.overlap_too_visits, 0)
        lines.append(
            f"{result.overlap_visits:,} visits touch the localization: "
            f"{result.overlap_too_visits:,} ToO follow-up and "
            f"{other:,} from the nominal cadence."
        )
    if nights:
        if result.trigger_night is not None:
            labels = [f"{n - result.trigger_night:+d}" for n in nights]
            lines.append(
                f"Nights shown, counted from the trigger night: {', '.join(labels)} "
                f"(survey nights {nights[0]}-{nights[-1]})."
            )
            if nights[0] < result.trigger_night:
                # Otherwise a negative night reads as a mislabelled epoch.
                lines.append(
                    "A negative night is *before* the trigger: the run starts from the "
                    "trigger's `day_obs`, so its first night can precede the alert. Those "
                    "visits are nominal cadence, not follow-up."
                )
        else:
            lines.append(f"Nights shown: {', '.join(str(n) for n in nights)}.")
    if shown < result.nights_with_overlap:
        # Never let a cap look like "that was all there was".
        lines.append(
            f":scissors: Showing the first {shown} of {result.nights_with_overlap} nights "
            f"(`sim.nightly_plots_max_nights`); the rest are in the run directory."
        )

    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"Simulated coverage: {result.source}"[:150],
            },
        },
        _section("\n".join(lines)),
    ]

    # The path is the point: it is where every artifact of this run lives.
    where = [f"*Simulation run in* `{result.job_dir or 'unknown'}`"]
    if result.visits_path:
        where.append(f"Visit database: `{result.visits_path}`")
    if config.sim.artifact_base_url and result.job_dir:
        name = result.job_dir.rstrip("/").rsplit("/", 1)[-1]
        where.append(f"<{config.sim.artifact_base_url.rstrip('/')}/{name}|browse artifacts>")
    blocks.append(_section("\n".join(where)))

    notes = []
    if result.opsim:
        notes.append(f"Visit history: {result.opsim}")
    if result.band_scheduler:
        notes.append(f"Filter carousel: {result.band_scheduler}")
    if notes:
        blocks.append(_context(" -- ".join(notes)))
    return blocks


def build_failure_blocks(
    what: str,
    error: BaseException | str,
    source: str = "",
    origin: str = "",
    log_tail: str = "",
) -> list[dict[str, Any]]:
    """Blocks announcing that chatterbox itself failed.

    A ToO is time-critical, so silence is the worst outcome: an alert that
    nobody hears about looks exactly like no alert at all. This says what broke
    where, in enough detail to start on it without going to the host first.

    Parameters
    ----------
    what : `str`
        What was being attempted, as a phrase: ``"handling a ToO record"``.
    error : `BaseException` or `str`
        The failure. An exception contributes its type as well as its message.
    source : `str`
        Event identifier, when it is known. Decoding is one of the things that
        can fail, so it often is not.
    origin : `str`
        Where the record came from -- a file path, a Kafka topic, an EFD topic.
    log_tail : `str`
        Last lines of a relevant log, included verbatim.

    Returns
    -------
    blocks : `list` [`dict`]
    """
    if isinstance(error, BaseException):
        detail = f"{type(error).__name__}: {error}"
    else:
        detail = str(error)

    header = ":rotating_light: *chatterbox failed while " + what + "*"
    lines = [header]
    if source:
        lines.append(f"Event: *{source}*")
    if origin:
        lines.append(f"Received from: `{origin}`")
    lines.append(f"```{detail[:1500]}```")
    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": (f"chatterbox failure: {source}" if source else "chatterbox failure")[:150],
            },
        },
        _section("\n".join(lines)),
    ]
    if log_tail.strip():
        blocks.append(_section("*Log tail*\n```" + log_tail.strip()[-1500:] + "```"))
    blocks.append(
        _context(
            "Posted by chatterbox's own error handler, so a failed alert is not silent. "
            "The service log has the full traceback."
        )
    )
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
