"""Tests for when a localization is up during the night."""

import healpy as hp
import numpy as np
import pytest
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import TimeDelta
from conftest import disc_reward_map, make_record

from chatterbox.astro.almanac import night_events
from chatterbox.astro.darkhours import dark_hours_map
from chatterbox.astro.skymap import localization_from_probability
from chatterbox.astro.visibility import _longest_run, field_visibility
from chatterbox.ingest.decode import decode_record
from chatterbox.slackbot.blocks import _visibility_lines

# Cerro Pachon. Used only by the independent check below, which deliberately
# does not go through the scheduler's own site model.
PACHON = EarthLocation(lat=-30.2446, lon=-70.7494, height=2663.0)


def trigger_at(ra, dec, radius=3.0):
    return decode_record(make_record(reward_map=disc_reward_map(ra, dec, radius)))


def visibility_of(trigger, moon=None, **dark_kwargs):
    events = night_events(when=trigger.event_time)
    if moon is not None:
        # Park the Moon somewhere chosen, so the avoidance test does not depend
        # on where it happens to be on the night the fixture alert falls.
        events.moon_ra_deg, events.moon_dec_deg = moon
    dark = dark_hours_map(events, nside=64, step_minutes=5.0, **dark_kwargs)
    visibility = field_visibility(
        events,
        trigger.localization,
        dark,
        trigger.geometry.centroid_ra_deg,
        trigger.geometry.centroid_dec_deg,
    )
    return trigger, events, dark, visibility


def altitude(trigger, when):
    centre = SkyCoord(trigger.geometry.centroid_ra_deg, trigger.geometry.centroid_dec_deg, unit="deg")
    return float(centre.transform_to(AltAz(obstime=when, location=PACHON)).alt.deg)


# --------------------------------------------------------------- run finding


def test_longest_run_picks_the_longest_stretch():
    flags = np.array([1, 1, 0, 1, 1, 1, 0, 1], dtype=bool)
    assert _longest_run(flags) == (3, 5)


def test_longest_run_handles_edges_and_nothing():
    assert _longest_run(np.array([1, 1, 1], dtype=bool)) == (0, 2)
    assert _longest_run(np.array([0, 0], dtype=bool)) is None
    assert _longest_run(np.array([], dtype=bool)) is None


# ------------------------------------------------------------------ accuracy


def test_the_centroid_window_matches_an_independent_altitude(rubin_scheduler):
    """Checked against astropy's AltAz, a different code path from production.

    The window edges should sit on the altitude limit and the peak should be
    the altitude the full transform gives, to within the sampling step.
    """
    trigger, _, dark, vis = visibility_of(trigger_at(60.0, -35.0))
    window = vis.centroid
    assert window is not None
    limit = dark.altitude_limit_deg

    # A rising edge, so within one 5-minute step of the limit.
    assert altitude(trigger, window.start) == pytest.approx(limit, abs=1.5)
    # Just before it the field is below the limit, just after it is above.
    assert altitude(trigger, window.start - TimeDelta(600.0, format="sec")) < limit
    assert altitude(trigger, window.start + TimeDelta(600.0, format="sec")) > limit
    assert altitude(trigger, vis.peak_time) == pytest.approx(vis.peak_altitude_deg, abs=0.6)


def test_the_window_agrees_with_the_dark_hours_map(rubin_scheduler):
    """Same definition of 'visible', so the hours cannot disagree."""
    trigger, _, dark, vis = visibility_of(trigger_at(60.0, -35.0))
    pixel = hp.ang2pix(
        dark.nside,
        trigger.geometry.centroid_ra_deg,
        trigger.geometry.centroid_dec_deg,
        lonlat=True,
    )
    assert vis.centroid.hours == pytest.approx(dark.hours[pixel], abs=0.2)


def test_windows_lie_inside_the_night_and_nest(rubin_scheduler):
    _, events, _, vis = visibility_of(trigger_at(60.0, -35.0, radius=8.0))
    start, end = events.observing_window(-12.0)
    for window in (vis.centroid, vis.any_part, vis.most):
        assert window is not None
        assert start - 1e-9 <= window.start.mjd < window.end.mjd <= end + 1e-9
    assert vis.most.hours <= vis.any_part.hours + 1e-9
    assert vis.any_part.start.mjd <= vis.most.start.mjd + 1e-9
    assert vis.most.end.mjd <= vis.any_part.end.mjd + 1e-9
    assert 0.0 < vis.best_fraction <= 1.0
    assert vis.any_part.start.mjd <= vis.best_time.mjd <= vis.any_part.end.mjd


def test_an_extended_region_is_up_longer_than_its_centroid(rubin_scheduler):
    """Some part of a large region is up before and after the middle of it."""
    _, _, _, vis = visibility_of(trigger_at(60.0, -35.0, radius=15.0))
    assert vis.any_part.hours > vis.centroid.hours


# --------------------------------------------------------------------- edges


def test_a_field_that_never_rises_has_no_window(rubin_scheduler):
    """Dec +60 peaks at the horizon from latitude -30."""
    _, _, _, vis = visibility_of(trigger_at(60.0, 60.0))
    assert vis.centroid is None
    assert vis.any_part is None and vis.most is None and vis.best_time is None
    assert vis.best_fraction == 0.0


def test_a_circumpolar_field_is_up_all_night(rubin_scheduler):
    """Two degrees from the pole it never drops below ~28 deg, so a limit of
    airmass 3 (19.5 deg) is held all night.
    """
    _, _, _, vis = visibility_of(trigger_at(10.0, -88.0, radius=1.0), airmass_limit=3.0)
    assert vis.centroid.whole_night
    assert vis.centroid.hours == pytest.approx(vis.window_hours, abs=0.1)


def test_a_rising_field_is_flagged_as_ending_at_dawn(rubin_scheduler):
    """'Until 10:24' means something else when that is just sunrise."""
    _, _, _, vis = visibility_of(trigger_at(60.0, -35.0))
    assert vis.centroid.ends_at_dawn and not vis.centroid.starts_at_dusk


def test_probability_in_the_tails_does_not_make_a_region_visible(rubin_scheduler):
    """A real skymap has faint weight everywhere; the region is the contour.

    Centred where nothing is ever up, so any visibility reported could only
    come from the tails reaching a part of the sky that is.
    """
    nside = 64
    centre = hp.ang2vec(60.0, 60.0, lonlat=True)
    vectors = np.array(hp.pix2vec(nside, np.arange(hp.nside2npix(nside))))
    separation = np.degrees(np.arccos(np.clip(vectors.T @ centre, -1.0, 1.0)))
    prob = np.exp(-0.5 * (separation / 3.0) ** 2) + 1e-9
    localization = localization_from_probability(prob, provenance="test", credible_level=0.9)

    record = make_record(reward_map=disc_reward_map(60.0, 60.0, 3.0))
    trigger = decode_record(record)
    events = night_events(when=trigger.event_time)
    dark = dark_hours_map(events, nside=nside)
    vis = field_visibility(events, localization, dark, 60.0, 60.0)
    assert vis.any_part is None, "tails reach the southern sky, which must not count"


def test_the_moon_avoidance_radius_is_honoured(rubin_scheduler):
    """The window must agree with a dark-hours map built to avoid the Moon."""
    trigger = trigger_at(60.0, -35.0, radius=1.0)
    centre = (trigger.geometry.centroid_ra_deg, trigger.geometry.centroid_dec_deg)
    _, _, _, free = visibility_of(trigger, moon=centre)
    _, _, _, avoided = visibility_of(trigger, moon=centre, moon_avoidance_deg=30.0)
    assert free.centroid is not None, "no avoidance radius, so the Moon is ignored"
    assert avoided.centroid is None, "the field sits on the Moon"
    assert avoided.any_part is None


# --------------------------------------------------------------------- slack


def test_the_post_says_when_the_field_is_up(rubin_scheduler):
    trigger, _, _, vis = visibility_of(trigger_at(60.0, -35.0))
    text = "\n".join(_visibility_lines(trigger, vis))
    assert "When the field is visible" in text
    assert "airmass < 2" in text and "altitude > 30 deg" in text
    assert "Centroid:" in text and "UTC [" in text, "UTC with Chile local beside it"
    assert "Highest at" in text and "altitude" in text
    assert "Any part of the localization:" in text
    assert "Most of it (over half the area)" in text, "the weight is named, not assumed"
    assert "(dawn)" in text, "a window cut off by sunrise says so"


def test_the_post_warns_when_the_field_never_rises(rubin_scheduler):
    trigger, _, _, vis = visibility_of(trigger_at(60.0, 60.0))
    text = "\n".join(_visibility_lines(trigger, vis))
    assert "centroid never reaches the airmass limit" in text
    assert "No part of the localization is up" in text


def test_the_post_says_up_all_night(rubin_scheduler):
    trigger, _, _, vis = visibility_of(trigger_at(10.0, -88.0, radius=1.0), airmass_limit=3.0)
    assert "Centroid: up all night" in "\n".join(_visibility_lines(trigger, vis))


def test_the_section_is_in_the_trigger_post_and_optional(rubin_scheduler, config):
    from chatterbox.plots.darkhours import region_hours_summary
    from chatterbox.slackbot.blocks import build_trigger_blocks
    from chatterbox.slackbot.client import render_blocks_as_text

    trigger, events, dark, vis = visibility_of(trigger_at(60.0, -35.0))
    stats = region_hours_summary(dark, trigger.localization)

    with_section = build_trigger_blocks(trigger, events, dark, stats, None, config, None, vis)
    without = build_trigger_blocks(trigger, events, dark, stats, None, config)
    assert "When the field is visible" in render_blocks_as_text(with_section)
    assert "When the field is visible" not in render_blocks_as_text(without)
    assert len(with_section) <= 50
    assert all(len(b["text"]["text"]) <= 3000 for b in with_section if b["type"] == "section")
