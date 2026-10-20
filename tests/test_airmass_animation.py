"""Tests for the animated airmass map."""

import numpy as np
import pytest
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from conftest import disc_reward_map, make_record
from PIL import Image

from chatterbox.app import process_trigger
from chatterbox.astro.almanac import night_events
from chatterbox.ingest.decode import decode_record
from chatterbox.plots.airmass_animation import (
    airmass_from_altitude,
    airmass_on_grid,
    frame_times,
    plot_airmass_animation,
)
from chatterbox.slackbot.client import SlackPoster

# Cerro Pachon, for the independent altitude check; not the scheduler's model.
PACHON = EarthLocation(lat=-30.2446, lon=-70.7494, height=2663.0)


def trigger_for(ra=60.0, dec=-35.0, radius=6.0):
    return decode_record(make_record(reward_map=disc_reward_map(ra, dec, radius)))


# -------------------------------------------------------------------- masking


def test_airmass_is_masked_above_the_limit_and_below_the_horizon():
    """The issue's rule: anything past airmass 3.8 is NaN, not a colour."""
    # 15.0 deg is airmass 3.86, over the limit; 15.5 deg is 3.74, under it.
    airmass = airmass_from_altitude([90.0, 30.0, 15.5, 15.0, 5.0, 0.0, -20.0], mask_above=3.8)
    assert airmass[0] == pytest.approx(1.0)
    assert airmass[1] == pytest.approx(2.0)
    assert airmass[2] == pytest.approx(3.74, abs=0.01)
    assert np.isnan(airmass[3:]).all(), "over the limit, on the horizon and below it are all NaN"


def test_the_mask_is_configurable():
    assert np.isfinite(airmass_from_altitude([15.0], mask_above=4.0)[0])
    assert np.isnan(airmass_from_altitude([15.0], mask_above=3.0)[0])


def test_airmass_on_the_sky_agrees_with_an_independent_altitude(rubin_scheduler):
    """Checked against astropy's AltAz, independent of production."""
    when = Time("2026-08-14T05:00:00")
    ra = np.array([60.0, 150.0, 10.0, 300.0])
    dec = np.array([-35.0, -60.0, -10.0, -45.0])
    ours = airmass_on_grid(ra, dec, when.mjd, mask_above=10.0)

    altaz = SkyCoord(ra, dec, unit="deg").transform_to(AltAz(obstime=when, location=PACHON))
    reference = np.where(altaz.alt.deg > 0, 1.0 / np.sin(np.radians(altaz.alt.deg)), np.nan)
    both = np.isfinite(ours) & np.isfinite(reference) & (reference < 3.0)
    assert both.any()
    assert ours[both] == pytest.approx(reference[both], abs=0.06)


# --------------------------------------------------------------------- frames


def test_frames_span_the_whole_night_inclusive(rubin_scheduler):
    events = night_events(when=Time("2026-08-14T02:00:00"))
    start, end = events.observing_window(-12.0)
    times = frame_times(events, step_minutes=10.0)
    assert times[0] == pytest.approx(start)
    assert times[-1] == pytest.approx(end), "the last frame is the end of the night, not a step short"
    assert np.all(np.diff(times) > 0)
    assert np.diff(times)[:-1] == pytest.approx(10.0 / 1440.0)


# ------------------------------------------------------------------ animation


def render(tmp_path, trigger, name="a.gif", **kwargs):
    events = night_events(when=trigger.event_time)
    settings = dict(step_minutes=120.0, dpi=40, fps=4)
    settings.update(kwargs)
    path = plot_airmass_animation(
        events,
        trigger.localization,
        tmp_path / name,
        centroid=(trigger.geometry.centroid_ra_deg, trigger.geometry.centroid_dec_deg),
        **settings,
    )
    return events, path


def test_the_animation_is_a_looping_gif_with_a_frame_per_time(tmp_path, rubin_scheduler):
    trigger = trigger_for()
    events, path = render(tmp_path, trigger)
    with Image.open(path) as gif:
        assert gif.format == "GIF"
        assert gif.n_frames == len(frame_times(events, 120.0))
        assert gif.n_frames > 3
        assert gif.info.get("loop") == 0, "loops forever"


def test_the_frames_actually_change_as_the_sky_turns(tmp_path, rubin_scheduler):
    """A static image saved as a GIF would pass every other check."""
    _, path = render(tmp_path, trigger_for())
    with Image.open(path) as gif:
        gif.seek(0)
        first = np.asarray(gif.convert("RGB"))
        gif.seek(gif.n_frames // 2)
        middle = np.asarray(gif.convert("RGB"))
    assert (first != middle).any(axis=-1).mean() > 0.02


def test_a_compact_localization_gets_a_zoomed_panel_and_a_large_one_does_not(tmp_path, rubin_scheduler):
    _, compact = render(tmp_path, trigger_for(radius=6.0), "compact.gif")
    _, large = render(tmp_path, trigger_for(radius=60.0), "large.gif")
    with Image.open(compact) as a, Image.open(large) as b:
        assert a.size[0] > b.size[0], "the zoom panel makes the figure wider"


# ------------------------------------------------------------------ in the app


@pytest.fixture
def animated_config(config):
    config.airmass_animation.enabled = True
    config.airmass_animation.step_minutes = 120.0
    config.airmass_animation.dpi = 40
    return config


def test_the_animation_is_posted_in_the_alerts_thread(record, animated_config, tmp_path, rubin_scheduler):
    import json

    poster = SlackPoster(animated_config, dry_run=True, output_dir=tmp_path / "posts")
    report = process_trigger(
        record, animated_config, poster=poster, post=True, run_sim=False, out_dir=tmp_path / "plots"
    )
    assert report.animation_thread is not None
    assert report.wait_for_simulation(
        timeout=300
    ), "the thread is joined with the rest of the background work"

    assert report.animation is not None and report.animation.is_file()
    payload = json.loads((tmp_path / "posts" / f"{report.trigger.source}_airmass.json").read_text())
    assert payload["files"] == [str(report.animation)]
    caption = payload["blocks"][0]["text"]["text"]
    assert "above airmass 3.8" in caption, "the mask is explained, not left as mysterious grey"
    assert report.events.day_obs in caption


def test_the_animation_does_not_run_when_switched_off_or_not_posting(
    record, animated_config, tmp_path, rubin_scheduler
):
    poster = SlackPoster(animated_config, dry_run=True, output_dir=tmp_path / "posts")
    declined = process_trigger(
        record,
        animated_config,
        poster=poster,
        post=True,
        run_sim=False,
        run_animation=False,
        out_dir=tmp_path / "a",
    )
    dry = process_trigger(
        record, animated_config, poster=poster, post=False, run_sim=False, out_dir=tmp_path / "b"
    )
    animated_config.airmass_animation.enabled = False
    disabled = process_trigger(
        record, animated_config, poster=poster, post=True, run_sim=False, out_dir=tmp_path / "c"
    )
    for report in (declined, dry, disabled):
        assert report.animation_thread is None
