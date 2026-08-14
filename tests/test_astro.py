"""Tests for skymap handling, the almanac and the dark-hours map.

The dark-hours map is new code with no prior implementation to compare against,
so it is validated against `astropy`'s full ``AltAz`` transform -- a completely
independent code path from the fast approximation used in production.
"""

import healpy as hp
import numpy as np
import pytest
from conftest import NSIDE, disc_reward_map

from chatterbox.astro.skymap import (
    contour_levels,
    credible_mask,
    geometry_from_mask,
    localization_from_probability,
    nest_to_ring,
)

# --------------------------------------------------------------------- skymap


def test_nest_to_ring_preserves_dtype_and_content():
    """healpy.reorder casts booleans, so the index mapping is used instead."""
    nested = disc_reward_map(60.0, -35.0, 9.8)
    ring = nest_to_ring(nested, NSIDE)
    assert ring.dtype == bool
    assert ring.sum() == nested.sum()
    # A round trip through healpy's float reorder must agree.
    expected = hp.reorder(nested.astype(float), n2r=True) > 0.5
    assert np.array_equal(ring, expected)


def test_nest_to_ring_rejects_wrong_length():
    with pytest.raises(ValueError, match="pixels"):
        nest_to_ring(np.zeros(100, dtype=bool), NSIDE)


def test_nest_to_ring_round_trip():
    nested = disc_reward_map(200.0, 10.0, 4.0)
    ring = nest_to_ring(nested, NSIDE)
    back = np.empty_like(ring)
    back[hp.ring2nest(NSIDE, np.arange(ring.size))] = ring
    assert np.array_equal(back, nested)


def test_chatterbox_does_not_import_astropy_healpix():
    """chatterbox's own code must not depend on astropy-healpix directly.

    It is still installed, because ``ligo.skymap`` needs it transitively for
    the sky projections and for reading GraceDB skymaps, but nothing here
    should reach for it: `pixel_corner_decs` covers our one use of it.
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "chatterbox"
    offenders = []
    for path in root.rglob("*.py"):
        for n, line in enumerate(path.read_text().splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "astropy_healpix" in stripped and ("import" in stripped):
                offenders.append(f"{path.relative_to(root)}:{n}: {stripped}")
    assert not offenders, "astropy_healpix imported in:\n" + "\n".join(offenders)


def test_pixel_corner_decs_matches_astropy_healpix():
    """Guard the replacement for ``astropy_healpix.boundaries_lonlat``.

    chatterbox no longer depends on astropy-healpix, but ligo.skymap pulls
    it in transitively, so when it is importable we check the two agree
    exactly. This is the reference test for the swap.
    """
    from chatterbox.astro.skymap import pixel_corner_decs

    try:
        from astropy_healpix import boundaries_lonlat
    except ImportError:
        pytest.skip("astropy-healpix is not importable")

    rng = np.random.default_rng(0)
    for nside in (1, 4, 32, 256):
        npix = hp.nside2npix(nside)
        pixels = rng.choice(npix, size=min(npix, 100), replace=False).astype(np.int64)
        reference = boundaries_lonlat(pixels, 1, nside, "nested")[1].to_value().ravel()
        assert pixel_corner_decs(nside, pixels) == pytest.approx(reference, abs=1e-12)


def test_pixel_corner_decs_golden_values():
    """Pin the values, so the check survives astropy-healpix going away.

    Taken from ``astropy_healpix.boundaries_lonlat([0], 1, 1, "nested")``,
    the implementation this replaced.
    """
    from chatterbox.astro.skymap import pixel_corner_decs

    # nside 1, NESTED pixel 0: a polar-cap pixel touching the north pole.
    decs = np.sort(pixel_corner_decs(1, [0]))
    assert decs == pytest.approx([0.0, 0.72972766, 0.72972766, np.pi / 2], abs=1e-8)
    # Its corners reach exactly the pole and exactly the equator.
    assert decs.max() == pytest.approx(np.pi / 2)
    assert decs.min() == pytest.approx(0.0)
    # The side corners sit at asin(2/3), the nside-1 polar-cap boundary.
    assert decs[1] == pytest.approx(np.arcsin(2.0 / 3.0), abs=1e-8)


def test_pixel_corner_decs_brackets_the_pixel_centre():
    """Corners must straddle the centre; that is why they are used."""
    from chatterbox.astro.skymap import pixel_corner_decs

    nside = 32
    for pixel in (0, 100, 5000, hp.nside2npix(nside) - 1):
        _, centre_dec = hp.pix2ang(nside, pixel, nest=True, lonlat=True)
        corners = np.degrees(pixel_corner_decs(nside, [pixel]))
        assert corners.min() <= centre_dec <= corners.max()


def test_pixel_corner_decs_accepts_a_scalar_and_respects_ordering():
    from chatterbox.astro.skymap import pixel_corner_decs

    scalar = pixel_corner_decs(8, 42)
    listed = pixel_corner_decs(8, [42])
    assert scalar.shape == (4,)
    assert scalar == pytest.approx(listed)

    # RING 42 is a different pixel from NESTED 42, so the two must differ.
    assert not np.allclose(pixel_corner_decs(8, [42], nest=False), listed)
    # ...and must equal the NESTED corners of the same physical pixel.
    same = hp.ring2nest(8, 42)
    assert pixel_corner_decs(8, [42], nest=False) == pytest.approx(pixel_corner_decs(8, [same]))


def test_pixel_corner_decs_in_range():
    from chatterbox.astro.skymap import pixel_corner_decs

    decs = pixel_corner_decs(16, np.arange(hp.nside2npix(16)))
    assert decs.size == hp.nside2npix(16) * 4
    assert decs.min() >= -np.pi / 2 - 1e-12
    assert decs.max() <= np.pi / 2 + 1e-12


def test_credible_mask_contains_at_least_the_requested_level():
    rng = np.random.default_rng(0)
    prob = rng.random(hp.nside2npix(16)) ** 4
    prob /= prob.sum()
    for level in (0.5, 0.9, 0.99):
        mask = credible_mask(prob, level)
        assert prob[mask].sum() >= level - 1e-12


def test_credible_mask_is_the_smallest_such_region():
    """Dropping the faintest included pixel must fall below the level.

    Exactly-representable values are used because the mask deliberately errs
    towards including one more pixel rather than reporting slightly less
    probability than asked for.
    """
    prob = np.zeros(hp.nside2npix(8))
    prob[:4] = [0.5, 0.25, 0.125, 0.125]
    mask = credible_mask(prob, 0.875)
    assert mask.sum() == 3

    included = np.sort(prob[mask])[::-1]
    assert included.sum() >= 0.875
    assert included[:-1].sum() < 0.875


def test_credible_mask_never_reports_less_than_asked():
    """With inexact sums the mask must round up, not down."""
    prob = np.zeros(hp.nside2npix(8))
    prob[:4] = [0.4, 0.3, 0.2, 0.1]
    mask = credible_mask(prob, 0.9)
    assert prob[mask].sum() >= 0.9


def test_credible_mask_on_a_zero_map():
    mask = credible_mask(np.zeros(48), 0.9)
    assert not mask.any()


def test_contour_levels_are_ascending_and_positive():
    rng = np.random.default_rng(1)
    prob = rng.random(hp.nside2npix(16)) ** 3
    prob /= prob.sum()
    levels = contour_levels(prob, (0.5, 0.9))
    assert levels == sorted(levels)
    assert all(level > 0 for level in levels)


def test_contour_levels_deduplicates_on_a_binary_map():
    """A uniform region gives the same threshold for every credible level."""
    prob = np.zeros(hp.nside2npix(16))
    prob[:100] = 1.0 / 100
    assert len(contour_levels(prob, (0.5, 0.9))) == 1


def test_geometry_of_an_empty_mask_is_degenerate_not_an_error():
    geometry = geometry_from_mask(np.zeros(hp.nside2npix(NSIDE), dtype=bool), NSIDE)
    assert geometry.n_pixels == 0
    assert geometry.area_deg2 == 0.0
    assert not np.isfinite(geometry.centroid_ra_deg)


def test_centroid_does_not_average_across_the_ra_wrap():
    """A region spanning RA=0 must not produce a centroid near RA=180."""
    mask = nest_to_ring(disc_reward_map(0.0, -30.0, 8.0), NSIDE)
    geometry = geometry_from_mask(mask, NSIDE)
    ra = geometry.centroid_ra_deg
    assert min(ra, 360.0 - ra) < 2.0, f"centroid RA {ra} is not near 0/360"


def test_localization_from_probability_normalizes():
    prob = np.abs(np.random.default_rng(2).normal(size=hp.nside2npix(16)))
    loc = localization_from_probability(prob, provenance="test")
    assert loc.prob_map.sum() == pytest.approx(1.0)
    assert loc.is_probability is True
    assert loc.quantity_name == "probability"


def test_localization_from_probability_rejects_a_zero_map():
    with pytest.raises(ValueError, match="zero"):
        localization_from_probability(np.zeros(hp.nside2npix(8)), provenance="test")


def test_vendored_skymap_flattens_a_multi_order_map():
    """Exercise the Skymap class vendored from the producer."""
    from chatterbox.astro.skymap import Skymap

    # Order 1 (nside 2) UNIQ indices are 4*4**1 = 16 through 79. Take four.
    uniq = np.array([16, 17, 18, 19], dtype=np.int64)
    densities = np.array([10.0, 1.0, 0.5, 0.1])
    skymap = Skymap(densities, uniq)

    flat = skymap.make_flat_binary_map(0.7, 3)
    assert flat.dtype == bool
    assert flat.size == hp.nside2npix(8)
    assert flat.any()
    # The highest-density pixel is order 1, so it upsamples to 4**2 = 16
    # order-3 pixels; the 70% region cannot be larger than all four parents.
    assert 16 <= flat.sum() <= 64

    area, min_dec = skymap.area_for_probability(0.9)
    assert area > 0
    assert -np.pi / 2 <= min_dec <= np.pi / 2

    # A denser pixel must be the maximum-probability coordinate.
    ra, dec = skymap.max_probability_coord()
    expected_ra, expected_dec = hp.pix2ang(2, 0, nest=True, lonlat=True)
    assert (ra, dec) == pytest.approx((expected_ra, expected_dec))


def test_vendored_skymap_area_grows_with_credible_level():
    from chatterbox.astro.skymap import Skymap

    uniq = np.arange(16, 80, dtype=np.int64)
    densities = np.linspace(10.0, 0.1, uniq.size)
    skymap = Skymap(densities, uniq)
    area_50, _ = skymap.area_for_probability(0.5)
    area_90, _ = skymap.area_for_probability(0.9)
    assert area_50 <= area_90


def test_kahan_adder_beats_naive_summation():
    """Compensated summation is why tiny probabilities still count."""
    from chatterbox.astro.skymap import KahanAdder

    adder = KahanAdder()
    for _ in range(10_000):
        adder += 1e-8
    naive = 0.0
    for _ in range(10_000):
        naive += 1e-8
    assert abs(float(adder) - 1e-4) <= abs(naive - 1e-4)
    assert float(adder) == pytest.approx(1e-4, rel=1e-12)


# -------------------------------------------------------------------- almanac


def test_night_events_fields(rubin_scheduler):
    from astropy.time import Time

    from chatterbox.astro.almanac import night_events

    events = night_events(when=Time("2026-08-14T02:00:00"))
    assert events.sun_n12_setting is not None
    assert events.sun_n12_rising is not None
    assert events.sun_n12_setting < events.sun_n12_rising
    # Nautical night at Cerro Pachon is between roughly 8 and 12 hours.
    assert 7.0 < events.night_length_hours < 13.0
    # Astronomical dark is always shorter than nautical night.
    assert events.dark_length_hours < events.night_length_hours
    assert 0.0 <= events.moon_phase <= 100.0


def test_daytime_trigger_reports_the_upcoming_night(rubin_scheduler):
    """A trigger during local daytime must not report a finished night.

    The almanac returns the night containing or *preceding* a time, so a
    midday trigger would otherwise be described with observability statistics
    for a night that had already ended.
    """
    from astropy.time import Time

    from chatterbox.astro.almanac import night_events

    # 15:18 UTC is 12:18 local at Cerro Pachon: the middle of the day.
    when = Time("2025-11-12T15:18:45")
    events = night_events(when=when)
    assert events.sun_n12_setting.mjd > when.mjd, "reported a night that already ended"
    assert events.day_obs == "2025-11-12"


def test_after_midnight_trigger_stays_on_the_current_night(rubin_scheduler):
    """Before dawn, the relevant night is the one already in progress."""
    from astropy.time import Time

    from chatterbox.astro.almanac import night_events

    when = Time("2026-08-14T05:00:00")  # 01:00 local, mid-night
    events = night_events(when=when)
    assert events.sun_n12_setting.mjd < when.mjd < events.sun_n12_rising.mjd


def test_prefer_next_night_can_be_disabled(rubin_scheduler):
    from astropy.time import Time

    from chatterbox.astro.almanac import night_events

    when = Time("2025-11-12T15:18:45")
    kept = night_events(when=when, prefer_next_night=False)
    advanced = night_events(when=when, prefer_next_night=True)
    assert kept.sun_n12_setting.mjd < advanced.sun_n12_setting.mjd


def test_explicit_day_obs_is_honoured(rubin_scheduler):
    from chatterbox.astro.almanac import night_events

    events = night_events(day_obs="2026-08-14")
    assert events.day_obs == "2026-08-14"
    assert events.sun_n12_setting is not None


def test_observing_window_rejects_unsupported_limits(rubin_scheduler):
    from astropy.time import Time

    from chatterbox.astro.almanac import night_events

    events = night_events(when=Time("2026-08-14T02:00:00"))
    with pytest.raises(ValueError, match="altitude limit"):
        events.observing_window(-15.0)


def test_moon_separation_handles_nan(rubin_scheduler):
    from astropy.time import Time

    from chatterbox.astro.almanac import moon_separation_deg, night_events

    events = night_events(when=Time("2026-08-14T02:00:00"))
    assert not np.isfinite(moon_separation_deg(float("nan"), -30.0, events))
    sep = moon_separation_deg(events.moon_ra_deg, events.moon_dec_deg, events)
    assert sep == pytest.approx(0.0, abs=1e-6)


def test_format_time_handles_missing_events():
    from chatterbox.astro.almanac import format_time

    assert format_time(None) == "--"


# ----------------------------------------------------------------- dark hours


def test_airmass_to_altitude():
    from chatterbox.astro.darkhours import airmass_to_altitude_deg

    assert airmass_to_altitude_deg(2.0) == pytest.approx(30.0)
    assert airmass_to_altitude_deg(1.0) == pytest.approx(90.0)
    with pytest.raises(ValueError, match="Airmass"):
        airmass_to_altitude_deg(0.5)


@pytest.fixture
def dark_map(rubin_scheduler):
    from astropy.time import Time

    from chatterbox.astro.almanac import night_events
    from chatterbox.astro.darkhours import dark_hours_map

    events = night_events(when=Time("2026-08-14T02:00:00"))
    return dark_hours_map(events, nside=32, step_minutes=5.0)


def test_dark_hours_never_exceed_the_night(dark_map):
    """A naive sample count over-reports; the total must be clipped."""
    assert dark_map.hours.max() <= dark_map.window_hours + 1e-9
    assert dark_map.hours.min() >= 0.0


def test_dark_hours_northern_sky_is_inaccessible(dark_map):
    """From latitude -30, altitude > 30 requires Dec below about +30."""
    ra, dec = hp.pix2ang(dark_map.nside, np.arange(dark_map.hours.size), lonlat=True)
    assert dark_map.hours[dec > 35.0].max() == 0.0
    assert dark_map.hours[dec < -80.0].max() > 0.0


def test_dark_hours_matches_astropy_altaz(dark_map):
    """Cross-check against a fully independent coordinate transform."""
    import astropy.units as u
    from astropy.coordinates import AltAz, EarthLocation, SkyCoord
    from astropy.time import Time

    location = EarthLocation.of_site("Cerro Pachon")
    start, end = dark_map.events.observing_window(dark_map.sun_alt_limit_deg)
    times = Time(np.arange(start, end, 1.0 / 60.0 / 24.0), format="mjd")

    rng = np.random.default_rng(7)
    pixels = rng.choice(dark_map.hours.size, 8, replace=False)
    ra, dec = hp.pix2ang(dark_map.nside, pixels, lonlat=True)

    for pixel, ra_deg, dec_deg in zip(pixels, ra, dec):
        altaz = SkyCoord(ra_deg * u.deg, dec_deg * u.deg).transform_to(
            AltAz(obstime=times, location=location)
        )
        reference = float((altaz.alt.deg > dark_map.altitude_limit_deg).sum()) / 60.0
        assert dark_map.hours[pixel] == pytest.approx(reference, abs=0.15)


def test_moon_avoidance_reduces_accessible_hours(rubin_scheduler):
    from astropy.time import Time

    from chatterbox.astro.almanac import night_events
    from chatterbox.astro.darkhours import dark_hours_map

    events = night_events(when=Time("2026-08-14T02:00:00"))
    plain = dark_hours_map(events, nside=32, step_minutes=10.0)
    avoided = dark_hours_map(events, nside=32, step_minutes=10.0, moon_avoidance_deg=40.0)
    assert avoided.hours.sum() < plain.hours.sum()
    assert (avoided.hours <= plain.hours + 1e-9).all()


def test_region_stats_and_weighting_agree_for_a_uniform_region(dark_map):
    from chatterbox.astro.skymap import localization_from_reward_map

    localization = localization_from_reward_map(disc_reward_map(60.0, -35.0, 9.8), NSIDE)
    stats = dark_map.stats_in_region(localization.prob_map > 0)
    weighted = dark_map.weighted_stats(localization.prob_map)
    # For a uniform-weight region the weighted mean is the plain mean.
    assert weighted["weighted_mean_hours"] == pytest.approx(stats["mean_hours"], rel=0.02)
    assert 0.0 <= stats["fraction_accessible"] <= 1.0


def test_region_stats_on_an_empty_mask(dark_map):
    stats = dark_map.stats_in_region(np.zeros(dark_map.hours.size, dtype=bool))
    assert stats["fraction_accessible"] == 0.0
    assert not np.isfinite(stats["max_hours"])
