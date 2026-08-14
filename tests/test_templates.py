"""Tests for per-band template coverage maps."""

import healpy as hp
import numpy as np
import pandas as pd
import pytest

from chatterbox.astro.skymap import localization_from_probability
from chatterbox.astro.templates import BANDS, build_template_maps, load_template_maps

NSIDE = 64


def visit_table(pointings):
    """``[(ra, dec, band), ...]`` in degrees, as a ConsDB-shaped table."""
    return pd.DataFrame([{"s_ra": ra, "s_dec": dec, "band": band} for ra, dec, band in pointings])


def point_localization(ra=60.0, dec=-35.0, radius=1.0, nside=NSIDE):
    prob = np.zeros(hp.nside2npix(nside))
    prob[hp.query_disc(nside, hp.ang2vec(ra, dec, lonlat=True), np.radians(radius), inclusive=True)] = 1.0
    return localization_from_probability(prob, provenance="test")


def test_build_accumulates_visits():
    visits = visit_table([(60.0, -35.0, "g"), (60.0, -35.0, "g"), (200.0, 10.0, "r")])
    coverage = build_template_maps(visits, nside=NSIDE)
    assert coverage.n_visits["g"] == 2
    assert coverage.n_visits["r"] == 1
    assert coverage.maps["g"].max() == 2
    assert coverage.maps["z"].max() == 0


def test_all_bands_are_present_even_when_unobserved():
    coverage = build_template_maps(visit_table([(60.0, -35.0, "g")]), nside=NSIDE)
    assert set(coverage.maps) == set(BANDS)
    assert coverage.area_deg2("u") == 0.0


def test_min_visits_gates_the_mask():
    visits = visit_table([(60.0, -35.0, "g")])
    once = build_template_maps(visits, nside=NSIDE, min_visits=1)
    twice = build_template_maps(visits, nside=NSIDE, min_visits=2)
    assert once.mask("g").any()
    assert not twice.mask("g").any()


def test_coverage_in_region():
    coverage = build_template_maps(visit_table([(60.0, -35.0, "g")]), nside=NSIDE)
    fractions = coverage.coverage_in_region(point_localization().prob_map, nside=NSIDE)
    assert fractions["g"] == pytest.approx(1.0, abs=0.02)
    assert fractions["r"] == 0.0
    assert set(fractions) == set(BANDS)


def test_coverage_in_region_resamples_across_resolutions():
    """The localization is usually at a different nside from the cache."""
    coverage = build_template_maps(visit_table([(60.0, -35.0, "g")]), nside=NSIDE)
    fine = point_localization(nside=256)
    fractions = coverage.coverage_in_region(fine.prob_map, nside=256)
    assert fractions["g"] == pytest.approx(1.0, abs=0.05)


def test_coverage_of_a_region_with_no_templates():
    coverage = build_template_maps(visit_table([(200.0, 10.0, "g")]), nside=NSIDE)
    fractions = coverage.coverage_in_region(point_localization().prob_map, nside=NSIDE)
    assert fractions["g"] == pytest.approx(0.0, abs=1e-9)


def test_round_trip_through_the_cache(tmp_path):
    coverage = build_template_maps(
        visit_table([(60.0, -35.0, "g"), (61.0, -35.0, "i")]),
        nside=NSIDE,
        source="unit test",
    )
    coverage.save(tmp_path / "templates")
    loaded = load_template_maps(tmp_path / "templates")

    assert loaded is not None
    assert loaded.nside == coverage.nside
    assert loaded.source == "unit test"
    assert loaded.built_at
    assert loaded.bands == coverage.bands
    for band in coverage.bands:
        assert np.array_equal(loaded.maps[band], coverage.maps[band])


def test_missing_cache_returns_none_rather_than_raising(tmp_path):
    """A missing cache must degrade to a post without the template panel."""
    assert load_template_maps(tmp_path / "absent") is None


def test_corrupt_cache_returns_none(tmp_path):
    cache = tmp_path / "templates"
    cache.mkdir()
    (cache / "meta.json").write_text("{not json")
    assert load_template_maps(cache) is None


def test_missing_columns_raise():
    with pytest.raises(KeyError, match="missing columns"):
        build_template_maps(pd.DataFrame({"ra": [1.0]}), nside=NSIDE)


def test_non_finite_pointings_are_dropped():
    visits = visit_table([(60.0, -35.0, "g"), (np.nan, -35.0, "g")])
    coverage = build_template_maps(visits, nside=NSIDE)
    assert coverage.n_visits["g"] == 1


def test_area_scales_with_the_field_of_view():
    visits = visit_table([(60.0, -35.0, "g")])
    small = build_template_maps(visits, nside=NSIDE, fov_radius_deg=1.0)
    large = build_template_maps(visits, nside=NSIDE, fov_radius_deg=2.0)
    assert large.area_deg2("g") > small.area_deg2("g")
    # A 1.75 degree disc is about 9.6 deg^2; check the scale is sane.
    default = build_template_maps(visits, nside=256)
    assert 7.0 < default.area_deg2("g") < 13.0
