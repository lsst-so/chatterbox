"""Tests for per-band localization coverage."""

import healpy as hp
import numpy as np
import pandas as pd
import pytest
from conftest import NSIDE, disc_reward_map

from chatterbox.astro.skymap import localization_from_probability, localization_from_reward_map
from chatterbox.sim.coverage import BANDS, band_coverage, band_coverage_by_epoch, coverage_curve, too_visits

NSIDE_MAP = 128


def make_visits(pointings, band="g", epoch=0, mjd0=61265.0):
    """Build a visit table as ``sim_runner`` writes it (RA/Dec radians)."""
    rows = []
    for n, (ra, dec) in enumerate(pointings):
        rows.append(
            {
                "RA": np.radians(ra),
                "dec": np.radians(dec),
                "band": band,
                "observation_reason": f"too_GW_case_B_C_0_i{epoch}",
                "mjd": mjd0 + n / 1440.0,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def point_localization():
    """A localization concentrated in a single small disc at (60, -35)."""
    prob = np.zeros(hp.nside2npix(NSIDE_MAP))
    pixels = hp.query_disc(NSIDE_MAP, hp.ang2vec(60.0, -35.0, lonlat=True), np.radians(1.0), inclusive=True)
    prob[pixels] = 1.0
    return localization_from_probability(prob, provenance="test", credible_level=0.9)


def test_covering_the_whole_localization_gives_one(point_localization):
    """A visit centred on a 1 degree region should capture nearly all of it."""
    visits = make_visits([(60.0, -35.0)], band="g")
    result = band_coverage(visits, point_localization)
    assert result.fractions["g"] == pytest.approx(1.0, abs=0.02)
    assert result.any_band == pytest.approx(1.0, abs=0.02)


def test_pointing_elsewhere_gives_zero(point_localization):
    visits = make_visits([(200.0, 40.0)], band="g")
    result = band_coverage(visits, point_localization)
    assert result.fractions["g"] == pytest.approx(0.0, abs=1e-9)


def test_all_bands_are_reported_even_when_unobserved(point_localization):
    """A band that was never observed must report 0, not vanish."""
    visits = make_visits([(60.0, -35.0)], band="g")
    result = band_coverage(visits, point_localization)
    assert set(result.fractions) == set(BANDS)
    assert result.fractions["u"] == 0.0
    assert result.n_visits["u"] == 0


def test_overlapping_pointings_are_not_double_counted(point_localization):
    """Two identical pointings must not report 200% coverage."""
    single = band_coverage(make_visits([(60.0, -35.0)]), point_localization)
    doubled = band_coverage(make_visits([(60.0, -35.0), (60.0, -35.0)]), point_localization)
    assert doubled.fractions["g"] == pytest.approx(single.fractions["g"], rel=1e-9)
    assert doubled.fractions["g"] <= 1.0 + 1e-9


def test_bands_are_independent(point_localization):
    """Each band gets its own copy of the map, so both can reach 100%."""
    visits = pd.concat(
        [make_visits([(60.0, -35.0)], band="g"), make_visits([(60.0, -35.0)], band="r")],
        ignore_index=True,
    )
    result = band_coverage(visits, point_localization)
    assert result.fractions["g"] == pytest.approx(result.fractions["r"], rel=1e-9)
    assert result.fractions["g"] > 0.9
    # ...but "any band" still cannot exceed 1.
    assert result.any_band <= 1.0 + 1e-9


def test_partial_coverage_accumulates():
    """Tiling a region larger than one field of view accumulates coverage.

    The localization has to be bigger than the 1.75 degree field of view for
    a single pointing to be genuinely partial.
    """
    wide = np.zeros(hp.nside2npix(NSIDE_MAP))
    wide[hp.query_disc(NSIDE_MAP, hp.ang2vec(60.0, -35.0, lonlat=True), np.radians(5.0), inclusive=True)] = (
        1.0
    )
    localization = localization_from_probability(wide, provenance="test")

    tiles = [(57.0, -35.0), (63.0, -35.0), (60.0, -32.0), (60.0, -38.0), (60.0, -35.0)]
    one = band_coverage(make_visits(tiles[:1]), localization).fractions["g"]
    three = band_coverage(make_visits(tiles[:3]), localization).fractions["g"]
    five = band_coverage(make_visits(tiles), localization).fractions["g"]

    assert 0.0 < one < three < five < 1.0


def test_quantity_follows_the_localization(point_localization):
    """A binary reward map yields area fractions; a real map yields
    probability.
    """
    visits = make_visits([(60.0, -35.0)])
    assert band_coverage(visits, point_localization).quantity == "probability"

    area_localization = localization_from_reward_map(disc_reward_map(60.0, -35.0, 3.0), NSIDE)
    area_result = band_coverage(visits, area_localization)
    assert area_result.quantity == "area"
    assert area_result.is_probability is False


def test_degrees_input_is_supported(point_localization):
    visits = make_visits([(60.0, -35.0)])
    visits["RA"] = np.degrees(visits["RA"])
    visits["dec"] = np.degrees(visits["dec"])
    result = band_coverage(visits, point_localization, radians=False)
    assert result.fractions["g"] == pytest.approx(1.0, abs=0.02)


def test_opsim_column_convention_is_detected(point_localization):
    """lsst_support.save_opsim writes fieldRA/fieldDec in degrees.

    Mistaking degrees for radians yields silently wrong coverage rather than an
    error, so the convention must be detected from the column names.
    """
    from chatterbox.sim.coverage import detect_pointing_columns

    visits = make_visits([(60.0, -35.0)]).rename(columns={"RA": "fieldRA", "dec": "fieldDec"})
    visits["fieldRA"] = np.degrees(visits["fieldRA"])
    visits["fieldDec"] = np.degrees(visits["fieldDec"])

    assert detect_pointing_columns(visits) == ("fieldRA", "fieldDec", False)
    result = band_coverage(visits, point_localization)
    assert result.fractions["g"] == pytest.approx(1.0, abs=0.02)


def test_raw_sim_runner_convention_is_detected(point_localization):
    """A raw ObservationArray uses RA/dec in radians."""
    from chatterbox.sim.coverage import detect_pointing_columns

    visits = make_visits([(60.0, -35.0)])
    assert detect_pointing_columns(visits) == ("RA", "dec", True)


def test_both_conventions_give_the_same_answer(point_localization):
    radian_visits = make_visits([(59.0, -35.0), (61.0, -35.0)])
    degree_visits = radian_visits.rename(columns={"RA": "fieldRA", "dec": "fieldDec"})
    degree_visits["fieldRA"] = np.degrees(degree_visits["fieldRA"])
    degree_visits["fieldDec"] = np.degrees(degree_visits["fieldDec"])

    a = band_coverage(radian_visits, point_localization).fractions
    b = band_coverage(degree_visits, point_localization).fractions
    assert a == pytest.approx(b)


def test_unknown_pointing_columns_raise(point_localization):
    from chatterbox.sim.coverage import detect_pointing_columns

    visits = make_visits([(60.0, -35.0)]).rename(columns={"RA": "boresight_x", "dec": "boresight_y"})
    with pytest.raises(KeyError, match="no recognized pointing columns"):
        detect_pointing_columns(visits)


def test_time_column_detection():
    from chatterbox.sim.coverage import detect_time_column

    assert detect_time_column(make_visits([(60.0, -35.0)])) == "mjd"
    renamed = make_visits([(60.0, -35.0)]).rename(columns={"mjd": "observationStartMJD"})
    assert detect_time_column(renamed) == "observationStartMJD"
    assert detect_time_column(make_visits([(60.0, -35.0)]).drop(columns=["mjd"])) is None


def test_coverage_curve_without_a_time_column(point_localization):
    visits = make_visits([(59.0, -35.0), (61.0, -35.0)]).drop(columns=["mjd"])
    times, cumulative = coverage_curve(visits, point_localization)["g"]
    assert np.array_equal(times, np.arange(2, dtype=float))
    assert (np.diff(cumulative) >= -1e-12).all()


def test_non_finite_pointings_are_skipped(point_localization):
    visits = make_visits([(60.0, -35.0), (60.0, -35.0)])
    visits.loc[1, "RA"] = np.nan
    result = band_coverage(visits, point_localization)
    assert result.n_visits["g"] == 1
    assert result.fractions["g"] > 0.9


def test_missing_columns_raise(point_localization):
    visits = make_visits([(60.0, -35.0)]).drop(columns=["band"])
    with pytest.raises(KeyError, match="band"):
        band_coverage(visits, point_localization)


def test_too_visits_filters_and_parses_epochs():
    too = make_visits([(60.0, -35.0)], epoch=3)
    other = make_visits([(10.0, 10.0)])
    other["observation_reason"] = "pairs_ri_33.0"
    visits = pd.concat([too, other], ignore_index=True)

    selected = too_visits(visits)
    assert len(selected) == 1
    assert selected["too_epoch"].iloc[0] == 3


def test_too_visits_can_filter_by_id():
    a = make_visits([(60.0, -35.0)])
    b = make_visits([(60.0, -35.0)])
    b["observation_reason"] = "too_GW_case_B_C_7_i0"
    visits = pd.concat([a, b], ignore_index=True)
    assert len(too_visits(visits, too_id="7")) == 1
    assert len(too_visits(visits, too_id="0")) == 1
    assert len(too_visits(visits)) == 2


def test_too_visits_requires_the_column():
    with pytest.raises(KeyError, match="observation_reason"):
        too_visits(pd.DataFrame({"band": ["g"]}))


def test_unparseable_reason_gets_epoch_minus_one():
    visits = make_visits([(60.0, -35.0)])
    visits["observation_reason"] = "too_no_epoch_suffix"
    assert too_visits(visits)["too_epoch"].iloc[0] == -1


def test_coverage_by_epoch_splits_the_total(point_localization):
    early = make_visits([(59.5, -35.0)], epoch=0)
    late = make_visits([(60.5, -35.0)], epoch=1)
    visits = too_visits(pd.concat([early, late], ignore_index=True))

    result = band_coverage_by_epoch(visits, point_localization)
    assert set(result.epochs) == {0, 1}
    # Each epoch alone covers no more than both together.
    for fractions in result.epochs.values():
        assert fractions["g"] <= result.fractions["g"] + 1e-9


def test_coverage_curve_is_monotonic(point_localization):
    visits = make_visits([(59.0, -35.0), (60.0, -35.0), (61.0, -35.0)])
    curves = coverage_curve(visits, point_localization)
    times, cumulative = curves["g"]
    assert times.size == cumulative.size == 3
    assert (np.diff(cumulative) >= -1e-12).all()
    assert cumulative[-1] <= 1.0 + 1e-9


def test_coverage_curve_omits_unobserved_bands(point_localization):
    curves = coverage_curve(make_visits([(60.0, -35.0)], band="g"), point_localization)
    assert set(curves) == {"g"}


def test_summary_line_lists_only_covered_bands(point_localization):
    result = band_coverage(make_visits([(60.0, -35.0)]), point_localization)
    line = result.summary_line()
    assert "g" in line
    assert "u" not in line


def test_as_percent(point_localization):
    result = band_coverage(make_visits([(60.0, -35.0)]), point_localization)
    assert result.as_percent()["g"] == pytest.approx(100.0 * result.fractions["g"])
