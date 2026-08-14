"""Tests for per-band localization coverage."""

import healpy as hp
import numpy as np
import pandas as pd
import pytest
from conftest import NSIDE, disc_reward_map

from chatterbox.astro.skymap import localization_from_probability, localization_from_reward_map
from chatterbox.sim.coverage import (
    BANDS,
    band_coverage,
    band_coverage_by_epoch,
    coverage_by_night,
    coverage_curve,
    too_visits,
)

NSIDE_MAP = 128


def make_visits(pointings, band="g", epoch=0, mjd0=61265.0, night=0, nights=None):
    """Build a visit table as ``sim_runner`` writes it (RA/Dec radians).

    ``nights`` assigns one night per pointing; ``night`` puts them all on the
    same one.
    """
    rows = []
    for n, (ra, dec) in enumerate(pointings):
        rows.append(
            {
                "RA": np.radians(ra),
                "dec": np.radians(dec),
                "band": band,
                "observation_reason": f"too_GW_case_B_C_0_i{epoch}",
                "mjd": mjd0 + n / 1440.0,
                "night": nights[n] if nights is not None else night,
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


# ------------------------------------------------------------- by night


@pytest.fixture
def spread_localization():
    """A localization wide enough that one visit cannot cover it all."""
    prob = np.zeros(hp.nside2npix(NSIDE_MAP))
    pixels = hp.query_disc(NSIDE_MAP, hp.ang2vec(60.0, -35.0, lonlat=True), np.radians(3.0), inclusive=True)
    prob[pixels] = 1.0
    return localization_from_probability(prob, provenance="test", credible_level=0.9)


def test_nights_are_ordered_and_only_those_with_visits(spread_localization):
    visits = make_visits([(59.0, -35.0), (61.0, -35.0), (60.0, -33.0)], nights=[12, 3, 12])
    nightly = coverage_by_night(visits, spread_localization)
    assert nightly.nights == [3, 12], "ordered, deduplicated, and night 4-11 absent"
    assert len(nightly.any_band_gained) == len(nightly.any_band_cumulative) == 2


def test_the_nights_add_up_to_the_total(spread_localization):
    """The invariant that makes the figure checkable against the numbers.

    Both zero a pixel once counted, so splitting the same visits by night must
    not change the answer -- otherwise the last point on the per-night figure
    would disagree with the coverage quoted in the thread.
    """
    visits = make_visits([(59.0, -35.0), (61.0, -35.0), (60.0, -33.0)], nights=[0, 1, 2])
    nightly = coverage_by_night(visits, spread_localization)
    total = band_coverage(visits, spread_localization)
    assert nightly.any_band_cumulative[-1] == pytest.approx(total.any_band, rel=1e-9)
    assert nightly.cumulative["g"][-1] == pytest.approx(total.fractions["g"], rel=1e-9)
    assert sum(nightly.any_band_gained) == pytest.approx(total.any_band, rel=1e-9)


def test_a_repeated_pointing_gains_nothing_on_the_second_night(spread_localization):
    """Coverage is not re-earned: night 2 sees the same sky as night 1."""
    visits = make_visits([(60.0, -35.0), (60.0, -35.0)], nights=[1, 2])
    nightly = coverage_by_night(visits, spread_localization)
    assert nightly.any_band_gained[0] > 0.05
    assert nightly.any_band_gained[1] == pytest.approx(0.0, abs=1e-12)
    assert nightly.any_band_cumulative[1] == pytest.approx(nightly.any_band_cumulative[0], rel=1e-9)


def test_cumulative_coverage_never_decreases(spread_localization):
    visits = make_visits([(59.0, -35.0), (60.0, -35.0), (61.0, -35.0)], nights=[1, 2, 3])
    nightly = coverage_by_night(visits, spread_localization)
    assert (np.diff(nightly.any_band_cumulative) >= -1e-12).all()
    assert all(gain >= -1e-12 for gain in nightly.any_band_gained)
    assert nightly.any_band_cumulative[-1] <= 1.0 + 1e-9


def test_bands_are_tracked_separately_by_night(spread_localization):
    """Two bands on the same night each get their own copy of the map."""
    visits = pd.concat(
        [
            make_visits([(60.0, -35.0)], band="g", night=1),
            make_visits([(60.0, -35.0)], band="r", night=1),
        ],
        ignore_index=True,
    )
    nightly = coverage_by_night(visits, spread_localization)
    assert set(nightly.gained) == {"g", "r"}, "unobserved bands are omitted, not zero-filled"
    assert nightly.gained["g"][0] == pytest.approx(nightly.gained["r"][0], rel=1e-9)
    # ...but the sky they share is counted once for the any-band total.
    assert nightly.any_band_gained[0] == pytest.approx(nightly.gained["g"][0], rel=1e-9)


def test_unusable_nights_are_dropped_not_lumped_together(spread_localization):
    visits = make_visits([(60.0, -35.0), (61.0, -35.0)], nights=[1, 2])
    visits.loc[1, "night"] = np.nan
    nightly = coverage_by_night(visits, spread_localization)
    assert nightly.nights == [1]


def test_no_usable_night_is_empty_rather_than_an_error(spread_localization):
    visits = make_visits([(60.0, -35.0)])
    visits["night"] = np.nan
    nightly = coverage_by_night(visits, spread_localization)
    assert nightly.nights == []
    assert nightly.as_dict() == {}
    assert nightly.summary_line() == "no nights with visits"


def test_the_night_column_is_required(spread_localization):
    visits = make_visits([(60.0, -35.0)]).drop(columns=["night"])
    with pytest.raises(KeyError, match="night"):
        coverage_by_night(visits, spread_localization)


def test_serialized_form_is_json_ready(spread_localization):
    """result.json is the process boundary, so keys must be strings."""
    import json

    visits = make_visits([(59.0, -35.0), (61.0, -35.0)], nights=[4, 9])
    nightly = coverage_by_night(visits, spread_localization)
    as_dict = nightly.as_dict()
    cumulative = nightly.cumulative_any_band()
    assert list(as_dict) == ["4", "9"]
    assert list(cumulative) == ["4", "9"]
    assert as_dict["4"]["g"] > 0
    json.dumps({"coverage_by_night": as_dict, "cumulative_by_night": cumulative})


def test_quantity_is_carried_through(spread_localization):
    """An area fraction must never be presented as a probability."""
    visits = make_visits([(60.0, -35.0)], night=1)
    assert coverage_by_night(visits, spread_localization).quantity == "probability"

    area = localization_from_reward_map(disc_reward_map(60.0, -35.0, 3.0), NSIDE)
    nightly = coverage_by_night(visits, area)
    assert nightly.quantity == "area"
    assert not nightly.is_probability


# --------------------------------------------------- counting from the trigger


def test_night_labels_count_from_the_trigger():
    from chatterbox.sim.coverage import night_label, nights_since_trigger

    assert nights_since_trigger([11, 12, 15], 12) == [-1, 0, 3]
    assert night_label(12, 12) == "+0"
    assert night_label(11, 12) == "-1"
    assert night_label(19, 12) == "+7"


def test_night_labels_fall_back_to_survey_nights():
    """An older result.json has no anchor; it must still render."""
    from chatterbox.sim.coverage import night_label, nights_since_trigger

    assert nights_since_trigger([11, 12], None) == [11, 12]
    assert night_label(11, None) == "11"


def test_coverage_is_reported_relative_to_the_trigger(spread_localization):
    visits = make_visits([(59.0, -35.0), (61.0, -35.0), (60.0, -33.0)], nights=[11, 12, 14])
    nightly = coverage_by_night(visits, spread_localization, reference_night=12)

    assert nightly.nights == [11, 12, 14], "the survey nights are kept"
    assert nightly.relative_nights == [-1, 0, 2]
    assert nightly.labels() == ["-1", "+0", "+2"]
    assert list(nightly.as_dict()) == ["-1", "0", "2"]
    assert list(nightly.cumulative_any_band()) == ["-1", "0", "2"]
    assert "night +0" in nightly.summary_line()


def test_a_pre_trigger_night_is_kept_not_discarded(spread_localization):
    """The run starts from the trigger's day_obs, so it can start early.

    Those visits are nominal cadence rather than follow-up, and they do cover
    localization area, so they are reported -- as night -1, not as an epoch.
    """
    visits = make_visits([(60.0, -35.0), (61.0, -35.0)], nights=[11, 12])
    nightly = coverage_by_night(visits, spread_localization, reference_night=12)
    assert nightly.relative_nights[0] == -1
    assert nightly.any_band_gained[0] > 0


def test_without_an_anchor_the_survey_nights_are_reported(spread_localization):
    visits = make_visits([(60.0, -35.0)], night=11)
    nightly = coverage_by_night(visits, spread_localization)
    assert nightly.reference_night is None
    assert list(nightly.as_dict()) == ["11"]
