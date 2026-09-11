"""Tests for per-band localization coverage."""

import healpy as hp
import numpy as np
import pandas as pd
import pytest
from conftest import NSIDE, disc_reward_map

from chatterbox.astro.skymap import (
    REGION_NSIDE_FLOOR,
    localization_from_probability,
    localization_from_reward_map,
    region_localizations,
)
from chatterbox.sim.coverage import (
    BANDS,
    any_band_curve,
    band_coverage,
    band_coverage_by_epoch,
    campaign_time,
    compare_coverage,
    coverage_by_night,
    coverage_curve,
    region_coverage,
    survey_visits,
    too_mask,
    too_visits,
)

NSIDE_MAP = 128


def make_visits(
    pointings,
    band="g",
    epoch=0,
    mjd0=61265.0,
    night=0,
    nights=None,
    note=None,
    reason=None,
    visittime=34.0,
    slewtime=6.0,
    exptime=30.0,
):
    """Build a visit table as ``sim_runner`` writes it (RA/Dec radians).

    ``nights`` assigns one night per pointing; ``night`` puts them all on the
    same one. ``note`` adds a ``scheduler_note`` column, which the real
    ``ObservationArray`` always has but which is left out by default so tests
    can exercise one tag at a time.
    """
    rows = []
    for n, (ra, dec) in enumerate(pointings):
        row = {
            "RA": np.radians(ra),
            "dec": np.radians(dec),
            "band": band,
            "observation_reason": reason if reason is not None else f"too_GW_case_B_C_0_i{epoch}",
            "mjd": mjd0 + n / 1440.0,
            "night": nights[n] if nights is not None else night,
            "visittime": visittime,
            "slewtime": slewtime,
            "exptime": exptime,
        }
        if note is not None:
            row["scheduler_note"] = note
        rows.append(row)
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


# ------------------------------------------ recognising the follow-up visits


def test_the_scheduler_note_alone_identifies_the_follow_up():
    """An opsim export need not carry observation_reason at all."""
    visits = make_visits([(60.0, -35.0)], note="ToO, GW_case_B_C, g, 0, i2").drop(
        columns=["observation_reason"]
    )
    selected = too_visits(visits)
    assert len(selected) == 1
    assert selected["too_epoch"].iloc[0] == 2, "the epoch is in the note too"


def test_either_tag_is_enough_and_the_union_is_taken(caplog):
    """A rename upstream should cost a warning, not an empty selection.

    Both fields are written by ``ToOScriptedSurvey`` for every follow-up visit,
    so a visit tagged by one and not the other means an upstream convention
    moved -- which is invisible if only one field is ever consulted.
    """
    tagged_both = make_visits([(60.0, -35.0)], note="ToO, GW_case_B_C, g, 0, i0")
    note_only = make_visits([(61.0, -35.0)], reason="pairs_ri_33", note="ToO, BBH, g, 0, i1")
    reason_only = make_visits([(62.0, -35.0)], note="long_gap_near_sun, 33")
    survey = make_visits([(10.0, 10.0)], reason="pairs_ri_33", note="pairs_ri_33")
    visits = pd.concat([tagged_both, note_only, reason_only, survey], ignore_index=True)

    with caplog.at_level("WARNING"):
        mask = too_mask(visits)
    assert mask.tolist() == [True, True, True, False]
    assert "tagged ToO by one field but not the other" in caplog.text


def test_a_survey_whose_name_merely_contains_too_is_not_follow_up():
    """`ToO` has to be a word in the note, or "tooth" would match."""
    visits = make_visits([(60.0, -35.0)], reason="tooth_fairy_survey", note="tooth_fairy_survey")
    assert not too_mask(visits).any()


def test_the_epoch_is_read_from_the_note_when_the_reason_has_none():
    """A survey configured with a constant reason still has epochs."""
    visits = make_visits([(60.0, -35.0)], reason="too_gw_case_b_c", note="ToO, GW_case_B_C, g, 0, i4")
    assert too_visits(visits)["too_epoch"].iloc[0] == 4


def test_survey_visits_is_the_complement_of_the_follow_up():
    too = make_visits([(60.0, -35.0)])
    survey = make_visits([(10.0, 10.0)], reason="pairs_ri_33")
    visits = pd.concat([too, survey], ignore_index=True)

    assert len(too_visits(visits)) == 1
    assert len(survey_visits(visits)) == 1
    assert len(too_visits(visits)) + len(survey_visits(visits)) == len(visits)


def test_recognising_follow_up_needs_at_least_one_tag():
    with pytest.raises(KeyError, match="observation_reason"):
        too_mask(pd.DataFrame({"band": ["g"]}))


# ----------------------------------------------------- ToO versus ToO + survey


def test_the_survey_contribution_is_reported_separately(spread_localization):
    """The point of issue #2: which number is the strategy, which is reality.

    The localization is wider than one field, and the survey visit sits on a
    part of it the follow-up never reached, so the survey's contribution is
    real rather than a repeat.
    """
    follow_up = make_visits([(58.5, -35.0)], band="g")
    survey = make_visits([(61.5, -35.0)], band="r", reason="pairs_ri_33")
    visits = pd.concat([follow_up, survey], ignore_index=True)

    comparison = compare_coverage(visits, spread_localization)
    assert comparison.too_n_visits == 1
    assert comparison.survey_n_visits == 1
    # The follow-up is g only, so r coverage can only come from the survey.
    assert comparison.too.fractions["r"] == 0.0
    assert comparison.combined.fractions["r"] > 0.0
    assert comparison.survey_gain["any"] > 0.0
    assert comparison.combined.any_band > comparison.too.any_band


def test_combined_coverage_is_never_below_the_follow_up_alone(spread_localization):
    """Painting is monotonic in the visit set, so this cannot be violated."""
    follow_up = make_visits([(60.0, -35.0), (62.0, -35.0)], band="g")
    survey = make_visits([(61.0, -34.0)], band="i", reason="pairs_ri_33")
    visits = pd.concat([follow_up, survey], ignore_index=True)

    comparison = compare_coverage(visits, spread_localization)
    for band in BANDS:
        assert comparison.combined.fractions[band] >= comparison.too.fractions[band] - 1e-9
        assert comparison.survey_gain[band] >= -1e-9
    assert comparison.survey_gain["any"] >= -1e-9


def test_a_survey_visit_repeating_the_follow_up_adds_nothing(point_localization):
    """Survey visits re-cover what the follow-up reached; only new area counts.

    This is why the combined number is not the sum of the two.
    """
    follow_up = make_visits([(60.0, -35.0)], band="g")
    survey = make_visits([(60.0, -35.0)], band="g", reason="pairs_ri_33")
    visits = pd.concat([follow_up, survey], ignore_index=True)

    comparison = compare_coverage(visits, point_localization)
    assert comparison.survey_gain["any"] == pytest.approx(0.0, abs=1e-6)
    assert comparison.combined.any_band == pytest.approx(comparison.too.any_band, abs=1e-6)


def test_the_comparison_ignores_visits_that_cannot_touch_the_localization(point_localization):
    follow_up = make_visits([(60.0, -35.0)], band="g")
    far_away = make_visits([(180.0, 40.0)], band="r", reason="pairs_ri_33")
    visits = pd.concat([follow_up, far_away], ignore_index=True)

    comparison = compare_coverage(visits, point_localization)
    assert comparison.survey_n_visits == 0
    assert comparison.survey_gain["any"] == pytest.approx(0.0, abs=1e-6)


def test_the_any_band_curve_ends_where_the_total_does(spread_localization):
    visits = too_visits(make_visits([(60.0, -35.0), (61.0, -35.0), (62.0, -34.0)], band="g"))
    times, cumulative = any_band_curve(visits, spread_localization)

    assert times.size == cumulative.size == 3
    assert np.all(np.diff(cumulative) >= -1e-12), "cumulative coverage cannot fall"
    total = band_coverage(visits, spread_localization).any_band
    assert cumulative[-1] == pytest.approx(total, abs=1e-9)


def test_the_any_band_curve_of_nothing_is_empty(point_localization):
    times, cumulative = any_band_curve(make_visits([]), point_localization)
    assert times.size == 0 and cumulative.size == 0


# ------------------------------------------------------------- campaign time


def test_campaign_time_charges_the_slew_as_well_as_the_visit():
    """A ToO interruption costs the survey the slew to the target too."""
    follow_up = make_visits([(60.0, -35.0), (60.5, -35.0)], visittime=34.0, slewtime=6.0)
    timing = campaign_time(follow_up)

    assert timing.n_too_visits == 2
    assert timing.too_seconds == pytest.approx(2 * 40.0)
    assert timing.too_open_shutter_s == pytest.approx(2 * 30.0)
    assert timing.measure == "visittime + slewtime"


def test_campaign_time_reports_the_share_of_the_nights_it_touched():
    """The denominator is the campaign's own nights, not the whole run.

    Including nights the campaign never touched would push the share towards
    zero and say nothing about the trade the scheduler actually made.
    """
    follow_up = make_visits([(60.0, -35.0)], night=11, visittime=34.0, slewtime=6.0)
    same_night = make_visits([(10.0, 10.0)] * 3, night=11, reason="pairs_ri_33", visittime=34.0, slewtime=6.0)
    other_night = make_visits(
        [(10.0, 10.0)] * 20, night=12, reason="pairs_ri_33", visittime=34.0, slewtime=6.0
    )
    visits = pd.concat([follow_up, same_night, other_night], ignore_index=True)

    timing = campaign_time(visits)
    assert timing.nights == 1
    assert timing.n_all_visits == 4, "only night 11 is in the denominator"
    assert timing.fraction == pytest.approx(1 / 4)


def test_campaign_time_is_keyed_from_the_trigger_night():
    visits = pd.concat(
        [
            make_visits([(60.0, -35.0)], night=11),
            make_visits([(60.5, -35.0)], night=12),
            make_visits([(10.0, 10.0)] * 3, night=12, reason="pairs_ri_33"),
        ],
        ignore_index=True,
    )
    timing = campaign_time(visits, reference_night=11)

    assert sorted(timing.by_night) == ["0", "1"]
    assert timing.by_night["0"]["fraction"] == pytest.approx(1.0), "night 0 was all follow-up"
    assert timing.by_night["1"]["fraction"] == pytest.approx(1 / 4)
    assert timing.by_night["1"]["n_too"] == 1
    for entry in timing.by_night.values():
        assert 0.0 <= entry["fraction"] <= 1.0


def test_campaign_time_splits_by_band_and_epoch():
    visits = pd.concat(
        [
            make_visits([(60.0, -35.0)], band="g", epoch=0),
            make_visits([(60.5, -35.0)], band="r", epoch=1, mjd0=61265.0 + 1 / 1440.0),
        ],
        ignore_index=True,
    )
    timing = campaign_time(visits)

    assert timing.by_band == {"g": pytest.approx(40.0), "r": pytest.approx(40.0)}
    assert timing.by_epoch == {"0": pytest.approx(40.0), "1": pytest.approx(40.0)}
    assert timing.span_hours == pytest.approx(1 / 60.0, abs=1e-6)


def test_campaign_time_says_when_it_could_only_count_exposures(caplog):
    """An undercount must not read as a cheap campaign."""
    visits = make_visits([(60.0, -35.0)]).drop(columns=["visittime", "slewtime"])
    with caplog.at_level("WARNING"):
        timing = campaign_time(visits)

    assert timing.too_seconds == pytest.approx(30.0)
    assert "no visittime or slewtime" in timing.measure
    assert "readout and slew" in caplog.text


def test_no_follow_up_visits_means_no_campaign_time():
    visits = make_visits([(10.0, 10.0)], reason="pairs_ri_33")
    timing = campaign_time(visits)

    assert timing.n_too_visits == 0
    assert timing.too_seconds == 0.0
    assert timing.by_night == {}


def test_campaign_time_survives_a_duplicated_index():
    """Concatenating without ignore_index must not mis-attribute time."""
    visits = pd.concat([make_visits([(60.0, -35.0)]), make_visits([(10.0, 10.0)] * 3, reason="pairs_ri_33")])
    assert not visits.index.is_unique, "the situation being guarded against"

    timing = campaign_time(visits)
    assert timing.n_too_visits == 1
    assert timing.too_seconds == pytest.approx(40.0)
    assert timing.fraction == pytest.approx(1 / 4)


def test_a_follow_up_visit_outside_the_contour_is_still_counted(caplog):
    """The bug the monotonicity guard caught: combined fell below ToO-only.

    Coverage is integrated over the whole map, but the overlap pre-selection
    tests the boresight against the *credible region* dilated by the field of
    view. A real skymap has faint probability outside its own 90% contour, so a
    follow-up pointing just beyond the contour still collects weight -- and
    dropping it from the combined set made the combined number smaller than the
    ToO-only number it is supposed to contain.
    """
    prob = np.zeros(hp.nside2npix(NSIDE_MAP))
    centre = hp.ang2vec(60.0, -35.0, lonlat=True)
    vectors = np.array(hp.pix2vec(NSIDE_MAP, np.arange(prob.size)))
    seps = np.degrees(np.arccos(np.clip(vectors.T @ centre, -1.0, 1.0)))
    prob = np.exp(-0.5 * (seps / 2.0) ** 2)  # tails everywhere, as BAYESTAR has
    prob /= prob.sum()
    localization = localization_from_probability(prob, provenance="test", credible_level=0.9)

    # Far enough out that the dilated contour does not contain the boresight,
    # near enough that the disc still lands on real probability.
    outside = make_visits([(60.0, -27.0)], band="g")
    from chatterbox.sim.coverage import visits_overlapping

    assert visits_overlapping(outside, localization).empty, "the setup being guarded"
    assert band_coverage(outside, localization).fractions["g"] > 0, "but it does cover weight"

    with caplog.at_level("WARNING"):
        comparison = compare_coverage(outside, localization)
    assert comparison.combined.any_band >= comparison.too.any_band
    assert comparison.survey_gain["any"] >= 0.0
    assert "should be impossible" not in caplog.text


# ---------------------------------------------------------------------------
# Coverage of the credible region itself (issue #1)
# ---------------------------------------------------------------------------


def gaussian_localization(sigma_deg=2.0, nside=NSIDE_MAP, credible_level=0.9):
    """A peaked map with tails everywhere, as a real BAYESTAR map has.

    A *uniform* map is no good for these tests: `credible_mask` sorts pixels by
    density, so on flat weight the 90% region it picks is an arbitrary
    scattering of pixels rather than a region.
    """
    centre = hp.ang2vec(60.0, -35.0, lonlat=True)
    vectors = np.array(hp.pix2vec(nside, np.arange(hp.nside2npix(nside))))
    seps = np.degrees(np.arccos(np.clip(vectors.T @ centre, -1.0, 1.0)))
    prob = np.exp(-0.5 * (seps / sigma_deg) ** 2)
    return localization_from_probability(prob / prob.sum(), provenance="test", credible_level=credible_level)


def test_the_region_is_the_credible_region_not_the_whole_map():
    """The denominator is the 90% region, and it holds ~90% of the map."""
    maps = region_localizations(gaussian_localization())
    assert maps.region_probability == pytest.approx(0.9, abs=0.01)
    assert maps.region_probability >= 0.9, "whole pixels, so never slightly under"
    assert maps.probability.prob_map.sum() == pytest.approx(1.0), "renormalized to the region"
    assert maps.area.prob_map.sum() == pytest.approx(1.0)
    assert 0 < maps.region_area_deg2 < 41253.0


def test_the_region_is_measured_at_the_resolution_floor():
    """A coarse map is regridded up, and the regrid conserves the region."""
    coarse = localization_from_reward_map(disc_reward_map(60.0, -35.0, 8.0), NSIDE)
    maps = region_localizations(coarse)
    assert maps.nside == REGION_NSIDE_FLOOR
    expected = int(coarse.prob_map.astype(bool).sum()) * hp.nside2pixarea(NSIDE, degrees=True)
    assert maps.region_area_deg2 == pytest.approx(expected, rel=1e-6)


def test_probability_and_area_agree_when_the_density_is_uniform():
    """The sanity check that the two numbers are the same measurement.

    With flat weight inside the region, the fraction of its probability and
    the fraction of its area under a set of visits are the same number, so any
    disagreement would mean the two maps were painted differently.
    """
    prob = np.zeros(hp.nside2npix(NSIDE_MAP))
    pixels = hp.query_disc(NSIDE_MAP, hp.ang2vec(60.0, -35.0, lonlat=True), np.radians(4.0), inclusive=True)
    prob[pixels] = 1.0
    # credible_level=1 so the region is the whole top hat rather than an
    # arbitrary 90% of equally-likely pixels.
    flat = localization_from_probability(prob, provenance="test", credible_level=1.0)

    visits = make_visits([(59.0, -35.0), (61.5, -34.0)])
    region = region_coverage(visits, flat)
    assert region.too_area == pytest.approx(region.too_probability, rel=1e-9)
    assert 0.05 < region.too_area < 0.95, "a partial cover, so the equality is not trivial"


def test_a_pointing_on_the_peak_buys_more_probability_than_area():
    """Why both numbers are reported rather than one standing for the other."""
    localization = gaussian_localization()
    region = region_coverage(make_visits([(60.0, -35.0)]), localization)
    assert region.too_probability > region.too_area * 2
    assert region.too_area < 0.35, "one field is a small part of a 90% region this size"


def test_tiling_the_region_covers_all_of_it():
    localization = gaussian_localization(sigma_deg=1.5)
    grid = [
        (60.0 + dx / np.cos(np.radians(35.0)), -35.0 + dy)
        for dx in np.arange(-6.0, 6.01, 1.5)
        for dy in np.arange(-6.0, 6.01, 1.5)
    ]
    region = region_coverage(make_visits(grid), localization)
    assert region.too_probability == pytest.approx(1.0, abs=0.01)
    assert region.too_area == pytest.approx(1.0, abs=0.01)


def test_probability_outside_the_region_does_not_count():
    """The distinction from the all-sky rows: tails are excluded here.

    A visit far enough out to miss the 90% contour still collects weight from
    a real map's tails, which is exactly what the all-sky number reports. The
    region number must not.
    """
    localization = gaussian_localization()
    outside = make_visits([(60.0, -25.0)])
    assert band_coverage(outside, localization).any_band > 0, "tails, as the setup needs"

    region = region_coverage(outside, localization)
    assert region.too_probability == pytest.approx(0.0, abs=1e-9)
    assert region.too_area == pytest.approx(0.0, abs=1e-9)


def test_the_region_probability_converts_back_to_an_all_sky_fraction():
    """`too_probability_absolute` is comparable to the all-sky row above it.

    Measured at the same resolution, the only thing separating the two is the
    probability the visits collected from outside the contour, so for fields
    sitting on the peak they nearly agree. The map is built at the region
    resolution here deliberately: at a coarser nside the all-sky pass paints
    inflated discs and runs ahead of the region number by much more than the
    tails accounts for, which is the effect
    `test_a_coarse_map_overstates_the_covered_area` pins.
    """
    localization = gaussian_localization(nside=REGION_NSIDE_FLOOR)
    visits = make_visits([(60.0, -35.0), (61.0, -36.0)])
    region = region_coverage(visits, localization)
    all_sky = band_coverage(visits, localization).any_band

    assert region.nside == localization.nside, "no regrid, so like for like"
    assert region.too_probability_absolute > 0
    assert region.too_probability_absolute <= all_sky + 1e-9, "the region is part of the map"
    assert region.too_probability_absolute == pytest.approx(all_sky, rel=0.05)


def test_the_combined_row_is_never_below_the_follow_up(caplog):
    localization = gaussian_localization()
    follow_up = make_visits([(60.0, -35.0)])
    survey = make_visits([(60.0, -38.0)], reason="pairs_gr_33")
    combined = pd.concat([follow_up, survey], ignore_index=True)

    with caplog.at_level("WARNING"):
        region = region_coverage(follow_up, localization, combined_visits=combined)
    assert region.combined_probability >= region.too_probability
    assert region.combined_area > region.too_area, "the survey visit reached new sky"
    assert region.combined_n_visits == 2
    assert "should be impossible" not in caplog.text


def test_the_reward_map_path_reports_area_only():
    """No probability skymap means no probability number, not a fake one."""
    localization = localization_from_reward_map(disc_reward_map(60.0, -35.0, 5.0), NSIDE)
    region = region_coverage(make_visits([(60.0, -35.0)]), localization)

    assert region.is_probability is False
    assert region.too_probability == 0.0, "not measured, rather than measured as zero"
    assert region.too_area > 0
    assert region.credible_level == pytest.approx(0.7), "the producer's level, not 0.9"


def test_a_coarse_map_overstates_the_covered_area():
    """What the resolution floor is defending against.

    `query_disc(inclusive=True)` keeps every pixel a field touches, so at the
    reward map's own nside 32 -- 3.4 deg^2 per pixel against a 9.6 deg^2 field
    -- the footprint inflates enough to flatter the strategy. The floor is the
    fix; this pins the effect so the constant is not quietly lowered.
    """
    localization = localization_from_reward_map(disc_reward_map(60.0, -35.0, 8.0), NSIDE)
    visits = make_visits([(60.0, -35.0), (62.0, -36.0)])

    coarse = region_coverage(visits, localization, nside=NSIDE)
    fine = region_coverage(visits, localization)

    assert coarse.region_area_deg2 == pytest.approx(fine.region_area_deg2, rel=1e-6)
    assert coarse.too_area > fine.too_area * 1.5
    assert fine.too_area > 0


def test_the_region_maps_work_with_the_per_band_machinery():
    """The maps are ordinary localizations, so nothing else needs rewriting."""
    maps = region_localizations(gaussian_localization())
    visits = pd.concat(
        [make_visits([(60.0, -35.0)], band="g"), make_visits([(61.0, -35.0)], band="r")],
        ignore_index=True,
    )
    per_band = band_coverage(visits, maps.probability)
    assert per_band.fractions["g"] > 0 and per_band.fractions["r"] > 0
    assert per_band.any_band <= per_band.fractions["g"] + per_band.fractions["r"] + 1e-9
    assert band_coverage(visits, maps.area).is_probability is False
