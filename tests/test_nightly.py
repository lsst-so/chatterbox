"""Tests for per-night visit coverage: selection, mapping, plots, posting."""

import json

import healpy as hp
import numpy as np
import pandas as pd
import pytest
from conftest import disc_reward_map

from chatterbox.astro.skymap import localization_from_probability, localization_from_reward_map
from chatterbox.plots.style import localization_extent, localization_levels, sky_projection
from chatterbox.sim.coverage import nightly_visit_maps, visits_overlapping
from chatterbox.sim.runner import SimResult
from chatterbox.slackbot.blocks import build_nightly_visits_blocks
from chatterbox.slackbot.client import render_blocks_as_text

NSIDE = 128


def disc_localization(ra=60.0, dec=-35.0, radius=3.0, nside=NSIDE, graded=False):
    """A localization covering a disc, optionally with a density gradient."""
    prob = np.zeros(hp.nside2npix(nside))
    pixels = hp.query_disc(nside, hp.ang2vec(ra, dec, lonlat=True), np.radians(radius), inclusive=True)
    if graded:
        centre = hp.ang2vec(ra, dec, lonlat=True)
        vectors = np.array(hp.pix2vec(nside, pixels))
        seps = np.degrees(np.arccos(np.clip(vectors.T @ centre, -1.0, 1.0)))
        prob[pixels] = np.exp(-(seps**2) / (2 * (radius / 2) ** 2))
    else:
        prob[pixels] = 1.0
    return localization_from_probability(prob, provenance="test", credible_level=0.9)


def visit_rows(pointings, band="g", night=1, reason="pairs_ri_33.0"):
    """Visits in the raw sim_runner convention: RA/dec in radians."""
    return pd.DataFrame(
        [
            {
                "RA": np.radians(ra),
                "dec": np.radians(dec),
                "band": band,
                "night": night,
                "observation_reason": reason,
                "mjd": 61000.0 + n / 1440.0,
            }
            for n, (ra, dec) in enumerate(pointings)
        ]
    )


# ------------------------------------------------------------------ selection


def test_overlapping_visit_is_selected():
    localization = disc_localization()
    selected = visits_overlapping(visit_rows([(60.0, -35.0)]), localization, nside=NSIDE)
    assert len(selected) == 1


def test_distant_visit_is_excluded():
    localization = disc_localization()
    selected = visits_overlapping(visit_rows([(200.0, 40.0)]), localization, nside=NSIDE)
    assert selected.empty


def test_visit_just_outside_still_overlaps_via_the_field_of_view():
    """A boresight off the region still overlaps if its FOV reaches in."""
    localization = disc_localization(radius=3.0)
    # 4.5 deg from centre: 1.5 deg outside the region, inside a 1.75 deg FOV.
    selected = visits_overlapping(visit_rows([(60.0, -30.5)]), localization, nside=NSIDE)
    assert len(selected) == 1


def test_non_too_visits_are_included():
    """This is the point: nominal-cadence visits contribute real coverage."""
    survey = visit_rows([(60.0, -35.0)], reason="pairs_ri_33.0")
    follow_up = visit_rows([(60.0, -35.0)], reason="too_GW_case_B_0_i0")
    both = pd.concat([survey, follow_up], ignore_index=True)

    selected = visits_overlapping(both, disc_localization(), nside=NSIDE)
    assert len(selected) == 2
    reasons = set(selected["observation_reason"])
    assert "pairs_ri_33.0" in reasons


def test_selection_uses_the_credible_region_not_every_non_zero_pixel():
    """A real probability map has non-zero tails almost everywhere.

    Selecting on ``prob > 0`` would match visits far outside the contour, and
    would make the dilation crawl over the whole sky.
    """
    nside = 64
    # Broad tail everywhere, plus a concentrated core.
    prob = np.full(hp.nside2npix(nside), 1e-9)
    core = hp.query_disc(nside, hp.ang2vec(60.0, -35.0, lonlat=True), np.radians(3.0), inclusive=True)
    prob[core] = 1.0
    localization = localization_from_probability(prob, provenance="tailed", credible_level=0.9)

    far = visit_rows([(200.0, 40.0)])
    assert visits_overlapping(far, localization, nside=nside).empty
    near = visit_rows([(60.0, -35.0)])
    assert len(visits_overlapping(near, localization, nside=nside)) == 1


def test_binary_reward_map_uses_its_whole_region():
    """A reward map is uniform, so every non-zero pixel is the region."""
    localization = localization_from_reward_map(disc_reward_map(60.0, -35.0, 9.8), 32)
    selected = visits_overlapping(visit_rows([(60.0, -35.0)]), localization, nside=32)
    assert len(selected) == 1


def test_empty_region_selects_nothing(caplog):
    empty = localization_from_reward_map(np.zeros(hp.nside2npix(32), dtype=bool), 32)
    with caplog.at_level("WARNING"):
        assert visits_overlapping(visit_rows([(60.0, -35.0)]), empty, nside=32).empty
    assert "no visit can overlap" in caplog.text


# -------------------------------------------------------------------- mapping


def test_nightly_maps_are_keyed_by_night_and_band():
    visits = pd.concat(
        [
            visit_rows([(60.0, -35.0)], band="g", night=3),
            visit_rows([(60.0, -35.0)], band="r", night=3),
            visit_rows([(60.0, -35.0)], band="g", night=4),
        ],
        ignore_index=True,
    )
    maps = nightly_visit_maps(visits, nside=NSIDE)
    assert sorted(maps) == [3, 4]
    assert sorted(maps[3]) == ["g", "r"]
    assert sorted(maps[4]) == ["g"]


def test_repeat_visits_accumulate():
    """Counts, not coverage: three visits to one field read 3, not 1."""
    visits = visit_rows([(60.0, -35.0)] * 3, band="i", night=2)
    counts = nightly_visit_maps(visits, nside=NSIDE)[2]["i"]
    assert counts.max() == 3


def test_nightly_maps_need_the_night_column():
    visits = visit_rows([(60.0, -35.0)]).drop(columns=["night"])
    with pytest.raises(KeyError, match="night"):
        nightly_visit_maps(visits, nside=NSIDE)


def test_nightly_maps_handle_the_opsim_convention():
    """save_opsim writes fieldRA/fieldDec in degrees."""
    visits = visit_rows([(60.0, -35.0)], band="z", night=5)
    visits = visits.rename(columns={"RA": "fieldRA", "dec": "fieldDec"})
    visits["fieldRA"] = np.degrees(visits["fieldRA"])
    visits["fieldDec"] = np.degrees(visits["fieldDec"])
    assert nightly_visit_maps(visits, nside=NSIDE)[5]["z"].max() == 1


# --------------------------------------------------------------------- framing


def test_extent_of_a_compact_region():
    ra, dec, radius = localization_extent(disc_localization(radius=3.0))
    assert ra == pytest.approx(60.0, abs=0.5)
    assert dec == pytest.approx(-35.0, abs=0.5)
    assert radius == pytest.approx(3.0, abs=0.5)


def test_compact_localization_is_drawn_zoomed():
    """A 12 deg^2 patch of coverage is a few pixels on an all-sky map."""
    projection, kwargs = sky_projection(disc_localization(radius=3.0))
    assert projection == "astro zoom"
    assert "center" in kwargs and "radius" in kwargs
    assert kwargs["radius"].to_value("deg") > 3.0


def test_large_localization_stays_all_sky():
    """Zooming a 60 deg region would silently crop it."""
    projection, kwargs = sky_projection(disc_localization(radius=60.0))
    assert projection == "astro degrees mollweide"
    assert kwargs == {}


def test_extent_of_an_empty_region_is_nan():
    empty = localization_from_reward_map(np.zeros(hp.nside2npix(32), dtype=bool), 32)
    ra, dec, radius = localization_extent(empty)
    assert not np.isfinite(radius)
    # And it falls back to all-sky rather than raising.
    assert sky_projection(empty)[0] == "astro degrees mollweide"


# -------------------------------------------------------------------- contours


def test_flat_probability_region_gets_a_boundary_level():
    """A uniform region has no gradient; a credible level fills it solid."""
    localization = disc_localization(graded=False)
    levels = localization_levels(localization)
    assert len(levels) == 1
    positive = localization.prob_map[localization.prob_map > 0]
    assert levels[0] < positive.min()


def test_graded_map_gets_real_credible_levels():
    """With a density gradient, the credible contours are meaningful."""
    localization = disc_localization(radius=6.0, graded=True)
    levels = localization_levels(localization)
    assert len(levels) >= 1
    assert levels == sorted(levels)
    positive = localization.prob_map[localization.prob_map > 0]
    # Real credible thresholds sit inside the distribution, not at its floor.
    assert all(level > positive.min() / 2 for level in levels)


def test_levels_are_never_at_or_above_the_peak():
    """A contour at the peak encloses nothing and renders as noise."""
    localization = disc_localization(radius=6.0, graded=True)
    peak = localization.prob_map.max()
    assert all(level < peak for level in localization_levels(localization))


def test_reward_map_still_traces_its_boundary():
    localization = localization_from_reward_map(disc_reward_map(60.0, -35.0, 5.0), 32)
    levels = localization_levels(localization)
    assert len(levels) == 1


# -------------------------------------------------------------------- plotting


def test_plot_writes_one_figure_per_night(tmp_path):
    from chatterbox.plots.nightly import plot_all_nights

    localization = disc_localization(graded=True)
    visits = pd.concat(
        [visit_rows([(60.0, -35.0), (60.5, -35.2)], band="g", night=n) for n in (3, 4, 5)],
        ignore_index=True,
    )
    maps = nightly_visit_maps(visits, nside=NSIDE)
    paths, nights, total = plot_all_nights(
        maps, localization, tmp_path, source="S1", alert_type="GW_case_B", max_nights=0
    )
    assert nights == [3, 4, 5]
    assert total == 3
    for path in paths:
        assert path.is_file()
        assert path.stat().st_size > 10_000


def test_plot_caps_the_number_of_nights_and_reports_the_total(tmp_path, caplog):
    """A 50-night BBH run must not upload 50 figures, nor hide the cap."""
    from chatterbox.plots.nightly import plot_all_nights

    visits = pd.concat(
        [visit_rows([(60.0, -35.0)], band="g", night=n) for n in range(1, 8)], ignore_index=True
    )
    maps = nightly_visit_maps(visits, nside=NSIDE)
    with caplog.at_level("WARNING"):
        paths, nights, total = plot_all_nights(
            maps, disc_localization(graded=True), tmp_path, source="S1", max_nights=2
        )
    assert len(paths) == 2
    assert nights == [1, 2]
    assert total == 7, "the pre-cap total must survive, so the post can say so"
    assert "first 2 of 7" in caplog.text


def test_plot_of_no_nights_is_not_an_error(tmp_path):
    from chatterbox.plots.nightly import plot_all_nights

    paths, nights, total = plot_all_nights({}, disc_localization(), tmp_path)
    assert paths == [] and nights == [] and total == 0


# --------------------------------------------------------------------- message


def sim_result(**overrides):
    defaults = dict(
        status="complete",
        source="S251112cm",
        alert_type="GW_case_B",
        nights=20,
        coverage={"g": 0.33, "i": 0.39},
        any_band=0.41,
        quantity="probability",
        total_visits=6994,
        too_visits=1972,
        job_dir="/sdf/home/s/seanmacb/.chatterbox/work/sim/S251112cm_20260814T173913",
        visits_path="/sdf/home/s/seanmacb/.chatterbox/work/sim/S251112cm_20260814T173913/visits.db",
        nightly_plots=["/tmp/a.png", "/tmp/b.png"],
        nightly_plot_nights=[11, 12],
        nights_with_overlap=20,
        overlap_visits=2356,
        overlap_too_visits=1804,
    )
    defaults.update(overrides)
    return SimResult(**defaults)


def test_message_names_the_run_directory(config):
    """The whole point of the text: where the simulation ran."""
    text = render_blocks_as_text(build_nightly_visits_blocks(sim_result(), config))
    assert "/sdf/home/s/seanmacb/.chatterbox/work/sim/S251112cm_20260814T173913" in text
    assert "Simulation run in" in text
    assert "visits.db" in text


def test_message_states_the_selection_rule(config):
    text = render_blocks_as_text(build_nightly_visits_blocks(sim_result(), config))
    assert "overlap the localization contour" in text
    assert "Nights shown: 11, 12" in text


def test_message_distinguishes_too_from_nominal_cadence(config):
    text = render_blocks_as_text(build_nightly_visits_blocks(sim_result(), config))
    assert "2,356 visits touch the localization" in text
    assert "1,804 ToO follow-up" in text
    assert "552 from the nominal cadence" in text


def test_message_splits_within_the_overlapping_set(config):
    """The split must not mix denominators.

    ``too_visits`` counts follow-up over the whole sky, so using it here could
    claim more follow-up visits than there are overlapping visits at all -- a
    3-night run where every visit is follow-up reported 1,134 of 1,108.
    """
    result = sim_result(total_visits=1134, too_visits=1134, overlap_visits=1108, overlap_too_visits=1108)
    text = render_blocks_as_text(build_nightly_visits_blocks(result, config))
    assert "1,108 visits touch the localization" in text
    assert "1,108 ToO follow-up" in text
    assert "0 from the nominal cadence" in text
    assert "1,134" not in text.split("Simulation run in")[0]


def test_message_admits_when_it_capped(config):
    text = render_blocks_as_text(build_nightly_visits_blocks(sim_result(), config))
    assert "Showing the first 2 of 20 nights" in text


def test_message_omits_the_cap_note_when_nothing_was_dropped(config):
    result = sim_result(nights_with_overlap=2)
    text = render_blocks_as_text(build_nightly_visits_blocks(result, config))
    assert "Showing the first" not in text


def test_message_blocks_are_valid(config):
    blocks = build_nightly_visits_blocks(sim_result(), config)
    json.dumps(blocks)
    assert len(blocks) <= 50
    for block in blocks:
        if block["type"] == "header":
            assert block["text"]["type"] == "plain_text"
            assert len(block["text"]["text"]) <= 150
        if block["type"] == "section":
            assert len(block["text"]["text"]) <= 3000


def test_message_survives_a_missing_job_dir(config):
    text = render_blocks_as_text(build_nightly_visits_blocks(sim_result(job_dir=""), config))
    assert "unknown" in text


def test_artifact_url_is_used_when_configured(config):
    config.sim.artifact_base_url = "https://usdf.example/chatterbox"
    text = render_blocks_as_text(build_nightly_visits_blocks(sim_result(), config))
    assert "https://usdf.example/chatterbox/S251112cm_20260814T173913" in text


# ------------------------------------------------------------------- posting


def test_nightly_plots_start_their_own_thread(config, tmp_path):
    """Not replies: a 20-night run would swamp the alert's own thread."""
    from chatterbox.slackbot.client import SlackPoster

    poster = SlackPoster(config, dry_run=True, output_dir=tmp_path)
    plots = []
    for name in ("n1.png", "n2.png"):
        path = tmp_path / name
        path.write_bytes(b"x")
        plots.append(str(path))

    blocks = build_nightly_visits_blocks(sim_result(nightly_plots=plots), config)
    poster.post(blocks, "nightly", files=plots, label="S1_nightly")

    payload = json.loads((tmp_path / "S1_nightly.json").read_text())
    # A top-level post carries no thread_ts; its files ride in its own thread.
    assert "thread_ts" not in payload
    assert len(payload["files"]) == 2
