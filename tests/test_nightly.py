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
from chatterbox.slackbot.blocks import build_sim_figures_blocks
from chatterbox.slackbot.client import PostedMessage, render_blocks_as_text

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
        curve_plot="/sdf/home/s/seanmacb/.chatterbox/work/sim/S251112cm_20260814T173913/coverage.png",
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
    text = render_blocks_as_text(build_sim_figures_blocks(sim_result(), config))
    assert "/sdf/home/s/seanmacb/.chatterbox/work/sim/S251112cm_20260814T173913" in text
    assert "Simulation run in" in text
    assert "visits.db" in text


def test_message_states_the_selection_rule(config):
    text = render_blocks_as_text(build_sim_figures_blocks(sim_result(), config))
    assert "overlap the localization contour" in text
    assert "Nights shown: 11, 12" in text


def test_message_distinguishes_too_from_nominal_cadence(config):
    text = render_blocks_as_text(build_sim_figures_blocks(sim_result(), config))
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
    text = render_blocks_as_text(build_sim_figures_blocks(result, config))
    assert "1,108 visits touch the localization" in text
    assert "1,108 ToO follow-up" in text
    assert "0 from the nominal cadence" in text
    assert "1,134" not in text.split("Simulation run in")[0]


def test_message_admits_when_it_capped(config):
    text = render_blocks_as_text(build_sim_figures_blocks(sim_result(), config))
    assert "Showing the first 2 of 20 nights" in text


def test_message_omits_the_cap_note_when_nothing_was_dropped(config):
    result = sim_result(nights_with_overlap=2)
    text = render_blocks_as_text(build_sim_figures_blocks(result, config))
    assert "Showing the first" not in text


def test_message_blocks_are_valid(config):
    blocks = build_sim_figures_blocks(sim_result(), config)
    json.dumps(blocks)
    assert len(blocks) <= 50
    for block in blocks:
        if block["type"] == "header":
            assert block["text"]["type"] == "plain_text"
            assert len(block["text"]["text"]) <= 150
        if block["type"] == "section":
            assert len(block["text"]["text"]) <= 3000


def test_message_survives_a_missing_job_dir(config):
    text = render_blocks_as_text(build_sim_figures_blocks(sim_result(job_dir=""), config))
    assert "unknown" in text


def test_artifact_url_is_used_when_configured(config):
    config.sim.artifact_base_url = "https://usdf.example/chatterbox"
    text = render_blocks_as_text(build_sim_figures_blocks(sim_result(), config))
    assert "https://usdf.example/chatterbox/S251112cm_20260814T173913" in text


# ------------------------------------------------------------------- posting


def test_figures_start_their_own_thread(config, tmp_path):
    """Not replies: a 20-night run would swamp the alert's own thread."""
    from chatterbox.slackbot.client import SlackPoster

    poster = SlackPoster(config, dry_run=True, output_dir=tmp_path)
    plots = []
    for name in ("n1.png", "n2.png"):
        path = tmp_path / name
        path.write_bytes(b"x")
        plots.append(str(path))

    blocks = build_sim_figures_blocks(sim_result(nightly_plots=plots), config)
    poster.post(blocks, "nightly", files=plots, label="S1_figures")

    payload = json.loads((tmp_path / "S1_figures.json").read_text())
    # A top-level post carries no thread_ts; its files ride in its own thread.
    assert "thread_ts" not in payload
    assert len(payload["files"]) == 2


def test_the_curve_travels_with_the_nightly_maps(config):
    """Asked for explicitly: every coverage figure in the one thread."""
    from chatterbox.slackbot.blocks import sim_figures

    result = sim_result()
    figures = sim_figures(result)
    assert figures[0] == result.curve_plot, "the cumulative curve reads first"
    assert figures[1:] == result.nightly_plots
    text = render_blocks_as_text(build_sim_figures_blocks(result, config))
    assert "Cumulative coverage" in text


def test_figures_are_empty_when_nothing_rendered(config):
    from chatterbox.slackbot.blocks import sim_figures

    assert sim_figures(sim_result(curve_plot=None, nightly_plots=[])) == []


def test_the_coverage_reply_says_where_the_figures_went(config):
    """Otherwise a reply with no image reads as "no figures were made"."""
    from chatterbox.slackbot.blocks import build_sim_reply_blocks

    text = render_blocks_as_text(build_sim_reply_blocks(sim_result(), config))
    assert "thread of its own" in text
    # And per-epoch coverage is not lost when that line is added.
    bare = sim_result(curve_plot=None, nightly_plots=[], coverage_by_epoch={"0": {"g": 0.3}})
    text = render_blocks_as_text(build_sim_reply_blocks(bare, config))
    assert "thread of its own" not in text
    assert "epoch 0: g 30%" in text


class RecordingPoster:
    """A poster that records calls instead of reaching Slack."""

    def __init__(self):
        self.posts = []
        self.replies = []
        self.offline = False
        self.output_dir = "/dev/null"

    def post(self, blocks, text, is_test=False, files=None, label="post"):
        from chatterbox.slackbot.client import PostedMessage

        self.posts.append({"blocks": blocks, "text": text, "files": files, "label": label})
        return PostedMessage(channel="#too", ts=f"{len(self.posts)}.0")

    def reply(self, parent, blocks, text, files=None, label="reply"):
        from chatterbox.slackbot.client import PostedMessage

        self.replies.append({"parent": parent, "text": text, "files": files, "label": label})
        return PostedMessage(channel=parent.channel, ts="9.9")


def test_post_sim_results_posts_the_figures(config):
    """The figures must actually be handed to the poster, with the message."""
    from chatterbox.app import post_sim_results

    poster = RecordingPoster()
    result = sim_result()
    parent = PostedMessage(channel="#too", ts="1.0")
    sent = post_sim_results(result, config, poster, parent=parent)

    assert len(poster.replies) == 1, "coverage belongs in the alert's thread"
    assert len(poster.posts) == 1, "the figures belong in a message of their own"
    assert poster.posts[0]["files"] == [result.curve_plot] + result.nightly_plots
    assert poster.posts[0]["label"] == "S251112cm_figures"
    # No figure rides in the alert's thread; they are all in the other message.
    assert poster.replies[0]["files"] is None
    assert len(sent) == 2


def test_figures_are_posted_without_a_parent_message(config):
    """Regression: the figures were gated on stage 1 having posted.

    They are a top-level message, so they need no parent -- but the old code
    returned early when there was none, which meant a dry run, or a stage-1
    post that failed, silently produced PNGs and no Slack message at all.
    """
    from chatterbox.app import post_sim_results

    poster = RecordingPoster()
    sent = post_sim_results(sim_result(), config, poster, parent=None)

    assert not poster.replies
    labels = [p["label"] for p in poster.posts]
    assert labels == ["S251112cm_sim", "S251112cm_figures"]
    assert len(sent) == 2


def test_a_failed_coverage_post_still_posts_the_figures(config):
    """Losing the text must not also lose the figures."""
    from chatterbox.app import post_sim_results

    poster = RecordingPoster()
    original = poster.post

    def fail_first(blocks, text, is_test=False, files=None, label="post"):
        if label.endswith("_sim"):
            raise RuntimeError("Slack said no")
        return original(blocks, text, is_test=is_test, files=files, label=label)

    poster.post = fail_first
    sent = post_sim_results(sim_result(), config, poster, parent=None)

    assert [p["label"] for p in poster.posts] == ["S251112cm_figures"]
    assert len(sent) == 1


def test_dry_run_writes_both_payloads(config, tmp_path):
    """How a real run is previewed before it goes to a channel."""
    from chatterbox.app import post_sim_results
    from chatterbox.slackbot.client import SlackPoster

    plots = []
    for name in ("n1.png", "n2.png"):
        path = tmp_path / name
        path.write_bytes(b"x")
        plots.append(str(path))

    poster = SlackPoster(config, dry_run=True, output_dir=tmp_path)
    post_sim_results(sim_result(curve_plot=None, nightly_plots=plots), config, poster)

    assert (tmp_path / "S251112cm_sim.json").is_file()
    figures = json.loads((tmp_path / "S251112cm_figures.json").read_text())
    assert figures["files"] == plots


def test_the_service_delivers_the_figures_when_stage_one_did_not_post(config, monkeypatch, tmp_path):
    """The same regression, at the site that had it: the service's own path."""
    from types import SimpleNamespace

    from chatterbox import app

    job = SimpleNamespace(job_dir=tmp_path, wait=lambda timeout=None: 0, tail_log=lambda lines=20: "")
    monkeypatch.setattr(app, "launch_simulation", lambda trigger, cfg: job)
    monkeypatch.setattr(app, "load_sim_result", lambda job_dir: sim_result())

    poster = RecordingPoster()
    trigger = SimpleNamespace(source="S251112cm", is_test=False)
    report = SimpleNamespace(posted=None)
    app._start_simulation(trigger, config, poster, report, sim_wait=True)

    assert [p["label"] for p in poster.posts] == ["S251112cm_sim", "S251112cm_figures"]


def test_test_alerts_route_both_messages(config):
    """An is_test simulation must not land in the operational channel."""
    from chatterbox.app import post_sim_results
    from chatterbox.slackbot.client import SlackPoster

    config.slack.channel = "#too"
    config.slack.test_channel = "#too-test"
    poster = SlackPoster(config, dry_run=True)
    assert poster.channel_for(is_test=True) == "#too-test"

    recorder = RecordingPoster()
    post_sim_results(sim_result(), config, recorder, is_test=True)
    assert all(p["blocks"] for p in recorder.posts)
    # The routing itself lives in the poster; what matters here is that the
    # flag reaches it for both messages.
    calls = []
    original = recorder.post

    def capture(blocks, text, is_test=False, files=None, label="post"):
        calls.append(is_test)
        return original(blocks, text, is_test=is_test, files=files, label=label)

    recorder.post = capture
    post_sim_results(sim_result(), config, recorder, is_test=True)
    assert calls == [True, True]


# ------------------------------------------------------- surviving the process


def background_sim(monkeypatch, tmp_path, delay_s=0.2):
    """A backgrounded simulation whose driver takes `delay_s` to finish."""
    import time
    from types import SimpleNamespace

    from chatterbox import app

    def wait(timeout=None):
        time.sleep(delay_s)
        return 0

    job = SimpleNamespace(job_dir=tmp_path, wait=wait, tail_log=lambda lines=20: "")
    monkeypatch.setattr(app, "launch_simulation", lambda trigger, cfg: job)
    monkeypatch.setattr(app, "load_sim_result", lambda job_dir: sim_result())
    return app, job


def test_a_backgrounded_simulation_is_joinable(config, monkeypatch, tmp_path):
    """Why nothing posted: the posting thread died with the process.

    ``launch_simulation`` detaches the driver with ``start_new_session``, so
    the simulation survives the CLI exiting and writes every PNG. The daemon
    thread waiting to post it does not. The report must therefore expose that
    thread, so a caller about to exit can wait for it.
    """
    from types import SimpleNamespace

    app, _ = background_sim(monkeypatch, tmp_path)
    poster = RecordingPoster()
    report = app.TriggerReport(
        trigger=SimpleNamespace(source="S251112cm", is_test=False),
        events=None,
        dark_hours=None,
        dark_stats={},
        template_coverage=None,
        blocks=[],
        text="",
    )
    app._start_simulation(report.trigger, config, poster, report, sim_wait=False)

    assert report.sim_thread is not None, "the caller cannot wait for what it cannot see"
    assert not poster.posts, "the driver has not finished yet"
    assert report.wait_for_simulation(timeout=30)
    assert [p["label"] for p in poster.posts] == ["S251112cm_sim", "S251112cm_figures"]


def test_waiting_reports_a_simulation_that_is_still_running(config, monkeypatch, tmp_path):
    from types import SimpleNamespace

    app, _ = background_sim(monkeypatch, tmp_path, delay_s=5.0)
    report = app.TriggerReport(
        trigger=SimpleNamespace(source="S251112cm", is_test=False),
        events=None,
        dark_hours=None,
        dark_stats={},
        template_coverage=None,
        blocks=[],
        text="",
    )
    app._start_simulation(report.trigger, config, RecordingPoster(), report, sim_wait=False)
    assert not report.wait_for_simulation(timeout=0.05)


def test_the_service_names_the_command_that_posts_a_stranded_run(config, monkeypatch, tmp_path, caplog):
    """A shutdown mid-simulation must leave a way to post it."""
    from types import SimpleNamespace

    app, job = background_sim(monkeypatch, tmp_path, delay_s=5.0)
    report = app.TriggerReport(
        trigger=SimpleNamespace(source="S251112cm", is_test=False),
        events=None,
        dark_hours=None,
        dark_stats={},
        template_coverage=None,
        blocks=[],
        text="",
    )
    app._start_simulation(report.trigger, config, RecordingPoster(), report, sim_wait=False)

    with caplog.at_level("WARNING"):
        app.drain_simulations([report], grace_s=0.05)
    assert f"chatterbox post-sim {tmp_path}" in caplog.text


# ------------------------------------------------------------------- post-sim


def finished_job_dir(tmp_path, **overrides):
    """A job directory as the driver leaves it, with real figure files."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    figures = []
    for name in ("coverage.png", "S251112cm_night011_visits.png"):
        path = tmp_path / name
        path.write_bytes(b"x")
        figures.append(str(path))
    result = sim_result(curve_plot=figures[0], nightly_plots=figures[1:], **overrides)
    (tmp_path / "result.json").write_text(json.dumps(result.__dict__))
    return tmp_path, figures


def test_post_sim_posts_a_finished_run(config, tmp_path, capsys):
    """The recovery path for a run whose launcher had already exited."""
    from chatterbox.cli import main

    job_dir, figures = finished_job_dir(tmp_path / "job")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"paths: {{work_dir: {tmp_path / 'work'}}}\n")

    assert main(["-c", str(config_path), "post-sim", str(job_dir), "--dry-run"]) == 0
    posts = tmp_path / "work" / "posts"
    payload = json.loads((posts / "S251112cm_figures.json").read_text())
    assert payload["files"] == figures
    assert (posts / "S251112cm_sim.json").is_file()


def test_post_sim_refuses_a_directory_with_no_result(config, tmp_path, capsys):
    from chatterbox.cli import main

    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"paths: {{work_dir: {tmp_path / 'work'}}}\n")
    unfinished = tmp_path / "still-running"
    unfinished.mkdir()

    assert main(["-c", str(config_path), "post-sim", str(unfinished), "--dry-run"]) == 1
    assert "has not finished" in capsys.readouterr().err


def test_post_sim_routes_a_test_alert(config, tmp_path):
    """result.json does not record is_test; job.json does."""
    from chatterbox.sim.runner import read_job_spec

    job_dir, _ = finished_job_dir(tmp_path / "job")
    (job_dir / "job.json").write_text(json.dumps({"source": "S251112cm", "is_test": True}))
    assert read_job_spec(job_dir)["is_test"] is True
    # And a job directory from before that field existed still posts.
    assert read_job_spec(tmp_path / "nonexistent") == {}
