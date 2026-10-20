"""Tests for Slack message construction and delivery."""

import json

import pytest

from chatterbox.ingest.decode import decode_record
from chatterbox.sim.runner import SimResult
from chatterbox.slackbot.blocks import build_sim_reply_blocks, build_trigger_blocks, plain_text_summary
from chatterbox.slackbot.client import SlackPoster, render_blocks_as_text


def block_text(blocks) -> str:
    """Concatenate every piece of text in a block payload."""
    return render_blocks_as_text(blocks)


@pytest.fixture
def rendered(record, config, rubin_scheduler):
    """Blocks for a trigger, with the almanac and dark-hours map computed."""
    from chatterbox.astro.almanac import night_events
    from chatterbox.astro.darkhours import dark_hours_map
    from chatterbox.plots.darkhours import region_hours_summary

    trigger = decode_record(record)
    events = night_events(when=trigger.event_time)
    dark = dark_hours_map(events, nside=32, step_minutes=10.0)
    stats = region_hours_summary(dark, trigger.localization)
    blocks = build_trigger_blocks(trigger, events, dark, stats, None, config)
    return trigger, blocks


def test_blocks_are_valid_shapes(rendered):
    _, blocks = rendered
    assert len(blocks) <= 50, "Slack rejects more than 50 blocks"
    for block in blocks:
        assert "type" in block
        if block["type"] == "section":
            assert len(block["text"]["text"]) <= 3000
        if block["type"] == "header":
            # Slack requires plain_text and at most 150 characters.
            assert block["text"]["type"] == "plain_text"
            assert len(block["text"]["text"]) <= 150


def test_blocks_are_json_serializable(rendered):
    _, blocks = rendered
    json.dumps(blocks)


def test_post_mentions_the_key_facts(rendered):
    trigger, blocks = rendered
    text = block_text(blocks)
    assert trigger.source in text
    assert trigger.alert_type in text
    assert "Gravitational wave" in text
    # Observability
    assert "Sun < -12 deg" in text
    assert "Moon" in text
    assert "Accessible dark hours" in text
    # The strategy the scheduler will follow
    assert "Planned Rubin follow-up" in text
    # Why it triggered
    assert "Why it triggered" in text


def test_area_based_numbers_are_labelled_as_area(rendered):
    """The bot must never present an area fraction as a probability."""
    trigger, blocks = rendered
    assert trigger.localization.is_probability is False
    text = block_text(blocks)
    assert "area" in text.lower()
    assert "*area* fractions" in text


def test_missing_template_cache_is_called_out(rendered):
    _, blocks = rendered
    assert "refresh-templates" in block_text(blocks)


def test_template_coverage_is_reported_when_available(record, config, rubin_scheduler):
    import healpy as hp
    import numpy as np

    from chatterbox.astro.almanac import night_events
    from chatterbox.astro.darkhours import dark_hours_map
    from chatterbox.astro.templates import TemplateCoverage
    from chatterbox.plots.darkhours import region_hours_summary

    trigger = decode_record(record)
    events = night_events(when=trigger.event_time)
    dark = dark_hours_map(events, nside=32, step_minutes=10.0)
    stats = region_hours_summary(dark, trigger.localization)

    nside = 64
    covered = np.zeros(hp.nside2npix(nside))
    covered[hp.query_disc(nside, hp.ang2vec(60.0, -35.0, lonlat=True), np.radians(6.0), inclusive=True)] = 1.0
    coverage = TemplateCoverage(
        maps={"i": covered},
        nside=nside,
        built_at="2026-08-13T00:00:00+00:00",
        source="/published/maps",
    )

    blocks = build_trigger_blocks(trigger, events, dark, stats, coverage, config)
    text = block_text(blocks)
    assert "Existing template coverage" in text
    assert "Cache refreshed" in text
    assert "/published/maps" in text
    # Bands with no published map must be called out rather than silently zero.
    assert "No coverage map published for" in text
    assert "u" in text


def test_test_alerts_are_marked(data_dir, config, rubin_scheduler):
    from chatterbox.astro.almanac import night_events
    from chatterbox.astro.darkhours import dark_hours_map
    from chatterbox.ingest.decode import load_record_file
    from chatterbox.plots.darkhours import region_hours_summary

    trigger = decode_record(load_record_file(data_dir / "gw_case_d_test.json"))
    events = night_events(when=trigger.event_time)
    dark = dark_hours_map(events, nside=32, step_minutes=10.0)
    stats = region_hours_summary(dark, trigger.localization)
    blocks = build_trigger_blocks(trigger, events, dark, stats, None, config)
    text = block_text(blocks)
    assert "TEST" in text
    assert "[TEST]" in blocks[0]["text"]["text"]


def test_unproduced_alert_types_are_flagged(record, config, rubin_scheduler):
    from chatterbox.astro.almanac import night_events
    from chatterbox.astro.darkhours import dark_hours_map
    from chatterbox.plots.darkhours import region_hours_summary

    record["alert_type"] = "SSO_night"
    trigger = decode_record(record)
    events = night_events(when=trigger.event_time)
    dark = dark_hours_map(events, nside=32, step_minutes=10.0)
    stats = region_hours_summary(dark, trigger.localization)
    blocks = build_trigger_blocks(trigger, events, dark, stats, None, config)
    text = block_text(blocks)
    assert "No current producer emits this alert type" in text
    assert "asteroid" in text.lower()


def test_plain_text_summary(record):
    trigger = decode_record(record)
    summary = plain_text_summary(trigger)
    assert trigger.source in summary
    assert "deg^2" in summary


def test_plain_text_summary_marks_tests(record):
    record["is_test"] = True
    assert plain_text_summary(decode_record(record)).startswith("[TEST]")


# ---------------------------------------------------------------- sim replies


def test_sim_reply_reports_every_band(config):
    result = SimResult(
        status="complete",
        source="S260814a",
        alert_type="GW_case_B",
        nights=20,
        coverage={"u": 0.0, "g": 0.91, "r": 0.42, "i": 0.42, "z": 0.0, "y": 0.0},
        n_visits={"g": 325, "r": 133, "i": 133},
        any_band=0.93,
        quantity="probability",
        total_visits=9912,
        too_visits=591,
        runtime_s=1234.0,
    )
    text = block_text(build_sim_reply_blocks(result, config))
    for band in ("u", "g", "r", "i", "z", "y"):
        # Including the bands with nothing, which is a result in itself.
        assert f"{band} " in text
    assert "u 0.0%" in text
    assert "91.0%" in text
    assert "probability" in text
    # A single unlabelled row invites both readings; the label is the point.
    assert "ToO follow-up only" in text


def test_sim_reply_flags_area_based_results(config):
    result = SimResult(
        status="complete",
        source="IceCube-260814A",
        alert_type="neutrino",
        nights=30,
        coverage={"g": 0.5},
        any_band=0.5,
        quantity="area",
    )
    text = block_text(build_sim_reply_blocks(result, config))
    assert "*area* fractions, not probabilities" in text


def test_sim_reply_reports_failure(config):
    result = SimResult(
        status="failed",
        source="S260814a",
        alert_type="GW_case_B",
        nights=20,
        error="scheduler config not found",
    )
    text = block_text(build_sim_reply_blocks(result, config))
    assert "did not produce coverage" in text
    assert "scheduler config not found" in text


def test_sim_reply_includes_the_band_carousel_note(config):
    result = SimResult(
        status="complete",
        source="S260814a",
        alert_type="GW_case_B",
        nights=20,
        coverage={"g": 0.5},
        any_band=0.5,
        quantity="probability",
        band_scheduler="SimpleBandSched backup applies",
    )
    assert "SimpleBandSched" in block_text(build_sim_reply_blocks(result, config))


# ------------------------------------------------------------------ failures


def test_failure_blocks_name_what_broke_and_where():
    from chatterbox.slackbot.blocks import build_failure_blocks

    blocks = build_failure_blocks(
        "handling a ToO record",
        KeyError("reward_map"),
        source="S251112cm",
        origin="lsst.scimma.too_alert on summit_efd",
        log_tail="Traceback (most recent call last):\n  boom",
    )
    text = render_blocks_as_text(blocks)
    assert "handling a ToO record" in text
    assert "S251112cm" in text
    assert "lsst.scimma.too_alert on summit_efd" in text
    assert "KeyError" in text, "the exception type is half the diagnosis"
    assert "boom" in text


def test_failure_blocks_are_valid_without_any_context():
    """Decoding is one of the things that fails, so the id is often unknown."""
    from chatterbox.slackbot.blocks import build_failure_blocks

    blocks = build_failure_blocks("monitoring for ToO alerts (efd)", "influx is unreachable")
    json.dumps(blocks)
    assert blocks[0]["type"] == "header"
    assert len(blocks[0]["text"]["text"]) <= 150
    for block in blocks:
        if block["type"] == "section":
            assert len(block["text"]["text"]) <= 3000
    assert "influx is unreachable" in render_blocks_as_text(blocks)


def test_a_huge_traceback_does_not_produce_an_invalid_block():
    from chatterbox.slackbot.blocks import build_failure_blocks

    blocks = build_failure_blocks("simulating S1", RuntimeError("x" * 20_000), log_tail="y" * 20_000)
    for block in blocks:
        if block["type"] == "section":
            assert len(block["text"]["text"]) <= 3000


def test_posting_a_failure_never_raises(config):
    """It is called from except blocks; it cannot add a second failure."""
    from chatterbox.app import post_failure

    class BrokenPoster:
        def post(self, *args, **kwargs):
            raise RuntimeError("Slack is down too")

    assert post_failure("handling a ToO record", ValueError("first"), BrokenPoster()) is None
    # And with no poster at all, as in a dry run.
    assert post_failure("handling a ToO record", ValueError("first"), None) is None


def test_a_failed_test_alert_goes_to_the_test_channel(config, tmp_path):
    from chatterbox.app import post_failure

    config.slack.test_channel = "#too-test"
    poster = SlackPoster(config, dry_run=True, output_dir=tmp_path)
    post_failure("handling a ToO record", ValueError("nope"), poster, source="S1", is_test=True)
    payload = json.loads((tmp_path / "S1_failure.json").read_text())
    assert payload["channel"] == "#too-test"


# -------------------------------------------------------------------- poster


def test_offline_poster_writes_a_payload(config, tmp_path):
    poster = SlackPoster(config, dry_run=True, output_dir=tmp_path)
    posted = poster.post([{"type": "divider"}], "hello", label="unit")
    assert posted.offline is True
    assert posted.can_thread is False
    payload = json.loads((tmp_path / "unit.json").read_text())
    assert payload["text"] == "hello"
    assert payload["channel"] == config.slack.channel


def test_offline_poster_does_not_need_slack_sdk(config, tmp_path, monkeypatch):
    """The whole pipeline must be runnable without slack-sdk installed."""
    monkeypatch.delenv(config.slack.bot_token_env, raising=False)
    poster = SlackPoster(config, output_dir=tmp_path)
    assert poster.offline is True
    assert poster.post([{"type": "divider"}], "hi", label="x").offline is True


def test_test_alerts_route_to_the_test_channel(config, tmp_path):
    config.slack.test_channel = "#too-test"
    poster = SlackPoster(config, dry_run=True, output_dir=tmp_path)
    assert poster.channel_for(is_test=True) == "#too-test"
    assert poster.channel_for(is_test=False) == config.slack.channel


def test_test_alerts_fall_back_to_the_main_channel(config, tmp_path):
    config.slack.test_channel = ""
    poster = SlackPoster(config, dry_run=True, output_dir=tmp_path)
    assert poster.channel_for(is_test=True) == config.slack.channel


def test_upload_of_a_missing_file_is_not_fatal(config, tmp_path):
    poster = SlackPoster(config, dry_run=True, output_dir=tmp_path)
    assert poster.upload(tmp_path / "nope.png", channel="#x") is False


def test_render_blocks_as_text_covers_all_block_types():
    blocks = [
        {"type": "header", "text": {"type": "plain_text", "text": "Title"}},
        {"type": "section", "text": {"type": "mrkdwn", "text": "Body"}},
        {"type": "divider"},
        {"type": "context", "elements": [{"type": "mrkdwn", "text": "Note"}]},
    ]
    text = render_blocks_as_text(blocks)
    assert "Title" in text and "Body" in text and "Note" in text
    assert "---" in text


def test_long_section_text_is_truncated():
    from chatterbox.slackbot.blocks import MAX_SECTION_CHARS, _section

    block = _section("x" * (MAX_SECTION_CHARS + 500))
    assert len(block["text"]["text"]) <= MAX_SECTION_CHARS
    assert block["text"]["text"].endswith("...")


def test_mention_is_prepended_for_real_alerts(config, tmp_path):
    config.slack.mention = ["!subteam^S123"]
    poster = SlackPoster(config, dry_run=True, output_dir=tmp_path)
    poster.post([{"type": "divider"}], "alert", is_test=False, label="real")
    payload = json.loads((tmp_path / "real.json").read_text())
    assert payload["text"].startswith("<!subteam^S123>")


def test_mention_is_suppressed_for_test_alerts(config, tmp_path):
    config.slack.mention = ["!subteam^S123"]
    poster = SlackPoster(config, dry_run=True, output_dir=tmp_path)
    poster.post([{"type": "divider"}], "alert", is_test=True, label="test")
    payload = json.loads((tmp_path / "test.json").read_text())
    assert not payload["text"].startswith("<!subteam")


# ---------------------------------------------------------------------------
# Tagging the shift crew (issue #6)
# ---------------------------------------------------------------------------

CREW_IDS = {"summit-sup-sci": "S0LEAD", "os-day-shift": "S0DAY", "os-night-shift": "S0NIGHT"}


def fresh_trigger(record, **changes):
    """A trigger received a minute after its event, so it is not stale."""
    from astropy.time import TimeDelta

    record = {**record, **changes}
    event = decode_record(record).event_time
    return decode_record(record, received_at=event + TimeDelta(60.0, format="sec"))


def crew_text(trigger, config, ids=CREW_IDS) -> str:
    from chatterbox.slackbot.blocks import crew_blocks

    return block_text(crew_blocks(trigger, config, ids))


def test_a_real_alert_tags_all_three_groups(record, config):
    text = crew_text(fresh_trigger(record), config)
    for group_id in CREW_IDS.values():
        assert f"<!subteam^{group_id}>" in text


def test_the_lead_is_told_they_decide(record, config):
    """Issue #6: the support scientist has authority over the event."""
    text = crew_text(fresh_trigger(record), config)
    lead_line = next(line for line in text.splitlines() if "S0LEAD" in line)
    assert "summit support scientist" in lead_line
    assert "authority to make decisions" in lead_line
    assert "S0DAY" not in lead_line and "S0NIGHT" not in lead_line


def test_the_observing_specialists_are_told_to_wait_for_the_lead(record, config):
    text = crew_text(fresh_trigger(record), config)
    notify_line = next(line for line in text.splitlines() if "S0DAY" in line)
    assert "S0NIGHT" in notify_line, "both shifts addressed together"
    assert "be aware" in notify_line
    assert "as soon as possible" in notify_line
    assert "await further instructions" in notify_line
    assert "S0LEAD" not in notify_line


def test_every_group_is_tagged_once(record, config):
    config.slack.crew.notify = ["os-day-shift", "summit-sup-sci", "os-night-shift"]
    text = crew_text(fresh_trigger(record), config)
    assert text.count("S0LEAD") == 1, "the lead is not also told to wait for themselves"


def test_the_crew_comes_straight_after_the_header(config, record, rubin_scheduler):
    """It is the part of the post somebody has to act on."""
    from chatterbox.astro.almanac import night_events
    from chatterbox.astro.darkhours import dark_hours_map
    from chatterbox.plots.darkhours import region_hours_summary

    trigger = fresh_trigger(record)
    events = night_events(when=trigger.event_time)
    dark = dark_hours_map(events, nside=32, step_minutes=10.0)
    stats = region_hours_summary(dark, trigger.localization)
    blocks = build_trigger_blocks(trigger, events, dark, stats, None, config, CREW_IDS)
    # The header is two blocks -- title, then source and timing -- and the
    # crew is the very next one, ahead of the divider and everything else.
    assert blocks[0]["type"] == "header"
    assert trigger.source in blocks[1]["text"]["text"]
    assert "S0LEAD" in blocks[2]["text"]["text"]
    assert len(blocks) <= 50


def test_a_test_alert_tags_nobody_and_says_so(record, config):
    trigger = fresh_trigger(record, is_test=True)
    text = crew_text(trigger, config)
    assert "<!subteam" not in text
    assert "shift was not tagged" in text
    assert "@summit-sup-sci" in text, "plain text, which pings nobody"


def test_an_old_event_tags_nobody_and_says_so(record, config):
    """A replay of an archived alert must not page the shift."""
    stale = decode_record(record)  # received now; the fixture event is from the past
    assert stale.age_s > 24 * 3600
    text = crew_text(stale, config)
    assert "<!subteam" not in text
    assert "old, so the shift was not tagged" in text


def test_the_age_limit_can_be_switched_off(record, config):
    config.slack.crew.max_age_hours = 0
    assert "<!subteam^S0LEAD>" in crew_text(decode_record(record), config)


def test_an_event_of_unknown_age_is_still_tagged(record, config):
    """When the age cannot be told, failing to tag is the worse error."""
    trigger = fresh_trigger(record)
    trigger.event_time = None
    assert "<!subteam^S0LEAD>" in crew_text(trigger, config)


def test_a_group_without_an_id_is_written_as_text_and_flagged(record, config):
    """A handle typed as text pings nobody; the post must not look tagged."""
    text = crew_text(fresh_trigger(record), config, ids={"summit-sup-sci": "S0LEAD"})
    assert "<!subteam^S0LEAD>" in text
    assert "@os-day-shift" in text and "<!subteam^S0DAY>" not in text
    assert "Not notified" in text
    assert "@os-day-shift, @os-night-shift" in text.split("Not notified")[1]


def test_a_dry_run_does_not_claim_the_post_would_not_notify(record, config):
    """ids=None means 'not looked up', which is not 'not found'."""
    from chatterbox.slackbot.blocks import crew_blocks

    text = block_text(crew_blocks(fresh_trigger(record), config, None))
    assert "@summit-sup-sci" in text
    assert "Not notified" not in text


def test_the_crew_can_be_switched_off(record, config):
    from chatterbox.slackbot.blocks import crew_blocks

    config.slack.crew.enabled = False
    assert crew_blocks(fresh_trigger(record), config, CREW_IDS) == []


def test_ids_from_the_config_are_used_without_asking_slack(config, tmp_path):
    config.slack.crew.ids = dict(CREW_IDS)
    poster = SlackPoster(config, dry_run=True, output_dir=tmp_path)
    assert poster.crew_ids() == CREW_IDS


def test_handles_are_resolved_through_usergroups_list(config, tmp_path, monkeypatch):
    calls = []

    class FakeClient:
        def usergroups_list(self):
            calls.append(1)
            return {
                "usergroups": [
                    {"handle": "summit-sup-sci", "id": "S0LEAD"},
                    {"handle": "os-day-shift", "id": "S0DAY"},
                    {"handle": "os-night-shift", "id": "S0NIGHT"},
                    {"handle": "unrelated", "id": "S0OTHER"},
                ]
            }

    monkeypatch.setenv(config.slack.bot_token_env, "xoxb-test")
    poster = SlackPoster(config, output_dir=tmp_path)
    monkeypatch.setattr(poster, "_get_client", lambda: FakeClient())

    assert poster.crew_ids() == CREW_IDS
    assert poster.crew_ids() == CREW_IDS
    assert len(calls) == 1, "looked up once, not once per alert"


def test_a_failed_lookup_loses_the_tag_and_not_the_alert(config, tmp_path, monkeypatch, caplog):
    class Refusing:
        def usergroups_list(self):
            raise RuntimeError("missing_scope")

    monkeypatch.setenv(config.slack.bot_token_env, "xoxb-test")
    config.slack.crew.ids = {"summit-sup-sci": "S0LEAD"}
    poster = SlackPoster(config, output_dir=tmp_path)
    monkeypatch.setattr(poster, "_get_client", lambda: Refusing())

    with caplog.at_level("ERROR"):
        ids = poster.crew_ids()
    assert ids == {"summit-sup-sci": "S0LEAD"}, "what is known survives"
    assert "usergroups:read" in caplog.text


def test_an_offline_poster_cannot_look_anything_up(config, tmp_path):
    assert SlackPoster(config, dry_run=True, output_dir=tmp_path).crew_ids() is None


def test_the_tag_reaches_the_posted_payload(record, config, tmp_path):
    """Through the poster: what goes to Slack carries the mention."""
    trigger = fresh_trigger(record)
    from chatterbox.slackbot.blocks import crew_blocks

    blocks = crew_blocks(trigger, config, CREW_IDS)
    poster = SlackPoster(config, dry_run=True, output_dir=tmp_path)
    poster.post(blocks, "alert", is_test=False, label="crew")
    payload = json.loads((tmp_path / "crew.json").read_text())
    assert "<!subteam^S0LEAD>" in payload["blocks"][0]["text"]["text"]
