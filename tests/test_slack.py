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
    import pandas as pd

    from chatterbox.astro.almanac import night_events
    from chatterbox.astro.darkhours import dark_hours_map
    from chatterbox.astro.templates import build_template_maps
    from chatterbox.plots.darkhours import region_hours_summary

    trigger = decode_record(record)
    events = night_events(when=trigger.event_time)
    dark = dark_hours_map(events, nside=32, step_minutes=10.0)
    stats = region_hours_summary(dark, trigger.localization)
    coverage = build_template_maps(
        pd.DataFrame([{"s_ra": 60.0, "s_dec": -35.0, "band": "i"}]), nside=64, source="test"
    )
    blocks = build_trigger_blocks(trigger, events, dark, stats, coverage, config)
    text = block_text(blocks)
    assert "Existing template coverage" in text
    assert "Cache built" in text


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
        assert f"{band} *" in text
    assert "91.0%" in text
    assert "probability" in text


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
