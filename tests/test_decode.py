"""Tests for decoding the producer's ``too_alert`` record."""

import json

import healpy as hp
import numpy as np
import pytest
from conftest import NSIDE, disc_reward_map, make_record

from chatterbox.ingest.decode import TOO_ALERT_FIELDS, decode_record, load_record_file


def test_decode_basic_fields(record):
    trigger = decode_record(record)
    assert trigger.source == "S260814a"
    assert trigger.alert_type == "GW_case_B"
    assert trigger.reward_map_nside == NSIDE
    assert trigger.reward_map.dtype == bool
    assert trigger.reward_map.size == hp.nside2npix(NSIDE)
    assert trigger.is_test is False
    assert trigger.is_update is False


def test_instrument_padding_is_stripped(record):
    """The schema pads to three entries; padding must not be displayed."""
    assert record["instrument"] == ["H1", "L1", ""]
    trigger = decode_record(record)
    assert trigger.clean_instruments == ["H1", "L1"]


@pytest.mark.parametrize("field", TOO_ALERT_FIELDS)
def test_missing_field_raises(record, field):
    del record[field]
    with pytest.raises(KeyError, match=field):
        decode_record(record)


def test_null_source_raises(record):
    """A null source cannot identify an event, and would not even serialize."""
    record["source"] = None
    with pytest.raises(ValueError, match="source"):
        decode_record(record)


def test_reward_map_length_must_match_nside(record):
    record["reward_map"] = record["reward_map"][:-1]
    with pytest.raises(ValueError, match="reward_map"):
        decode_record(record)


def test_integer_reward_map_is_accepted(record):
    """JSON round-trips sometimes turn booleans into 0/1."""
    expected = decode_record(record).reward_map
    record["reward_map"] = [int(x) for x in record["reward_map"]]
    assert np.array_equal(decode_record(record).reward_map, expected)


@pytest.mark.parametrize(
    "raw",
    [
        "2026-08-14T02:00:00.000Z",
        "2026-08-14T02:00:00Z",
        "2026-08-14T02:00:00.000",
        "2026-08-14 02:00:00",
    ],
)
def test_event_time_formats(record, raw):
    """The producer passes this field through unparsed, so formats vary."""
    record["event_trigger_timestamp"] = raw
    trigger = decode_record(record)
    assert trigger.event_time is not None
    assert trigger.event_time.strftime("%Y-%m-%d %H:%M") == "2026-08-14 02:00"


def test_unparseable_event_time_does_not_drop_the_alert(record):
    record["event_trigger_timestamp"] = "sometime last Tuesday"
    trigger = decode_record(record)
    assert trigger.event_time is None
    assert trigger.latency_s is None
    assert trigger.age_s is None


def test_latency_uses_producer_timestamp(record):
    record["event_trigger_timestamp"] = "2026-08-14T02:00:00.000Z"
    # 90 seconds after the event.
    record["timestamp"] = int((1786.0 * 0 + 1786000000) * 1000)
    trigger = decode_record(record)
    expected = record["timestamp"] / 1e3 - trigger.event_time.unix
    assert trigger.latency_s == pytest.approx(expected)


def test_geometry_matches_the_reward_map(record):
    trigger = decode_record(record)
    n_set = int(np.count_nonzero(trigger.reward_map))
    pixel_area = hp.nside2pixarea(NSIDE, degrees=True)
    assert trigger.geometry.n_pixels == n_set
    assert trigger.geometry.area_deg2 == pytest.approx(n_set * pixel_area)
    # The centroid of a disc at (60, -35) should be close to its centre.
    assert trigger.geometry.centroid_ra_deg == pytest.approx(60.0, abs=1.0)
    assert trigger.geometry.centroid_dec_deg == pytest.approx(-35.0, abs=1.0)


def test_declination_range_uses_pixel_corners(record):
    """Corner-based limits must be at least as wide as centre-based ones."""
    trigger = decode_record(record)
    ring = np.flatnonzero(np.take(trigger.reward_map, hp.ring2nest(NSIDE, np.arange(hp.nside2npix(NSIDE)))))
    _, centre_dec = hp.pix2ang(NSIDE, ring, lonlat=True)
    assert trigger.geometry.dec_min_deg <= centre_dec.min()
    assert trigger.geometry.dec_max_deg >= centre_dec.max()


def test_localization_from_reward_map_is_area_not_probability(record):
    """The record has no probability density; this must not imply one."""
    trigger = decode_record(record)
    assert trigger.localization.is_probability is False
    assert trigger.localization.quantity_name == "area"
    assert trigger.localization.prob_map.sum() == pytest.approx(1.0)
    # Uniform weight inside the region.
    nonzero = trigger.localization.prob_map[trigger.localization.prob_map > 0]
    assert nonzero.min() == pytest.approx(nonzero.max())


def test_raw_excludes_the_reward_map(record):
    """The reward map is 12,288 entries; in `raw` it would bloat logs."""
    trigger = decode_record(record)
    assert "reward_map" not in trigger.raw
    assert trigger.raw["source"] == "S260814a"


def test_committed_fixtures_decode(data_dir):
    for name in ("gw_case_b", "sn_galactic", "neutrino", "gw_case_d_test"):
        trigger = decode_record(load_record_file(data_dir / f"{name}.json"))
        assert trigger.geometry.area_deg2 > 0
        assert trigger.localization.prob_map.sum() == pytest.approx(1.0)


def test_test_flag_is_preserved(data_dir):
    trigger = decode_record(load_record_file(data_dir / "gw_case_d_test.json"))
    assert trigger.is_test is True
    # MS/TS prefixes mark mock and test superevents.
    assert trigger.source.startswith("MS")


def test_load_record_file_unwraps_envelopes(tmp_path):
    """Confluent REST exports wrap records; the file sender does not."""
    record = make_record()
    plain = tmp_path / "plain.json"
    plain.write_text(json.dumps(record))
    listed = tmp_path / "listed.json"
    listed.write_text(json.dumps([record]))
    wrapped = tmp_path / "wrapped.json"
    wrapped.write_text(json.dumps({"records": [{"value": record}]}))

    for path in (plain, listed, wrapped):
        assert load_record_file(path)["source"] == record["source"]


def test_load_record_file_rejects_unknown_suffix(tmp_path):
    path = tmp_path / "record.parquet"
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="Unsupported"):
        load_record_file(path)


def test_circular_localization_decodes(sn_record):
    """SuperK alerts carry a circular region built by cone search."""
    trigger = decode_record(sn_record)
    assert trigger.alert_type == "SN_Galactic"
    assert trigger.clean_instruments == ["Super-K"]
    # A 3.5 degree circle is ~38 deg^2, quantized up by nside-32 pixels.
    assert 30 < trigger.geometry.area_deg2 < 130
    assert trigger.geometry.gal_b_abs_min_deg < 5.0


def test_nside_other_than_32_is_accepted():
    """nside is read from the record, not assumed, in case the producer
    changes.
    """
    reward = disc_reward_map(120.0, -40.0, 5.0, nside=64)
    trigger = decode_record(make_record(reward_map=reward, nside=64))
    assert trigger.reward_map_nside == 64
    assert trigger.localization.nside == 64
