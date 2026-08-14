"""Tests for the EFD ingest source.

The EFD is the default monitoring path, and the thing that can go wrong is
subtle: InfluxDB has no arrays, so the reward map arrives as one column per
pixel and has to be reassembled. A mistake there does not raise -- it produces
a plausible-looking localization somewhere else on the sky.
"""

import numpy as np
import pandas as pd
import pytest
from conftest import make_record

from chatterbox.ingest.decode import TOO_ALERT_FIELDS, decode_record
from chatterbox.ingest.efd import EFD_DATABASE, EFD_TOPIC, EfdTooAlertSource, record_from_efd_row

WRITE_TIME = pd.Timestamp("2026-08-14T02:00:05.500Z")


def flatten(record, when=WRITE_TIME, shuffle=False, seed=0):
    """Turn a producer record into a row shaped like the EFD returns one.

    One column per reward-map pixel and per instrument slot, exactly as Influx
    stores it, with the time as the row's index.
    """
    flat = {
        "source": record["source"],
        "alert_type": record["alert_type"],
        "event_trigger_timestamp": record["event_trigger_timestamp"],
        "reward_map_nside": np.int64(record["reward_map_nside"]),
        "is_test": np.bool_(record["is_test"]),
        "is_update": np.bool_(record["is_update"]),
        "timestamp": np.int64(record["timestamp"]),
    }
    for n, value in enumerate(record["instrument"]):
        flat[f"instrument{n}"] = value
    for n, value in enumerate(record["reward_map"]):
        flat[f"reward_map{n}"] = np.bool_(value)

    names = list(flat)
    if shuffle:
        # Influx makes no promise about column order, and lexical order is
        # actively wrong: reward_map10 sorts before reward_map2.
        rng = np.random.default_rng(seed)
        rng.shuffle(names)
    row = pd.Series({name: flat[name] for name in names}, name=pd.Timestamp(when))
    return row


def frame_of(rows):
    """A DataFrame as ``select_time_series`` returns it, time-indexed."""
    return pd.DataFrame([r.to_dict() for r in rows], index=[r.name for r in rows])


class FakeEfdClient:
    """An EFD client that returns canned frames and records its calls."""

    def __init__(self, frames, error=None):
        self.frames = list(frames)
        self.error = error
        self.calls = []
        self.db_name = "efd"
        self.closed = False

    async def select_time_series(self, topic, fields, start, end, *args, **kwargs):
        self.calls.append({"topic": topic, "fields": fields, "start": start, "end": end})
        if self.error is not None:
            raise self.error
        if not self.frames:
            return pd.DataFrame()
        return self.frames.pop(0)

    async def close(self):
        self.closed = True


# -------------------------------------------------------------- rebuilding


def test_a_flattened_row_round_trips_to_a_record():
    """The reassembled record must be indistinguishable from the original."""
    record = make_record(source="S260814a", instruments=["H1", "L1"])
    rebuilt = record_from_efd_row(flatten(record))

    assert not [f for f in TOO_ALERT_FIELDS if f not in rebuilt], "all nine fields present"
    assert rebuilt["source"] == record["source"]
    assert rebuilt["alert_type"] == record["alert_type"]
    assert rebuilt["reward_map_nside"] == record["reward_map_nside"]
    assert rebuilt["timestamp"] == record["timestamp"]
    assert rebuilt["is_test"] is False and rebuilt["is_update"] is False
    assert np.array_equal(np.asarray(rebuilt["reward_map"], dtype=bool), record["reward_map"])


def test_the_rebuilt_record_decodes_to_the_same_trigger():
    """The real check: it survives the decoder the other transports use."""
    record = make_record()
    direct = decode_record(record)
    from_efd = decode_record(record_from_efd_row(flatten(record)))

    assert from_efd.source == direct.source
    # The EFD path drops the schema's padding, since an unwritten instrument
    # slot is a null column rather than an empty string. Both agree on what an
    # instrument actually is.
    assert from_efd.clean_instruments == direct.clean_instruments
    assert from_efd.geometry.area_deg2 == pytest.approx(direct.geometry.area_deg2)
    assert from_efd.geometry.centroid_dec_deg == pytest.approx(direct.geometry.centroid_dec_deg)
    assert np.array_equal(from_efd.reward_map, direct.reward_map)


def test_pixels_are_ordered_numerically_not_lexically():
    """reward_map10 must not land where reward_map2 belongs.

    Sorting the column names as text scrambles the map into a mask that is
    still the right size and still plausible, which is why this is asserted
    against a shuffled row rather than trusted.
    """
    record = make_record()
    rebuilt = record_from_efd_row(flatten(record, shuffle=True))
    assert np.array_equal(np.asarray(rebuilt["reward_map"], dtype=bool), record["reward_map"])


def test_the_nside_column_is_not_mistaken_for_a_pixel():
    record = make_record()
    rebuilt = record_from_efd_row(flatten(record))
    assert len(rebuilt["reward_map"]) == len(record["reward_map"])
    assert rebuilt["reward_map_nside"] == record["reward_map_nside"]


def test_null_pixels_are_false_not_true(caplog):
    """NaN casts to True, which would silently add sky to the region."""
    record = make_record()
    row = flatten(record)
    lit = [n for n, v in enumerate(record["reward_map"]) if v][:3]
    row = row.astype(object)
    for n in lit:
        row[f"reward_map{n}"] = np.nan

    with caplog.at_level("WARNING"):
        rebuilt = record_from_efd_row(row)
    assert "null" in caplog.text
    rebuilt_map = np.asarray(rebuilt["reward_map"], dtype=bool)
    assert not rebuilt_map[lit].any(), "a null pixel must not become part of the localization"
    assert rebuilt_map.sum() == int(np.asarray(record["reward_map"]).sum()) - len(lit)


def test_a_row_is_read_at_its_own_resolution():
    """A frame mixing resolutions pads the smaller rows with nulls.

    The row's own ``reward_map_nside`` is authoritative, not the width of the
    frame it arrived in -- otherwise the map comes out the wrong length and the
    alert is dropped by the decoder.
    """
    record = make_record()
    row = flatten(record).astype(object)
    npix = len(record["reward_map"])
    for extra in range(npix, npix + 50):
        row[f"reward_map{extra}"] = np.nan

    rebuilt = record_from_efd_row(row)
    assert len(rebuilt["reward_map"]) == npix
    assert np.array_equal(np.asarray(rebuilt["reward_map"], dtype=bool), record["reward_map"])
    assert decode_record(rebuilt).source == record["source"]


def test_an_all_null_map_is_an_error_not_an_empty_localization():
    record = make_record()
    row = flatten(record).astype(object)
    for n in range(len(record["reward_map"])):
        row[f"reward_map{n}"] = np.nan
    with pytest.raises(ValueError, match="is null"):
        record_from_efd_row(row)


def test_a_row_with_no_pixel_columns_is_an_error():
    """A query that returned no pixels is not an empty localization."""
    row = pd.Series({"source": "S1", "reward_map_nside": 32}, name=WRITE_TIME)
    with pytest.raises(ValueError, match="no reward_map"):
        record_from_efd_row(row)


def test_empty_instrument_slots_are_dropped():
    """The producer pads to three slots; the empties are not instruments."""
    record = make_record(instruments=["H1"])
    assert record["instrument"] == ["H1", "", ""]
    assert record_from_efd_row(flatten(record))["instrument"] == ["H1"]


def test_a_missing_timestamp_falls_back_to_the_write_time(caplog):
    """Better a slightly-late timestamp than a dropped alert."""
    record = make_record()
    row = flatten(record).drop(labels=["timestamp"])
    with caplog.at_level("WARNING"):
        rebuilt = record_from_efd_row(row)
    assert "no 'timestamp'" in caplog.text
    assert rebuilt["timestamp"] == int(WRITE_TIME.timestamp() * 1000)
    # And it is still decodable, which is the point of the fallback.
    assert decode_record(rebuilt).source == record["source"]


# ------------------------------------------------------------------ source


def test_the_source_queries_the_configured_topic_and_database():
    record = make_record()
    client = FakeEfdClient([frame_of([flatten(record)])])
    source = EfdTooAlertSource(poll_interval_s=0.0, once=True, client=client)

    delivered = list(source)
    assert len(delivered) == 1
    call = client.calls[0]
    assert call["topic"] == EFD_TOPIC
    assert call["fields"] == "*", "12,288 pixel columns are not worth naming individually"
    assert client.db_name == EFD_DATABASE, "the alerts are not in the default database"
    assert call["start"] <= call["end"]


def test_delivered_records_carry_their_provenance():
    record = make_record()
    client = FakeEfdClient([frame_of([flatten(record)])])
    source = EfdTooAlertSource(efd_name="usdf_efd", poll_interval_s=0.0, once=True, client=client)

    ((_, metadata),) = list(source)
    assert metadata["transport"] == "efd"
    assert metadata["topic"] == EFD_TOPIC
    assert "usdf_efd" in metadata["origin"]
    assert str(WRITE_TIME.year) in metadata["efd_time"]


def test_an_alert_is_delivered_once_even_though_the_window_overlaps():
    """Influx ranges are inclusive, so the last row comes back next poll."""
    record = make_record()
    row = flatten(record)
    client = FakeEfdClient([frame_of([row]), frame_of([row])])
    source = EfdTooAlertSource(poll_interval_s=0.0, once=True, client=client)

    assert len(list(source)) == 1
    assert list(source) == [], "the same row must not be re-posted"
    assert len(client.calls) == 2


def test_two_alerts_in_one_poll_come_out_oldest_first():
    early = flatten(make_record(source="S1"), when="2026-08-14T02:00:00Z")
    late = flatten(make_record(source="S2"), when="2026-08-14T03:00:00Z")
    client = FakeEfdClient([frame_of([late, early])])
    source = EfdTooAlertSource(poll_interval_s=0.0, once=True, client=client)

    assert [r["source"] for r, _ in source] == ["S1", "S2"]


def test_one_unreadable_row_does_not_take_down_the_rest():
    good = flatten(make_record(source="S2"), when="2026-08-14T03:00:00Z")
    broken = pd.Series(
        {"source": "S1", "reward_map_nside": 32},
        name=pd.Timestamp("2026-08-14T02:00:00Z"),
    )
    client = FakeEfdClient([frame_of([broken, good])])
    source = EfdTooAlertSource(poll_interval_s=0.0, once=True, client=client)

    assert [r["source"] for r, _ in source] == ["S2"]


def test_an_empty_poll_yields_nothing():
    client = FakeEfdClient([pd.DataFrame()])
    source = EfdTooAlertSource(poll_interval_s=0.0, once=True, client=client)
    assert list(source) == []


def test_a_missing_dependency_fails_immediately_not_after_five_retries(monkeypatch):
    """Retrying a permanent error only delays the diagnosis by a minute.

    The client is built before the retry loop, so a missing lsst-efd-client or
    a misconfigured instance name is reported at once. Patched rather than
    inferred from the environment, so this holds on an RSP host too -- where a
    real client would otherwise be built and the EFD actually queried.
    """
    calls = {"n": 0}

    def explode(self):
        calls["n"] += 1
        raise ImportError("EFD ingest requires lsst-efd-client (pip install lsst-efd-client)")

    monkeypatch.setattr(EfdTooAlertSource, "_build_client", explode)
    source = EfdTooAlertSource(poll_interval_s=30.0, max_consecutive_errors=5)
    with pytest.raises(ImportError, match="lsst-efd-client"):
        next(iter(source))
    assert calls["n"] == 1, "a permanent error must not be retried five times"


def test_a_persistent_outage_is_raised_so_the_service_can_report_it(caplog):
    """One failure is a blip; five in a row is something to say out loud."""
    client = FakeEfdClient([], error=RuntimeError("influx is down"))
    source = EfdTooAlertSource(poll_interval_s=0.0, max_consecutive_errors=3, client=client)

    with caplog.at_level("ERROR"):
        with pytest.raises(RuntimeError, match="3 consecutive EFD queries failed"):
            list(source)
    assert "influx is down" in caplog.text
    assert len(client.calls) == 3


def test_a_failure_that_recovers_is_not_fatal():
    """The blip case: the query that follows a failure is still delivered."""
    record = make_record()
    client = FakeEfdClient([frame_of([flatten(record)])])

    calls = {"n": 0}
    original = client.select_time_series

    async def fail_once(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TimeoutError("first query timed out")
        return await original(*args, **kwargs)

    client.select_time_series = fail_once
    source = EfdTooAlertSource(poll_interval_s=0.0, max_consecutive_errors=3, client=client)

    delivered = []
    for item in source:
        delivered.append(item)
        break
    assert len(delivered) == 1


def test_closing_releases_the_loop_but_not_a_borrowed_client():
    """A client handed in is the caller's to close, not the source's."""
    client = FakeEfdClient([pd.DataFrame()])
    source = EfdTooAlertSource(poll_interval_s=0.0, once=True, client=client)
    list(source)
    source.close()
    assert not client.closed
    assert source._loop is None
    # Closing twice must not raise: the service closes in a finally block.
    source.close()


def test_lookback_reaches_before_startup():
    """Off by default, because chatterbox does not remember what it posted."""
    from chatterbox.config import Config

    assert Config().ingest.efd_lookback_s == 0.0

    client = FakeEfdClient([pd.DataFrame()])
    source = EfdTooAlertSource(poll_interval_s=0.0, once=True, lookback_s=600.0, client=client)
    list(source)
    start, end = client.calls[0]["start"], client.calls[0]["end"]
    assert (end - start).sec == pytest.approx(600.0, abs=5.0)
