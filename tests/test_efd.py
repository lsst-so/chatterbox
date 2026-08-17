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

    def __init__(self, frames, error=None, efd_name="summit_efd", fields=None):
        self.frames = list(frames)
        self.error = error
        self.calls = []
        self.field_calls = []
        self.db_name = "efd"
        # A real client knows which instance it opened, and that is what the
        # source reports.
        self.efd_name = efd_name
        self.closed = False
        # ``get_fields`` reports the topic's field names, which the source then
        # lists in the query. Default to the columns the canned rows carry --
        # the fields a real topic would report -- falling back to a minimal set
        # so a frame-less poll still has fields to ask for, exactly as a topic
        # that already holds data always does.
        if fields is None:
            fields = []
            for frame in self.frames:
                if not frame.empty:
                    fields = [str(c) for c in frame.columns]
                    break
            if not fields:
                fields = ["source", "alert_type", "timestamp", "reward_map_nside", "reward_map0"]
        self.fields = list(fields)

    async def get_fields(self, topic):
        self.field_calls.append(topic)
        return list(self.fields)

    async def select_time_series(self, topic, fields, start, end, *args, **kwargs):
        self.calls.append({"topic": topic, "fields": fields, "start": start, "end": end})
        if self.error is not None:
            raise self.error
        if not self.frames:
            return pd.DataFrame()
        return self.frames.pop(0)

    async def close(self):
        self.closed = True


class TimeAwareFakeEfdClient:
    """A fake that honours the query window and models the relay's ingest lag.

    Unlike `FakeEfdClient`, which returns canned frames regardless of the window,
    this one filters by ``[start, end]`` on each row's own time index, and hides
    a row until a given poll number -- the relay writing it later than its
    timestamp. That is what a plain marching window trips over and the revisit
    re-read is meant to survive.
    """

    def __init__(self, rows, fields=None):
        # rows: list of (row_series, visible_from_call), 0-based poll index.
        self.rows = list(rows)
        self.calls = []
        self.field_calls = []
        self.db_name = "efd"
        self.efd_name = "base_efd"
        self.closed = False
        if fields is None:
            cols: set[str] = set()
            for row, _ in self.rows:
                cols.update(str(c) for c in row.index)
            fields = sorted(cols) or ["source"]
        self.fields = list(fields)

    async def get_fields(self, topic):
        self.field_calls.append(topic)
        return list(self.fields)

    async def select_time_series(self, topic, fields, start, end, *args, **kwargs):
        from astropy.time import Time

        poll = len(self.calls)  # this call's 0-based index
        self.calls.append({"topic": topic, "fields": fields, "start": start, "end": end})
        visible = []
        for row, visible_from in self.rows:
            when = Time(row.name.to_pydatetime())
            if visible_from <= poll and start <= when <= end:
                visible.append(row)
        if not visible:
            return pd.DataFrame()
        return frame_of(visible)

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
    # The fields are listed explicitly, from get_fields, exactly as the summit
    # notebooks issue the query: a quoted "*" is not an InfluxQL wildcard, so
    # the old "*" matched nothing and every poll came back empty.
    assert client.field_calls == [EFD_TOPIC], "the field list is fetched from the topic"
    assert isinstance(call["fields"], list), "fields are named, not queried as '*'"
    assert "source" in call["fields"] and "reward_map0" in call["fields"]
    assert client.db_name == EFD_DATABASE, "the alerts are not in the default database"
    assert call["start"] <= call["end"]


def test_the_fields_are_named_from_get_fields_not_a_wildcard():
    """The query lists the topic's fields, exactly as the notebooks do it.

    A quoted "*" is not an InfluxQL wildcard -- the client quotes every field
    it is handed -- so ``select_time_series(topic, "*", ...)`` asked for a field
    literally named "*", matched nothing, and no alert was ever delivered. The
    fix is to pass the field list ``get_fields`` reports.
    """
    record = make_record(instruments=["H1", "L1"])
    row = flatten(record)
    client = FakeEfdClient([frame_of([row])])
    source = EfdTooAlertSource(poll_interval_s=0.0, once=True, client=client)

    list(source)
    (call,) = client.calls
    assert call["fields"] != "*"
    # Every field the row carries is asked for, pixels included.
    assert set(row.index) <= set(call["fields"])


def test_the_field_list_is_fetched_once_and_reused_across_polls():
    """get_fields is not re-run every poll: the topic's schema does not move."""
    first = flatten(make_record(source="S1"), when="2026-08-14T02:00:00Z")
    second = flatten(make_record(source="S2"), when="2026-08-14T04:00:00Z")
    client = FakeEfdClient([frame_of([first]), frame_of([second])])
    source = EfdTooAlertSource(poll_interval_s=0.0, once=True, client=client)

    assert len(list(source)) == 1
    assert len(list(source)) == 1, "the second poll delivers the second alert"
    assert len(client.calls) == 2, "two polls happened"
    assert client.field_calls == [EFD_TOPIC], "but the field list was fetched only once"


def test_a_topic_with_no_fields_yet_is_skipped_then_recovers(caplog):
    """An empty topic must not latch the source to 'nothing'.

    ``get_fields`` returns nothing until an alert has been written. That empty
    result is not cached, so once one appears the source picks it up rather than
    forever asking for no fields -- and while it is empty, no fieldless query is
    ever sent.
    """
    record = make_record()
    real_fields = [str(c) for c in flatten(record).index]
    client = FakeEfdClient([pd.DataFrame(), frame_of([flatten(record)])], fields=real_fields)

    calls = {"n": 0}

    async def fields_later(topic):
        # Empty until an alert has been written, then the real schema.
        calls["n"] += 1
        return [] if calls["n"] == 1 else list(real_fields)

    client.get_fields = fields_later
    source = EfdTooAlertSource(poll_interval_s=0.0, client=client)

    with caplog.at_level("WARNING"):
        delivered = next(iter(source))

    assert "no fields yet" in caplog.text
    assert delivered[0]["source"] == record["source"]
    assert calls["n"] >= 2, "the empty result is retried, not cached"


def test_delivered_records_carry_their_provenance():
    record = make_record()
    client = FakeEfdClient([frame_of([flatten(record)])], efd_name="usdf_efd")
    source = EfdTooAlertSource(efd_name="usdf_efd", poll_interval_s=0.0, once=True, client=client)

    ((_, metadata),) = list(source)
    assert metadata["transport"] == "efd"
    assert metadata["topic"] == EFD_TOPIC
    assert metadata["site"] == "usdf_efd", "which EFD an alert came from is provenance"
    assert metadata["database"] == EFD_DATABASE
    assert "usdf_efd" in metadata["origin"]
    assert str(WRITE_TIME.year) in metadata["efd_time"]


def test_the_site_is_named_before_anything_is_polled(caplog):
    """An operator reading the log has to see which EFD is being watched."""
    client = FakeEfdClient([pd.DataFrame()], efd_name="usdf_efd")
    source = EfdTooAlertSource(efd_name="usdf_efd", poll_interval_s=0.0, once=True, client=client)
    with caplog.at_level("INFO"):
        list(source)
    assert "Watching the usdf_efd EFD" in caplog.text
    assert EFD_DATABASE in caplog.text


def test_the_site_is_taken_from_the_client_not_the_config():
    """A client that names itself is what is actually being read.

    ``efd_name`` is required to *build* a client, but a client handed in for
    testing carries its own instance name, and that -- not the configured
    string -- is what the source then reports.
    """
    client = FakeEfdClient([pd.DataFrame()])
    client.efd_name = "summit_efd"
    source = EfdTooAlertSource(efd_name="", poll_interval_s=0.0, once=True, client=client)

    assert source.site == "unset", "nothing is known before connecting"
    list(source)
    assert source.site == "summit_efd"
    assert source.describe() == f"{EFD_TOPIC} on summit_efd (database {EFD_DATABASE})"


def test_an_empty_efd_name_is_an_error_without_a_client():
    """The host-default fallback is gone: a named instance is required.

    With no client handed in, the source builds its own with
    ``EfdClient(efd_name, db_name=...)``; an empty name has nothing to open, so
    it is rejected before anything is polled rather than silently defaulted.
    """
    source = EfdTooAlertSource(efd_name="", poll_interval_s=0.0, once=True)
    with pytest.raises(ValueError, match="must name an EFD instance"):
        next(iter(source))


def test_an_anonymous_client_reports_what_was_configured():
    """A client that does not name itself must not make the site a lie."""
    client = FakeEfdClient([pd.DataFrame()])
    del client.efd_name  # nothing to resolve from
    source = EfdTooAlertSource(efd_name="base_efd", poll_interval_s=0.0, once=True, client=client)
    list(source)
    assert source.site == "base_efd"


def test_a_site_that_disagrees_with_the_config_is_flagged(caplog):
    """Reading the wrong EFD quietly is worse than reading it loudly."""
    client = FakeEfdClient([pd.DataFrame()], efd_name="summit_efd")
    source = EfdTooAlertSource(efd_name="usdf_efd", poll_interval_s=0.0, once=True, client=client)
    with caplog.at_level("WARNING"):
        list(source)
    assert "ingest.efd_name is 'usdf_efd'" in caplog.text
    assert source.site == "summit_efd", "the client is what is actually being read"


def test_the_description_names_site_database_and_topic():
    source = EfdTooAlertSource(efd_name="idf_efd", topic="lsst.scimma.too_alert_test")
    assert source.describe() == "lsst.scimma.too_alert_test on idf_efd (database lsst.scimma)"


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
    # revisit_s isolated to 0 here so the span measures the lookback alone.
    source = EfdTooAlertSource(
        poll_interval_s=0.0, once=True, lookback_s=600.0, revisit_s=0.0, client=client
    )
    list(source)
    start, end = client.calls[0]["start"], client.calls[0]["end"]
    assert (end - start).sec == pytest.approx(600.0, abs=5.0)


def test_every_poll_re_reads_a_trailing_window_for_late_alerts():
    """The window reaches back lookback + revisit, not just to the watermark.

    An alert's EFD time index is the producer's send time, but the relay makes
    it queryable later. A plain forward-marching window sweeps past the
    timestamp before the row appears; re-reading a trailing ``revisit_s`` behind
    the watermark is what keeps a late alert from being lost.
    """
    client = FakeEfdClient([pd.DataFrame()])
    source = EfdTooAlertSource(
        poll_interval_s=0.0, once=True, lookback_s=30.0, revisit_s=900.0, client=client
    )
    list(source)
    start, end = client.calls[0]["start"], client.calls[0]["end"]
    assert (end - start).sec == pytest.approx(930.0, abs=5.0), "lookback + revisit"


def test_a_late_arriving_row_is_caught_and_delivered_once():
    """The fix, end to end: a row visible only on a later poll is still posted.

    ``TimeAwareFakeEfdClient`` honours the query window and models the relay lag
    by hiding the row until the second poll and stamping it a few seconds in the
    past. With a plain ``[watermark, now]`` window the marching watermark would
    have moved past that timestamp and the row would be lost; the trailing
    revisit re-read catches it, and the dedup posts it exactly once.
    """
    when = pd.Timestamp.now(tz="UTC") - pd.Timedelta(seconds=5)
    row = flatten(make_record(source="LATE"), when=when)
    client = TimeAwareFakeEfdClient([(row, 1)])  # visible from the 2nd poll on
    source = EfdTooAlertSource(
        poll_interval_s=0.0, lookback_s=0.0, revisit_s=300.0, client=client
    )

    delivered = []
    for item in source:
        delivered.append(item)
        break
    assert len(delivered) == 1
    assert delivered[0][0]["source"] == "LATE"
    assert len(client.calls) >= 2, "it took a re-read on a later poll to see it"
