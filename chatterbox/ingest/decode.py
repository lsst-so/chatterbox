"""Decode a ``too_alert`` record into a `~chatterbox.models.Trigger`.

The record is the Avro schema defined by
``rubin-ToO-producer/output_schema.json``. It has exactly nine fields and no
nullable types, but records reach chatterbox through several transports (Avro
over Kafka, JSON from the file sender, a hand-written replay fixture), so
decoding is defensive about types while still insisting on the required fields.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from astropy.time import Time

from ..astro.skymap import geometry_from_mask, localization_from_reward_map, nest_to_ring
from ..models import Trigger

__all__ = ["TOO_ALERT_FIELDS", "decode_record", "load_record_file", "REWARD_MAP_CREDIBLE_LEVEL"]

logger = logging.getLogger(__name__)

#: Fields required by the producer's output schema.
TOO_ALERT_FIELDS = (
    "source",
    "instrument",
    "alert_type",
    "event_trigger_timestamp",
    "reward_map",
    "reward_map_nside",
    "is_test",
    "is_update",
    "timestamp",
)

#: Credible level the published reward map represents.
#:
#: ``forward_alerts.py`` calls ``make_flat_binary_map(0.7, ...)`` for every
#: alert class, while *selecting* on the 90% area for GW. The value is volatile
#: upstream -- it has been 0.9, 0.8 and 0.7 -- so it is named here rather than
#: being implied by a hardcoded string in message text.
REWARD_MAP_CREDIBLE_LEVEL = 0.7


def _parse_event_time(raw: Any) -> Time | None:
    """Parse the event trigger timestamp, tolerating upstream format drift.

    The producer passes this field through verbatim from the incoming alert
    with no parsing, so its exact format depends on which messenger sent it.
    A failure here must not drop the alert.
    """
    if raw in (None, ""):
        return None
    text = str(raw).strip()
    try:
        # Time understands ISO-8601 with or without a trailing Z, but not both
        # a Z and a space separator, so normalize first.
        normalized = text.replace("Z", "").replace(" ", "T")
        return Time(normalized, format="isot", scale="utc")
    except Exception:
        pass
    try:
        return Time(text)
    except Exception as exc:
        logger.warning("Could not parse event_trigger_timestamp %r: %s", raw, exc)
        return None


def _as_bool_map(raw: Any) -> np.ndarray:
    """Coerce a reward map to a boolean array.

    Avro gives a list of booleans; JSON round-trips may give 0/1 integers or a
    list of lists from a DataFrame export.
    """
    arr = np.asarray(raw)
    if arr.dtype == object:
        arr = np.asarray(arr.tolist())
    return arr.astype(bool).ravel()


def decode_record(record: dict[str, Any], received_at: Time | None = None) -> Trigger:
    """Turn a raw ToO record into a `~chatterbox.models.Trigger`.

    Parameters
    ----------
    record : `dict`
        Decoded ``too_alert`` record.
    received_at : `~astropy.time.Time`, optional
        When chatterbox received it. Defaults to now.

    Returns
    -------
    trigger : `Trigger`

    Raises
    ------
    KeyError
        If a required schema field is absent.
    ValueError
        If the reward map length disagrees with ``reward_map_nside``.
    """
    missing = [f for f in TOO_ALERT_FIELDS if f not in record]
    if missing:
        raise KeyError(f"ToO record is missing required field(s) {missing}")

    source = record["source"]
    if source is None:
        # The producer's base identifier path can yield None, which would not
        # even serialize against its own schema; fail loudly rather than post
        # an unidentifiable alert.
        raise ValueError("ToO record has a null 'source'; cannot identify the event")

    nside = int(record["reward_map_nside"])
    reward_map = _as_bool_map(record["reward_map"])
    expected = 12 * nside * nside
    if reward_map.size != expected:
        raise ValueError(
            f"reward_map has {reward_map.size} pixels but reward_map_nside={nside} implies {expected}"
        )

    instruments = list(record["instrument"] or [])

    localization = localization_from_reward_map(reward_map, nside, credible_level=REWARD_MAP_CREDIBLE_LEVEL)
    geometry = geometry_from_mask(nest_to_ring(reward_map, nside), nside)

    trigger = Trigger(
        source=str(source),
        alert_type=str(record["alert_type"]),
        instruments=[str(i) for i in instruments],
        event_trigger_timestamp=str(record["event_trigger_timestamp"]),
        reward_map=reward_map,
        reward_map_nside=nside,
        is_test=bool(record["is_test"]),
        is_update=bool(record["is_update"]),
        producer_timestamp_ms=int(record["timestamp"]),
        localization=localization,
        geometry=geometry,
        event_time=_parse_event_time(record["event_trigger_timestamp"]),
        received_at=Time.now() if received_at is None else received_at,
        raw={k: v for k, v in record.items() if k != "reward_map"},
    )
    logger.info(
        "Decoded %s (%s): %.0f deg^2, dec %.1f to %.1f, test=%s update=%s",
        trigger.source,
        trigger.alert_type,
        geometry.area_deg2,
        geometry.dec_min_deg,
        geometry.dec_max_deg,
        trigger.is_test,
        trigger.is_update,
    )
    return trigger


def load_record_file(path: str | Path) -> dict[str, Any]:
    """Read a single ToO record from disk.

    Supports the JSON written by the producer's ``FileSender`` and Avro files
    written by its Kafka path (via ``fastavro``, when installed).

    Parameters
    ----------
    path : `str` or `pathlib.Path`
        File to read.

    Returns
    -------
    record : `dict`
    """
    path = Path(path).expanduser()
    if path.suffix in (".json", ".txt"):
        with open(path) as f:
            data = json.load(f)
        # Some exports wrap the record in a list or a {"records": [...]}
        # envelope.
        if isinstance(data, list):
            if not data:
                raise ValueError(f"{path} contains an empty list")
            return data[0]
        if isinstance(data, dict) and "records" in data and isinstance(data["records"], list):
            return data["records"][0].get("value", data["records"][0])
        return data

    if path.suffix == ".avro":
        try:
            import fastavro
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("Reading .avro records requires fastavro (pip install fastavro)") from exc
        with open(path, "rb") as f:
            records = list(fastavro.reader(f))
        if not records:
            raise ValueError(f"{path} contains no Avro records")
        return records[0]

    raise ValueError(f"Unsupported record file type: {path.suffix} ({path})")
