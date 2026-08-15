"""Watch the EFD for new ``too_alert`` records.

The producer's alerts also land in the Engineering Facilities Database, in the
``lsst.scimma`` InfluxDB database as the ``too_alert`` measurement. That makes
the EFD a monitoring path that needs no broker subscription and no credentials
beyond the ones a summit or USDF notebook already has, which is how the summit
analysis notebooks read these records today.

InfluxDB has no array types, so a record arrives *flattened*: the boolean
reward map becomes one column per pixel (``reward_map0`` ... ``reward_mapN``,
12,288 of them at nside 32) and the three-element instrument array becomes
``instrument0``, ``instrument1``, ``instrument2``. `record_from_efd_row` puts
that back together into the nine-field record the rest of chatterbox decodes,
so nothing downstream has to know which transport an alert arrived on.

Notes
-----
Two details of the reconstruction matter and are easy to get wrong:

- The pixel columns must be ordered *numerically*, not lexically:
  ``reward_map10`` sorts before ``reward_map2`` as text, which would scramble
  the localization into an unrecognizable but perfectly plausible-looking mask.
- A field Influx has no value for comes back as ``NaN``, and ``NaN`` casts to
  ``True``. Left alone that silently *adds* sky to the credible region, so
  nulls are counted and forced to False rather than cast.
"""

import asyncio
import logging
import re
import time
from collections import deque
from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd
from astropy.time import Time, TimeDelta

from .source import TooAlertSource

__all__ = [
    "EFD_DATABASE",
    "EFD_TOPIC",
    "EfdTooAlertSource",
    "record_from_efd_row",
]

logger = logging.getLogger(__name__)

#: InfluxDB database the SCiMMA alerts are written to. This is *not* the
#: default database an EFD client connects to, which holds the SAL topics.
EFD_DATABASE = "lsst.scimma"

#: Measurement carrying the producer's ``too_alert`` records.
EFD_TOPIC = "lsst.scimma.too_alert"

#: One pixel of the flattened reward map. Anchored so ``reward_map_nside`` is
#: not mistaken for pixel data.
_PIXEL_RE = re.compile(r"^reward_map(\d+)$")

#: One element of the flattened instrument array.
_INSTRUMENT_RE = re.compile(r"^instrument(\d+)$")

#: Fields the record carries as plain scalars.
_SCALAR_FIELDS = (
    "source",
    "alert_type",
    "event_trigger_timestamp",
    "reward_map_nside",
    "is_test",
    "is_update",
    "timestamp",
)


def _plain(value: Any) -> Any:
    """Convert a numpy or pandas scalar to something JSON can hold."""
    if value is None or (np.isscalar(value) and pd.isna(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def record_from_efd_row(row: pd.Series, when: Any = None) -> dict[str, Any]:
    """Rebuild a ``too_alert`` record from one flattened EFD row.

    Parameters
    ----------
    row : `pandas.Series`
        One row of the DataFrame returned by
        ``EfdClient.select_time_series("lsst.scimma.too_alert", ...)``.
    when : optional
        The row's own timestamp, used only to fill in ``timestamp`` when the
        column is absent. Defaults to `row.name`, which is the Influx time
        index.

    Returns
    -------
    record : `dict`
        The nine fields of the producer's output schema, ready for
        `chatterbox.ingest.decode.decode_record`.

    Raises
    ------
    ValueError
        When the row carries no reward-map pixels at all, which means the query
        did not return the pixel columns rather than that the map was empty.
    """
    pixels: list[tuple[int, str]] = []
    instruments: list[tuple[int, str]] = []
    for name in row.index:
        text = str(name)
        pixel = _PIXEL_RE.match(text)
        if pixel is not None:
            pixels.append((int(pixel.group(1)), text))
            continue
        instrument = _INSTRUMENT_RE.match(text)
        if instrument is not None:
            instruments.append((int(instrument.group(1)), text))

    if not pixels:
        candidates = [str(n) for n in row.index if "reward_map" in str(n)]
        raise ValueError(
            "EFD row carries no reward_map<pixel> columns, so the localization "
            f"cannot be rebuilt; reward_map-like columns present: {candidates[:5]}"
        )

    # Numerically, not lexically: reward_map10 must not precede reward_map2.
    ordered = [name for _, name in sorted(pixels)]

    # A frame holding rows of two different resolutions has the union of both
    # pixel columns, so the lower-resolution rows are padded with nulls beyond
    # their own length. Trust the row's own nside over the frame's width.
    nside = _plain(row["reward_map_nside"]) if "reward_map_nside" in row.index else None
    if nside is not None:
        expected = 12 * int(nside) * int(nside)
        if len(ordered) > expected:
            logger.info(
                "EFD row has %d pixel columns but nside=%d implies %d; using the first %d",
                len(ordered),
                int(nside),
                expected,
                expected,
            )
            ordered = ordered[:expected]

    values = row[ordered].to_numpy()
    missing = pd.isna(values)
    if missing.all():
        # Not an empty localization -- the producer never publishes one. This
        # row simply has no map in it, which happens when a frame carries rows
        # from more than one measurement shape.
        raise ValueError(
            f"every one of the {values.size} reward_map pixels in this EFD row is null; "
            "there is no localization to rebuild"
        )
    if missing.any():
        # Forced to False rather than cast: NaN would become True and quietly
        # grow the credible region.
        logger.warning(
            "%d of %d reward_map pixels are null in the EFD row; treating them as False",
            int(missing.sum()),
            values.size,
        )
        values = np.where(missing, False, values)
    reward_map = [int(bool(v)) for v in values]

    record: dict[str, Any] = {
        "reward_map": reward_map,
        "instrument": [
            str(_plain(row[name])) for _, name in sorted(instruments) if _plain(row[name]) not in (None, "")
        ],
    }
    for field in _SCALAR_FIELDS:
        if field in row.index:
            record[field] = _plain(row[field])

    # ``timestamp`` is the producer's own send time in milliseconds. The record
    # is unusable without it, and the Influx write time is the closest thing
    # available, so fall back to that rather than dropping the alert.
    if record.get("timestamp") is None:
        stamp = pd.Timestamp(row.name if when is None else when)
        record["timestamp"] = int(stamp.timestamp() * 1000.0)
        logger.warning("EFD row has no 'timestamp'; using its write time instead")

    for field in ("is_test", "is_update"):
        record[field] = bool(record.get(field) or False)
    if record.get("reward_map_nside") is not None:
        record["reward_map_nside"] = int(record["reward_map_nside"])

    return record


class EfdTooAlertSource(TooAlertSource):
    """Poll the EFD for ToO records written since the last poll.

    Parameters
    ----------
    efd_name : `str`
        EFD instance to connect to: ``summit_efd``, ``usdf_efd``, ``idf_efd``
        or ``base_efd``. Empty uses the client's own default for the host.
    database : `str`
        InfluxDB database holding the alerts. The default, `EFD_DATABASE`, is
        not where the SAL topics live, so it has to be set explicitly on the
        client.
    topic : `str`
        Measurement to query.
    poll_interval_s : `float`
        Seconds between queries.
    lookback_s : `float`
        How far before startup to look on the first query. ``0`` means "only
        alerts that arrive from now on", which is the safe default: chatterbox
        keeps no record of what it has already posted, so a longer lookback
        makes a restart re-post recent alerts.
    once : `bool`
        Query once and return, for tests and cron-style deployments.
    max_consecutive_errors : `int`
        Give up after this many failed queries in a row. A single failure is a
        blip and is retried; a persistent one is an outage that the service
        should report rather than sit quietly through.
    client : optional
        A ready EFD client, mainly for testing. When given, no client is built
        and `efd_name` is unused.

    Notes
    -----
    ``select_time_series`` is a coroutine, and the rest of chatterbox is
    synchronous, so the source owns a private event loop for the client's
    lifetime rather than starting one per query.
    """

    def __init__(
        self,
        efd_name: str = "summit_efd",
        database: str = EFD_DATABASE,
        topic: str = EFD_TOPIC,
        poll_interval_s: float = 10.0,
        lookback_s: float = 0.0,
        once: bool = False,
        max_consecutive_errors: int = 5,
        client: Any = None,
    ) -> None:
        self.efd_name = efd_name
        self.database = database
        self.topic = topic
        self.poll_interval_s = poll_interval_s
        self.lookback_s = lookback_s
        self.once = once
        self.max_consecutive_errors = max_consecutive_errors
        self._client = client
        self._owns_client = client is None
        self._prepared = False
        #: Instance the client actually resolved to, which is the only way to
        #: know which site an empty ``efd_name`` picked. Filled on connect.
        self._resolved_site: str | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._watermark: Time | None = None
        # Influx range queries are inclusive at the boundary, so the row that
        # set the watermark comes back again on the next poll. Bounded because
        # this runs for weeks; alerts are rare, so the cap is never reached in
        # practice and exists only so it cannot grow without limit.
        self._seen: set[tuple[str, Any]] = set()
        self._order: deque[tuple[str, Any]] = deque(maxlen=1000)

    # ------------------------------------------------------------- plumbing

    @property
    def site(self) -> str:
        """Which EFD instance this source reads.

        Resolved from the client once it is open, because ``efd_name`` may be
        empty -- ``lsst.summit.utils`` then picks an instance for the host, and
        the configuration alone cannot say which. Before connecting, and for a
        client that does not name itself, this falls back to what was asked
        for, and says so rather than inventing a site.
        """
        if self._resolved_site:
            return self._resolved_site
        configured = (self.efd_name or "").strip()
        return configured or "unknown (the host default)"

    def describe(self) -> str:
        """One line naming the site, database and topic being polled.

        Used in the startup log, on every record's metadata, and in the failure
        posted to Slack, so "monitoring stopped" always says *what* stopped.
        """
        return f"{self.topic} on {self.site} (database {self.database})"

    def _note_site(self, client: Any) -> None:
        """Record the instance the client resolved to, if it names itself.

        The client wins over the configuration, because it is what is actually
        being read. A disagreement is worth a warning rather than a silent
        preference: it means the configured name did not take effect.
        """
        for attribute in ("efd_name", "name", "site"):
            value = getattr(client, attribute, None)
            if isinstance(value, str) and value.strip():
                self._resolved_site = value.strip()
                configured = (self.efd_name or "").strip()
                if configured and configured != self._resolved_site:
                    logger.warning(
                        "ingest.efd_name is %r but the client opened %r; reading %r",
                        configured,
                        self._resolved_site,
                        self._resolved_site,
                    )
                return
        # Not fatal: the configured name is still reported, and a client that
        # keeps its instance private is not something to fail a poll over.
        logger.debug("EFD client does not name its instance; reporting the configured name")

    def _build_client(self) -> Any:
        """Construct an EFD client, preferring the summit helper."""
        name = (self.efd_name or "").strip()
        try:
            from lsst.summit.utils.efdUtils import makeEfdClient

            client = makeEfdClient(name) if name else makeEfdClient()
            self._note_site(client)
            logger.info(
                "Opened the %s EFD via lsst.summit.utils (ingest.efd_name=%r)",
                self.site,
                name,
            )
        except ImportError:
            try:
                from lsst_efd_client import EfdClient
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    "EFD ingest requires lsst-efd-client (pip install lsst-efd-client), "
                    "or lsst.summit.utils on an RSP host"
                ) from exc
            if not name:
                raise ValueError(
                    "ingest.efd_name must name an EFD instance (summit_efd, usdf_efd, "
                    "idf_efd, base_efd) when lsst.summit.utils is not available to pick "
                    "a default for this host"
                )
            client = EfdClient(name)
            self._note_site(client)
            logger.info("Opened the %s EFD via lsst_efd_client", self.site)
        return client

    def _ensure_client(self) -> Any:
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
        if self._client is None:

            async def build() -> Any:
                # Built with the loop running: some client versions bind an
                # aiohttp session at construction time.
                return self._build_client()

            self._client = self._loop.run_until_complete(build())
        # Outside the branch above so it applies to a client handed in as well:
        # which database to read is this source's business, and the alerts are
        # not in the default one. A client left pointing at the default returns
        # nothing at all rather than complaining. The same goes for the site --
        # a client supplied by a caller is as worth naming as one built here.
        if not self._prepared:
            self._note_site(self._client)
            if self.database:
                self._client.db_name = self.database
            self._prepared = True
        return self._client

    def _query(self, start: Time, end: Time) -> pd.DataFrame:
        """One ``select_time_series`` call, run on the private loop."""
        client = self._ensure_client()
        assert self._loop is not None
        # "*" rather than a field list: at nside 32 the explicit list is over
        # 12,000 names long, and the client would only expand it again.
        frame = self._loop.run_until_complete(client.select_time_series(self.topic, "*", start, end))
        if frame is None:
            return pd.DataFrame()
        return frame

    def _remember(self, key: tuple[str, Any]) -> None:
        """Record a delivered alert, evicting the oldest when full."""
        if len(self._order) == self._order.maxlen:
            self._seen.discard(self._order[0])
        self._order.append(key)
        self._seen.add(key)

    # ---------------------------------------------------------------- source

    def _new_rows(self, frame: pd.DataFrame) -> list[tuple[Any, pd.Series]]:
        """Rows not yet delivered, oldest first."""
        if frame is None or frame.empty:
            return []
        fresh = []
        for index, row in frame.sort_index().iterrows():
            source = str(row["source"]) if "source" in row.index else ""
            key = (source, str(index))
            if key in self._seen:
                continue
            self._remember(key)
            fresh.append((index, row))
        return fresh

    def __iter__(self) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
        # Connect before the retry loop: a missing lsst-efd-client or a
        # misconfigured instance name is permanent, and retrying it five times
        # only delays saying so by a minute. It also resolves the site, so
        # everything below can name it.
        self._ensure_client()
        origin = self.describe()
        # Start the window before now when asked, so an alert that arrived
        # during a restart is not missed.
        self._watermark = Time.now() - TimeDelta(self.lookback_s, format="sec")
        logger.info(
            "Watching the %s EFD for %s (database %s) from %s, every %.0f s",
            self.site,
            self.topic,
            self.database,
            self._watermark.isot,
            self.poll_interval_s,
        )

        errors = 0
        while True:
            end = Time.now()
            try:
                frame = self._query(self._watermark, end)
                errors = 0
            except Exception as exc:
                errors += 1
                logger.error(
                    "EFD query %d/%d failed: %s",
                    errors,
                    self.max_consecutive_errors,
                    exc,
                )
                if errors >= self.max_consecutive_errors:
                    raise RuntimeError(
                        f"{errors} consecutive EFD queries failed; last error: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
                if self.once:
                    return
                time.sleep(self.poll_interval_s)
                continue

            for index, row in self._new_rows(frame):
                try:
                    record = record_from_efd_row(row, when=index)
                except Exception as exc:
                    logger.error("Could not rebuild a record from an EFD row at %s: %s", index, exc)
                    continue
                yield record, {
                    "transport": "efd",
                    "site": self.site,
                    "database": self.database,
                    "topic": self.topic,
                    "origin": origin,
                    "efd_time": str(index),
                }

            # Advance only after the rows are out: an exception mid-iteration
            # would otherwise skip past alerts that were never delivered.
            self._watermark = end
            if self.once:
                return
            time.sleep(self.poll_interval_s)

    def close(self) -> None:
        """Release the client and the private event loop."""
        client = self._client
        if client is not None and self._owns_client and self._loop is not None:
            closer = getattr(client, "close", None)
            if callable(closer):
                try:
                    result = closer()
                    if asyncio.iscoroutine(result):
                        self._loop.run_until_complete(result)
                except Exception as exc:
                    logger.warning("Closing the EFD client failed: %s", exc)
        if self._loop is not None:
            self._loop.close()
            self._loop = None
        if self._owns_client:
            self._client = None
