"""Transports that deliver ToO records to chatterbox.

``forward_alerts.py`` can emit through four senders (stdout, files, Kafka via
hop, and the Confluent REST proxy), so chatterbox supports the two that a
downstream consumer can realistically attach to, plus a replay source for
testing:

- `KafkaTooAlertSource` subscribes to the producer's output topic with
  hop-client, which is required because the producer frames records as
  ``hop.models.AvroBlob`` and adds its own headers.
- `FileTooAlertSource` watches the directory the file sender writes
  ``{source}.json`` into.
- `ReplaySource` yields explicit paths once and stops.

Every source yields ``(record, metadata)`` and exposes ``mark_done`` so Kafka
offsets are only committed after a record has been fully handled.
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ..config import Config, IngestConfig

__all__ = [
    "TooAlertSource",
    "KafkaTooAlertSource",
    "FileTooAlertSource",
    "ReplaySource",
    "make_source",
]

logger = logging.getLogger(__name__)


class TooAlertSource(ABC):
    """A stream of raw ToO records."""

    @abstractmethod
    def __iter__(self) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
        """Yield ``(record, metadata)`` pairs."""

    def mark_done(self, metadata: dict[str, Any]) -> None:
        """Acknowledge a record. Default is a no-op."""

    def close(self) -> None:
        """Release any resources."""


class ReplaySource(TooAlertSource):
    """Yield records from explicit file paths, then stop.

    Parameters
    ----------
    paths : sequence of `str` or `pathlib.Path`
        Record files to read, in order.
    """

    def __init__(self, paths) -> None:
        self.paths = [Path(p).expanduser() for p in paths]

    def __iter__(self) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
        from .decode import load_record_file

        for path in self.paths:
            logger.info("Replaying %s", path)
            yield load_record_file(path), {"origin": str(path), "transport": "replay"}


class FileTooAlertSource(TooAlertSource):
    """Watch a directory for record files written by the producer.

    Parameters
    ----------
    watch_dir : `str`
        Directory to poll.
    poll_interval_s : `float`
        Seconds between scans.
    once : `bool`
        Process the files already present and return, instead of polling
        forever. Useful for tests and for a cron-style deployment.

    Notes
    -----
    The producer's ``FileSender`` writes ``{source}.json`` and *overwrites* it
    when the same event is re-sent, so files are tracked by
    ``(path, mtime, size)``. That way a rewritten record is picked up again
    rather than being mistaken for one already handled.
    """

    def __init__(self, watch_dir: str, poll_interval_s: float = 2.0, once: bool = False) -> None:
        self.watch_dir = Path(watch_dir).expanduser()
        self.poll_interval_s = poll_interval_s
        self.once = once
        self._seen: set[tuple[str, float, int]] = set()

    def _scan(self) -> list[Path]:
        if not self.watch_dir.is_dir():
            return []
        found = []
        for path in sorted(self.watch_dir.iterdir()):
            if path.suffix not in (".json", ".avro"):
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            key = (str(path), stat.st_mtime, stat.st_size)
            if key in self._seen:
                continue
            self._seen.add(key)
            found.append(path)
        return found

    def __iter__(self) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
        from .decode import load_record_file

        self.watch_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Watching %s for ToO records", self.watch_dir)
        while True:
            for path in self._scan():
                try:
                    record = load_record_file(path)
                except Exception as exc:
                    logger.error("Could not read %s: %s", path, exc)
                    continue
                yield record, {"origin": str(path), "transport": "files"}
            if self.once:
                return
            time.sleep(self.poll_interval_s)


class KafkaTooAlertSource(TooAlertSource):
    """Subscribe to the producer's output topic with hop-client.

    Parameters
    ----------
    url : `str`
        hop URL, e.g. ``kafka://kafka.scimma.org/lsst.rubin-too-alerts``.
    group_id : `str`
        Consumer group. A distinct group from the producer's own consumers is
        required so chatterbox sees every record.
    allow_tests : `bool`
        When False, hop is asked to drop messages carrying the ``_test`` header
        before they reach chatterbox.

    Notes
    -----
    Authentication is delegated entirely to hop-client's own credential lookup
    (``~/.config/hop/auth.toml``), matching how the producer does it: it also
    constructs ``hop.io.Stream()`` with no arguments.
    """

    def __init__(self, url: str, group_id: str = "chatterbox", allow_tests: bool = True) -> None:
        if not url:
            raise ValueError(
                "No Kafka URL configured. Set ingest.kafka_url, e.g. "
                "'kafka://<broker>/<topic>'. Topic names are deployment "
                "configuration and are not defined in the producer repository."
            )
        self.url = url
        self.group_id = group_id
        self.allow_tests = allow_tests
        self._stream = None

    def _open(self):
        try:
            import hop
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("Kafka ingest requires hop-client (pip install hop-client)") from exc
        logger.info("Opening %s as group %s", self.url, self.group_id)
        return hop.io.Stream(auth=True).open(
            self.url,
            mode="r",
            group_id=self.group_id,
            ignoretest=not self.allow_tests,
        )

    def __iter__(self) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
        self._stream = self._open()
        for message, metadata in self._stream.read(metadata=True, autocommit=False):
            for record in _unwrap(message):
                meta = {
                    "transport": "kafka",
                    "topic": getattr(metadata, "topic", None),
                    "offset": getattr(metadata, "offset", None),
                    "partition": getattr(metadata, "partition", None),
                    "headers": getattr(metadata, "headers", None),
                    "_raw_metadata": metadata,
                }
                yield record, meta

    def mark_done(self, metadata: dict[str, Any]) -> None:
        """Commit the offset for a handled record."""
        raw = metadata.get("_raw_metadata")
        if self._stream is not None and raw is not None:
            self._stream.mark_done(raw)

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None


def _unwrap(message: Any) -> list[dict[str, Any]]:
    """Extract record dicts from a hop message.

    hop wraps payloads in model objects (``AvroBlob``, ``JSONBlob``) whose
    ``content`` is either a single record or a list of them.
    """
    content = getattr(message, "content", message)
    if isinstance(content, (bytes, bytearray)):
        return [json.loads(content)]
    if isinstance(content, str):
        return [json.loads(content)]
    if isinstance(content, dict):
        return [content]
    if isinstance(content, list):
        return [c for c in content if isinstance(c, dict)]
    logger.error("Cannot interpret message content of type %s", type(content))
    return []


def make_source(config: Config, paths=None) -> TooAlertSource:
    """Build the ingest source described by the configuration.

    Parameters
    ----------
    config : `Config`
        Loaded configuration.
    paths : sequence, optional
        Explicit record paths. Forces a `ReplaySource` regardless of
        ``ingest.kind``.

    Returns
    -------
    source : `TooAlertSource`
    """
    if paths:
        return ReplaySource(paths)

    ingest: IngestConfig = config.ingest
    kind = ingest.kind.lower()
    if kind == "kafka":
        return KafkaTooAlertSource(
            url=ingest.kafka_url,
            group_id=ingest.kafka_group_id,
            allow_tests=ingest.allow_tests,
        )
    if kind == "files":
        return FileTooAlertSource(ingest.watch_dir, ingest.poll_interval_s)
    if kind == "replay":
        raise ValueError("ingest.kind='replay' requires explicit paths")
    raise ValueError(f"Unknown ingest.kind {ingest.kind!r}; expected 'kafka', 'files' or 'replay'")
