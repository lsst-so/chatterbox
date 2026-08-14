"""Shared test fixtures.

Records are built programmatically where the shape is what matters, and read
from ``tests/data`` where realism matters. The committed fixtures were
generated
from a real GraceDB skymap by ``scripts/make_fixture.py`` using the same
``Skymap.make_flat_binary_map`` code path the producer uses.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import healpy as hp
import numpy as np
import pytest

DATA_DIR = Path(__file__).parent / "data"

#: The producer hardcodes order 5 in every filter.
NSIDE = 32


@pytest.fixture
def data_dir() -> Path:
    """Directory holding committed record fixtures."""
    return DATA_DIR


def disc_reward_map(ra: float, dec: float, radius_deg: float, nside: int = NSIDE) -> np.ndarray:
    """A circular reward map in NESTED ordering, as the producer emits."""
    flat = np.zeros(hp.nside2npix(nside), dtype=bool)
    ring = hp.query_disc(nside, hp.ang2vec(ra, dec, lonlat=True), np.radians(radius_deg), inclusive=True)
    flat[hp.ring2nest(nside, ring)] = True
    return flat


def make_record(
    source: str = "S260814a",
    alert_type: str = "GW_case_B",
    reward_map: np.ndarray | None = None,
    nside: int = NSIDE,
    instruments: list[str] | None = None,
    event_time: str = "2026-08-14T02:00:00.000Z",
    is_test: bool = False,
    is_update: bool = False,
    timestamp_ms: int | None = None,
) -> dict:
    """Build a ``too_alert`` record matching the producer's output schema.

    The instrument list is padded to exactly three entries, as
    ``AlertFilter.process`` does.
    """
    if reward_map is None:
        reward_map = disc_reward_map(60.0, -35.0, 9.8, nside)
    instruments = list(instruments or ["H1", "L1"])
    while len(instruments) < 3:
        instruments.append("")
    instruments = instruments[:3]
    if timestamp_ms is None:
        timestamp_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    return {
        "source": source,
        "instrument": instruments,
        "alert_type": alert_type,
        "event_trigger_timestamp": event_time,
        "reward_map": [bool(x) for x in reward_map],
        "reward_map_nside": int(nside),
        "is_test": bool(is_test),
        "is_update": bool(is_update),
        "timestamp": int(timestamp_ms),
    }


@pytest.fixture
def record() -> dict:
    """A minimal valid GW record."""
    return make_record()


@pytest.fixture
def gw_record(data_dir: Path) -> dict:
    """The realistic GW record derived from S251112cm's skymap."""
    return json.loads((data_dir / "gw_case_b.json").read_text())


@pytest.fixture
def sn_record(data_dir: Path) -> dict:
    """A galactic supernova record with a circular localization."""
    return json.loads((data_dir / "sn_galactic.json").read_text())


@pytest.fixture
def neutrino_record(data_dir: Path) -> dict:
    """A neutrino record with a single-field-of-view localization."""
    return json.loads((data_dir / "neutrino.json").read_text())


@pytest.fixture(autouse=True)
def no_slack_token(monkeypatch):
    """Make it impossible for the suite to post to a real channel.

    ``run_service`` builds its own `SlackPoster` from the environment, so a
    developer with ``SLACK_BOT_TOKEN`` exported would otherwise have tests --
    including the ones that deliberately provoke failures -- posting to the
    channel. Without a token the poster is offline and writes payloads to the
    temporary work directory instead.
    """
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)


@pytest.fixture
def config(tmp_path):
    """A Config pointing entirely at a temporary directory."""
    from chatterbox.config import Config

    cfg = Config()
    cfg.paths.work_dir = str(tmp_path / "work")
    cfg.templates.cache_dir = str(tmp_path / "templates")
    cfg.enrich.cache_dir = str(tmp_path / "skymaps")
    # Tests must never reach the network or spawn a simulation.
    cfg.enrich.gracedb = False
    cfg.sim.enabled = False
    return cfg


@pytest.fixture
def rubin_scheduler():
    """Skip a test when rubin_scheduler is unavailable."""
    return pytest.importorskip("rubin_scheduler", reason="requires rubin_scheduler")
