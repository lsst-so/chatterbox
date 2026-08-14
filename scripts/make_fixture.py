#!/usr/bin/env python3
"""Generate a realistic ToO record fixture from a real GraceDB skymap.

The reward map is produced by the same ``Skymap.make_flat_binary_map`` code
path
``forward_alerts.py`` uses, so the fixture is a faithful example of what the
producer emits rather than an invented shape.

Examples
--------
Build a GW fixture from a superevent (downloads the skymap once)::

    python scripts/make_fixture.py --superevent S251112cm \\
        --alert-type GW_case_B --out tests/data/gw_case_b.json

Build a circular-localization fixture, as SuperK galactic SNe alerts produce::

    python scripts/make_fixture.py --circle 265.0 -29.0 3.5 \\
        --alert-type SN_Galactic --source SK_20260813_01 \\
        --out tests/data/sn_galactic.json
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import healpy as hp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chatterbox.astro.skymap import Skymap
from chatterbox.ingest.decode import REWARD_MAP_CREDIBLE_LEVEL

TARGET_ORDER = 5  # nside 32, as hardcoded in every producer filter


def reward_map_from_skymap(path: str, credible_level: float) -> np.ndarray:
    """Build the producer's binary reward map from a multi-order skymap."""
    from astropy.table import Table

    table = Table.read(path)
    skymap = Skymap(table["PROBDENSITY"], table["UNIQ"])
    return skymap.make_flat_binary_map(credible_level, TARGET_ORDER)


def reward_map_from_circle(ra: float, dec: float, radius: float) -> np.ndarray:
    """Build a reward map covering a circular localization, NESTED at nside 32.

    Mirrors ``SuperKAlertFilter.generate_scheduling_data``, which marks every
    pixel *touched* by the circle.
    """
    nside = 1 << TARGET_ORDER
    flat = np.zeros(hp.nside2npix(nside), dtype=bool)
    ring = hp.query_disc(nside, hp.ang2vec(ra, dec, lonlat=True), np.radians(radius), inclusive=True)
    flat[hp.ring2nest(nside, ring)] = True
    return flat


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--superevent", help="GraceDB superevent id to download a skymap for")
    group.add_argument("--skymap", help="Local multi-order skymap FITS file")
    group.add_argument(
        "--circle",
        nargs=3,
        type=float,
        metavar=("RA", "DEC", "RADIUS"),
        help="Circular localization in degrees",
    )
    parser.add_argument("--alert-type", required=True, help="alert_type label to emit")
    parser.add_argument("--source", help="source id (defaults to the superevent id)")
    parser.add_argument("--instruments", nargs="*", default=None, help="Instrument names")
    parser.add_argument("--event-time", help="ISO event time (defaults to now)")
    parser.add_argument("--is-test", action="store_true")
    parser.add_argument("--is-update", action="store_true")
    parser.add_argument(
        "--credible-level",
        type=float,
        default=REWARD_MAP_CREDIBLE_LEVEL,
        help=f"Credible level for the reward map (producer uses {REWARD_MAP_CREDIBLE_LEVEL})",
    )
    parser.add_argument("--out", required=True, help="Output JSON path")
    args = parser.parse_args()

    if args.circle:
        reward = reward_map_from_circle(*args.circle)
        source = args.source or "SYNTHETIC"
        instruments = args.instruments or ["Super-K"]
    else:
        path = args.skymap
        if args.superevent:
            import time

            from chatterbox.config import EnrichConfig
            from chatterbox.ingest.enrich_gracedb import _client, _download_skymap

            cfg = EnrichConfig()
            cache = Path(cfg.cache_dir).expanduser()
            downloaded = _download_skymap(
                _client(cfg), args.superevent, cache, time.monotonic() + cfg.timeout_s
            )
            if downloaded is None:
                print(f"Could not download a skymap for {args.superevent}", file=sys.stderr)
                return 1
            path = str(downloaded)
        reward = reward_map_from_skymap(path, args.credible_level)
        source = args.source or args.superevent or "SYNTHETIC"
        instruments = args.instruments or ["H1", "L1", "V1"]

    # The producer pads or truncates the instrument list to exactly three.
    instruments = list(instruments)
    while len(instruments) < 3:
        instruments.append("")
    instruments = instruments[:3]

    event_time = args.event_time or datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    record = {
        "source": source,
        "instrument": instruments,
        "alert_type": args.alert_type,
        "event_trigger_timestamp": event_time,
        "reward_map": [bool(x) for x in reward],
        "reward_map_nside": 1 << TARGET_ORDER,
        "is_test": bool(args.is_test),
        "is_update": bool(args.is_update),
        "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record))
    area = reward.sum() * hp.nside2pixarea(1 << TARGET_ORDER, degrees=True)
    print(f"Wrote {out}: {source} ({args.alert_type}), {int(reward.sum())} pixels = {area:.0f} deg^2")
    return 0


if __name__ == "__main__":
    sys.exit(main())
