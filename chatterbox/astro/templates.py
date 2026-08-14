"""Per-band LSST template coverage as HEALPix maps.

Coverage is expressed as one visit-count map per band, following the recipe
used in ``plotMaker``: paint a disc of the LSSTCam field-of-view radius around
each visit boresight and accumulate. A pixel is considered to have a template
once it reaches ``min_visits``.

Maps are built offline by ``chatterbox refresh-templates`` and cached on disk,
so the alert path only ever loads them. That keeps a ConsDB query off the
low-latency path, and the cache's build time is reported in Slack so staleness
is visible rather than silent.

Notes
-----
The disc approximation ignores the real focal-plane geometry and chip gaps, so
it slightly overestimates coverage near field edges. It is what the existing
analysis notebooks use, and it keeps the template map and the localization on
the same footing.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import healpy as hp
import numpy as np
import pandas as pd

__all__ = [
    "TemplateCoverage",
    "build_template_maps",
    "load_template_maps",
    "fetch_visits_consdb",
    "fetch_visits_csv",
    "BANDS",
]

logger = logging.getLogger(__name__)

#: LSST bands in canonical order.
BANDS = ("u", "g", "r", "i", "z", "y")

_META_NAME = "meta.json"


@dataclass
class TemplateCoverage:
    """Per-band visit-count maps in RING ordering.

    Attributes
    ----------
    maps : `dict` [`str`, `numpy.ndarray`]
        Visit counts per pixel, keyed by single-character band name.
    nside : `int`
        Map resolution.
    min_visits : `int`
        Visits required for a pixel to count as having a template.
    built_at : `str`
        ISO-8601 UTC timestamp of when the maps were built.
    source : `str`
        Where the visits came from, shown in Slack.
    n_visits : `dict` [`str`, `int`]
        Number of visits that went into each band's map.
    fov_radius_deg : `float`
        Field-of-view radius used when painting visits.
    """

    maps: dict[str, np.ndarray]
    nside: int
    min_visits: int = 1
    built_at: str = ""
    source: str = ""
    n_visits: dict[str, int] = field(default_factory=dict)
    fov_radius_deg: float = 1.75

    @property
    def bands(self) -> list[str]:
        """Bands present in the cache, in canonical order."""
        return [b for b in BANDS if b in self.maps]

    def mask(self, band: str) -> np.ndarray:
        """Boolean map of pixels with a template in a band."""
        return self.maps[band] >= self.min_visits

    def area_deg2(self, band: str) -> float:
        """Total sky area with a template in a band."""
        return float(self.mask(band).sum() * hp.nside2pixarea(self.nside, degrees=True))

    def coverage_in_region(self, prob_map: np.ndarray, nside: int | None = None) -> dict[str, float]:
        """Fraction of a localization's weight that already has templates.

        Parameters
        ----------
        prob_map : `numpy.ndarray`
            Normalized weight per pixel, RING ordering. Resampled to this
            cache's resolution when necessary.
        nside : `int`, optional
            Resolution of `prob_map`; inferred when omitted.

        Returns
        -------
        fractions : `dict` [`str`, `float`]
            Fraction in 0-1 per band. Bands absent from the cache report 0.
        """
        prob_map = np.asarray(prob_map, dtype=float)
        in_nside = nside if nside is not None else hp.get_nside(prob_map)
        if in_nside != self.nside:
            # power=-2 conserves the summed weight across resolution changes.
            prob_map = hp.ud_grade(prob_map, self.nside, order_in="RING", order_out="RING", power=-2)
        total = prob_map.sum()
        if total <= 0:
            return {b: 0.0 for b in BANDS}
        weights = prob_map / total
        result = {}
        for band in BANDS:
            result[band] = float(weights[self.mask(band)].sum()) if band in self.maps else 0.0
        return result

    # ------------------------------------------------------------------ I/O

    def save(self, cache_dir: str | Path) -> Path:
        """Write the maps and metadata to a cache directory."""
        out = Path(cache_dir).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        for band, m in self.maps.items():
            np.save(out / f"{band}.npy", m.astype(np.int32))
        meta = {
            "nside": self.nside,
            "min_visits": self.min_visits,
            "built_at": self.built_at,
            "source": self.source,
            "n_visits": self.n_visits,
            "fov_radius_deg": self.fov_radius_deg,
            "bands": self.bands,
        }
        (out / _META_NAME).write_text(json.dumps(meta, indent=2))
        logger.info("Wrote template coverage for %s to %s", ",".join(self.bands), out)
        return out


def load_template_maps(cache_dir: str | Path) -> TemplateCoverage | None:
    """Load cached per-band template maps.

    Returns
    -------
    coverage : `TemplateCoverage` or None
        None when the cache is absent or unreadable, so the caller can post
        without the template panel rather than failing the whole alert.
    """
    path = Path(cache_dir).expanduser()
    meta_path = path / _META_NAME
    if not meta_path.is_file():
        logger.warning("No template coverage cache at %s; run 'chatterbox refresh-templates'", path)
        return None
    try:
        meta = json.loads(meta_path.read_text())
        maps = {}
        for band in meta.get("bands", list(BANDS)):
            band_path = path / f"{band}.npy"
            if band_path.is_file():
                maps[band] = np.load(band_path)
        if not maps:
            logger.warning("Template cache at %s contains no band maps", path)
            return None
        return TemplateCoverage(
            maps=maps,
            nside=int(meta["nside"]),
            min_visits=int(meta.get("min_visits", 1)),
            built_at=meta.get("built_at", ""),
            source=meta.get("source", ""),
            n_visits=meta.get("n_visits", {}),
            fov_radius_deg=float(meta.get("fov_radius_deg", 1.75)),
        )
    except Exception as exc:
        logger.warning("Could not read template cache at %s: %s", path, exc)
        return None


# ------------------------------------------------------------------ building


def build_template_maps(
    visits: pd.DataFrame,
    nside: int = 256,
    fov_radius_deg: float = 1.75,
    min_visits: int = 1,
    bands: tuple[str, ...] = BANDS,
    ra_col: str = "s_ra",
    dec_col: str = "s_dec",
    band_col: str = "band",
    source: str = "",
) -> TemplateCoverage:
    """Accumulate per-band visit-count maps from a visit table.

    Parameters
    ----------
    visits : `pandas.DataFrame`
        One row per visit, with boresight RA/Dec in **degrees** and a band.
    nside : `int`
        Output resolution.
    fov_radius_deg : `float`
        Radius of the disc painted around each boresight.
    min_visits : `int`
        Stored on the result; visits required for a template to exist.
    bands : `tuple` [`str`]
        Bands to build.
    ra_col, dec_col, band_col : `str`
        Column names. Defaults match a ConsDB visit export.
    source : `str`
        Description recorded in the cache metadata.

    Returns
    -------
    coverage : `TemplateCoverage`
    """
    from datetime import datetime, timezone

    missing = [c for c in (ra_col, dec_col, band_col) if c not in visits.columns]
    if missing:
        raise KeyError(f"Visit table is missing columns {missing}; have {list(visits.columns)}")

    npix = hp.nside2npix(nside)
    radius = np.radians(fov_radius_deg)
    maps: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}

    for band in bands:
        rows = visits[visits[band_col] == band]
        nvis_map = np.zeros(npix, dtype=np.int32)
        ra = np.asarray(rows[ra_col], dtype=float)
        dec = np.asarray(rows[dec_col], dtype=float)
        good = np.isfinite(ra) & np.isfinite(dec)
        if good.sum() != ra.size:
            logger.warning("Dropping %d %s-band visits with non-finite pointings", ra.size - good.sum(), band)
        ra, dec = ra[good], dec[good]
        if ra.size:
            vecs = hp.ang2vec(ra, dec, lonlat=True)
            for vec in vecs:
                nvis_map[hp.query_disc(nside, vec, radius, inclusive=True)] += 1
        maps[band] = nvis_map
        counts[band] = int(ra.size)
        logger.info(
            "Band %s: %d visits -> %.0f deg^2 with >=%d visits",
            band,
            ra.size,
            (nvis_map >= min_visits).sum() * hp.nside2pixarea(nside, degrees=True),
            min_visits,
        )

    return TemplateCoverage(
        maps=maps,
        nside=nside,
        min_visits=min_visits,
        built_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        source=source,
        n_visits=counts,
        fov_radius_deg=fov_radius_deg,
    )


def fetch_visits_csv(path: str | Path) -> pd.DataFrame:
    """Read a visit table exported from ConsDB as CSV or ECSV.

    Parameters
    ----------
    path : `str` or `pathlib.Path`
        File to read. ``.ecsv`` is read through astropy.

    Returns
    -------
    visits : `pandas.DataFrame`
    """
    path = Path(path).expanduser()
    if path.suffix == ".ecsv":
        from astropy.table import Table

        return Table.read(path).to_pandas()
    return pd.read_csv(path)


def fetch_visits_consdb(
    instrument: str = "lsstcam",
    site: str = "usdf",
    tokenfile: str = "~/.lsst/usdf_rsp",
    t_start: str | None = None,
    t_end: str | None = None,
) -> pd.DataFrame:
    """Query ConsDB for the visit history, via ``rubin_nights``.

    Parameters
    ----------
    instrument : `str`
        ConsDB instrument name.
    site : `str`
        One of the endpoints known to ``rubin_nights`` (``usdf``, ``summit``).
    tokenfile : `str`
        Path to the RSP token file.
    t_start, t_end : `str`, optional
        ISO times bounding the query. Defaults to the full history.

    Returns
    -------
    visits : `pandas.DataFrame`
        With at least ``band``, ``s_ra`` and ``s_dec`` columns.
    """
    from astropy.time import Time
    from rubin_nights import connections
    from rubin_nights.consdb_query import ConsDbFastAPI, ConsDbTap

    tokenfile = str(Path(tokenfile).expanduser())
    endpoints = connections.get_clients(tokenfile=tokenfile, site=site)

    start = Time(t_start) if t_start else None
    end = Time(t_end) if t_end else None

    consdb = endpoints.get("consdb")
    if consdb is None:
        # get_clients did not supply one; fall back to constructing directly.
        base = connections.api_endpoints.get(site)
        token = Path(tokenfile).read_text().strip()
        try:
            consdb = ConsDbTap(api_base=base, token=token)
        except Exception:
            consdb = ConsDbFastAPI(api_base=base, token=token)

    visits = consdb.get_visits(instrument=instrument, t_start=start, t_end=end)
    logger.info("ConsDB returned %d visits for %s", len(visits), instrument)
    return visits
