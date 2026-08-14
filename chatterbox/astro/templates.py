"""Per-band LSST template coverage as HEALPix maps.

Coverage is not computed here. It is produced upstream by the incremental
templates tooling, which writes one HEALPix FITS map per band at a known path,
named by `DEFAULT_MAP_PATTERN`. chatterbox reads those files and nothing else.

``chatterbox refresh-templates`` copies them into a local cache so the alert
path never touches the shared filesystem the originals live on, and the cache's
build time is reported in Slack so staleness is visible rather than silent.

Notes
-----
The maps are binary: 1 where a template exists, 0 where it does not. Anything
non-zero is treated as covered, and `load_source_maps` warns if a file turns
out to hold something else, since that would mean the upstream format changed.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import healpy as hp
import numpy as np

__all__ = [
    "TemplateCoverage",
    "load_template_maps",
    "load_source_maps",
    "read_coverage_map",
    "map_filename",
    "BANDS",
    "DEFAULT_MAP_PATTERN",
]

logger = logging.getLogger(__name__)

#: LSST bands in canonical order.
BANDS = ("u", "g", "r", "i", "z", "y")

#: Filename convention used by the incremental templates tooling, e.g.
#: ``template_coverage_healpix_y_nside64.fits``.
DEFAULT_MAP_PATTERN = "template_coverage_healpix_{band}_nside{nside}.fits"

_META_NAME = "meta.json"


def map_filename(band: str, nside: int, pattern: str = DEFAULT_MAP_PATTERN) -> str:
    """Filename of the coverage map for one band.

    Parameters
    ----------
    band : `str`
        Single-character band name.
    nside : `int`
        Map resolution, which appears in the filename.
    pattern : `str`
        Format string with ``{band}`` and ``{nside}`` fields.

    Returns
    -------
    name : `str`
    """
    return pattern.format(band=band, nside=nside)


@dataclass
class TemplateCoverage:
    """Per-band template coverage maps in RING ordering.

    Attributes
    ----------
    maps : `dict` [`str`, `numpy.ndarray`]
        Coverage value per pixel, keyed by single-character band name.
    nside : `int`
        Map resolution.
    built_at : `str`
        ISO-8601 UTC timestamp of when the cache was refreshed.
    source : `str`
        Where the maps came from, shown in Slack.
    band_files : `dict` [`str`, `str`]
        The source file each band was read from.
    """

    maps: dict[str, np.ndarray]
    nside: int
    built_at: str = ""
    source: str = ""
    band_files: dict[str, str] = field(default_factory=dict)

    @property
    def bands(self) -> list[str]:
        """Bands present, in canonical order."""
        return [b for b in BANDS if b in self.maps]

    @property
    def missing_bands(self) -> list[str]:
        """Bands with no coverage map available."""
        return [b for b in BANDS if b not in self.maps]

    def mask(self, band: str) -> np.ndarray:
        """Boolean map of pixels with a template in a band.

        The published maps are binary, so any non-zero value means covered.
        This also makes NaN (were it ever used for "no data") read as
        uncovered rather than raising.
        """
        return np.asarray(self.maps[band]) > 0

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
            Fraction in 0-1 per band. Bands with no map report 0.
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
        """Write the maps and metadata to a local cache directory."""
        out = Path(cache_dir).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        for band, m in self.maps.items():
            np.save(out / f"{band}.npy", np.asarray(m, dtype=np.float32))
        meta = {
            "nside": self.nside,
            "built_at": self.built_at,
            "source": self.source,
            "band_files": self.band_files,
            "bands": self.bands,
        }
        (out / _META_NAME).write_text(json.dumps(meta, indent=2))
        logger.info("Wrote template coverage for %s to %s", ",".join(self.bands), out)
        return out


def read_coverage_map(path: str | Path) -> np.ndarray:
    """Read one band's HEALPix coverage map as a RING-ordered array.

    Parameters
    ----------
    path : `str` or `pathlib.Path`
        FITS file written by the incremental templates tooling.

    Returns
    -------
    coverage : `numpy.ndarray`
        RING ordering. ``healpy`` converts from NESTED when the file's
        ``ORDERING`` keyword says so, so either is accepted.
    """
    # nest=False asks healpy for RING output regardless of how the file is
    # stored; dtype=None preserves whatever the tooling wrote.
    return np.asarray(hp.read_map(str(path), nest=False, dtype=None))


def load_source_maps(
    maps_dir: str | Path,
    nside: int,
    bands: tuple[str, ...] = BANDS,
    pattern: str = DEFAULT_MAP_PATTERN,
) -> TemplateCoverage:
    """Read the per-band coverage maps from the directory they land in.

    Parameters
    ----------
    maps_dir : `str` or `pathlib.Path`
        Directory holding one FITS map per band.
    nside : `int`
        Resolution, used both to build the filenames and to validate the maps.
    bands : `tuple` [`str`]
        Bands to look for.
    pattern : `str`
        Filename pattern with ``{band}`` and ``{nside}`` fields.

    Returns
    -------
    coverage : `TemplateCoverage`

    Raises
    ------
    FileNotFoundError
        If the directory does not exist, or no band's map could be read.
        A missing *individual* band is logged and skipped, since coverage
        is built up band by band upstream.
    """
    directory = Path(maps_dir).expanduser()
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Template coverage directory not found: {directory}. Set "
            "templates.maps_dir to the directory the coverage maps are published to."
        )

    maps: dict[str, np.ndarray] = {}
    band_files: dict[str, str] = {}
    expected_npix = hp.nside2npix(nside)

    for band in bands:
        path = directory / map_filename(band, nside, pattern)
        if not path.is_file():
            logger.warning("No %s-band coverage map at %s", band, path)
            continue
        try:
            coverage = read_coverage_map(path)
        except Exception as exc:
            logger.error("Could not read %s: %s", path, exc)
            continue
        if coverage.size != expected_npix:
            logger.error(
                "%s has %d pixels, expected %d for nside=%d; skipping",
                path,
                coverage.size,
                expected_npix,
                nside,
            )
            continue
        # The maps are meant to be binary. Anything else still works, since
        # non-zero counts as covered, but it means the upstream format
        # changed -- worth saying out loud rather than silently reinterpreting.
        distinct = np.unique(coverage[np.isfinite(coverage)])
        if not np.all(np.isin(distinct, (0, 1))):
            logger.warning(
                "%s is not binary (values %s...); treating any non-zero pixel as covered",
                path.name,
                np.array2string(distinct[:5], precision=3),
            )

        maps[band] = coverage
        band_files[band] = str(path)
        logger.info(
            "Read %s-band coverage from %s (%.0f deg^2 covered)",
            band,
            path.name,
            (coverage > 0).sum() * hp.nside2pixarea(nside, degrees=True),
        )

    if not maps:
        raise FileNotFoundError(
            f"No template coverage maps found in {directory} matching "
            f"{pattern.format(band='<band>', nside=nside)}"
        )

    from datetime import datetime, timezone

    missing = [b for b in bands if b not in maps]
    if missing:
        logger.warning("No coverage map for band(s): %s", ", ".join(missing))

    return TemplateCoverage(
        maps=maps,
        nside=nside,
        built_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        source=str(directory),
        band_files=band_files,
    )


def load_template_maps(cache_dir: str | Path) -> TemplateCoverage | None:
    """Load the local cache written by ``chatterbox refresh-templates``.

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
            built_at=meta.get("built_at", ""),
            source=meta.get("source", ""),
            band_files=meta.get("band_files", {}),
        )
    except Exception as exc:
        logger.warning("Could not read template cache at %s: %s", path, exc)
        return None
