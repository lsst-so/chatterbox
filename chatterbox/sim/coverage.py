"""Fraction of a localization covered by simulated visits, per band.

This is the reusable form of an analysis that previously existed only as a
notebook cell. The algorithm:

1. Take an independent copy of the localization map per band, so each band's
   number is a self-contained "what fraction did *this* band reach".
2. For each visit, paint a disc of the LSSTCam field-of-view radius, add the
   enclosed weight to the running total, then zero those pixels so overlapping
   pointings are not double-counted.
3. Report the total per band, iterating over all six LSST bands explicitly so a
   band that was never observed reports 0 rather than disappearing.

Whether the result is a *probability* or an *area* fraction depends entirely on
the map handed in; `~chatterbox.models.Localization.is_probability` records
which, and `CoverageResult.quantity` carries it through to the caller.
"""

import copy
import logging
import re
from dataclasses import dataclass, field

import healpy as hp
import numpy as np
import pandas as pd

from ..models import Localization

__all__ = [
    "BANDS",
    "FOV_RADIUS_DEG",
    "CoverageResult",
    "too_visits",
    "band_coverage",
    "band_coverage_by_epoch",
    "coverage_curve",
    "detect_pointing_columns",
    "detect_time_column",
]

logger = logging.getLogger(__name__)

#: LSST bands, in canonical order. Iterated explicitly so unobserved bands
#: report zero coverage instead of being omitted.
BANDS = ("u", "g", "r", "i", "z", "y")

#: LSSTCam field-of-view radius used to approximate each visit's footprint.
#: This ignores the real focal-plane geometry and chip gaps.
FOV_RADIUS_DEG = 1.75

#: ToO visits are tagged ``too_{target_name_base}_{too_id}_i{epoch}`` in
#: ``observation_reason`` by ``ToOScriptedSurvey``. ``target_name`` is a
#: comma-joined list of every survey that claimed the field, so it is not a
#: reliable filter; ``observation_reason`` is.
_REASON_RE = re.compile(r"^too_(?P<base>.+)_i(?P<epoch>\d+)$")

#: Pointing column conventions, in preference order, as
#: ``(ra_col, dec_col, in_radians)``.
#:
#: A raw ``sim_runner`` ObservationArray uses ``RA``/``dec`` in **radians**,
#: while ``lsst_support.save_opsim`` converts to the opsim schema with
#: ``fieldRA``/``fieldDec`` in **degrees**. Both reach this code depending on
#: how
#: the visits were saved, and mistaking one for the other silently produces
#: nonsense coverage rather than an error, so the convention is detected.
_POINTING_CONVENTIONS = (
    ("RA", "dec", True),
    ("fieldRA", "fieldDec", False),
    ("ra", "decl", False),
)

#: Time column conventions, in preference order.
_TIME_COLUMNS = ("mjd", "observationStartMJD", "exp_midpt_mjd", "obs_start_mjd")


@dataclass
class CoverageResult:
    """Per-band coverage of a localization by a set of visits.

    Attributes
    ----------
    fractions : `dict` [`str`, `float`]
        Fraction of the localization weight covered per band, 0-1.
    n_visits : `dict` [`str`, `int`]
        Visits used per band.
    quantity : `str`
        ``"probability"`` or ``"area"``, from the localization's provenance.
    is_probability : `bool`
        True when `fractions` are genuine probabilities.
    any_band : `float`
        Fraction covered by at least one visit in any band.
    nside : `int`
        Resolution the calculation ran at.
    fov_radius_deg : `float`
        Field-of-view radius used.
    epochs : `dict`
        Optional per-epoch breakdown, filled by `band_coverage_by_epoch`.
    """

    fractions: dict[str, float]
    n_visits: dict[str, int]
    quantity: str
    is_probability: bool
    any_band: float
    nside: int
    fov_radius_deg: float = FOV_RADIUS_DEG
    epochs: dict[int, dict[str, float]] = field(default_factory=dict)

    def as_percent(self) -> dict[str, float]:
        """Coverage per band expressed as a percentage."""
        return {band: 100.0 * value for band, value in self.fractions.items()}

    def summary_line(self) -> str:
        """One-line summary, e.g. ``"g 41%, r 62%, i 100%"``."""
        parts = [f"{b} {100 * v:.0f}%" for b, v in self.fractions.items() if v > 0]
        return ", ".join(parts) if parts else "no coverage"


def too_visits(visits: pd.DataFrame, too_id: str | None = None) -> pd.DataFrame:
    """Select the ToO follow-up visits from a simulation output.

    Parameters
    ----------
    visits : `pandas.DataFrame`
        Simulated observations, as written by ``sim_runner``.
    too_id : `str`, optional
        Restrict to one ToO id, matching the ``_{id}_i{epoch}`` part of
        ``observation_reason``.

    Returns
    -------
    selected : `pandas.DataFrame`
        With an added integer ``too_epoch`` column.
    """
    if "observation_reason" not in visits.columns:
        raise KeyError(f"Visit table has no 'observation_reason' column; have {list(visits.columns)}")
    reason = visits["observation_reason"].astype(str)
    mask = reason.str.startswith("too_")
    if too_id is not None:
        mask &= reason.str.contains(f"_{too_id}_i", regex=False)
    selected = visits.loc[mask].copy()
    epochs = []
    for value in selected["observation_reason"].astype(str):
        match = _REASON_RE.match(value)
        epochs.append(int(match.group("epoch")) if match else -1)
    selected["too_epoch"] = epochs
    logger.info("Selected %d ToO visits out of %d", len(selected), len(visits))
    return selected


def detect_pointing_columns(visits: pd.DataFrame) -> tuple[str, str, bool]:
    """Identify the boresight columns and their units.

    Parameters
    ----------
    visits : `pandas.DataFrame`
        Visit table from any of the supported writers.

    Returns
    -------
    ra_col, dec_col : `str`
        Column names.
    radians : `bool`
        True when the values are in radians.

    Raises
    ------
    KeyError
        When no known convention is present.
    """
    for ra_col, dec_col, radians in _POINTING_CONVENTIONS:
        if ra_col in visits.columns and dec_col in visits.columns:
            logger.debug("Using pointing columns %s/%s (radians=%s)", ra_col, dec_col, radians)
            return ra_col, dec_col, radians
    known = ", ".join(f"{r}/{d}" for r, d, _ in _POINTING_CONVENTIONS)
    raise KeyError(
        f"Visit table has no recognized pointing columns (expected one of {known}); "
        f"got {sorted(visits.columns)[:20]}..."
    )


def detect_time_column(visits: pd.DataFrame) -> str | None:
    """Identify the visit time column, or None if there is not one."""
    for name in _TIME_COLUMNS:
        if name in visits.columns:
            return name
    return None


def _pointings(
    visits: pd.DataFrame,
    ra_col: str,
    dec_col: str,
    radians: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract finite pointings in degrees."""
    if ra_col not in visits.columns or dec_col not in visits.columns:
        raise KeyError(f"Visit table is missing {ra_col!r}/{dec_col!r}; have {list(visits.columns)}")
    ra = np.asarray(visits[ra_col], dtype=float)
    dec = np.asarray(visits[dec_col], dtype=float)
    if radians:
        ra, dec = np.degrees(ra), np.degrees(dec)
    good = np.isfinite(ra) & np.isfinite(dec)
    if good.sum() != ra.size:
        logger.warning("Ignoring %d visits with non-finite pointings", int(ra.size - good.sum()))
    return ra[good], dec[good]


def _accumulate(
    prob_map: np.ndarray,
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    nside: int,
    radius_rad: float,
) -> tuple[float, list[float]]:
    """Sum weight under a sequence of pointings, zeroing as it goes.

    Returns the total and the per-visit increments, so a cumulative curve can
    be built without repeating the work.
    """
    working = copy.deepcopy(prob_map)
    increments = []
    total = 0.0
    for ra, dec in zip(ra_deg, dec_deg):
        idx = hp.query_disc(nside, hp.ang2vec(ra, dec, lonlat=True), radius_rad, inclusive=True)
        gained = float(working[idx].sum())
        working[idx] = 0.0
        total += gained
        increments.append(gained)
    return total, increments


def band_coverage(
    visits: pd.DataFrame,
    localization: Localization,
    fov_radius_deg: float = FOV_RADIUS_DEG,
    bands: tuple[str, ...] = BANDS,
    band_col: str = "band",
    ra_col: str | None = None,
    dec_col: str | None = None,
    radians: bool | None = None,
) -> CoverageResult:
    """Fraction of a localization covered per band.

    Parameters
    ----------
    visits : `pandas.DataFrame`
        Visits to account for. Pass the output of `too_visits` to restrict to
        the ToO follow-up rather than the whole survey.
    localization : `Localization`
        Localization to integrate. Its ``is_probability`` flag determines
        whether the result is a probability or an area fraction.
    fov_radius_deg : `float`
        Field-of-view radius painted around each pointing.
    bands : `tuple` [`str`]
        Bands to report, always all of them by default.
    band_col : `str`
        Band column name.
    ra_col, dec_col : `str`, optional
        Pointing column names. Detected from the table when omitted, which
        handles both the raw ``sim_runner`` and the opsim conventions.
    radians : `bool`, optional
        Whether RA/Dec are in radians. Detected alongside the column names when
        omitted.

    Returns
    -------
    coverage : `CoverageResult`
    """
    prob_map = np.asarray(localization.prob_map, dtype=float)
    nside = localization.nside
    radius = np.radians(fov_radius_deg)

    if ra_col is None or dec_col is None:
        detected_ra, detected_dec, detected_radians = detect_pointing_columns(visits)
        ra_col = ra_col or detected_ra
        dec_col = dec_col or detected_dec
        if radians is None:
            radians = detected_radians
    elif radians is None:
        radians = True

    if band_col not in visits.columns:
        raise KeyError(f"Visit table has no {band_col!r} column; have {list(visits.columns)}")
    band_values = visits[band_col].astype(str)

    fractions: dict[str, float] = {}
    counts: dict[str, int] = {}
    for band in bands:
        rows = visits.loc[band_values == band]
        if rows.empty:
            fractions[band] = 0.0
            counts[band] = 0
            continue
        ra, dec = _pointings(rows, ra_col, dec_col, radians)
        total, _ = _accumulate(prob_map, ra, dec, nside, radius)
        fractions[band] = float(total)
        counts[band] = int(ra.size)

    # "Any band" needs a single pass over all visits so overlap between bands
    # is not counted twice.
    ra_all, dec_all = _pointings(visits, ra_col, dec_col, radians)
    any_band, _ = _accumulate(prob_map, ra_all, dec_all, nside, radius)

    result = CoverageResult(
        fractions=fractions,
        n_visits=counts,
        quantity=localization.quantity_name,
        is_probability=localization.is_probability,
        any_band=float(any_band),
        nside=nside,
        fov_radius_deg=fov_radius_deg,
    )
    logger.info(
        "Coverage (%s): %s; any band %.1f%%",
        result.quantity,
        result.summary_line(),
        100 * result.any_band,
    )
    return result


def band_coverage_by_epoch(
    visits: pd.DataFrame,
    localization: Localization,
    **kwargs,
) -> CoverageResult:
    """Per-band coverage plus a breakdown by follow-up epoch.

    Requires the ``too_epoch`` column added by `too_visits`.

    Parameters
    ----------
    visits : `pandas.DataFrame`
        Output of `too_visits`.
    localization : `Localization`
        Localization to integrate.
    **kwargs
        Forwarded to `band_coverage`.

    Returns
    -------
    coverage : `CoverageResult`
        With `CoverageResult.epochs` populated.
    """
    result = band_coverage(visits, localization, **kwargs)
    if "too_epoch" not in visits.columns:
        logger.warning("No 'too_epoch' column; skipping the per-epoch breakdown")
        return result
    for epoch in sorted(e for e in visits["too_epoch"].unique() if e >= 0):
        subset = visits.loc[visits["too_epoch"] == epoch]
        per_epoch = band_coverage(subset, localization, **kwargs)
        result.epochs[int(epoch)] = per_epoch.fractions
    return result


def coverage_curve(
    visits: pd.DataFrame,
    localization: Localization,
    fov_radius_deg: float = FOV_RADIUS_DEG,
    bands: tuple[str, ...] = BANDS,
    band_col: str = "band",
    ra_col: str | None = None,
    dec_col: str | None = None,
    time_col: str | None = None,
    radians: bool | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Cumulative coverage against time, per band.

    Parameters
    ----------
    visits : `pandas.DataFrame`
        Visits to account for.
    localization : `Localization`
        Localization to integrate.
    fov_radius_deg : `float`
        Field-of-view radius.
    bands : `tuple` [`str`]
        Bands to compute.
    band_col : `str`
        Band column name.
    ra_col, dec_col, time_col : `str`, optional
        Column names, detected from the table when omitted.
    radians : `bool`, optional
        Whether RA/Dec are in radians, detected when omitted.

    Returns
    -------
    curves : `dict` [`str`, `tuple`]
        ``band -> (mjd, cumulative_fraction)``. Bands with no visits are
        omitted, since there is no curve to draw.
    """
    prob_map = np.asarray(localization.prob_map, dtype=float)
    nside = localization.nside
    radius = np.radians(fov_radius_deg)

    if ra_col is None or dec_col is None:
        detected_ra, detected_dec, detected_radians = detect_pointing_columns(visits)
        ra_col = ra_col or detected_ra
        dec_col = dec_col or detected_dec
        if radians is None:
            radians = detected_radians
    elif radians is None:
        radians = True
    if time_col is None:
        time_col = detect_time_column(visits)

    band_values = visits[band_col].astype(str)

    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for band in bands:
        rows = visits.loc[band_values == band]
        if rows.empty:
            continue
        rows = rows.sort_values(time_col) if time_col is not None else rows
        ra, dec = _pointings(rows, ra_col, dec_col, radians)
        if ra.size == 0:
            continue
        _, increments = _accumulate(prob_map, ra, dec, nside, radius)
        times = (
            np.asarray(rows[time_col], dtype=float)[: len(increments)]
            if time_col is not None
            else np.arange(len(increments), dtype=float)
        )
        curves[band] = (times, np.cumsum(increments))
    return curves
