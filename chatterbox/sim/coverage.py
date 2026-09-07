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
    "CampaignTime",
    "CoverageComparison",
    "CoverageResult",
    "NightlyCoverage",
    "any_band_curve",
    "campaign_time",
    "compare_coverage",
    "too_epoch_of",
    "too_mask",
    "too_visits",
    "survey_visits",
    "band_coverage",
    "band_coverage_by_epoch",
    "coverage_by_night",
    "coverage_curve",
    "detect_pointing_columns",
    "detect_time_column",
    "night_label",
    "nights_since_trigger",
    "visits_overlapping",
    "nightly_visit_maps",
]

logger = logging.getLogger(__name__)

#: LSST bands, in canonical order. Iterated explicitly so unobserved bands
#: report zero coverage instead of being omitted.
BANDS = ("u", "g", "r", "i", "z", "y")

#: LSSTCam field-of-view radius used to approximate each visit's footprint.
#: This ignores the real focal-plane geometry and chip gaps.
FOV_RADIUS_DEG = 1.75

#: ToO visits carry two independent marks, both written by
#: ``ToOScriptedSurvey._new_event``:
#:
#: * ``observation_reason = too_{target_name_base}_{too_id}_i{epoch}``
#: * ``scheduler_note = "{survey_name}, {band}, {too_id}, i{epoch}"``, where
#:   every ToO survey in ``ts_fbs_utils`` names itself ``"ToO, ..."``.
#:
#: Either alone identifies the follow-up, and both are checked: a rename
#: upstream then costs a warning rather than a silently empty selection.
#: ``target_name`` is *not* usable -- it is a comma-joined list of every survey
#: that claimed the field.
#:
#: Passing ``observation_reason`` to the survey constructor appends a
#: ``TrackingInfoDetailer``, which only fills fields that are still empty, so
#: the per-observation value above -- epoch suffix included -- survives.
_REASON_RE = re.compile(r"^too_(?P<base>.+)_i(?P<epoch>\d+)$")

#: Loose test for the ToO prefix of ``observation_reason``, which the epoch
#: regex above is too strict for: a survey configured with a constant reason
#: has no ``_i{epoch}`` suffix but is still follow-up.
_REASON_PREFIX_RE = re.compile(r"^too_", re.IGNORECASE)

#: ``ToO`` as a whole word in ``scheduler_note``, so ``tooth_survey`` does not
#: match. Anchored on non-letters rather than ``\b`` because the note's
#: separators are commas and spaces.
_NOTE_RE = re.compile(r"(?:^|[^A-Za-z])too(?:[^A-Za-z]|$)", re.IGNORECASE)

#: Trailing ``i{epoch}`` of either field. ``scheduler_note`` is a 60-character
#: field, so a long survey name plus a long ToO id can truncate this away --
#: which is why ``observation_reason`` is read first.
_EPOCH_RE = re.compile(r"i(\d+)\s*$")

#: Columns a ToO visit can be recognised by, in the order they are trusted.
_TAG_COLUMNS = ("observation_reason", "scheduler_note")

#: Per-visit time columns. ``visittime`` is the on-sky visit (exposures plus
#: readout); ``slewtime`` is what the survey paid to get there, which a ToO
#: interruption is directly responsible for. Both are in the
#: ``ObservationArray`` dtype.
_VISIT_TIME_COLUMNS = ("visittime", "slewtime")

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


def nights_since_trigger(nights, reference_night: int | None) -> list[int]:
    """Convert survey night indices to nights since the trigger.

    ``night`` in a simulation output is `rubin_scheduler`'s *survey* night
    counter, which counts from ``SURVEY_START_MJD`` and so has no relation to
    when the alert fired -- a November 2025 event lands on night 11 or 12
    whatever the follow-up looks like. Everything a reader cares about is
    relative: night 0 is the trigger night, night 1 the one after it. Negative
    values are real and worth seeing, since the simulation starts from the
    trigger's ``day_obs`` and can therefore include the night *before* the
    trigger, whose visits are nominal cadence rather than follow-up.

    Parameters
    ----------
    nights : sequence of `int`
        Survey night indices.
    reference_night : `int` or None
        Survey night the trigger fell on. None leaves the numbers alone, so a
        result from before this was recorded still renders.

    Returns
    -------
    relative : `list` [`int`]
    """
    if reference_night is None:
        return [int(n) for n in nights]
    return [int(n) - int(reference_night) for n in nights]


def night_label(night: int, reference_night: int | None) -> str:
    """Label one night: ``"+0"`` for the trigger night, ``"-1"`` before it.

    Falls back to the bare survey night when there is no reference, so the
    label is never silently ambiguous about which counting is in use.
    """
    if reference_night is None:
        return str(int(night))
    return f"{int(night) - int(reference_night):+d}"


@dataclass
class NightlyCoverage:
    """Localization coverage broken down by simulation night.

    Two views of the same run: how much each band *added* on each night, and
    how much had been covered in total by the end of it. The last cumulative
    value equals what `band_coverage` reports for the same visits, because both
    zero a pixel once it has been counted.

    Attributes
    ----------
    nights : `list` [`int`]
        Survey nights with visits, in order. Nights with none are absent.
    gained : `dict` [`str`, `list` [`float`]]
        ``band -> fraction added on each night``, parallel to `nights`. Bands
        with no visits anywhere are omitted.
    cumulative : `dict` [`str`, `list` [`float`]]
        ``band -> fraction covered through each night``.
    any_band_gained, any_band_cumulative : `list` [`float`]
        The same, counting a pixel once no matter which band reached it.
    quantity : `str`
        ``"probability"`` or ``"area"``, from the localization's provenance.
    is_probability : `bool`
        True when these are genuine probabilities.
    reference_night : `int` or None
        Survey night the trigger fell on, so `nights` can be reported relative
        to it. Everything user-facing uses the relative numbers.
    """

    nights: list[int] = field(default_factory=list)
    gained: dict[str, list[float]] = field(default_factory=dict)
    cumulative: dict[str, list[float]] = field(default_factory=dict)
    any_band_gained: list[float] = field(default_factory=list)
    any_band_cumulative: list[float] = field(default_factory=list)
    quantity: str = "area"
    is_probability: bool = False
    reference_night: int | None = None

    @property
    def relative_nights(self) -> list[int]:
        """`nights`, counted from the trigger rather than the survey start."""
        return nights_since_trigger(self.nights, self.reference_night)

    def labels(self) -> list[str]:
        """One label per night, e.g. ``["-1", "+0", "+1"]``."""
        return [night_label(night, self.reference_night) for night in self.nights]

    def as_dict(self) -> dict[str, dict[str, float]]:
        """``night -> band -> fraction gained``, with JSON-ready keys.

        Keyed by nights since the trigger, so ``"0"`` is the trigger night and
        ``"-1"`` the night before. Add ``trigger_night`` from the result to
        recover the survey night.
        """
        return {
            str(night): {band: values[n] for band, values in self.gained.items() if values[n] > 0}
            for n, night in enumerate(self.relative_nights)
        }

    def cumulative_any_band(self) -> dict[str, float]:
        """``nights since trigger -> total fraction covered through it``."""
        return {str(night): self.any_band_cumulative[n] for n, night in enumerate(self.relative_nights)}

    def summary_line(self) -> str:
        """One-line progression, e.g. ``"night +0: 13%, night +1: 28%"``."""
        parts = [
            f"night {label}: {100 * self.any_band_cumulative[n]:.0f}%"
            for n, label in enumerate(self.labels())
        ]
        return ", ".join(parts) if parts else "no nights with visits"


def too_mask(visits: pd.DataFrame, too_id: str | None = None) -> pd.Series:
    """Boolean mask of the ToO follow-up visits.

    Recognises a visit by ``observation_reason`` starting with ``too_`` *or*
    ``scheduler_note`` naming ``ToO`` -- see `_TAG_COLUMNS`. When both columns
    are present but disagree the union is used and the disagreement is logged,
    because that is the signal an upstream tag changed and it is otherwise
    invisible.

    Parameters
    ----------
    visits : `pandas.DataFrame`
        Simulated observations, as written by ``sim_runner``.
    too_id : `str`, optional
        Restrict to one ToO id, matching the ``_{id}_i{epoch}`` part of
        ``observation_reason`` or the ``, {id}, i{epoch}`` tail of
        ``scheduler_note``.

    Returns
    -------
    mask : `pandas.Series` of `bool`
        Indexed like `visits`.

    Raises
    ------
    KeyError
        When neither tag column is present, since then nothing can be said
        about which visits were follow-up.
    """
    present = [column for column in _TAG_COLUMNS if column in visits.columns]
    if not present:
        raise KeyError(
            f"Visit table has none of {list(_TAG_COLUMNS)}, so ToO visits cannot be "
            f"identified; have {list(visits.columns)}"
        )

    masks: dict[str, pd.Series] = {}
    if "observation_reason" in present:
        reason = visits["observation_reason"].astype(str)
        masks["observation_reason"] = reason.str.match(_REASON_PREFIX_RE)
    if "scheduler_note" in present:
        note = visits["scheduler_note"].astype(str)
        masks["scheduler_note"] = note.str.contains(_NOTE_RE)

    mask = pd.Series(False, index=visits.index)
    for column_mask in masks.values():
        mask |= column_mask.fillna(False)

    if len(masks) == 2:
        left, right = masks["observation_reason"], masks["scheduler_note"]
        disagreements = int((left != right).sum())
        if disagreements:
            logger.warning(
                "%d visit(s) are tagged ToO by one field but not the other "
                "(observation_reason: %d, scheduler_note: %d); taking the union. "
                "An upstream tag may have been renamed",
                disagreements,
                int(left.sum()),
                int(right.sum()),
            )

    if too_id is not None:
        wanted = pd.Series(False, index=visits.index)
        if "observation_reason" in present:
            wanted |= visits["observation_reason"].astype(str).str.contains(f"_{too_id}_i", regex=False)
        if "scheduler_note" in present:
            wanted |= visits["scheduler_note"].astype(str).str.contains(f", {too_id}, i", regex=False)
        mask &= wanted

    return mask


def too_epoch_of(reason: str, note: str = "") -> int:
    """Which follow-up epoch a visit belongs to, or ``-1`` if unknown.

    ``observation_reason`` is read first and ``scheduler_note`` used as a
    fallback, so an epoch is still recovered when a survey was configured with
    a constant reason that carries no ``_i{epoch}`` suffix.
    """
    match = _REASON_RE.match(str(reason))
    if match is not None:
        return int(match.group("epoch"))
    for value in (reason, note):
        found = _EPOCH_RE.search(str(value))
        if found is not None:
            return int(found.group(1))
    return -1


def too_visits(visits: pd.DataFrame, too_id: str | None = None) -> pd.DataFrame:
    """Select the ToO follow-up visits from a simulation output.

    Parameters
    ----------
    visits : `pandas.DataFrame`
        Simulated observations, as written by ``sim_runner``.
    too_id : `str`, optional
        Restrict to one ToO id.

    Returns
    -------
    selected : `pandas.DataFrame`
        With an added integer ``too_epoch`` column, ``-1`` where no epoch could
        be read from either tag.

    See Also
    --------
    survey_visits : the complement, everything the nominal survey did.
    """
    mask = too_mask(visits, too_id=too_id)
    selected = visits.loc[mask].copy()
    reasons = (
        selected["observation_reason"].astype(str)
        if "observation_reason" in selected.columns
        else pd.Series("", index=selected.index)
    )
    notes = (
        selected["scheduler_note"].astype(str)
        if "scheduler_note" in selected.columns
        else pd.Series("", index=selected.index)
    )
    selected["too_epoch"] = [too_epoch_of(r, n) for r, n in zip(reasons, notes)]
    unknown = int((selected["too_epoch"] < 0).sum())
    if unknown:
        logger.warning(
            "%d of %d ToO visits carry no epoch in either tag; they are reported "
            "outside the per-epoch breakdown",
            unknown,
            len(selected),
        )
    logger.info("Selected %d ToO visits out of %d", len(selected), len(visits))
    return selected


def survey_visits(visits: pd.DataFrame) -> pd.DataFrame:
    """Select everything that is *not* tagged ToO follow-up.

    The nominal-cadence visits. These cover real localization area too, which
    is why the coverage a trigger actually receives is measured from the union
    rather than from the follow-up alone; see `compare_coverage`.

    Parameters
    ----------
    visits : `pandas.DataFrame`
        Simulated observations.

    Returns
    -------
    selected : `pandas.DataFrame`
    """
    selected = visits.loc[~too_mask(visits)].copy()
    logger.info("Selected %d nominal-survey visits out of %d", len(selected), len(visits))
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


def _paint(
    working: np.ndarray,
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    nside: int,
    radius_rad: float,
) -> list[float]:
    """Sum weight under a sequence of pointings, zeroing `working` as it goes.

    The map is modified in place, so successive calls with the same map
    continue where the last one stopped -- which is what makes a night-by-night
    breakdown add up to the same total as one pass over every visit.
    """
    increments = []
    for ra, dec in zip(ra_deg, dec_deg):
        idx = hp.query_disc(nside, hp.ang2vec(ra, dec, lonlat=True), radius_rad, inclusive=True)
        gained = float(working[idx].sum())
        working[idx] = 0.0
        increments.append(gained)
    return increments


def _accumulate(
    prob_map: np.ndarray,
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    nside: int,
    radius_rad: float,
) -> tuple[float, list[float]]:
    """Sum weight under a sequence of pointings against a fresh copy.

    Returns the total and the per-visit increments, so a cumulative curve can
    be built without repeating the work.
    """
    increments = _paint(copy.deepcopy(prob_map), ra_deg, dec_deg, nside, radius_rad)
    return float(sum(increments)), increments


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


@dataclass
class CoverageComparison:
    """ToO-only coverage beside the coverage the localization actually got.

    Attributes
    ----------
    too : `CoverageResult`
        From the tagged follow-up visits alone. This is the number that
        measures the *strategy*.
    combined : `CoverageResult`
        From the follow-up plus every nominal-cadence visit that landed on the
        localization. This is what the event actually received.
    survey_gain : `dict` [`str`, `float`]
        ``combined - too`` per band, and under the key ``"any"`` for the
        any-band totals.
    too_n_visits, survey_n_visits : `int`
        Visits behind each side, counted within the same overlapping set.

    Notes
    -----
    `combined` is **not** ``too`` plus the survey's own coverage: survey visits
    re-cover pixels the follow-up already reached, so only the difference
    `survey_gain` is new. That difference is the honest statement of what the
    nominal cadence added, and the number to weigh a strategy against.
    """

    too: CoverageResult
    combined: CoverageResult
    survey_gain: dict[str, float] = field(default_factory=dict)
    too_n_visits: int = 0
    survey_n_visits: int = 0
    #: The visits `combined` was measured from, so a caller drawing a curve of
    #: the same quantity uses the same rows rather than rebuilding the set and
    #: risking a curve that disagrees with the total.
    combined_visits: pd.DataFrame | None = field(default=None, repr=False, compare=False)

    def summary_line(self) -> str:
        """One line contrasting the two any-band totals."""
        return (
            f"ToO only {100 * self.too.any_band:.1f}%, "
            f"with the nominal survey {100 * self.combined.any_band:.1f}% "
            f"({100 * self.survey_gain.get('any', 0.0):+.1f} points)"
        )


@dataclass
class CampaignTime:
    """Observing time the ToO campaign consumed.

    Attributes
    ----------
    too_seconds : `float`
        Total time attributable to the follow-up: on-sky visit time plus the
        slew to each pointing.
    too_open_shutter_s : `float`
        Exposure time alone, without slew or readout.
    all_seconds : `float`
        The same measure over every simulated visit on the nights the campaign
        touched, which is the denominator of `fraction`.
    fraction : `float`
        ``too_seconds / all_seconds``, 0-1. The share of observing time the
        campaign took from the survey on those nights.
    n_too_visits, n_all_visits : `int`
        Visit counts behind those totals.
    nights : `int`
        Nights the campaign placed visits on.
    span_hours : `float`
        First to last follow-up visit.
    by_night : `dict` [`str`, `dict`]
        ``nights since trigger -> {"too_s", "all_s", "n_too", "fraction"}``,
        keyed like `NightlyCoverage.as_dict`.
    by_band, by_epoch : `dict` [`str`, `float`]
        Follow-up seconds per band and per epoch.
    measure : `str`
        Which columns the totals came from, so a fallback is never mistaken for
        the full accounting.
    """

    too_seconds: float = 0.0
    too_open_shutter_s: float = 0.0
    all_seconds: float = 0.0
    fraction: float = 0.0
    n_too_visits: int = 0
    n_all_visits: int = 0
    nights: int = 0
    span_hours: float = 0.0
    by_night: dict[str, dict[str, float]] = field(default_factory=dict)
    by_band: dict[str, float] = field(default_factory=dict)
    by_epoch: dict[str, float] = field(default_factory=dict)
    measure: str = "visittime + slewtime"

    def summary_line(self) -> str:
        """One line, e.g. ``"38 min over 3 nights -- 7.4% of the time"``."""
        return (
            f"{self.n_too_visits:,} visits, {self.too_seconds / 60.0:.0f} min "
            f"over {self.nights} night{'s' if self.nights != 1 else ''} "
            f"-- {100 * self.fraction:.1f}% of the observing time on those nights"
        )


def compare_coverage(
    visits: pd.DataFrame,
    localization: Localization,
    fov_radius_deg: float = FOV_RADIUS_DEG,
    nside: int | None = None,
    overlapping: pd.DataFrame | None = None,
    **kwargs,
) -> CoverageComparison:
    """Measure the localization's coverage with and without the survey.

    Parameters
    ----------
    visits : `pandas.DataFrame`
        *All* simulated visits, follow-up and nominal cadence alike.
    localization : `Localization`
        Localization to integrate.
    fov_radius_deg : `float`
        Field-of-view radius, used both for the overlap test and the painting.
    nside : `int`, optional
        Resolution of the overlap pre-selection. Defaults to the
        localization's.
    overlapping : `pandas.DataFrame`, optional
        The result of `visits_overlapping` for these visits, when a caller has
        already computed it. Saves repeating the dilation.
    **kwargs
        Forwarded to `band_coverage` and `band_coverage_by_epoch`.

    Returns
    -------
    comparison : `CoverageComparison`

    Notes
    -----
    The nominal-cadence side is pre-selected with `visits_overlapping` so the
    extra pass stays proportional to the localization rather than to the
    survey. The follow-up is then added back unconditionally, because that
    pre-selection is *not* exact for the follow-up: it tests the boresight
    against the credible region dilated by the field of view, while coverage is
    integrated over the whole map -- and a real skymap has faint probability
    outside its own 90% contour. A follow-up pointing just outside the contour
    still collects weight, so dropping it would put the combined number *below*
    the ToO-only number.
    """
    follow_up = too_visits(visits)
    too = band_coverage_by_epoch(follow_up, localization, fov_radius_deg=fov_radius_deg, **kwargs)

    if overlapping is None:
        overlapping = visits_overlapping(
            visits, localization, fov_radius_deg=fov_radius_deg, nside=nside or localization.nside
        )
    # Every follow-up visit, plus the nominal-cadence ones that can reach the
    # localization. Taking the union rather than the pre-selection alone is
    # what makes `combined >= too` hold by construction; see the note above.
    nominal = overlapping.loc[~too_mask(overlapping)] if not overlapping.empty else overlapping
    survey_n = int(len(nominal))
    combined_rows = pd.concat([follow_up, nominal]) if survey_n else follow_up
    if survey_n:
        combined = band_coverage(combined_rows, localization, fov_radius_deg=fov_radius_deg, **kwargs)
    else:
        combined = too

    gain = {band: combined.fractions.get(band, 0.0) - too.fractions.get(band, 0.0) for band in BANDS}
    gain["any"] = combined.any_band - too.any_band
    # Painting is monotonic in the visit set, so a negative gain is impossible
    # and means the two sides were measured against different maps.
    negative = {band: value for band, value in gain.items() if value < -1e-9}
    if negative:
        logger.warning(
            "Combined coverage is below ToO-only coverage for %s, which should be "
            "impossible; the two were measured inconsistently",
            ", ".join(sorted(negative)),
        )

    comparison = CoverageComparison(
        too=too,
        combined=combined,
        survey_gain=gain,
        too_n_visits=int(len(follow_up)),
        survey_n_visits=survey_n,
        combined_visits=combined_rows,
    )
    logger.info("Coverage (%s): %s", too.quantity, comparison.summary_line())
    return comparison


def campaign_time(
    visits: pd.DataFrame,
    reference_night: int | None = None,
    night_col: str = "night",
    band_col: str = "band",
    time_col: str | None = None,
) -> CampaignTime:
    """Observing time the ToO follow-up took, and what share of the night.

    Parameters
    ----------
    visits : `pandas.DataFrame`
        *All* simulated visits. The denominator is every visit on the nights
        the campaign touched, so the follow-up alone is not enough.
    reference_night : `int`, optional
        Survey night the trigger fell on, so `CampaignTime.by_night` is keyed
        by nights since the trigger.
    night_col, band_col : `str`
        Column names.
    time_col : `str`, optional
        Time column for the campaign span, detected when omitted.

    Returns
    -------
    timing : `CampaignTime`

    Notes
    -----
    Cost is ``visittime + slewtime``: the visit itself plus the slew that
    reaching the target required, since an interruption charges the survey for
    both. When those columns are absent -- an opsim export need not carry
    ``slewtime`` -- the fallback is ``exptime * nexp``, and
    `CampaignTime.measure` says which was used.
    """
    # A positional index, so a caller that concatenated frames without
    # ``ignore_index`` cannot make two visits share a label -- which would
    # mis-attribute time between the follow-up and the survey.
    visits = visits.reset_index(drop=True)
    follow_up = too_visits(visits)
    if follow_up.empty:
        logger.info("No ToO visits, so no campaign time to account for")
        return CampaignTime()

    cost, measure = _visit_seconds(visits)
    shutter, _ = _visit_seconds(visits, open_shutter_only=True)
    too_rows = too_mask(visits)

    nights = (
        pd.to_numeric(visits[night_col], errors="coerce")
        if night_col in visits.columns
        else pd.Series(np.nan, index=visits.index)
    )
    campaign_nights = sorted({int(n) for n in nights[too_rows].dropna().unique()})

    # Only the campaign's own nights are in the denominator: including nights
    # it never touched would dilute the share towards zero and say nothing
    # about the trade the scheduler actually made.
    on_campaign_nights = (
        nights.isin(campaign_nights) if campaign_nights else pd.Series(True, index=visits.index)
    )
    too_seconds = float(cost[too_rows].sum())
    all_seconds = float(cost[on_campaign_nights].sum())

    by_night: dict[str, dict[str, float]] = {}
    for night in campaign_nights:
        here = nights == night
        night_too = float(cost[here & too_rows].sum())
        night_all = float(cost[here].sum())
        label = str(nights_since_trigger([night], reference_night)[0])
        by_night[label] = {
            "too_s": night_too,
            "all_s": night_all,
            "n_too": int((here & too_rows).sum()),
            "fraction": night_too / night_all if night_all > 0 else 0.0,
        }

    by_band: dict[str, float] = {}
    if band_col in follow_up.columns:
        bands = follow_up[band_col].astype(str)
        for band in BANDS:
            seconds = float(cost[follow_up.index[bands == band]].sum())
            if seconds > 0:
                by_band[band] = seconds

    by_epoch: dict[str, float] = {}
    if "too_epoch" in follow_up.columns:
        for epoch in sorted(e for e in follow_up["too_epoch"].unique() if e >= 0):
            by_epoch[str(int(epoch))] = float(cost[follow_up.index[follow_up["too_epoch"] == epoch]].sum())

    span_hours = 0.0
    time_col = time_col or detect_time_column(visits)
    if time_col is not None and time_col in follow_up.columns:
        times = pd.to_numeric(follow_up[time_col], errors="coerce").dropna()
        if len(times):
            span_hours = float(times.max() - times.min()) * 24.0

    timing = CampaignTime(
        too_seconds=too_seconds,
        too_open_shutter_s=float(shutter[too_rows].sum()),
        all_seconds=all_seconds,
        fraction=too_seconds / all_seconds if all_seconds > 0 else 0.0,
        n_too_visits=int(len(follow_up)),
        n_all_visits=int(on_campaign_nights.sum()),
        nights=len(campaign_nights),
        span_hours=span_hours,
        by_night=by_night,
        by_band=by_band,
        by_epoch=by_epoch,
        measure=measure,
    )
    logger.info("Campaign time (%s): %s", measure, timing.summary_line())
    return timing


def _visit_seconds(visits: pd.DataFrame, open_shutter_only: bool = False) -> tuple[pd.Series, str]:
    """Seconds attributable to each visit, and which columns they came from.

    Returns
    -------
    seconds : `pandas.Series`
        Indexed like `visits`.
    measure : `str`
        Human-readable description of the columns used.
    """
    zero = pd.Series(0.0, index=visits.index)

    def column(name: str) -> pd.Series:
        if name not in visits.columns:
            return zero
        return pd.to_numeric(visits[name], errors="coerce").fillna(0.0)

    # exptime is already the total on-sky exposure of the visit in the
    # scheduler's accounting, so nexp must not multiply it -- that would double
    # every two-snap visit.
    exposure = column("exptime")
    if open_shutter_only:
        return exposure, "exptime"

    present = [name for name in _VISIT_TIME_COLUMNS if name in visits.columns]
    if not present:
        logger.warning(
            "Visit table has neither %s; falling back to exptime, which omits " "readout and slew overheads",
            " nor ".join(_VISIT_TIME_COLUMNS),
        )
        return exposure, "exptime (no visittime or slewtime column)"

    total = zero.copy()
    for name in present:
        total = total + column(name)
    if len(present) < len(_VISIT_TIME_COLUMNS):
        logger.info("Campaign time from %s only", " + ".join(present))
    return total, " + ".join(present)


def any_band_curve(
    visits: pd.DataFrame,
    localization: Localization,
    fov_radius_deg: float = FOV_RADIUS_DEG,
    ra_col: str | None = None,
    dec_col: str | None = None,
    time_col: str | None = None,
    radians: bool | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative coverage against time, counting a pixel once in any band.

    The single-curve companion to `coverage_curve`, used to draw the ToO-only
    and ToO-plus-survey totals on one axis.

    Returns
    -------
    times : `numpy.ndarray`
        MJD of each visit, in order.
    cumulative : `numpy.ndarray`
        Fraction covered through that visit.
    """
    if visits.empty:
        return np.array([]), np.array([])

    if ra_col is None or dec_col is None:
        detected_ra, detected_dec, detected_radians = detect_pointing_columns(visits)
        ra_col = ra_col or detected_ra
        dec_col = dec_col or detected_dec
        if radians is None:
            radians = detected_radians
    elif radians is None:
        radians = True
    time_col = time_col or detect_time_column(visits)

    rows = visits.sort_values(time_col) if time_col is not None else visits
    ra, dec = _pointings(rows, ra_col, dec_col, radians)
    if ra.size == 0:
        return np.array([]), np.array([])
    _, increments = _accumulate(
        np.asarray(localization.prob_map, dtype=float),
        ra,
        dec,
        localization.nside,
        np.radians(fov_radius_deg),
    )
    times = (
        np.asarray(rows[time_col], dtype=float)[: len(increments)]
        if time_col is not None
        else np.arange(len(increments), dtype=float)
    )
    return times, np.cumsum(increments)


def coverage_by_night(
    visits: pd.DataFrame,
    localization: Localization,
    fov_radius_deg: float = FOV_RADIUS_DEG,
    bands: tuple[str, ...] = BANDS,
    band_col: str = "band",
    night_col: str = "night",
    ra_col: str | None = None,
    dec_col: str | None = None,
    radians: bool | None = None,
    reference_night: int | None = None,
) -> NightlyCoverage:
    """Coverage gained on each simulated night, per band.

    `coverage_curve` answers "how much, by when" against wall-clock time;
    this answers "how much did each night buy", on the same night index the
    per-night visit maps are labelled with. Both totals agree with
    `band_coverage` because all three zero a pixel once it is counted.

    Parameters
    ----------
    visits : `pandas.DataFrame`
        Visits to account for. Pass the output of `too_visits` so the totals
        match the coverage reported for the run.
    localization : `Localization`
        Localization to integrate.
    fov_radius_deg : `float`
        Field-of-view radius painted around each pointing.
    bands : `tuple` [`str`]
        Bands to consider.
    band_col, night_col : `str`
        Column names. ``night`` is the simulation's own night index.
    ra_col, dec_col, radians : optional
        Pointing columns and units, detected when omitted.
    reference_night : `int`, optional
        Survey night the trigger fell on, so the result can report nights
        relative to it. See `nights_since_trigger`.

    Returns
    -------
    nightly : `NightlyCoverage`
        Empty when no visit carries a usable night.
    """
    if night_col not in visits.columns:
        raise KeyError(f"Visit table has no {night_col!r} column; have {sorted(visits.columns)[:20]}...")

    if ra_col is None or dec_col is None:
        detected_ra, detected_dec, detected_radians = detect_pointing_columns(visits)
        ra_col = ra_col or detected_ra
        dec_col = dec_col or detected_dec
        if radians is None:
            radians = detected_radians
    elif radians is None:
        radians = True

    prob_map = np.asarray(localization.prob_map, dtype=float)
    nside = localization.nside
    radius = np.radians(fov_radius_deg)

    # A night that cannot be read as a number cannot be placed on the axis, so
    # those rows are dropped rather than lumped into some arbitrary night.
    night_values = pd.to_numeric(visits[night_col], errors="coerce")
    usable = visits.loc[night_values.notna()]
    if usable.empty:
        logger.warning("No visits carry a usable %r; skipping the per-night breakdown", night_col)
        return NightlyCoverage(
            quantity=localization.quantity_name,
            is_probability=localization.is_probability,
            reference_night=reference_night,
        )
    night_values = night_values.loc[usable.index].astype(int)
    nights = sorted(int(n) for n in night_values.unique())
    band_values = usable[band_col].astype(str) if band_col in usable.columns else None

    def walk(rows: pd.DataFrame, row_nights: pd.Series) -> tuple[list[float], list[float]]:
        """Gained and cumulative fractions over `nights`, in order."""
        working = copy.deepcopy(prob_map)
        gained, cumulative, running = [], [], 0.0
        for night in nights:
            subset = rows.loc[row_nights == night]
            night_total = 0.0
            if not subset.empty:
                ra, dec = _pointings(subset, ra_col, dec_col, radians)
                if ra.size:
                    night_total = float(sum(_paint(working, ra, dec, nside, radius)))
            running += night_total
            gained.append(night_total)
            cumulative.append(running)
        return gained, cumulative

    per_band_gained: dict[str, list[float]] = {}
    per_band_cumulative: dict[str, list[float]] = {}
    if band_values is not None:
        for band in bands:
            rows = usable.loc[band_values == band]
            if rows.empty:
                continue
            gained, cumulative = walk(rows, night_values.loc[rows.index])
            per_band_gained[band] = gained
            per_band_cumulative[band] = cumulative

    any_gained, any_cumulative = walk(usable, night_values)

    nightly = NightlyCoverage(
        nights=nights,
        gained=per_band_gained,
        cumulative=per_band_cumulative,
        any_band_gained=any_gained,
        any_band_cumulative=any_cumulative,
        quantity=localization.quantity_name,
        is_probability=localization.is_probability,
        reference_night=reference_night,
    )
    logger.info(
        "Coverage by night (%s): %s",
        nightly.quantity,
        nightly.summary_line(),
    )
    return nightly


def visits_overlapping(
    visits: pd.DataFrame,
    localization: Localization,
    fov_radius_deg: float = FOV_RADIUS_DEG,
    nside: int | None = None,
    ra_col: str | None = None,
    dec_col: str | None = None,
    radians: bool | None = None,
) -> pd.DataFrame:
    """Select visits whose field of view touches the localization.

    This is a wider net than `too_visits`: it catches nominal-cadence visits
    that happen to land on the localization as well as the tagged ToO
    follow-up, which is what you want when asking "what coverage does this
    localization actually get".

    Parameters
    ----------
    visits : `pandas.DataFrame`
        All simulated visits.
    localization : `Localization`
        Localization to test against. The region is the one its contour
        encloses, via `chatterbox.astro.skymap.localization_region`, so the
        selection agrees with what the plots show.
    fov_radius_deg : `float`
        Field-of-view radius. A visit counts as overlapping when its disc
        intersects the region.
    nside : `int`, optional
        Resolution for the overlap test. Defaults to the localization's.
    ra_col, dec_col, radians : optional
        Pointing columns and units, detected when omitted.

    Returns
    -------
    selected : `pandas.DataFrame`
        The overlapping rows, in input order.

    Notes
    -----
    Rather than a disc query per visit, the region is dilated once by the
    field-of-view radius and each visit is then a single pixel lookup: a visit
    overlaps exactly when its boresight falls inside the dilated region. That
    turns an O(visits x pixels) scan into one dilation plus O(visits).
    """
    if ra_col is None or dec_col is None:
        detected_ra, detected_dec, detected_radians = detect_pointing_columns(visits)
        ra_col = ra_col or detected_ra
        dec_col = dec_col or detected_dec
        if radians is None:
            radians = detected_radians
    elif radians is None:
        radians = True

    from ..astro.skymap import localization_region

    nside = nside or localization.nside
    region = np.flatnonzero(localization_region(localization, nside))
    if region.size == 0:
        logger.warning("Localization region is empty; no visit can overlap it")
        return visits.iloc[:0].copy()

    dilated = np.zeros(hp.nside2npix(nside), dtype=bool)
    radius = np.radians(fov_radius_deg)
    # pix2vec returns (x, y, z) arrays; transpose to one vector per pixel.
    for vec in np.array(hp.pix2vec(nside, region)).T:
        dilated[hp.query_disc(nside, vec, radius, inclusive=True)] = True

    ra = np.asarray(visits[ra_col], dtype=float)
    dec = np.asarray(visits[dec_col], dtype=float)
    if radians:
        ra, dec = np.degrees(ra), np.degrees(dec)
    finite = np.isfinite(ra) & np.isfinite(dec)

    overlaps = np.zeros(len(visits), dtype=bool)
    pixels = hp.ang2pix(nside, ra[finite], dec[finite], lonlat=True)
    overlaps[finite] = dilated[pixels]

    selected = visits.loc[overlaps].copy()
    logger.info(
        "%d of %d visits overlap the localization (FOV %.2f deg, nside %d)",
        len(selected),
        len(visits),
        fov_radius_deg,
        nside,
    )
    return selected


def nightly_visit_maps(
    visits: pd.DataFrame,
    nside: int = 256,
    fov_radius_deg: float = FOV_RADIUS_DEG,
    bands: tuple[str, ...] = BANDS,
    band_col: str = "band",
    night_col: str = "night",
    ra_col: str | None = None,
    dec_col: str | None = None,
    radians: bool | None = None,
) -> dict[int, dict[str, np.ndarray]]:
    """Per-night, per-band visit-count maps.

    Unlike `band_coverage`, which zeroes pixels to avoid double counting, these
    are plain counts: a pixel visited three times in a night reads 3. That is
    what makes them useful to look at -- depth of coverage, not just extent.

    Parameters
    ----------
    visits : `pandas.DataFrame`
        Visits to map, normally the output of `too_visits`.
    nside : `int`
        Resolution of the output maps. Kept modest by default because these are
        for plotting, not accounting.
    fov_radius_deg : `float`
        Field-of-view radius painted around each pointing.
    bands : `tuple` [`str`]
        Bands to consider.
    band_col, night_col : `str`
        Column names. ``night`` is the simulation's own night index;
        ``day_obs`` is present but unpopulated for simulated visits, so it
        cannot be used.
    ra_col, dec_col, radians : optional
        Pointing columns and units, detected when omitted.

    Returns
    -------
    maps : `dict` [`int`, `dict` [`str`, `numpy.ndarray`]]
        ``night -> band -> counts`` in RING ordering. Only nights and bands
        with visits appear.
    """
    if night_col not in visits.columns:
        raise KeyError(f"Visit table has no {night_col!r} column; have {sorted(visits.columns)[:20]}...")

    if ra_col is None or dec_col is None:
        detected_ra, detected_dec, detected_radians = detect_pointing_columns(visits)
        ra_col = ra_col or detected_ra
        dec_col = dec_col or detected_dec
        if radians is None:
            radians = detected_radians
    elif radians is None:
        radians = True

    npix = hp.nside2npix(nside)
    radius = np.radians(fov_radius_deg)
    band_values = visits[band_col].astype(str)

    result: dict[int, dict[str, np.ndarray]] = {}
    for night, night_rows in visits.groupby(night_col):
        per_band: dict[str, np.ndarray] = {}
        for band in bands:
            rows = night_rows.loc[band_values.loc[night_rows.index] == band]
            if rows.empty:
                continue
            ra, dec = _pointings(rows, ra_col, dec_col, radians)
            if ra.size == 0:
                continue
            counts = np.zeros(npix, dtype=np.int32)
            for one_ra, one_dec in zip(ra, dec):
                idx = hp.query_disc(nside, hp.ang2vec(one_ra, one_dec, lonlat=True), radius, inclusive=True)
                counts[idx] += 1
            per_band[band] = counts
        if per_band:
            result[int(night)] = per_band

    logger.info(
        "Built visit-count maps for %d night(s) at nside %d",
        len(result),
        nside,
    )
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
