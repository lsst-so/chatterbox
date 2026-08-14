"""Keep a fresh copy of the visit history the simulation starts from.

The scheduler needs the visits already taken before the simulated night, so
that a ToO is scheduled on top of a realistic survey state rather than a blank
sky. ``lsst_survey_sim.simulate_lsst.fetch_previous_visits`` is the tool that
gets them from ConsDB; this module wraps it in an age-aware cache so nobody has
to hand-maintain an opsim file.

The cache is refreshed when it is missing, older than
``sim.opsim_max_age_hours`` (once a night by default), or when it was fetched
for an earlier ``day_obs`` than the one being simulated -- that last case
matters because ``fetch_previous_visits`` returns visits strictly *before* the
day it is asked for, so a cache built for an earlier night is missing data.

If a refresh fails but a cache exists, the cached visits are used and the
staleness is reported rather than failing the simulation: an out-of-date survey
state is far better than no coverage estimate at all.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

__all__ = ["OpsimCache", "ensure_opsim", "fetch_opsim", "default_day_obs"]

logger = logging.getLogger(__name__)

_META_SUFFIX = ".meta.json"


def default_day_obs(when: datetime | None = None) -> int:
    """The ``day_obs`` that captures every visit taken so far.

    ``fetch_previous_visits`` returns visits *before* the day it is given, so
    the day after the current one is what covers tonight.

    Parameters
    ----------
    when : `datetime.datetime`, optional
        Reference time, defaults to now (UTC).

    Returns
    -------
    day_obs : `int`
        In ``YYYYMMDD`` form.
    """
    when = when or datetime.now(timezone.utc)
    return int((when + timedelta(days=1)).strftime("%Y%m%d"))


@dataclass
class OpsimCache:
    """A cached visit history and how it came to be.

    Attributes
    ----------
    path : `pathlib.Path`
        The Parquet file holding the visits.
    day_obs : `int`
        The ``day_obs`` it was fetched for; it holds visits before this day.
    fetched_at : `str`
        ISO-8601 UTC timestamp of the fetch.
    n_visits : `int`
        Number of visits it contains.
    refreshed : `bool`
        True when this run fetched it, False when a cached copy was reused.
    stale_reason : `str`
        Why a stale cache was reused, empty when the cache is current. Surfaced
        in the Slack reply so an old survey state is never invisible.
    """

    path: Path
    day_obs: int
    fetched_at: str
    n_visits: int
    refreshed: bool = False
    stale_reason: str = ""

    @property
    def age_hours(self) -> float:
        """Hours since the cache was fetched, or NaN if unknown."""
        try:
            fetched = datetime.fromisoformat(self.fetched_at)
        except ValueError:
            return float("nan")
        if fetched.tzinfo is None:
            fetched = fetched.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - fetched).total_seconds() / 3600.0

    def describe(self) -> str:
        """One-line provenance for logs and the Slack reply."""
        age = self.age_hours
        age_text = "unknown age" if age != age else f"{age:.1f} h old"
        what = "refreshed" if self.refreshed else "cached"
        text = f"{self.n_visits:,} visits ({what}, {age_text}, through day_obs {self.day_obs})"
        if self.stale_reason:
            text += f" -- {self.stale_reason}"
        return text


def _meta_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + _META_SUFFIX)


def _read_meta(path: Path) -> dict | None:
    meta_path = _meta_path(path)
    if not (path.is_file() and meta_path.is_file()):
        return None
    try:
        return json.loads(meta_path.read_text())
    except Exception as exc:
        logger.warning("Could not read opsim cache metadata %s: %s", meta_path, exc)
        return None


def fetch_opsim(day_obs: int, tokenfile: str | None, site: str = "usdf"):
    """Fetch the visit history from ConsDB with the scheduler's own tool.

    Parameters
    ----------
    day_obs : `int`
        Fetch visits before this ``day_obs``.
    tokenfile : `str` or None
        RSP token file. None uses the ``ACCESS_TOKEN`` environment variable.
    site : `str`
        ConsDB site, which must match where the token came from.

    Returns
    -------
    visits : `pandas.DataFrame` or None
        Opsim-formatted visits, or None when ConsDB has none.
    """
    from lsst_survey_sim.simulate_lsst import fetch_previous_visits

    resolved = str(Path(tokenfile).expanduser()) if tokenfile else None
    logger.info("Fetching visits before day_obs %d from ConsDB at %s", day_obs, site)
    return fetch_previous_visits(day_obs, resolved, site=site)


def ensure_opsim(
    cache_path: str | Path,
    day_obs: int,
    tokenfile: str | None = None,
    site: str = "usdf",
    max_age_hours: float = 24.0,
    force: bool = False,
) -> tuple["object", OpsimCache]:
    """Return the visit history, refreshing the cache when it is out of date.

    Parameters
    ----------
    cache_path : `str` or `pathlib.Path`
        Where the cached visits live.
    day_obs : `int`
        The night being simulated; the cache must cover visits before it.
    tokenfile : `str` or None
        RSP token file for ConsDB.
    site : `str`
        ConsDB site.
    max_age_hours : `float`
        Refresh a cache older than this. ``0`` always refreshes.
    force : `bool`
        Refresh regardless of age.

    Returns
    -------
    visits : `pandas.DataFrame`
        The visit history.
    cache : `OpsimCache`
        Provenance, including whether a stale copy had to be reused.

    Raises
    ------
    RuntimeError
        If the visits could not be fetched and no cache exists to fall back on.
    """
    import pandas as pd

    cache_path = Path(cache_path).expanduser()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    meta = _read_meta(cache_path)

    reason = ""
    if force:
        reason = "refresh forced"
    elif meta is None:
        reason = "no cache yet"
    elif int(meta.get("day_obs", 0)) < day_obs:
        # The cache holds visits before an earlier night, so it is missing
        # everything observed since -- a correctness issue, not staleness.
        reason = f"cache covers only day_obs {meta.get('day_obs')} < {day_obs}"
    else:
        cached = OpsimCache(
            path=cache_path,
            day_obs=int(meta["day_obs"]),
            fetched_at=meta.get("fetched_at", ""),
            n_visits=int(meta.get("n_visits", 0)),
        )
        age = cached.age_hours
        if max_age_hours <= 0:
            reason = "max_age_hours is 0"
        elif age != age:  # NaN
            reason = "cache age unknown"
        elif age > max_age_hours:
            reason = f"cache is {age:.1f} h old (limit {max_age_hours:g} h)"

    if not reason:
        logger.info("Using cached visit history %s: %s", cache_path, cached.describe())
        return pd.read_parquet(cache_path), cached

    logger.info("Refreshing visit history: %s", reason)
    try:
        visits = fetch_opsim(day_obs, tokenfile, site)
    except Exception as exc:
        visits = None
        fetch_error = str(exc)
        logger.error("Could not fetch visits from ConsDB: %s", exc)
    else:
        fetch_error = ""

    if visits is not None and len(visits) > 0:
        # Parquet rather than HDF5: the opsim table has ~250 columns including
        # strings, which HDF5 stores as pickled objects -- a 168 MB file and a
        # PerformanceWarning, versus a few MB here with real string types.
        visits.to_parquet(cache_path, index=False)
        fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        _meta_path(cache_path).write_text(
            json.dumps(
                {"day_obs": int(day_obs), "fetched_at": fetched_at, "n_visits": int(len(visits))},
                indent=2,
            )
        )
        cache = OpsimCache(
            path=cache_path,
            day_obs=int(day_obs),
            fetched_at=fetched_at,
            n_visits=int(len(visits)),
            refreshed=True,
        )
        logger.info("Cached visit history to %s: %s", cache_path, cache.describe())
        return visits, cache

    # The fetch failed or came back empty. An out-of-date survey state still
    # produces a useful coverage estimate, so fall back rather than give up.
    if meta is not None and cache_path.is_file():
        detail = fetch_error or "ConsDB returned no visits"
        cache = OpsimCache(
            path=cache_path,
            day_obs=int(meta.get("day_obs", 0)),
            fetched_at=meta.get("fetched_at", ""),
            n_visits=int(meta.get("n_visits", 0)),
            stale_reason=f"could not refresh ({detail}), using the cached copy",
        )
        logger.warning("Using a stale visit history: %s", cache.describe())
        return pd.read_parquet(cache_path), cache

    raise RuntimeError(
        "Could not fetch the visit history from ConsDB and no cache exists at "
        f"{cache_path}. "
        + (f"The fetch failed with: {fetch_error}. " if fetch_error else "ConsDB returned no visits. ")
        + "Check sim.opsim_tokenfile and sim.opsim_site, or run "
        "'chatterbox refresh-opsim' where ConsDB is reachable."
    )
