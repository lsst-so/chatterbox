"""Accessible dark hours per HEALPix pixel.

For each pixel, count the hours during the night for which Rubin could observe
it: airmass below a limit (default 2, i.e. altitude above 30 degrees) while the
Sun is below an altitude limit (default -12 degrees).

Nothing in the sibling repositories computed this, so it is implemented here.
The vectorized altitude evaluation follows the pattern used for DDF
observability in the ToO sim drivers, and the airmass-to-altitude conversion is
the usual ``arcsin(1 / X)``.
"""

import logging
from dataclasses import dataclass

import healpy as hp
import numpy as np

from .almanac import NightEvents

__all__ = ["DarkHoursMap", "dark_hours_map", "airmass_to_altitude_deg"]

logger = logging.getLogger(__name__)


def airmass_to_altitude_deg(airmass: float) -> float:
    """Minimum altitude in degrees corresponding to a maximum airmass.

    Uses the plane-parallel relation ``X = 1 / sin(alt)``, which is what the
    scheduler's own airmass limits assume.

    Parameters
    ----------
    airmass : `float`
        Airmass limit, must be >= 1.

    Returns
    -------
    altitude : `float`
        Altitude in degrees.
    """
    if airmass < 1.0:
        raise ValueError(f"Airmass must be >= 1, got {airmass}")
    return float(np.degrees(np.arcsin(1.0 / airmass)))


@dataclass
class DarkHoursMap:
    """Hours of accessible dark time per pixel for one night.

    Attributes
    ----------
    hours : `numpy.ndarray`
        Accessible hours per pixel, RING ordering.
    nside : `int`
        Map resolution.
    events : `NightEvents`
        The night this map describes.
    airmass_limit : `float`
        Airmass limit applied.
    sun_alt_limit_deg : `float`
        Sun altitude limit defining the window.
    altitude_limit_deg : `float`
        Altitude limit derived from `airmass_limit`.
    window_hours : `float`
        Length of the Sun-altitude window, i.e. the maximum any pixel can have.
    moon_avoidance_deg : `float`
        Moon avoidance radius applied, 0 when disabled.
    """

    hours: np.ndarray
    nside: int
    events: NightEvents
    airmass_limit: float
    sun_alt_limit_deg: float
    altitude_limit_deg: float
    window_hours: float
    moon_avoidance_deg: float = 0.0

    def stats_in_region(self, mask: np.ndarray) -> dict[str, float]:
        """Summarize accessible hours inside a credible region.

        Parameters
        ----------
        mask : `numpy.ndarray`
            Boolean mask in RING ordering. Resampled with
            ``healpy.ud_grade`` if its resolution differs from this map's.

        Returns
        -------
        stats : `dict`
            ``max_hours``, ``mean_hours``, ``median_hours`` and
            ``fraction_accessible`` (fraction of the region reaching airmass
            below the limit at any point in the window).
        """
        mask = np.asarray(mask, dtype=bool)
        if mask.size != self.hours.size:
            mask = hp.ud_grade(mask.astype(float), self.nside, order_in="RING", order_out="RING") > 0
        if not mask.any():
            return {
                "max_hours": float("nan"),
                "mean_hours": float("nan"),
                "median_hours": float("nan"),
                "fraction_accessible": 0.0,
            }
        values = self.hours[mask]
        return {
            "max_hours": float(values.max()),
            "mean_hours": float(values.mean()),
            "median_hours": float(np.median(values)),
            "fraction_accessible": float(np.count_nonzero(values) / values.size),
        }

    def weighted_stats(self, prob_map: np.ndarray) -> dict[str, float]:
        """Accessible hours weighted by localization probability (or area).

        Parameters
        ----------
        prob_map : `numpy.ndarray`
            Normalized weight per pixel in RING ordering, resampled if needed.

        Returns
        -------
        stats : `dict`
            ``weighted_mean_hours`` and ``weight_accessible``, the fraction of
            total weight lying on pixels with any accessible dark time.
        """
        prob_map = np.asarray(prob_map, dtype=float)
        if prob_map.size != self.hours.size:
            # ud_grade preserves the sum for power=-2, which is what we want
            # when redistributing probability between resolutions.
            prob_map = hp.ud_grade(prob_map, self.nside, order_in="RING", order_out="RING", power=-2)
        total = prob_map.sum()
        if total <= 0:
            return {"weighted_mean_hours": float("nan"), "weight_accessible": 0.0}
        weights = prob_map / total
        return {
            "weighted_mean_hours": float(np.sum(weights * self.hours)),
            "weight_accessible": float(np.sum(weights[self.hours > 0])),
        }


def dark_hours_map(
    events: NightEvents,
    nside: int = 64,
    step_minutes: float = 5.0,
    airmass_limit: float = 2.0,
    sun_alt_limit_deg: float = -12.0,
    moon_avoidance_deg: float = 0.0,
) -> DarkHoursMap:
    """Compute accessible dark hours per pixel for one night.

    Parameters
    ----------
    events : `NightEvents`
        Night to evaluate, from `chatterbox.astro.almanac.night_events`.
    nside : `int`
        Output map resolution.
    step_minutes : `float`
        Time sampling. 5 minutes is accurate to a few minutes per pixel and
        takes well under a second at nside 64.
    airmass_limit : `float`
        Maximum airmass counted as accessible.
    sun_alt_limit_deg : `float`
        Sun altitude bounding the window; -12 or -18.
    moon_avoidance_deg : `float`
        When positive, samples closer than this to the Moon are not counted.

    Returns
    -------
    dark_hours : `DarkHoursMap`

    Notes
    -----
    Each sample contributes the step length, but the total is clipped to the
    window length: a naive count of samples over-reports by up to one step
    because both endpoints can satisfy the condition.
    """
    from rubin_scheduler.utils import Site, approx_ra_dec2_alt_az

    site = Site("LSST")
    start_mjd, end_mjd = events.observing_window(sun_alt_limit_deg)
    window_hours = (end_mjd - start_mjd) * 24.0

    step_days = step_minutes / (60.0 * 24.0)
    times = np.arange(start_mjd, end_mjd, step_days)
    if times.size == 0:
        raise ValueError(f"Observing window for {events.day_obs} is empty")

    npix = hp.nside2npix(nside)
    ra, dec = hp.pix2ang(nside, np.arange(npix), lonlat=True)

    altitude_limit = airmass_to_altitude_deg(airmass_limit)

    # approx_ra_dec2_alt_az returns DEGREES. Broadcasting (npix, 1) against
    # (ntime,) evaluates the whole grid in one vectorized call.
    alt, az = approx_ra_dec2_alt_az(
        ra[:, None],
        dec[:, None],
        site.latitude,
        site.longitude,
        times[None, :],
    )
    del az
    accessible = alt > altitude_limit

    if moon_avoidance_deg > 0:
        from rubin_scheduler.utils import angular_separation

        moon_sep = np.degrees(
            angular_separation(
                np.radians(ra[:, None]),
                np.radians(dec[:, None]),
                np.radians(events.moon_ra_deg),
                np.radians(events.moon_dec_deg),
            )
        )
        # The Moon moves slowly enough over one night that a single position is
        # adequate for an avoidance annulus at this resolution.
        accessible &= moon_sep > moon_avoidance_deg

    hours = accessible.sum(axis=1) * (step_minutes / 60.0)
    np.clip(hours, 0.0, window_hours, out=hours)

    logger.debug(
        "Dark-hours map: nside=%d ntime=%d window=%.2f h max=%.2f h",
        nside,
        times.size,
        window_hours,
        hours.max(),
    )

    return DarkHoursMap(
        hours=hours,
        nside=nside,
        events=events,
        airmass_limit=airmass_limit,
        sun_alt_limit_deg=sun_alt_limit_deg,
        altitude_limit_deg=altitude_limit,
        window_hours=window_hours,
        moon_avoidance_deg=moon_avoidance_deg,
    )
