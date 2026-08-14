"""Sun and Moon statistics for the night a trigger belongs to.

Built on ``rubin_scheduler.site_models.Almanac``, which is the same source the
scheduler itself uses, so the times quoted in Slack match the times the
scheduler plans against.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from astropy.time import Time

__all__ = [
    "CHILE_TZ",
    "NightEvents",
    "night_events",
    "moon_separation_deg",
    "format_time",
]

logger = logging.getLogger(__name__)

CHILE_TZ = ZoneInfo("Chile/Continental")


def _time_or_none(mjd: float | None) -> Time | None:
    """Wrap an MJD as a `~astropy.time.Time`, mapping NaN and None to None.

    The almanac reports NaN for a moonrise or moonset that does not occur
    during the night, which is a normal outcome, not an error.
    """
    if mjd is None:
        return None
    value = float(mjd)
    if not np.isfinite(value):
        return None
    return Time(value, format="mjd", scale="utc")


def format_time(t: Time | None, tz: ZoneInfo | None = None) -> str:
    """Format a time for Slack, or ``"--"`` when the event does not occur."""
    if t is None:
        return "--"
    if tz is None:
        return t.utc.strftime("%Y-%m-%d %H:%M UTC")
    return t.to_datetime(timezone=tz).strftime("%H:%M")


@dataclass
class NightEvents:
    """Sun and Moon events for a single observing night.

    All times are UTC. ``None`` means the event does not occur during the
    night, which happens routinely for moonrise and moonset.
    """

    day_obs: str
    sunset: Time | None
    sun_n12_setting: Time | None
    sun_n18_setting: Time | None
    sun_n18_rising: Time | None
    sun_n12_rising: Time | None
    sunrise: Time | None
    moonrise: Time | None
    moonset: Time | None
    moon_phase: float
    moon_ra_deg: float
    moon_dec_deg: float

    @property
    def night_length_hours(self) -> float:
        """Hours between the -12 degree evening and morning crossings."""
        if self.sun_n12_setting is None or self.sun_n12_rising is None:
            return float("nan")
        return float((self.sun_n12_rising.mjd - self.sun_n12_setting.mjd) * 24.0)

    @property
    def dark_length_hours(self) -> float:
        """Hours between the -18 degree evening and morning crossings."""
        if self.sun_n18_setting is None or self.sun_n18_rising is None:
            return float("nan")
        return float((self.sun_n18_rising.mjd - self.sun_n18_setting.mjd) * 24.0)

    def observing_window(self, sun_alt_limit_deg: float = -12.0) -> tuple[float, float]:
        """MJD bounds of the window for a given Sun altitude limit.

        Parameters
        ----------
        sun_alt_limit_deg : `float`
            Either -12 or -18; other values raise.

        Returns
        -------
        start, end : `float`
            MJD bounds.
        """
        if sun_alt_limit_deg == -12.0:
            start, end = self.sun_n12_setting, self.sun_n12_rising
        elif sun_alt_limit_deg == -18.0:
            start, end = self.sun_n18_setting, self.sun_n18_rising
        else:
            raise ValueError(
                f"Unsupported Sun altitude limit {sun_alt_limit_deg}; the almanac provides -12 and -18"
            )
        if start is None or end is None:
            raise ValueError(f"Almanac has no {sun_alt_limit_deg} degree crossings for {self.day_obs}")
        return float(start.mjd), float(end.mjd)


def night_events(
    when: Time | None = None,
    day_obs: str | None = None,
    prefer_next_night: bool = True,
) -> NightEvents:
    """Sun and Moon events for the observing night relevant to a time.

    Parameters
    ----------
    when : `~astropy.time.Time`, optional
        A time during (or on the day of) the night of interest. Defaults to
        now.
    day_obs : `str`, optional
        Explicit ``YYYY-MM-DD`` evening date, overriding `when`.
    prefer_next_night : `bool`
        When `when` falls after the night's -12 degree morning crossing -- that
        is, the observable part of that night is already over -- advance to the
        following night. This is almost always what a ToO report should show,
        since a trigger arriving during local daytime is followed up that
        evening, not retroactively.

    Returns
    -------
    events : `NightEvents`

    Notes
    -----
    A trigger arriving after local midnight but before dawn still belongs to
    the night that started the previous evening, and is left there. The almanac
    is queried by MJD rather than by deriving a calendar date locally, because
    the mapping from instant to observing night depends on the site longitude.
    """
    from rubin_scheduler.site_models import Almanac
    from rubin_scheduler.utils import Site

    site = Site("LSST")
    try:
        almanac = Almanac()
    except (FileNotFoundError, OSError) as exc:
        # rubin_scheduler resolves its data through RUBIN_SIM_DATA_DIR and
        # falls back to $HOME/rubin_sim_data without complaint, so the useful
        # thing to report is which directory it actually looked in.
        data_dir = os.environ.get("RUBIN_SIM_DATA_DIR") or f"{Path.home() / 'rubin_sim_data'} (default)"
        raise RuntimeError(
            f"Could not load the almanac from {data_dir}: {exc}. "
            "Set sim.rubin_sim_data to a rubin_sim_data tree containing "
            "site_models, or run 'scheduler_download_data --dirs site_models'."
        ) from exc

    if day_obs is not None:
        info = almanac.get_sunset_info(evening_date=day_obs, longitude=site.longitude_rad)
        label = day_obs
    else:
        when = Time.now() if when is None else when
        info = almanac.get_sunset_info(mjd=float(when.mjd), longitude=site.longitude_rad)

        # get_sunset_info returns the night containing or preceding the time,
        # so a daytime trigger lands on a night that has already finished.
        # Advance in half-day steps until the night has observable time left.
        # Stepping rather than computing an offset avoids depending on exactly
        # where the almanac places its night boundary.
        if prefer_next_night and float(when.mjd) > float(info["sun_n12_rising"]):
            logger.info(
                "Trigger at %s is after this night's -12 deg morning crossing; "
                "reporting the following night",
                when.utc.iso,
            )
            probe = float(when.mjd)
            for _ in range(4):
                probe += 0.5
                candidate = almanac.get_sunset_info(mjd=probe, longitude=site.longitude_rad)
                if float(candidate["sun_n12_rising"]) > float(when.mjd):
                    info = candidate
                    break
            else:
                logger.warning(
                    "Could not find a night after %s; falling back to the night the almanac "
                    "returned for the trigger time",
                    when.utc.iso,
                )

        # Label the night by the UTC date of its sunset, which is the Rubin
        # day_obs convention for Cerro Pachon.
        sunset = _time_or_none(info["sunset"])
        label = sunset.utc.strftime("%Y-%m-%d") if sunset is not None else "unknown"

    positions = almanac.get_sun_moon_positions(float(info["sun_n12_setting"]))
    # rubin_scheduler 3.x returns a dict of arrays here; older versions a
    # structured array. Both index by name, so only the scalar squeeze differs.
    moon_phase = float(np.atleast_1d(positions["moon_phase"])[0])
    moon_ra = float(np.degrees(np.atleast_1d(positions["moon_RA"])[0]))
    moon_dec = float(np.degrees(np.atleast_1d(positions["moon_dec"])[0]))

    return NightEvents(
        day_obs=label,
        sunset=_time_or_none(info["sunset"]),
        sun_n12_setting=_time_or_none(info["sun_n12_setting"]),
        sun_n18_setting=_time_or_none(info["sun_n18_setting"]),
        sun_n18_rising=_time_or_none(info["sun_n18_rising"]),
        sun_n12_rising=_time_or_none(info["sun_n12_rising"]),
        sunrise=_time_or_none(info["sunrise"]),
        moonrise=_time_or_none(info["moonrise"]),
        moonset=_time_or_none(info["moonset"]),
        moon_phase=moon_phase,
        moon_ra_deg=moon_ra,
        moon_dec_deg=moon_dec,
    )


def moon_separation_deg(ra_deg: float, dec_deg: float, events: NightEvents) -> float:
    """Angular separation between a coordinate and the Moon at nautical dusk.

    Parameters
    ----------
    ra_deg, dec_deg : `float`
        Target coordinates in degrees.
    events : `NightEvents`
        Night whose Moon position should be used.

    Returns
    -------
    separation : `float`
        Degrees, or NaN if the target coordinate is not finite.
    """
    if not (np.isfinite(ra_deg) and np.isfinite(dec_deg)):
        return float("nan")
    from rubin_scheduler.utils import angular_separation

    return float(
        angular_separation(
            np.radians(ra_deg),
            np.radians(dec_deg),
            np.radians(events.moon_ra_deg),
            np.radians(events.moon_dec_deg),
        )
        * 180.0
        / np.pi
    )
