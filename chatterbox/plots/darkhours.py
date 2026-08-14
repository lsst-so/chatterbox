"""Localization contour over the accessible-dark-hours map."""

import logging
from pathlib import Path

import healpy as hp
import numpy as np

from ..astro.darkhours import DarkHoursMap
from ..deps import require
from ..models import Localization
from .style import PROJECTION, add_galactic_plane, localization_levels, mark_coord, use_headless_backend

__all__ = ["plot_dark_hours"]

logger = logging.getLogger(__name__)


def plot_dark_hours(
    dark_hours: DarkHoursMap,
    localization: Localization,
    out_path: str | Path,
    title: str | None = None,
    centroid: tuple[float, float] | None = None,
    dpi: int = 130,
) -> Path:
    """Draw the dark-hours map with the localization contour on top.

    Parameters
    ----------
    dark_hours : `DarkHoursMap`
        Accessible hours per pixel for the night.
    localization : `Localization`
        Localization whose contour is overlaid.
    out_path : `str` or `pathlib.Path`
        PNG destination.
    title : `str`, optional
        Plot title. A description of the night is generated when omitted.
    centroid : `tuple` [`float`, `float`], optional
        RA/Dec in degrees to mark.
    dpi : `int`
        Output resolution.

    Returns
    -------
    path : `pathlib.Path`
        The file written.
    """
    use_headless_backend()
    # Registers the 'astro ... mollweide' projections. require() names the
    # interpreter, since a wrong-environment install is the usual cause.
    require("ligo.skymap.plot")
    from matplotlib import pyplot as plt

    events = dark_hours.events
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(11, 6), dpi=dpi)
    ax = plt.axes(projection=PROJECTION)

    # Mask pixels that never reach the airmass limit so they read as "not
    # accessible" rather than as the bottom of the colour scale.
    hours = np.array(dark_hours.hours, dtype=float)
    display = np.where(hours > 0, hours, np.nan)

    vmax = float(np.nanmax(display)) if np.isfinite(display).any() else 1.0
    # nested=False: chatterbox works in RING ordering throughout.
    img = ax.imshow_hpx(display, nested=False, cmap="viridis", vmin=0.0, vmax=vmax)

    levels = localization_levels(localization)
    if levels:
        ax.contour_hpx(
            localization.prob_map,
            nested=False,
            levels=levels,
            colors=["red"],
            linewidths=[1.6],
            zorder=5,
        )
    else:
        logger.warning("No contour drawn for %s", out_path.name)

    add_galactic_plane(ax)
    if centroid is not None:
        mark_coord(ax, centroid[0], centroid[1], label="Localization centroid", color="red")

    ax.grid(alpha=0.25)
    cbar = fig.colorbar(img, ax=ax, location="bottom", pad=0.06, shrink=0.8, aspect=40)
    cbar.set_label(
        f"Accessible dark hours (airmass < {dark_hours.airmass_limit:g}, "
        f"Sun < {dark_hours.sun_alt_limit_deg:g} deg)"
    )

    if title is None:
        stats = dark_hours.stats_in_region(localization.prob_map > 0)
        title = (
            f"Night of {events.day_obs}: {dark_hours.window_hours:.1f} h with Sun < "
            f"{dark_hours.sun_alt_limit_deg:g} deg\n"
            f"Localization: max {stats['max_hours']:.1f} h, mean {stats['mean_hours']:.1f} h accessible; "
            f"{stats['fraction_accessible']:.0%} of the region reaches airmass < {dark_hours.airmass_limit:g}"
        )
    ax.set_title(title, fontsize=10)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="upper right", fontsize=7, framealpha=0.8)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", out_path)
    return out_path


def region_hours_summary(dark_hours: DarkHoursMap, localization: Localization) -> dict[str, float]:
    """Combine masked and weighted dark-hour statistics for a localization.

    Returns
    -------
    stats : `dict`
        Keys from `DarkHoursMap.stats_in_region` plus
        `DarkHoursMap.weighted_stats`.
    """
    mask = localization.prob_map > 0
    if mask.size != dark_hours.hours.size:
        mask = hp.ud_grade(mask.astype(float), dark_hours.nside, order_in="RING", order_out="RING") > 0
    stats = dark_hours.stats_in_region(mask)
    stats.update(dark_hours.weighted_stats(localization.prob_map))
    return stats
