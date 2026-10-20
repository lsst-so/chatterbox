"""An animation of airmass across the sky through the first night.

Each frame is the airmass of the whole sky at one moment, with the
localization's contour drawn over it, so an observer can watch the field rise,
cross and set against the airmass the scheduler will accept. Airmass above a
cut-off is masked out as NaN rather than painted: the far end of a colour scale
would otherwise make a field on the horizon look like a field that is merely a
little worse.

Airmass is computed straight from altitude on a sky grid, not read from a
HEALPix map, so the zoomed panel is as smooth as the all-sky one. The relation
is the dark-hours map's own, ``X = 1 / sin(altitude)``, from the same
``approx_ra_dec2_alt_az``, so a field that is "visible" in the post is the same
field that has a colour here.
"""

import logging
from pathlib import Path

import numpy as np

from ..astro.almanac import CHILE_TZ, NightEvents, format_time
from ..deps import require
from ..models import Localization
from .style import PROJECTION, localization_levels, mark_coord, sky_projection, use_headless_backend

__all__ = [
    "DEFAULT_MASK_AIRMASS",
    "airmass_from_altitude",
    "airmass_on_grid",
    "frame_times",
    "plot_airmass_animation",
]

logger = logging.getLogger(__name__)

#: Airmass above which a position is masked. 3.8 is an altitude of about 15
#: degrees, near the lowest the telescope can point.
DEFAULT_MASK_AIRMASS = 3.8


def airmass_from_altitude(altitude_deg, mask_above: float = DEFAULT_MASK_AIRMASS) -> np.ndarray:
    """Airmass from altitude, with unobservable positions set to NaN.

    Parameters
    ----------
    altitude_deg : array-like
        Altitude above the horizon, degrees.
    mask_above : `float`
        Airmass beyond which a position is masked.

    Returns
    -------
    airmass : `numpy.ndarray`
        ``1 / sin(altitude)``, NaN at or below the horizon -- where the
        relation is meaningless -- and wherever it exceeds `mask_above`.
    """
    altitude = np.asarray(altitude_deg, dtype=float)
    airmass = np.full(altitude.shape, np.nan)
    up = altitude > 0.0
    airmass[up] = 1.0 / np.sin(np.radians(altitude[up]))
    airmass[airmass > mask_above] = np.nan
    return airmass


def airmass_on_grid(ra_deg, dec_deg, mjd: float, mask_above: float = DEFAULT_MASK_AIRMASS) -> np.ndarray:
    """Airmass at Rubin for a set of sky positions at one time.

    Parameters
    ----------
    ra_deg, dec_deg : array-like
        Positions, degrees, any matching shape.
    mjd : `float`
        Time, UTC MJD.
    mask_above : `float`
        Airmass beyond which a position is masked to NaN.

    Returns
    -------
    airmass : `numpy.ndarray`
        Same shape as the inputs.
    """
    from rubin_scheduler.utils import Site, approx_ra_dec2_alt_az

    site = Site("LSST")
    ra = np.asarray(ra_deg, dtype=float)
    dec = np.asarray(dec_deg, dtype=float)
    altitude, _ = approx_ra_dec2_alt_az(ra.ravel(), dec.ravel(), site.latitude, site.longitude, float(mjd))
    return airmass_from_altitude(np.reshape(altitude, ra.shape), mask_above)


def frame_times(
    events: NightEvents, step_minutes: float = 10.0, sun_alt_limit_deg: float = -12.0
) -> np.ndarray:
    """Frame times across a night's observing window.

    Parameters
    ----------
    events : `NightEvents`
        The night.
    step_minutes : `float`
        Spacing between frames.
    sun_alt_limit_deg : `float`
        Sun altitude bounding the window; -12 or -18.

    Returns
    -------
    mjd : `numpy.ndarray`
        From the window's start to its end inclusive, so the last frame is the
        end of the night and not a step short of it.
    """
    start, end = events.observing_window(sun_alt_limit_deg)
    step = step_minutes / (60.0 * 24.0)
    times = np.arange(start, end, step)
    return np.append(times, end)


def _zoom_grid(ra_deg: float, dec_deg: float, radius_deg: float, n: int = 90):
    """A fine RA/Dec grid around a point, for the zoomed panel.

    RA offsets are stretched by ``1 / cos(dec)`` so the grid is square on the
    sky. Longitudes are left unwrapped (they may run below 0 or above 360) so a
    field across RA = 0 stays one contiguous mesh.
    """
    half = radius_deg * 1.2
    dy = np.linspace(-half, half, n)
    dx = np.linspace(-half, half, n) / max(np.cos(np.radians(dec_deg)), 0.05)
    return np.meshgrid(ra_deg + dx, np.clip(dec_deg + dy, -90.0, 90.0))


def plot_airmass_animation(
    events: NightEvents,
    localization: Localization,
    out_path: str | Path,
    mask_above: float = DEFAULT_MASK_AIRMASS,
    step_minutes: float = 10.0,
    sun_alt_limit_deg: float = -12.0,
    centroid: tuple[float, float] | None = None,
    fps: int = 8,
    dpi: int = 80,
) -> Path:
    """Animate airmass over the first night, with the localization contour.

    Parameters
    ----------
    events : `NightEvents`
        The night to animate.
    localization : `Localization`
        Whose contour is drawn on every frame.
    out_path : `str` or `pathlib.Path`
        GIF destination. A GIF rather than a video because Slack plays it
        inline and no encoder has to be installed.
    mask_above : `float`
        Airmass beyond which the map is masked to NaN.
    step_minutes : `float`
        Time between frames.
    sun_alt_limit_deg : `float`
        Sun altitude bounding the night; -12 matches the rest of the post.
    centroid : `tuple` [`float`, `float`], optional
        RA/Dec in degrees to mark and to report the airmass of.
    fps : `int`
        Playback speed.
    dpi : `int`
        Frame resolution. Kept modest: a night is dozens of frames.

    Returns
    -------
    path : `pathlib.Path`
        The GIF written.

    Notes
    -----
    A compact localization gets a second, zoomed panel beside the all-sky one,
    because on an all-sky map a few square degrees is a few pixels. A large or
    disjoint one gets the all-sky panel alone, for the same reason
    `sky_projection` gives: a zoom would crop it.
    """
    use_headless_backend()
    require("ligo.skymap.plot")
    import matplotlib
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    times = frame_times(events, step_minutes, sun_alt_limit_deg)

    projection, zoom_kwargs = sky_projection(localization)
    zoomed = projection != PROJECTION

    # A bare `Figure` rather than `plt.figure`: this runs in a background
    # thread while the next alert's first post is being drawn on the main one,
    # and pyplot's global figure registry is not thread-safe. Nothing here
    # touches it, so nothing has to be locked and the first post never waits.
    fig = Figure(figsize=(12.5 if zoomed else 9.0, 5.6), dpi=dpi)
    FigureCanvasAgg(fig)
    grid = fig.add_gridspec(1, 2 if zoomed else 1, width_ratios=[1.5, 1.0] if zoomed else [1.0])
    panels = [fig.add_subplot(grid[0, 0], projection=PROJECTION)]
    if zoomed:
        panels.append(fig.add_subplot(grid[0, 1], projection=projection, **zoom_kwargs))

    # One mesh per panel, updated in place each frame: redrawing a pcolormesh
    # per frame is what makes animations like this crawl.
    lon, lat = np.meshgrid(np.linspace(0.0, 360.0, 361), np.linspace(-90.0, 90.0, 181))
    grids = [(lon, lat)]
    if zoomed:
        radius = zoom_kwargs["radius"].to_value("deg")
        centre = zoom_kwargs["center"]
        grids.append(_zoom_grid(centre.ra.deg, centre.dec.deg, radius))

    cmap = matplotlib.colormaps["viridis_r"].copy()
    cmap.set_bad("0.93")
    meshes = []
    for ax, (glon, glat) in zip(panels, grids):
        blank = np.full(glon[:-1, :-1].shape, np.nan)
        meshes.append(
            ax.pcolormesh(
                glon,
                glat,
                blank,
                transform=ax.get_transform("world"),
                cmap=cmap,
                vmin=1.0,
                vmax=mask_above,
                shading="flat",
            )
        )

    levels = localization_levels(localization)
    for ax in panels:
        if levels:
            ax.contour_hpx(
                localization.prob_map,
                nested=False,
                levels=levels,
                colors=["red"],
                linewidths=[1.6],
                zorder=5,
            )
        if centroid is not None:
            mark_coord(ax, centroid[0], centroid[1], color="red")
        ax.grid(alpha=0.25)
    panels[0].set_title("All sky", fontsize=9)
    if zoomed:
        panels[1].set_title("Localization", fontsize=9)

    cbar = fig.colorbar(
        meshes[0], ax=panels, location="bottom", pad=0.18, shrink=0.6, aspect=40, extend="neither"
    )
    cbar.set_label(f"Airmass (grey: below the horizon or above {mask_above:g})")
    headline = fig.suptitle("", fontsize=11)

    def centroid_text(mjd: float) -> str:
        if centroid is None:
            return ""
        value = float(airmass_on_grid([centroid[0]], [centroid[1]], mjd, mask_above)[0])
        return f"; centroid airmass {value:.2f}" if np.isfinite(value) else "; centroid not observable"

    def draw(index: int):
        mjd = float(times[index])
        for mesh, (glon, glat) in zip(meshes, grids):
            # Cell centres, since the mesh is drawn flat over cell corners.
            clon = 0.25 * (glon[:-1, :-1] + glon[1:, :-1] + glon[:-1, 1:] + glon[1:, 1:])
            clat = 0.25 * (glat[:-1, :-1] + glat[1:, :-1] + glat[:-1, 1:] + glat[1:, 1:])
            mesh.set_array(airmass_on_grid(clon, clat, mjd, mask_above).ravel())
        from astropy.time import Time

        when = Time(mjd, format="mjd", scale="utc")
        headline.set_text(
            f"Night of {events.day_obs}: {when.utc.strftime('%H:%M')} UTC "
            f"[{format_time(when, CHILE_TZ)} Chile]{centroid_text(mjd)}"
        )
        return meshes

    animation = FuncAnimation(fig, draw, frames=len(times), blit=False)
    animation.save(out_path, writer=PillowWriter(fps=fps))
    logger.info("Wrote %s (%d frames)", out_path, len(times))
    return out_path
