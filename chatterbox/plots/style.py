"""Shared plotting helpers: projection, overlays and contour levels.

All plots use ``ligo.skymap.plot``'s WCS axes so that HEALPix maps and sky
coordinates can be drawn in the same frame, matching the projections used in
the existing analysis notebooks.
"""

import logging

import numpy as np

from ..models import Localization

__all__ = [
    "use_headless_backend",
    "PROJECTION",
    "GALACTIC_BUFFER_DEG",
    "add_galactic_plane",
    "localization_levels",
    "mark_coord",
    "BAND_COLORS",
]

logger = logging.getLogger(__name__)

#: Default all-sky projection.
PROJECTION = "astro degrees mollweide"

#: Half-width of the shaded Galactic plane region, degrees.
GALACTIC_BUFFER_DEG = 10.0

#: Band colours, matching the analysis notebooks.
BAND_COLORS = {
    "u": "indigo",
    "g": "forestgreen",
    "r": "dodgerblue",
    "i": "red",
    "z": "violet",
    "y": "orange",
}

_backend_set = False


def use_headless_backend() -> None:
    """Select a non-interactive matplotlib backend, once per process.

    Required because the bot renders plots from a service with no display.
    """
    global _backend_set
    if _backend_set:
        return
    import matplotlib

    matplotlib.use("Agg")
    _backend_set = True


def add_galactic_plane(ax, buffer_deg: float = GALACTIC_BUFFER_DEG) -> None:
    """Overlay the Galactic plane and a +/- buffer in Galactic latitude.

    Parameters
    ----------
    ax : `matplotlib.axes.Axes`
        A ``ligo.skymap`` WCS axes.
    buffer_deg : `float`
        Latitude offset for the dashed limits.

    Notes
    -----
    Coordinates are transformed with ``ax.get_transform("world")`` and plotted
    at their true ICRS positions. The versions of this overlay in the notebooks
    subtract 90 degrees from RA to compensate for a projection centre; that
    offset is a bug and is deliberately not reproduced here.
    """
    from astropy.coordinates import SkyCoord

    ell = np.linspace(0.0, 360.0, 721)
    specs = (
        (0.0, {"ls": "--", "color": "black", "lw": 0.8, "label": "Galactic plane"}),
        (
            buffer_deg,
            {
                "ls": "--",
                "color": "black",
                "lw": 0.6,
                "alpha": 0.5,
                "label": f"|b| = {buffer_deg:.0f} deg",
            },
        ),
        (-buffer_deg, {"ls": "--", "color": "black", "lw": 0.6, "alpha": 0.5}),
    )
    for b, kwargs in specs:
        coords = SkyCoord(l=ell * 0 + ell, b=np.full_like(ell, b), unit="deg", frame="galactic").icrs
        ra = coords.ra.deg
        dec = coords.dec.deg
        # Break the line where it wraps in RA so the projection does not draw a
        # horizontal streak across the whole map.
        jumps = np.flatnonzero(np.abs(np.diff(ra)) > 180.0)
        segments = np.split(np.arange(ra.size), jumps + 1)
        for n, seg in enumerate(segments):
            if seg.size < 2:
                continue
            kw = dict(kwargs)
            if n > 0:
                kw.pop("label", None)
            ax.plot(ra[seg], dec[seg], transform=ax.get_transform("world"), **kw)


def localization_levels(localization: Localization, levels=(0.5, 0.9)) -> list[float]:
    """Contour levels that trace a localization.

    For a real probability map these are the density thresholds enclosing the
    requested credible levels. For the producer's binary reward map every level
    has the same threshold, so a single level at half the in-region weight is
    returned instead, which traces the region boundary.

    Parameters
    ----------
    localization : `Localization`
        Map to draw.
    levels : `tuple` [`float`]
        Credible levels, used only for probability maps.

    Returns
    -------
    thresholds : `list` [`float`]
        Ascending, suitable for ``contour_hpx``. Empty if nothing can be drawn.
    """
    from ..astro.skymap import contour_levels

    if localization.is_probability:
        return contour_levels(localization.prob_map, levels)

    nonzero = localization.prob_map[localization.prob_map > 0]
    if nonzero.size == 0:
        logger.warning("Localization has no non-zero pixels; no contour will be drawn")
        return []
    return [float(nonzero.min()) / 2.0]


def mark_coord(ax, ra_deg: float, dec_deg: float, label: str | None = None, **kwargs) -> None:
    """Mark a sky position with a reticle."""
    if not (np.isfinite(ra_deg) and np.isfinite(dec_deg)):
        return
    style = {"marker": "+", "color": "black", "markersize": 9, "markeredgewidth": 1.5}
    style.update(kwargs)
    if label:
        style["label"] = label
    ax.plot(ra_deg, dec_deg, transform=ax.get_transform("world"), linestyle="none", **style)
