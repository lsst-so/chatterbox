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
    "localization_extent",
    "sky_projection",
    "MAX_ZOOM_RADIUS_DEG",
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

    Notes
    -----
    A map that is *flat* inside its region -- the producer's reward map, or a
    uniform prior -- has no density gradient to contour: asking for the level
    enclosing 90% lands exactly on the plateau, and matplotlib then paints the
    whole region rather than its outline. Such maps get a single level just
    below the plateau, which traces the boundary instead.
    """
    from ..astro.skymap import contour_levels

    prob = np.asarray(localization.prob_map, dtype=float)
    positive = prob[prob > 0]
    if positive.size == 0:
        logger.warning("Localization has no non-zero pixels; no contour will be drawn")
        return []

    # np.unique on a large map is affordable here and unambiguous.
    flat_region = np.unique(positive).size == 1
    if flat_region or not localization.is_probability:
        return [float(positive.min()) / 2.0]

    # Drop any level at or above the peak: a contour there encloses nothing.
    peak = float(positive.max())
    thresholds = [level for level in contour_levels(prob, levels) if level < peak]
    if not thresholds:
        return [float(positive.min()) / 2.0]
    return thresholds


#: A localization whose region fits inside this radius is drawn zoomed in.
#: Beyond it, a zoom would clip and the all-sky view is the honest choice.
MAX_ZOOM_RADIUS_DEG = 25.0


def localization_extent(localization: Localization) -> tuple[float, float, float]:
    """Centroid and angular size of a localization's enclosed region.

    Parameters
    ----------
    localization : `Localization`
        Localization to measure.

    Returns
    -------
    ra_deg, dec_deg : `float`
        Centroid, by vector mean so a region spanning RA 0 does not average to
        RA 180.
    radius_deg : `float`
        Greatest angular distance from that centroid to any region pixel. NaN
        when the region is empty.
    """
    import healpy as hp

    from ..astro.skymap import localization_region

    mask = localization_region(localization)
    pixels = np.flatnonzero(mask)
    if pixels.size == 0:
        return float("nan"), float("nan"), float("nan")

    nside = hp.get_nside(mask)
    vectors = np.array(hp.pix2vec(nside, pixels))
    mean = vectors.mean(axis=1)
    norm = np.linalg.norm(mean)
    if norm == 0:
        return float("nan"), float("nan"), float("nan")
    mean /= norm
    ra, dec = hp.vec2ang(mean, lonlat=True)
    # A dot product with the centroid separates every pixel at once.
    cosines = np.clip(vectors.T @ mean, -1.0, 1.0)
    radius = float(np.degrees(np.arccos(cosines.min())))
    return float(ra[0]), float(dec[0]), radius


def sky_projection(localization: Localization, max_zoom_radius_deg: float = MAX_ZOOM_RADIUS_DEG):
    """Projection and keyword arguments suited to a localization's size.

    A compact localization is unreadable on an all-sky map -- a 12 deg^2 patch
    is a few screen pixels -- so it gets a zoomed frame. A large or disjoint
    region, like a long GW arc, gets the all-sky view, because a zoom would
    silently crop part of it.

    Parameters
    ----------
    localization : `Localization`
        Localization to frame.
    max_zoom_radius_deg : `float`
        Largest region radius still drawn zoomed in.

    Returns
    -------
    projection : `str`
        Projection name for ``plt.subplot``.
    kwargs : `dict`
        Extra keyword arguments, empty for the all-sky case.
    """
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    ra, dec, radius = localization_extent(localization)
    if not np.isfinite(radius) or radius > max_zoom_radius_deg:
        return PROJECTION, {}
    # Pad so the contour is not flush against the frame.
    return "astro zoom", {
        "center": SkyCoord(ra * u.deg, dec * u.deg),
        "radius": max(radius * 1.4, 1.0) * u.deg,
    }


def mark_coord(ax, ra_deg: float, dec_deg: float, label: str | None = None, **kwargs) -> None:
    """Mark a sky position with a reticle."""
    if not (np.isfinite(ra_deg) and np.isfinite(dec_deg)):
        return
    style = {"marker": "+", "color": "black", "markersize": 9, "markeredgewidth": 1.5}
    style.update(kwargs)
    if label:
        style["label"] = label
    ax.plot(ra_deg, dec_deg, transform=ax.get_transform("world"), linestyle="none", **style)
