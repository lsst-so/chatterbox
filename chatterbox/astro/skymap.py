"""Skymap handling: multi-order parsing, credible regions and geometry.

`KahanAdder` and `Skymap` are vendored from ``rubin-ToO-producer``'s
``forward_alerts.py`` (the ``Skymap`` class at line 92) so that chatterbox
parses GraceDB multi-order maps exactly the way the producer does, and so the
areas quoted in Slack match the areas the producer cut on. They are reproduced
with their behaviour intact; only formatting and docstrings differ.

Everything else here is chatterbox's own, and works in **RING** ordering.
The producer's ``reward_map`` arrives NESTED, so it is converted on the way in
(the same conversion ``lsst_survey_sim`` applies before handing a map to the
feature-based scheduler).
"""

import logging
import math

import healpy as hp
import numpy as np
from astropy.coordinates import SkyCoord
from astropy_healpix import boundaries_lonlat

from ..models import Geometry, Localization

__all__ = [
    "KahanAdder",
    "Skymap",
    "nest_to_ring",
    "credible_mask",
    "contour_levels",
    "geometry_from_mask",
    "localization_from_reward_map",
    "localization_from_probability",
]

logger = logging.getLogger(__name__)

#: Square degrees per steradian.
DEG2_PER_SR = (180.0 / math.pi) ** 2


class KahanAdder:
    """Compensated summation, so long runs of tiny probabilities still count.

    Vendored from ``rubin-ToO-producer/forward_alerts.py``.
    """

    def __init__(self) -> None:
        self.sum = 0.0
        self.comp = 0.0

    def __float__(self) -> float:
        return self.sum

    def __iadd__(self, value: float) -> "KahanAdder":
        y = float(value) - self.comp
        t = self.sum + y
        self.comp = (t - self.sum) - y
        self.sum = t
        return self

    def __eq__(self, value: object) -> bool:
        return self.sum == value

    def __ne__(self, value: object) -> bool:
        return self.sum != value

    def __lt__(self, value: float) -> bool:
        return self.sum < value

    def __le__(self, value: float) -> bool:
        return self.sum <= value

    def __gt__(self, value: float) -> bool:
        return self.sum > value

    def __ge__(self, value: float) -> bool:
        return self.sum >= value


class Skymap:
    """A HEALPix multi-order (UNIQ/NUNIQ) probability skymap.

    Vendored from ``rubin-ToO-producer/forward_alerts.py`` so that credible
    areas reported by chatterbox agree with the producer's own cuts.

    Parameters
    ----------
    densities : array-like
        Probability densities, as in a multi-order map's ``PROBDENSITY``.
    u_indices : array-like
        Matching HEALPix ``UNIQ`` indices.
    drop_trivial_probabilities : `bool`
        Zero the densities of pixels that cannot move the running total, which
        makes a flattened map far more compressible.
    """

    def __init__(self, densities, u_indices, drop_trivial_probabilities: bool = True) -> None:
        self.data = np.array(
            sorted(zip(densities, u_indices), key=lambda entry: -entry[0]),
            dtype=[("prob_density", "f8"), ("uniq_index", "i8")],
        )

        summed_prob = KahanAdder()
        self.pixel_areas: dict[int, float] = {}
        for entry in self.data:
            order = math.floor(math.log2(entry[1] / 4) / 2)
            if drop_trivial_probabilities and summed_prob >= 1.0:
                entry[0] = 0
                continue
            if order not in self.pixel_areas:
                self.pixel_areas[order] = math.pi / (3 << (order << 1))
            summed_prob += entry[0] * self.pixel_areas[order]

    def area_for_probability(self, target_probability: float) -> tuple[float, float]:
        """Area of the highest-density region summing to a probability.

        Parameters
        ----------
        target_probability : `float`
            Credible level, e.g. 0.9.

        Returns
        -------
        area : `float`
            Sky area in steradians.
        min_dec : `float`
            Minimum declination in radians touched by any corner of any pixel
            in that region. This is the quantity the producer applies its
            30 degree visibility cut to.
        """
        summed_prob = KahanAdder()
        summed_area = KahanAdder()
        n_indices: dict[int, list[int]] = {}
        for p_dens, u_idx in self.data:
            order = math.floor(math.log2(u_idx / 4) / 2)
            area = self.pixel_areas[order]
            summed_prob += p_dens * area
            summed_area += area
            n_idx = u_idx - (4 << (2 * order))
            n_indices.setdefault(1 << order, []).append(n_idx)
            if summed_prob >= target_probability:
                break

        # boundaries_lonlat handles only one nside at a time, so batch by
        # order.
        min_dec = 100.0
        for nside, indices in n_indices.items():
            corner_decs = boundaries_lonlat(indices, 1, nside, "nested")[1].to_value().flatten()
            min_dec = min(min_dec, float(np.min(corner_decs)))
        return summed_area.sum, min_dec

    def make_flat_map(self) -> np.ndarray:
        """Flatten to a NESTED density map at the map's maximum order."""
        flat_order = max(self.pixel_areas.keys())
        flat_map = np.zeros(12 << (flat_order << 1))
        for p_dens, u_idx in self.data:
            order = math.floor(math.log2(u_idx / 4) / 2)
            n_idx = u_idx - (4 << (2 * order))
            min_idx = n_idx << (2 * (flat_order - order))
            idx_range = 1 << (2 * (flat_order - order))
            flat_map[min_idx : (min_idx + idx_range)] = p_dens
        return flat_map

    def make_flat_binary_map(self, target_probability: float, target_order: int | None = None) -> np.ndarray:
        """Flatten the credible region to a NESTED boolean map.

        This is the function that produces the ``reward_map`` chatterbox
        receives, reproduced so tests can build realistic fixtures.
        """
        if target_order is None:
            target_order = max(self.pixel_areas.keys())
        flat_map = np.zeros(12 << (target_order << 1), dtype=bool)

        summed_prob = KahanAdder()
        for p_dens, u_idx in self.data:
            order = math.floor(math.log2(u_idx / 4) / 2)
            n_idx = u_idx - (4 << (2 * order))
            if order < target_order:
                min_idx = n_idx << (2 * (target_order - order))
                idx_range = 1 << (2 * (target_order - order))
                flat_map[min_idx : (min_idx + idx_range)] = True
            else:
                flat_map[n_idx >> (2 * (order - target_order))] = True
            summed_prob += p_dens * self.pixel_areas[order]
            if summed_prob >= target_probability:
                break
        return flat_map

    def max_probability_coord(self) -> tuple[float, float]:
        """RA and Dec in degrees of the highest-density pixel."""
        # self.data is sorted by decreasing density, so the first row wins.
        u_idx = int(self.data["uniq_index"][0])
        order = math.floor(math.log2(u_idx / 4) / 2)
        n_idx = u_idx - (4 << (2 * order))
        ra, dec = hp.pix2ang(1 << order, n_idx, nest=True, lonlat=True)
        return float(ra), float(dec)


def nest_to_ring(nested: np.ndarray, nside: int) -> np.ndarray:
    """Reorder a full-sky map from NESTED to RING, preserving dtype.

    Parameters
    ----------
    nested : `numpy.ndarray`
        Full-sky map in NESTED ordering.
    nside : `int`
        Resolution of the map.

    Returns
    -------
    ring : `numpy.ndarray`
        The same values in RING ordering.

    Notes
    -----
    ``healpy.reorder`` casts booleans, so the index mapping is applied directly
    instead. This keeps the producer's boolean reward map boolean.
    """
    nested = np.asarray(nested)
    npix = hp.nside2npix(nside)
    if nested.size != npix:
        raise ValueError(f"Map has {nested.size} pixels, expected {npix} for nside={nside}")
    ring = np.empty_like(nested)
    ring[hp.nest2ring(nside, np.arange(npix))] = nested
    return ring


def credible_mask(prob_map: np.ndarray, level: float = 0.9) -> np.ndarray:
    """Boolean mask of the smallest region containing a given probability.

    Parameters
    ----------
    prob_map : `numpy.ndarray`
        Probability per pixel (need not be normalized).
    level : `float`
        Credible level, e.g. 0.9.

    Returns
    -------
    mask : `numpy.ndarray`
        True for pixels inside the credible region.
    """
    prob_map = np.asarray(prob_map, dtype=float)
    total = prob_map.sum()
    if total <= 0:
        return np.zeros(prob_map.size, dtype=bool)
    order = np.argsort(prob_map)[::-1]
    cumulative = np.cumsum(prob_map[order]) / total
    # searchsorted gives the first index reaching the level; include it so the
    # region contains at least `level`, never slightly less.
    cutoff = int(np.searchsorted(cumulative, level))
    cutoff = min(cutoff, prob_map.size - 1)
    mask = np.zeros(prob_map.size, dtype=bool)
    mask[order[: cutoff + 1]] = True
    return mask


def contour_levels(prob_map: np.ndarray, levels=(0.5, 0.9)) -> list[float]:
    """Density thresholds enclosing the given credible levels.

    The result is suitable for ``ligo.skymap`` axes' ``contour_hpx``, sorted
    ascending as matplotlib requires.

    Returns
    -------
    thresholds : `list` [`float`]
        One density threshold per requested level. Levels that cannot be
        reached (for example on a degenerate map) are omitted.
    """
    prob_map = np.asarray(prob_map, dtype=float)
    total = prob_map.sum()
    if total <= 0:
        return []
    sorted_prob = np.sort(prob_map)[::-1]
    cumulative = np.cumsum(sorted_prob) / total
    thresholds = []
    for level in levels:
        idx = int(np.searchsorted(cumulative, level))
        if idx >= sorted_prob.size:
            continue
        thresholds.append(float(sorted_prob[idx]))
    # Deduplicate: a binary map gives the same threshold for every level.
    unique = sorted(set(thresholds))
    return [t for t in unique if t > 0]


def geometry_from_mask(mask: np.ndarray, nside: int) -> Geometry:
    """Derive sky geometry from a boolean credible-region mask in RING order.

    Declination limits use pixel *corners* rather than centres, matching the
    producer's own ``min_dec`` computation, so the reported range is not
    optimistic by half a pixel.

    Parameters
    ----------
    mask : `numpy.ndarray`
        Boolean, RING ordering, full sky.
    nside : `int`
        Map resolution.

    Returns
    -------
    geometry : `Geometry`
    """
    mask = np.asarray(mask, dtype=bool)
    pixels = np.flatnonzero(mask)
    pixel_area = hp.nside2pixarea(nside, degrees=True)
    if pixels.size == 0:
        logger.warning("Credible-region mask is empty; geometry will be degenerate")
        return Geometry(
            area_deg2=0.0,
            dec_min_deg=float("nan"),
            dec_max_deg=float("nan"),
            centroid_ra_deg=float("nan"),
            centroid_dec_deg=float("nan"),
            gal_b_abs_min_deg=float("nan"),
            gal_b_abs_max_deg=float("nan"),
            n_pixels=0,
            pixel_area_deg2=pixel_area,
        )

    ra, dec = hp.pix2ang(nside, pixels, lonlat=True)

    # Corner declinations, so a region's true extent is not understated.
    corners = hp.boundaries(nside, pixels, step=1)  # (npix, 3, 4)
    corner_dec = np.degrees(np.arcsin(np.clip(corners[:, 2, :], -1.0, 1.0)))
    dec_min = float(corner_dec.min())
    dec_max = float(corner_dec.max())

    # Vector mean, so a region spanning RA = 0 does not average to RA = 180.
    vecs = hp.pix2vec(nside, pixels)
    mean_vec = np.array([np.mean(v) for v in vecs])
    norm = np.linalg.norm(mean_vec)
    if norm == 0:
        centroid_ra, centroid_dec = float("nan"), float("nan")
    else:
        c_ra, c_dec = hp.vec2ang(mean_vec / norm, lonlat=True)
        centroid_ra, centroid_dec = float(c_ra[0]), float(c_dec[0])

    gal_b = np.abs(SkyCoord(ra=ra, dec=dec, unit="deg").galactic.b.deg)

    return Geometry(
        area_deg2=float(pixels.size * pixel_area),
        dec_min_deg=dec_min,
        dec_max_deg=dec_max,
        centroid_ra_deg=centroid_ra,
        centroid_dec_deg=centroid_dec,
        gal_b_abs_min_deg=float(gal_b.min()),
        gal_b_abs_max_deg=float(gal_b.max()),
        n_pixels=int(pixels.size),
        pixel_area_deg2=pixel_area,
    )


def localization_from_reward_map(
    reward_map: np.ndarray,
    nside: int,
    credible_level: float = 0.7,
) -> Localization:
    """Build a `Localization` from the producer's binary reward map.

    Parameters
    ----------
    reward_map : `numpy.ndarray`
        Boolean map in **NESTED** ordering, as carried by the ToO record.
    nside : `int`
        ``reward_map_nside`` from the record (32 in current deployments).
    credible_level : `float`
        Credible level the region represents. ``forward_alerts.py`` publishes
        the 70% region for every alert class.

    Returns
    -------
    localization : `Localization`
        With ``is_probability=False``: the weight is uniform inside the region,
        so summing it measures fractional *area*, not probability.
    """
    ring = nest_to_ring(np.asarray(reward_map, dtype=bool), nside)
    n_set = int(ring.sum())
    prob = np.zeros(ring.size, dtype=float)
    if n_set:
        prob[ring] = 1.0 / n_set
    else:
        logger.warning("Reward map contains no set pixels")
    area = n_set * hp.nside2pixarea(nside, degrees=True)
    provenance = (
        f"producer reward map, {credible_level:.0%} credible region, "
        f"nside={nside} ({area:.0f} deg^2, uniform weight)"
    )
    return Localization(
        prob_map=prob,
        nside=nside,
        is_probability=False,
        credible_level=credible_level,
        provenance=provenance,
    )


def localization_from_probability(
    prob_map: np.ndarray,
    provenance: str,
    credible_level: float = 0.9,
    nested: bool = False,
) -> Localization:
    """Build a `Localization` from a real probability map.

    Parameters
    ----------
    prob_map : `numpy.ndarray`
        Probability per pixel, full sky.
    provenance : `str`
        Where the map came from, shown in Slack.
    credible_level : `float`
        Credible level used when drawing the primary contour.
    nested : `bool`
        True if `prob_map` is in NESTED ordering and needs converting.

    Returns
    -------
    localization : `Localization`
        With ``is_probability=True``.
    """
    prob_map = np.asarray(prob_map, dtype=float)
    nside = hp.get_nside(prob_map)
    if nested:
        prob_map = nest_to_ring(prob_map, nside)
    total = prob_map.sum()
    if total <= 0:
        raise ValueError("Probability map sums to zero")
    return Localization(
        prob_map=prob_map / total,
        nside=nside,
        is_probability=True,
        credible_level=credible_level,
        provenance=provenance,
    )
