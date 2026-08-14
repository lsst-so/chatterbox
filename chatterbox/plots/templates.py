"""Localization contour over existing per-band template coverage."""

import logging
from pathlib import Path

import numpy as np

from ..astro.templates import TemplateCoverage
from ..models import Localization
from .style import (
    BAND_COLORS,
    PROJECTION,
    add_galactic_plane,
    localization_levels,
    use_headless_backend,
)

__all__ = ["plot_template_coverage"]

logger = logging.getLogger(__name__)


def _binary_cmap(color: str):
    """A one-colour colormap, for maps that only say covered or not."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("binary_coverage", [color, color])


def plot_template_coverage(
    coverage: TemplateCoverage,
    localization: Localization,
    out_path: str | Path,
    bands: list[str] | None = None,
    dpi: int = 130,
    skip_empty: bool = True,
) -> Path | None:
    """Draw one panel per band: template coverage with the localization.

    Parameters
    ----------
    coverage : `TemplateCoverage`
        Per-band binary coverage maps.
    localization : `Localization`
        Localization whose contour is overlaid on every panel.
    out_path : `str` or `pathlib.Path`
        PNG destination.
    bands : `list` [`str`], optional
        Bands to draw. Defaults to every band in the cache.
    dpi : `int`
        Output resolution.
    skip_empty : `bool`
        Omit bands with no coverage anywhere, rather than drawing blank panels.

    Returns
    -------
    path : `pathlib.Path` or None
        The file written, or None when there was nothing to draw.
    """
    use_headless_backend()
    import ligo.skymap.plot  # noqa: F401  (registers the projections)
    from matplotlib import pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    candidates = bands if bands is not None else coverage.bands
    if skip_empty:
        drawn = [b for b in candidates if b in coverage.maps and coverage.maps[b].any()]
    else:
        drawn = [b for b in candidates if b in coverage.maps]

    if not drawn:
        logger.warning("No band has any template coverage; skipping the template plot")
        return None

    fractions = coverage.coverage_in_region(localization.prob_map, nside=localization.nside)
    levels = localization_levels(localization)

    ncols = 2 if len(drawn) > 1 else 1
    nrows = int(np.ceil(len(drawn) / ncols))
    fig = plt.figure(figsize=(7.0 * ncols, 3.9 * nrows), dpi=dpi)

    for n, band in enumerate(drawn, start=1):
        ax = fig.add_subplot(nrows, ncols, n, projection=PROJECTION)
        # The maps are binary, so a graded colour scale would imply a precision
        # that does not exist: covered pixels get one solid colour, and
        # uncovered ones are left blank via NaN.
        covered = np.where(coverage.mask(band), 1.0, np.nan)
        ax.imshow_hpx(
            covered,
            nested=False,
            cmap=_binary_cmap(BAND_COLORS.get(band, "tab:red")),
            vmin=0,
            vmax=1,
        )
        if levels:
            ax.contour_hpx(
                localization.prob_map,
                nested=False,
                levels=levels,
                colors=["black"],
                linewidths=[1.4],
                zorder=5,
            )
        add_galactic_plane(ax)
        ax.grid(alpha=0.2)
        quantity = localization.quantity_name
        ax.set_title(
            f"{band} band: {coverage.area_deg2(band):,.0f} deg$^2$ with templates\n"
            f"{fractions[band]:.1%} of localization {quantity} has templates",
            fontsize=9,
        )

    stamp = f" (refreshed {coverage.built_at})" if coverage.built_at else ""
    fig.suptitle(
        f"Existing LSST template coverage vs localization{stamp}\n"
        "Shaded: template exists. Black contour: localization.",
        fontsize=11,
    )
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", out_path)
    return out_path
