"""Per-night, per-band simulated visit counts with the localization contour.

One figure per night of the simulation, one panel per band that was observed,
showing how many visits each pixel received with the localization drawn on top.
This is the "what did the scheduler actually do, night by night" view, as
opposed to the single cumulative number `chatterbox.sim.coverage` reports.

Visit counts are genuine counts here, so a graded colour scale is meaningful --
unlike the binary template maps, which get a single solid colour.
"""

import logging
from pathlib import Path

import healpy as hp
import numpy as np

from ..deps import require
from ..models import Localization
from .style import add_galactic_plane, localization_levels, sky_projection, use_headless_backend

__all__ = ["plot_nightly_visits", "plot_all_nights"]

logger = logging.getLogger(__name__)


def plot_nightly_visits(
    night: int,
    band_maps: dict[str, np.ndarray],
    localization: Localization,
    out_path: str | Path,
    source: str = "",
    alert_type: str = "",
    dpi: int = 130,
) -> Path | None:
    """Draw one night's per-band visit counts with the localization contour.

    Parameters
    ----------
    night : `int`
        Simulation night index, used in the title.
    band_maps : `dict` [`str`, `numpy.ndarray`]
        Visit counts per pixel, keyed by band, RING ordering.
    localization : `Localization`
        Localization whose contour is overlaid on every panel.
    out_path : `str` or `pathlib.Path`
        PNG destination.
    source, alert_type : `str`
        Event identifiers for the title.
    dpi : `int`
        Output resolution.

    Returns
    -------
    path : `pathlib.Path` or None
        The file written, or None when there was nothing to draw.
    """
    if not band_maps:
        return None

    use_headless_backend()
    require("ligo.skymap.plot")
    from matplotlib import pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bands = [b for b in ("u", "g", "r", "i", "z", "y") if b in band_maps]
    levels = localization_levels(localization)
    # A compact localization is illegible all-sky: a 12 deg^2 patch of coverage
    # is a handful of screen pixels. Large or disjoint regions stay all-sky.
    projection, projection_kwargs = sky_projection(localization)

    # A shared colour scale makes bands comparable within the night.
    vmax = max(1, int(max(np.max(band_maps[b]) for b in bands)))

    ncols = 2 if len(bands) > 1 else 1
    nrows = int(np.ceil(len(bands) / ncols))
    # Extra height for the header: with a single panel the suptitle and the
    # axes title otherwise land on top of each other.
    fig = plt.figure(figsize=(7.0 * ncols, 3.9 * nrows + 0.8), dpi=dpi)

    img = None
    for n, band in enumerate(bands, start=1):
        ax = fig.add_subplot(nrows, ncols, n, projection=projection, **projection_kwargs)
        counts = np.asarray(band_maps[band], dtype=float)
        # Blank the unvisited sky rather than painting it the bottom colour.
        display = np.where(counts > 0, counts, np.nan)
        img = ax.imshow_hpx(display, nested=False, cmap="viridis", vmin=0, vmax=vmax)
        if levels:
            ax.contour_hpx(
                localization.prob_map,
                nested=False,
                levels=levels,
                colors=["red"],
                linewidths=[1.4],
                zorder=5,
            )
        add_galactic_plane(ax)
        ax.grid(alpha=0.2)

        covered = _covered_fraction(counts, localization)
        area = float((counts > 0).sum() * hp.nside2pixarea(hp.get_nside(counts), degrees=True))
        ax.set_title(
            f"{band} band: {int(counts.max())} visits deep over {area:,.0f} deg$^2$\n"
            f"{covered:.1%} of localization {localization.quantity_name} touched",
            fontsize=9,
        )

    # Reserve room for the suptitle *before* adding the colorbar: the colorbar
    # repositions its parent axes, and adjusting afterwards would leave it
    # sitting on top of the maps. WCS axes do not play well with tight_layout,
    # so the margin is set explicitly.
    fig.subplots_adjust(top=0.82 if nrows == 1 else 0.90, bottom=0.10)

    if img is not None:
        cbar = fig.colorbar(img, ax=fig.axes, location="bottom", pad=0.08, shrink=0.6, aspect=40)
        ticks = np.unique(np.linspace(0, vmax, min(vmax + 1, 8), dtype=int))
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([str(int(t)) for t in ticks])
        # The contour legend lives here rather than in a second suptitle line,
        # which is what used to collide with the panel titles.
        cbar.set_label("Simulated visits per pixel (red contour: localization)")

    title = f"Night {night}"
    if source:
        title += f" of the {source} simulation"
    if alert_type:
        title += f" ({alert_type})"
    fig.suptitle(title, fontsize=12)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", out_path)
    return out_path


def _covered_fraction(counts: np.ndarray, localization: Localization) -> float:
    """Fraction of localization weight under at least one visit."""
    prob = np.asarray(localization.prob_map, dtype=float)
    nside = hp.get_nside(counts)
    if prob.size != counts.size:
        # power=-2 conserves the summed weight across the resolution change.
        prob = hp.ud_grade(prob, nside, order_in="RING", order_out="RING", power=-2)
    total = prob.sum()
    if total <= 0:
        return 0.0
    return float(prob[counts > 0].sum() / total)


def plot_all_nights(
    nightly_maps: dict[int, dict[str, np.ndarray]],
    localization: Localization,
    out_dir: str | Path,
    source: str = "",
    alert_type: str = "",
    max_nights: int = 10,
    dpi: int = 130,
) -> tuple[list[Path], list[int], int]:
    """Render one figure per night, oldest first.

    Parameters
    ----------
    nightly_maps : `dict`
        ``night -> band -> counts``, from
        `chatterbox.sim.coverage.nightly_visit_maps`.
    localization : `Localization`
        Localization to overlay.
    out_dir : `str` or `pathlib.Path`
        Directory to write into.
    source, alert_type : `str`
        Event identifiers for the titles.
    max_nights : `int`
        Cap on the number of figures, so a 50-night BBH simulation does not
        produce 50 uploads. ``0`` means no cap. The number dropped is returned
        rather than silently swallowed.
    dpi : `int`
        Output resolution.

    Returns
    -------
    paths : `list` [`pathlib.Path`]
        Figures written.
    nights : `list` [`int`]
        The nights they correspond to.
    total_nights : `int`
        Nights with visits *before* the cap, so a caller can say what it left
        out.
    """
    out_dir = Path(out_dir)
    ordered = sorted(nightly_maps)
    total = len(ordered)
    chosen = ordered if max_nights <= 0 else ordered[:max_nights]
    if len(chosen) < total:
        logger.warning(
            "Plotting the first %d of %d nights with visits (sim.nightly_plots_max_nights)",
            len(chosen),
            total,
        )

    paths: list[Path] = []
    nights: list[int] = []
    for night in chosen:
        name = f"{source or 'sim'}_night{night:03d}_visits.png"
        try:
            path = plot_nightly_visits(
                night,
                nightly_maps[night],
                localization,
                out_dir / name,
                source=source,
                alert_type=alert_type,
                dpi=dpi,
            )
        except Exception as exc:
            logger.error("Could not render night %d: %s", night, exc)
            continue
        if path is not None:
            paths.append(path)
            nights.append(night)
    return paths, nights, total
