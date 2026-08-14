"""Localization coverage gained on each night of a simulation.

The cumulative curve in `chatterbox.plots.coverage` answers "how much, by
when"; this answers "what did each night buy". Bars are what a single night
added, the line over them is the running total across all bands, and both are
percentages of the same localization, so they share one axis.

Every night appears here, including nights the per-night visit maps leave out
when `sim.nightly_plots_max_nights` caps them.
"""

import logging
from pathlib import Path

import numpy as np

from ..models import Localization
from ..sim.coverage import NightlyCoverage
from .style import BAND_COLORS, use_headless_backend

__all__ = ["plot_coverage_by_night"]

logger = logging.getLogger(__name__)


def plot_coverage_by_night(
    nightly: NightlyCoverage,
    localization: Localization,
    out_path: str | Path,
    source: str = "",
    alert_type: str = "",
    dpi: int = 130,
) -> Path | None:
    """Plot per-night coverage gain per band, with the running total over it.

    Parameters
    ----------
    nightly : `NightlyCoverage`
        From `chatterbox.sim.coverage.coverage_by_night`.
    localization : `Localization`
        Used only to label the axis honestly as probability or area.
    out_path : `str` or `pathlib.Path`
        PNG destination.
    source : `str`
        Event identifier for the title.
    alert_type : `str`
        Alert type for the title.
    dpi : `int`
        Output resolution.

    Returns
    -------
    path : `pathlib.Path` or None
        None when there was nothing to plot.
    """
    if not nightly.nights:
        logger.warning("No nights with visits; nothing to plot")
        return None

    use_headless_backend()
    from matplotlib import pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Counted from the trigger, not from the survey start: "night 11" says
    # nothing, "+1" says the night after the alert.
    nights = nightly.relative_nights
    labels = nightly.labels()
    x = np.arange(len(nights), dtype=float)
    bands = [b for b in ("u", "g", "r", "i", "z", "y") if b in nightly.gained]

    # Widen with the run so a 50-night BBH simulation does not compress into
    # slivers, but keep a floor so a 3-night run is not a postage stamp.
    width = min(24.0, max(8.5, 0.55 * len(nights) + 3.0))
    fig, ax = plt.subplots(figsize=(width, 5.0), dpi=dpi)

    if bands:
        bar_width = 0.8 / len(bands)
        for n, band in enumerate(bands):
            offset = (n - (len(bands) - 1) / 2.0) * bar_width
            ax.bar(
                x + offset,
                [100.0 * v for v in nightly.gained[band]],
                width=bar_width,
                color=BAND_COLORS.get(band),
                label=f"{band} band",
                zorder=2,
            )

    ax.step(
        x,
        [100.0 * v for v in nightly.any_band_cumulative],
        where="mid",
        color="0.2",
        lw=1.8,
        marker="o",
        ms=3.5,
        label="cumulative, any band",
        zorder=3,
    )
    # The end point is the number quoted in the thread; say it out loud so the
    # figure can be checked against the text. It goes to the right of the last
    # night rather than above it, which a run that saturates would push off the
    # top of the axes.
    final = 100.0 * nightly.any_band_cumulative[-1]
    ax.annotate(
        f"{final:.1f}%",
        xy=(x[-1], final),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
        fontsize=9,
        color="0.2",
        fontweight="bold",
    )

    for level in (50.0, 90.0):
        ax.axhline(level, color="0.7", ls=":", lw=1.0, zorder=0)

    quantity = localization.quantity_name
    ax.set_ylabel(f"Localization {quantity} [%]")
    ax.set_xlabel("Nights since the trigger" if nightly.reference_night is not None else "Survey night")
    ax.set_ylim(0, 100)
    # Room on the right for the final-value label.
    ax.set_xlim(-0.7, len(nights) - 0.3 + 0.5)
    # Label every night when they fit, then thin out rather than overlap.
    stride = max(1, int(np.ceil(len(nights) / 25.0)))
    ax.set_xticks(x[::stride])
    ax.set_xticklabels(labels[::stride])
    if any(n < 0 for n in nights):
        # A night before the trigger is nominal cadence, not follow-up. Mark
        # the boundary rather than letting it read as an early epoch.
        boundary = float(np.searchsorted(np.asarray(nights), 0)) - 0.5
        ax.axvline(boundary, color="0.4", ls="--", lw=1.0, zorder=1)
        ax.text(
            boundary - 0.1,
            98.0,
            "trigger ",
            ha="right",
            va="top",
            fontsize=8,
            color="0.4",
            rotation=90,
        )
    ax.grid(alpha=0.25, axis="y")
    # "best" rather than a fixed corner: where the axes are empty depends
    # entirely on how fast this particular run covered the localization.
    ax.legend(loc="best", fontsize=9, ncol=2)

    title = "Coverage gained per night"
    if source:
        title += f" for {source}"
    if alert_type:
        title += f" ({alert_type})"
    title += "\nBars: added that night, per band. Line: running total, any band."
    if nightly.reference_night is not None:
        # The survey night is what the visit database and the per-night map
        # filenames use, so it has to stay findable from here.
        title += f"\nNight 0 is the trigger night (survey night {nightly.reference_night})"
    if not localization.is_probability:
        title += "\nArea fraction, not probability: no probability skymap was available"
    ax.set_title(title, fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", out_path)
    return out_path
