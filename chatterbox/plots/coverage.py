"""Cumulative localization coverage against time, per band."""

import logging
from pathlib import Path

import numpy as np

from ..models import Localization
from .style import BAND_COLORS, use_headless_backend

__all__ = ["plot_coverage_curve"]

logger = logging.getLogger(__name__)


def plot_coverage_curve(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    localization: Localization,
    out_path: str | Path,
    source: str = "",
    alert_type: str = "",
    event_mjd: float | None = None,
    too_total: tuple[np.ndarray, np.ndarray] | None = None,
    combined_total: tuple[np.ndarray, np.ndarray] | None = None,
    dpi: int = 130,
) -> Path | None:
    """Plot cumulative coverage per band against hours since the trigger.

    Parameters
    ----------
    curves : `dict`
        ``band -> (mjd, cumulative_fraction)`` from
        `chatterbox.sim.coverage.coverage_curve`.
    localization : `Localization`
        Used only to label the axis honestly as probability or area.
    out_path : `str` or `pathlib.Path`
        PNG destination.
    source : `str`
        Event identifier for the title.
    alert_type : `str`
        Alert type for the title.
    event_mjd : `float`, optional
        Trigger time; when given the x axis is hours since the trigger.
    too_total : `tuple`, optional
        ``(mjd, cumulative_fraction)`` counting every ToO visit once in any
        band, from `chatterbox.sim.coverage.any_band_curve`.
    combined_total : `tuple`, optional
        The same for the ToO follow-up *plus* the nominal-cadence visits that
        landed on the localization. Drawn dashed beside `too_total`, so the gap
        between the two lines is what the survey added.
    dpi : `int`
        Output resolution.

    Returns
    -------
    path : `pathlib.Path` or None
        None when there was nothing to plot.
    """
    if not curves:
        logger.warning("No coverage curves to plot")
        return None

    use_headless_backend()
    from matplotlib import pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=dpi)
    for band, (times, cumulative) in curves.items():
        if times.size == 0:
            continue
        x = (times - event_mjd) * 24.0 if event_mjd is not None else times
        # Start every curve at zero coverage so the first visit's jump is
        # visible.
        x = np.concatenate([[x[0]], x])
        y = np.concatenate([[0.0], 100.0 * cumulative])
        ax.step(x, y, where="post", label=f"{band} band", color=BAND_COLORS.get(band), lw=1.8)

    # The two any-band totals go on top of the per-band curves in black, since
    # they are sums of them rather than another band. Both are carried out to
    # the same right edge: one set of visits ending earlier than the other is
    # not a fall in coverage, and a step plot that simply stops reads like one.
    ends = [
        np.asarray(curve[0], dtype=float)[-1]
        for curve in (too_total, combined_total)
        if curve is not None and len(curve[0])
    ]
    right = max(ends) if ends else None

    def draw_total(curve, label, style):
        if curve is None:
            return False
        times, cumulative = curve
        times = np.asarray(times, dtype=float)
        if times.size == 0:
            return False
        x = (times - event_mjd) * 24.0 if event_mjd is not None else times
        x = np.concatenate([[x[0]], x])
        y = np.concatenate([[0.0], 100.0 * np.asarray(cumulative, dtype=float)])
        if right is not None:
            edge = (right - event_mjd) * 24.0 if event_mjd is not None else right
            if edge > x[-1]:
                x = np.concatenate([x, [edge]])
                y = np.concatenate([y, [y[-1]]])
        ax.step(x, y, where="post", label=label, color="0.15", ls=style, lw=2.2, zorder=5)
        return True

    drew_totals = draw_total(too_total, "any band, ToO only", "-")
    # Only worth a second line when the survey actually added something; two
    # coincident lines would read as a rendering fault.
    if combined_total is not None and too_total is not None:
        gap = float(np.max(combined_total[1]) - np.max(too_total[1])) if len(too_total[1]) else 1.0
        if gap > 1e-4:
            drew_totals |= draw_total(combined_total, "any band, + nominal survey", "--")

    for level in (50.0, 90.0):
        ax.axhline(level, color="0.7", ls=":", lw=1.0, zorder=0)

    quantity = localization.quantity_name
    ax.set_ylabel(f"Localization {quantity} covered [%]")
    ax.set_xlabel("Hours since trigger" if event_mjd is not None else "MJD")
    # A run that covers everything is a real outcome; without the headroom its
    # line sits on the frame and reads as clipped.
    ax.set_ylim(0, 102)
    ax.grid(alpha=0.25)
    # The legend labels carry what the two black lines mean; a separate caption
    # would only collide with a run that saturates early.
    ax.legend(loc="lower right", fontsize=9)

    title = "Expected coverage of the localization"
    if source:
        title += f" for {source}"
    if alert_type:
        title += f" ({alert_type})"
    if not localization.is_probability:
        title += "\nArea fraction, not probability: no probability skymap was available"
    ax.set_title(title, fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", out_path)
    return out_path
