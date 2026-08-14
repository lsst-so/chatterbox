"""Plot rendering for Slack posts."""

from .coverage import plot_coverage_curve
from .darkhours import plot_dark_hours
from .nightly import plot_all_nights, plot_nightly_visits
from .nightly_coverage import plot_coverage_by_night
from .style import add_galactic_plane, localization_levels, use_headless_backend
from .templates import plot_template_coverage

__all__ = [
    "use_headless_backend",
    "add_galactic_plane",
    "localization_levels",
    "plot_dark_hours",
    "plot_coverage_curve",
    "plot_coverage_by_night",
    "plot_nightly_visits",
    "plot_all_nights",
    "plot_template_coverage",
]
