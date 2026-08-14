"""Astronomical computation: skymaps, almanac, dark hours, template coverage."""

from .almanac import NightEvents, moon_separation_deg, night_events
from .darkhours import DarkHoursMap, dark_hours_map
from .skymap import (
    KahanAdder,
    Skymap,
    contour_levels,
    credible_mask,
    geometry_from_mask,
    localization_from_probability,
    localization_from_reward_map,
    nest_to_ring,
)
from .templates import TemplateCoverage, load_source_maps, load_template_maps

__all__ = [
    "KahanAdder",
    "Skymap",
    "nest_to_ring",
    "credible_mask",
    "contour_levels",
    "geometry_from_mask",
    "localization_from_reward_map",
    "localization_from_probability",
    "NightEvents",
    "night_events",
    "moon_separation_deg",
    "DarkHoursMap",
    "dark_hours_map",
    "TemplateCoverage",
    "load_template_maps",
    "load_source_maps",
]
