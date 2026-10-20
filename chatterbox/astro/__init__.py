"""Astronomical computation: skymaps, almanac, dark hours, template coverage."""

from .almanac import NightEvents, moon_separation_deg, night_events
from .darkhours import DarkHoursMap, dark_hours_map
from .skymap import (
    REGION_NSIDE_FLOOR,
    KahanAdder,
    RegionMaps,
    Skymap,
    contour_levels,
    credible_mask,
    geometry_from_mask,
    localization_from_probability,
    localization_from_reward_map,
    localization_region,
    nest_to_ring,
    pixel_corner_decs,
    region_localizations,
)
from .templates import TemplateCoverage, load_source_maps, load_template_maps
from .visibility import FieldVisibility, Window, field_visibility

__all__ = [
    "KahanAdder",
    "Skymap",
    "nest_to_ring",
    "pixel_corner_decs",
    "credible_mask",
    "contour_levels",
    "geometry_from_mask",
    "localization_from_reward_map",
    "localization_from_probability",
    "localization_region",
    "RegionMaps",
    "REGION_NSIDE_FLOOR",
    "region_localizations",
    "NightEvents",
    "night_events",
    "moon_separation_deg",
    "DarkHoursMap",
    "dark_hours_map",
    "TemplateCoverage",
    "load_template_maps",
    "load_source_maps",
    "FieldVisibility",
    "Window",
    "field_visibility",
]
