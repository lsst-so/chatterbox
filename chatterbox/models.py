"""Normalized in-memory representation of a ToO trigger.

These containers deliberately keep the *provenance* of every derived quantity.
The producer's output record contains only a binary credible region, so a
percentage computed from it is an **area** fraction, not a probability. When
GraceDB enrichment succeeds the same field becomes a true probability. Every
consumer therefore needs `Localization.is_probability` to label its output
honestly, and `Localization.quantity_name` exists so message text cannot drift
from the data.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from astropy.time import Time

__all__ = [
    "Localization",
    "Geometry",
    "GwEnrichment",
    "Trigger",
]


@dataclass
class Localization:
    """A sky localization ready for plotting and coverage accounting.

    Parameters
    ----------
    prob_map : `numpy.ndarray`
        Normalized weight per pixel in **RING** ordering, summing to 1. When
        `is_probability` is False this is uniform across the credible region,
        so summing it measures fractional area rather than probability.
    nside : `int`
        HEALPix resolution of `prob_map`.
    is_probability : `bool`
        True when `prob_map` came from a real probability skymap.
    credible_level : `float`
        Credible level the region represents, e.g. 0.7 for the producer's
        reward map.
    provenance : `str`
        Human-readable description of where the map came from, shown in Slack.
    """

    prob_map: np.ndarray
    nside: int
    is_probability: bool
    credible_level: float
    provenance: str

    @property
    def quantity_name(self) -> str:
        """What summing `prob_map` measures: 'probability' or 'area'."""
        return "probability" if self.is_probability else "area"

    @property
    def npix(self) -> int:
        """Number of pixels in the map."""
        return int(self.prob_map.size)


@dataclass
class Geometry:
    """Sky geometry of the credible region.

    All angles are in degrees. `dec_min_deg` is what the producer's 30 degree
    visibility cut is applied to upstream, so it is worth surfacing.
    """

    area_deg2: float
    dec_min_deg: float
    dec_max_deg: float
    centroid_ra_deg: float
    centroid_dec_deg: float
    gal_b_abs_min_deg: float
    gal_b_abs_max_deg: float
    n_pixels: int
    pixel_area_deg2: float


@dataclass
class GwEnrichment:
    """Quantities recovered from GraceDB for a gravitational-wave alert.

    Every field is optional: enrichment is best-effort and time-bounded, and a
    partial result is still worth reporting.
    """

    superevent_id: str
    skymap_path: str | None = None
    distance_mean_mpc: float | None = None
    distance_std_mpc: float | None = None
    area_50_deg2: float | None = None
    area_90_deg2: float | None = None
    far_hz: float | None = None
    #: p(BNS), p(NSBH), p(BBH), p(Terrestrial). Empty for searches that do not
    #: report one, e.g. the sub-solar-mass search.
    classification: dict[str, float] = field(default_factory=dict)
    #: HasNS, HasRemnant, HasMassGap, HasSSM.
    properties: dict[str, float] = field(default_factory=dict)
    #: Median source-frame chirp mass, from mchirp_source.json.
    chirp_mass_median: float | None = None
    #: P(chirp mass > 44 Msun), the quantity BBH_case_A cuts on.
    chirp_mass_prob_above_44: float | None = None
    group: str | None = None
    pipeline: str | None = None
    search: str | None = None
    significant: bool | None = None
    #: alert_type of the notice the metadata came from
    #: (PRELIMINARY/INITIAL/UPDATE).
    notice_alert_type: str | None = None
    gracedb_url: str | None = None
    #: Populated when enrichment was attempted and failed, for the Slack note.
    error: str | None = None

    @property
    def far_per_year(self) -> float | None:
        """False alarm rate expressed as events per year."""
        if self.far_hz is None:
            return None
        return self.far_hz * 3.15576e7

    @property
    def inverse_far_years(self) -> float | None:
        """One over the false alarm rate, in years -- the usual way to quote
        it.
        """
        per_year = self.far_per_year
        if per_year is None or per_year <= 0:
            return None
        return 1.0 / per_year

    @property
    def most_likely_class(self) -> tuple[str, float] | None:
        """Highest-probability CBC classification, or None if unavailable."""
        if not self.classification:
            return None
        name, value = max(self.classification.items(), key=lambda kv: kv[1])
        return name, float(value)


@dataclass
class Trigger:
    """A single ToO alert, decoded and augmented.

    The first block of fields maps one-to-one onto the producer's Avro schema
    (``rubin-ToO-producer/output_schema.json``); the rest is derived locally.
    """

    # --- straight from the too_alert record ---
    source: str
    alert_type: str
    instruments: list[str]
    event_trigger_timestamp: str
    reward_map: np.ndarray
    reward_map_nside: int
    is_test: bool
    is_update: bool
    producer_timestamp_ms: int

    # --- derived ---
    localization: Localization
    geometry: Geometry
    event_time: Time | None = None
    received_at: Time | None = None
    enrichment: GwEnrichment | None = None
    #: The decoded record as received, for debugging and replay.
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def latency_s(self) -> float | None:
        """Seconds between the physical event and the producer forwarding it.

        Returns
        -------
        latency : `float` or None
            None when `event_time` could not be parsed. This is the only
            latency proxy available: the producer never emits the alert
            submission time.
        """
        if self.event_time is None:
            return None
        return self.producer_timestamp_ms / 1e3 - self.event_time.unix

    @property
    def age_s(self) -> float | None:
        """Seconds between the physical event and chatterbox receiving it."""
        if self.event_time is None or self.received_at is None:
            return None
        return self.received_at.unix - self.event_time.unix

    @property
    def clean_instruments(self) -> list[str]:
        """Instrument list with the schema's padding removed.

        The producer pads or truncates `instrument` to exactly three entries,
        so empty strings are padding and must not be displayed.
        """
        return [i for i in self.instruments if i]
