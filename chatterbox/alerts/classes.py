"""Registry mapping ``alert_type`` labels onto messenger class and strategy.

The ``alert_type`` string is the only classification information the producer's
output record carries, so this table is what turns it into something a human
can read: which messenger it came from, why it passed the producer's cuts, and
what Rubin will actually do about it.

Follow-up strategies are read from ``ts_fbs_utils`` at runtime so they cannot
drift from what the scheduler really runs; a vendored snapshot is used when
that package is unavailable (for example in a bot-only environment).

Notes
-----
Only eight labels can actually be emitted by ``forward_alerts.py``:
``GW_case_B``, ``GW_case_D``, ``GW_case_large``, ``lensed_BNS_case_A``,
``lensed_BNS_case_B``, ``BBH_case_A``, ``neutrino`` and ``SN_Galactic``. The
remaining entries exist because ``ts_fbs_utils`` defines strategies for them
and a producer may start emitting them; they are marked ``produced=False`` so
a reader cannot mistake them for live coverage.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache

__all__ = [
    "MessengerClass",
    "StrategyEpoch",
    "FollowupStrategy",
    "AlertClass",
    "REGISTRY",
    "get_alert_class",
    "get_strategy",
    "sim_nights_for",
]

logger = logging.getLogger(__name__)

#: Steradians per square degree, for reporting the producer's cuts in deg^2.
_SR_PER_DEG2 = 3.0461741978670857e-4


class MessengerClass(Enum):
    """Broad messenger category, used to pick the Slack renderer."""

    GRAVITATIONAL_WAVE = "Gravitational wave"
    NEUTRINO = "High-energy neutrino"
    GALACTIC_SN = "Galactic supernova"
    SOLAR_SYSTEM = "Potentially hazardous asteroid"

    @property
    def emoji(self) -> str:
        """Emoji used in the Slack header for this messenger."""
        return {
            "GRAVITATIONAL_WAVE": ":ocean:",
            "NEUTRINO": ":snowflake:",
            "GALACTIC_SN": ":star2:",
            "SOLAR_SYSTEM": ":comet:",
        }[self.name]


@dataclass(frozen=True)
class StrategyEpoch:
    """One scheduled follow-up epoch.

    Parameters
    ----------
    t_hours : `float`
        Hours after the ToO trigger time.
    bands : `str`
        Concatenated band letters requested at this epoch, e.g. ``"gri"``.
    nvis : `int`
        Number of visits requested per band.
    exptime : `float`
        Exposure time in seconds.
    """

    t_hours: float
    bands: str
    nvis: int
    exptime: float

    @property
    def band_list(self) -> list[str]:
        """The requested bands as a list of single-character names."""
        return list(self.bands)


@dataclass
class FollowupStrategy:
    """The scheduler's planned response to one alert type."""

    alert_type: str
    epochs: list[StrategyEpoch]
    #: True when read live from ts_fbs_utils, False when using the snapshot.
    from_ts_fbs_utils: bool = False

    @property
    def bands(self) -> list[str]:
        """All bands the strategy requests, in canonical LSST order."""
        wanted = {b for epoch in self.epochs for b in epoch.band_list}
        return [b for b in ("u", "g", "r", "i", "z", "y") if b in wanted]

    @property
    def span_hours(self) -> float:
        """Hours from the first to the last scheduled epoch."""
        if not self.epochs:
            return 0.0
        return max(e.t_hours for e in self.epochs) - min(e.t_hours for e in self.epochs)

    def visits_per_band(self) -> dict[str, int]:
        """Total requested visits summed per band across all epochs."""
        totals: dict[str, int] = {}
        for epoch in self.epochs:
            for band in epoch.band_list:
                totals[band] = totals.get(band, 0) + epoch.nvis
        return totals

    @property
    def total_visits(self) -> int:
        """Total requested visits across all bands and epochs."""
        return sum(self.visits_per_band().values())


@dataclass(frozen=True)
class AlertClass:
    """Static description of one ``alert_type`` label.

    Parameters
    ----------
    alert_type : `str`
        The label as emitted by ``forward_alerts.py``.
    messenger : `MessengerClass`
        Which messenger produced it.
    display_name : `str`
        Short human-readable name for the Slack header.
    criteria : `str`
        Why this label was assigned, in terms of the producer's cuts. Shown so
        a reader can tell what the alert is without opening the producer
        source.
    sim_nights : `int`
        Nights to simulate for this class.
    produced : `bool`
        False when no current producer code path emits this label.
    """

    alert_type: str
    messenger: MessengerClass
    display_name: str
    criteria: str
    sim_nights: int
    produced: bool = True
    notes: str = ""


def _far_text() -> str:
    return "FAR < 3.17e-8 Hz (about 1 per year)"


REGISTRY: dict[str, AlertClass] = {
    "GW_case_B": AlertClass(
        alert_type="GW_case_B",
        messenger=MessengerClass.GRAVITATIONAL_WAVE,
        display_name="BNS/NSBH, well localized (gold)",
        criteria=(
            f"p(BNS)+p(NSBH) >= 0.9, {_far_text()}, 90% area < 100 deg^2, " "HasNS >= 0.5, HasRemnant >= 0.5"
        ),
        sim_nights=20,
    ),
    "GW_case_D": AlertClass(
        alert_type="GW_case_D",
        messenger=MessengerClass.GRAVITATIONAL_WAVE,
        display_name="BNS/NSBH, moderately localized (silver)",
        criteria=(
            f"p(BNS)+p(NSBH) >= 0.9, {_far_text()}, 90% area 100-500 deg^2, "
            "HasNS >= 0.5, HasRemnant >= 0.5"
        ),
        sim_nights=20,
    ),
    "GW_case_large": AlertClass(
        alert_type="GW_case_large",
        messenger=MessengerClass.GRAVITATIONAL_WAVE,
        display_name="BNS/NSBH, very large localization",
        criteria="p(BNS)+p(NSBH) >= 0.9 and 90% area >= 1000 deg^2 (no FAR cut applied)",
        sim_nights=20,
        notes="The producer logs this branch as not definitively implemented.",
    ),
    "BBH_case_A": AlertClass(
        alert_type="BBH_case_A",
        messenger=MessengerClass.GRAVITATIONAL_WAVE,
        display_name="BBH, well localized",
        criteria=("90% area < 20 deg^2, mean distance < 6000 Mpc, " "P(chirp mass > 44 Msun) > 0.8"),
        sim_nights=50,
    ),
    "lensed_BNS_case_A": AlertClass(
        alert_type="lensed_BNS_case_A",
        messenger=MessengerClass.GRAVITATIONAL_WAVE,
        display_name="Candidate lensed BNS, larger localization",
        criteria=(f"HasMassGap >= 0.9, p(NSBH) < 0.1, {_far_text()}, 90% area 15-900 deg^2"),
        sim_nights=20,
    ),
    "lensed_BNS_case_B": AlertClass(
        alert_type="lensed_BNS_case_B",
        messenger=MessengerClass.GRAVITATIONAL_WAVE,
        display_name="Candidate lensed BNS, tight localization",
        criteria=f"HasMassGap >= 0.9, p(NSBH) < 0.1, {_far_text()}, 90% area < 15 deg^2",
        sim_nights=20,
    ),
    "neutrino": AlertClass(
        alert_type="neutrino",
        messenger=MessengerClass.NEUTRINO,
        display_name="High-energy neutrino",
        criteria=(
            "p_astro >= 0.5, |galactic b| > 10 deg, "
            f"70% area < {0.002924 / _SR_PER_DEG2:.1f} deg^2 (one camera field of view)"
        ),
        sim_nights=30,
    ),
    "SN_Galactic": AlertClass(
        alert_type="SN_Galactic",
        messenger=MessengerClass.GALACTIC_SN,
        display_name="Galactic supernova",
        criteria=(
            "Circular 68% localization with area <= 100 deg^2 "
            "(radius <= 5.64 deg); no declination cut is applied upstream"
        ),
        sim_nights=5,
        notes="Tiles continuously in i band at 1 s and 15 s until a counterpart is identified.",
    ),
    "SSO_night": AlertClass(
        alert_type="SSO_night",
        messenger=MessengerClass.SOLAR_SYSTEM,
        display_name="Potentially hazardous asteroid (night)",
        criteria="Not yet defined: no producer emits this label.",
        sim_nights=5,
        produced=False,
        notes="No asteroid/PHA handler exists in rubin-ToO-producer; rendering is untested on real data.",
    ),
    "SSO_twilight": AlertClass(
        alert_type="SSO_twilight",
        messenger=MessengerClass.SOLAR_SYSTEM,
        display_name="Potentially hazardous asteroid (twilight)",
        criteria="Not yet defined: no producer emits this label.",
        sim_nights=5,
        produced=False,
        notes="No asteroid/PHA handler exists in rubin-ToO-producer; rendering is untested on real data.",
    ),
    # Assigned by the producer's criteria blocks but never emitted on their
    # own;
    # included so an unexpected record still renders with the right messenger.
    "GW_case_A": AlertClass(
        alert_type="GW_case_A",
        messenger=MessengerClass.GRAVITATIONAL_WAVE,
        display_name="Gravitational wave (case A)",
        criteria="Not currently reachable in the producer.",
        sim_nights=20,
        produced=False,
    ),
    "GW_case_C": AlertClass(
        alert_type="GW_case_C",
        messenger=MessengerClass.GRAVITATIONAL_WAVE,
        display_name="Gravitational wave, unidentified type",
        criteria="Assigned by the producer but not accompanied by a pass decision.",
        sim_nights=20,
        produced=False,
    ),
    "GW_case_E": AlertClass(
        alert_type="GW_case_E",
        messenger=MessengerClass.GRAVITATIONAL_WAVE,
        display_name="Gravitational wave, unidentified type",
        criteria="Assigned by the producer but not accompanied by a pass decision.",
        sim_nights=20,
        produced=False,
    ),
}


def get_alert_class(alert_type: str) -> AlertClass:
    """Look up an alert class, guessing the messenger for unknown labels.

    Parameters
    ----------
    alert_type : `str`
        Label from the ToO record.

    Returns
    -------
    alert_class : `AlertClass`
        A registry entry, or a synthesized entry for an unrecognized label so
        that an unexpected record still produces a usable post.
    """
    if alert_type in REGISTRY:
        return REGISTRY[alert_type]

    logger.warning("Unrecognized alert_type %r; falling back to a generic description", alert_type)
    lowered = alert_type.lower()
    if "neutrino" in lowered:
        messenger, nights = MessengerClass.NEUTRINO, 30
    elif "sn_galactic" in lowered or "supernova" in lowered:
        messenger, nights = MessengerClass.GALACTIC_SN, 5
    elif "sso" in lowered or "asteroid" in lowered:
        messenger, nights = MessengerClass.SOLAR_SYSTEM, 5
    elif "bbh" in lowered:
        messenger, nights = MessengerClass.GRAVITATIONAL_WAVE, 50
    else:
        # Everything the producer emits that is not covered above is a GW
        # class, including the lensed_BNS labels.
        messenger, nights = MessengerClass.GRAVITATIONAL_WAVE, 20
    return AlertClass(
        alert_type=alert_type,
        messenger=messenger,
        display_name=alert_type.replace("_", " "),
        criteria="Unknown label: not present in the chatterbox registry.",
        sim_nights=nights,
        produced=False,
        notes="This label is not known to chatterbox; the messenger was inferred from its name.",
    )


def sim_nights_for(alert_type: str) -> int:
    """Nights to simulate for an alert type.

    Notes
    -----
    Mirrors ``getSimNightsFromCSVEntry`` in the ToO sim drivers, but adds the
    ``lensed_BNS_*`` labels, which that function raises ``ValueError`` on.
    """
    return get_alert_class(alert_type).sim_nights


# Snapshot of ts_fbs_utils strategies, used when that package is unavailable.
# Times are hours after the trigger. Kept in sync with
# lsst.ts.fbs.utils.maintel.too_surveys.gen_too_surveys.
_SNAPSHOT: dict[str, list[tuple[float, str, int, float]]] = {
    "GW_case_A": [(0, "ugrizy", 3, 120.0), (24, "ugrizy", 1, 120.0), (48, "ugrizy", 1, 120.0)],
    "GW_case_B": [
        (0, "gri", 4, 30.0),
        (2, "gri", 4, 30.0),
        (4, "gri", 4, 30.0),
        (24, "ri", 6, 30.0),
        (48, "ri", 6, 30.0),
        (72, "ri", 6, 30.0),
    ],
    "GW_case_D": [(0, "gi", 1, 30.0), (24, "gi", 4, 30.0), (48, "gi", 4, 30.0), (72, "gi", 4, 30.0)],
    "GW_case_large": [
        (0, "gi", 1, 30.0),
        (48, "gi", 1, 30.0),
        (96, "gi", 1, 30.0),
        (144, "gi", 1, 30.0),
    ],
    "BBH_case_A": [
        (0, "rzi", 1, 30.0),
        (48, "rzi", 1, 30.0),
        (168, "rzi", 1, 30.0),
        (216, "rzi", 1, 30.0),
        (936, "rzi", 1, 30.0),
    ],
    "lensed_BNS_case_A": [
        (1.0, "g", 1, 30.0),
        (1.0, "r", 3, 30.0),
        (25.0, "g", 1, 30.0),
        (25.0, "r", 3, 30.0),
        (49.0, "g", 1, 30.0),
        (49.0, "r", 3, 30.0),
    ],
    "lensed_BNS_case_B": [
        (1.0, "g", 180, 30.0),
        (1.0, "r", 120, 30.0),
        (25.0, "g", 180, 30.0),
        (25.0, "r", 120, 30.0),
        (49.0, "g", 180, 30.0),
        (49.0, "r", 120, 30.0),
    ],
    "neutrino": [
        (0.0, "u", 1, 30.0),
        (0.0, "g", 4, 30.0),
        (0.25, "r", 1, 30.0),
        (0.0, "z", 1, 30.0),
        (24.0, "g", 4, 30.0),
        (24.0, "r", 1, 30.0),
        (144.0, "g", 4, 30.0),
        (144.0, "rz", 1, 30.0),
    ],
    "SSO_night": [(0.0, "r", 1, 30.0), (0.55, "r", 1, 30.0), (1.1, "r", 1, 30.0)],
    "SSO_twilight": [(0.0, "z", 2, 15.0), (0.167, "z", 2, 15.0), (0.333, "z", 2, 15.0)],
    "SN_Galactic": [(0.0, "i", 1, exp) for exp in (1.0, 15.0) * 8],
}

# Aliases: ts_fbs_utils groups several labels onto one survey.
_SNAPSHOT["GW_case_C"] = _SNAPSHOT["GW_case_B"]
_SNAPSHOT["GW_case_E"] = _SNAPSHOT["GW_case_D"]
_SNAPSHOT["BBH_case_B"] = _SNAPSHOT["BBH_case_A"]
_SNAPSHOT["BBH_case_C"] = _SNAPSHOT["BBH_case_A"]


@lru_cache(maxsize=1)
def _live_strategies() -> dict[str, list[StrategyEpoch]] | None:
    """Read the strategy table out of ts_fbs_utils, or None if unavailable.

    Notes
    -----
    ``ToOScriptedSurvey.times`` is stored in **days**; it is converted to hours
    here. The result is cached because building the surveys costs ~1 s.
    """
    try:
        import healpy as hp
        import numpy as np
        from lsst.ts.fbs.utils.maintel import too_surveys
    except ImportError as exc:  # pragma: no cover - depends on environment
        logger.info("ts_fbs_utils unavailable (%s); using the vendored strategy snapshot", exc)
        return None

    try:
        import warnings

        nside = 32
        with warnings.catch_warnings():
            # gen_too_surveys passes a deprecated kwarg internally; the warning
            # is about upstream's own call, not ours.
            warnings.simplefilter("ignore", FutureWarning)
            surveys = too_surveys.gen_too_surveys(
                nside=nside,
                detailer_list=[],
                too_footprint=np.ones(hp.nside2npix(nside)),
            )
    except Exception as exc:  # pragma: no cover - upstream API drift
        logger.warning("Could not build ToO surveys from ts_fbs_utils (%s); using snapshot", exc)
        return None

    table: dict[str, list[StrategyEpoch]] = {}
    for survey in surveys:
        try:
            times = np.asarray(survey.times, dtype=float) * 24.0
            bands = list(survey.bands_at_times)
            nvis = list(survey.nvis)
            exptimes = list(survey.exptimes)
            labels = list(survey.too_types_to_follow)
        except AttributeError as exc:  # pragma: no cover - upstream API drift
            logger.warning("Skipping a ToO survey with unexpected attributes: %s", exc)
            continue
        epochs = [
            StrategyEpoch(t_hours=float(t), bands=str(b), nvis=int(n), exptime=float(e))
            for t, b, n, e in zip(times, bands, nvis, exptimes)
        ]
        for label in labels:
            table[label] = epochs
    logger.debug("Loaded %d ToO strategies from ts_fbs_utils", len(table))
    return table


def get_strategy(alert_type: str) -> FollowupStrategy:
    """Return the scheduler's follow-up strategy for an alert type.

    Prefers the live ``ts_fbs_utils`` definition and falls back to the vendored
    snapshot. An unknown label yields an empty strategy rather than raising, so
    a novel alert still produces a post.
    """
    live = _live_strategies()
    if live is not None and alert_type in live:
        return FollowupStrategy(alert_type=alert_type, epochs=live[alert_type], from_ts_fbs_utils=True)

    snapshot = _SNAPSHOT.get(alert_type)
    if snapshot is None:
        logger.warning("No follow-up strategy known for alert_type %r", alert_type)
        return FollowupStrategy(alert_type=alert_type, epochs=[])
    epochs = [StrategyEpoch(t_hours=t, bands=b, nvis=n, exptime=e) for t, b, n, e in snapshot]
    return FollowupStrategy(alert_type=alert_type, epochs=epochs, from_ts_fbs_utils=False)
