"""Best-effort enrichment of gravitational-wave triggers from GraceDB.

The producer's output record carries no probability density, distance, FAR or
classification: it computes them, cuts on them, logs them and discards them.
For a GW alert the ``source`` field is the GraceDB superevent id, so the real
skymap and event properties can be recovered from the public API.

This is strictly optional and time-bounded. If anything fails the trigger keeps
its area-based localization and the failure is recorded in
`~chatterbox.models.GwEnrichment.error` so the Slack post can say so, rather
than quietly presenting an area fraction as a probability.
"""

import logging
import re
import time
from pathlib import Path

import healpy as hp
import numpy as np

from ..astro.skymap import credible_mask, geometry_from_mask, localization_from_probability
from ..config import EnrichConfig
from ..models import GwEnrichment, Trigger

__all__ = [
    "is_superevent_id",
    "enrich_gravitational_wave",
    "SKYMAP_PREFERENCE",
    "WORKING_NSIDE",
]

logger = logging.getLogger(__name__)

#: GraceDB superevent ids: S for real candidates, MS/TS for mock and test.
_SUPEREVENT_RE = re.compile(r"^(?:S|MS|TS)\d{6}[a-z]+$")

#: Skymap files to try, best first. Multi-order maps are smallest to download.
SKYMAP_PREFERENCE = (
    "bayestar.multiorder.fits",
    "bayestar.fits.gz",
    "bayestar.fits",
    "Bilby.multiorder.fits",
    "bilby.multiorder.fits",
)

#: Resolution the probability map is degraded to. Full-resolution BAYESTAR maps
#: can be nside 2048 (50M pixels); this keeps plotting and coverage fast while
#: staying far finer than the nside-32 reward map it replaces.
WORKING_NSIDE = 512


def is_superevent_id(source: str) -> bool:
    """True if a ``source`` looks like a GraceDB superevent id."""
    return bool(_SUPEREVENT_RE.match(source or ""))


def _client(config: EnrichConfig):
    """Construct an unauthenticated GraceDB client for public data."""
    from ligo.gracedb.rest import GraceDb

    # Public superevents need no credentials; asking for them would fail on a
    # host with no certificate. fail_if_noauth=False keeps it anonymous.
    return GraceDb(service_url=config.gracedb_service_url, fail_if_noauth=False)


def _download_skymap(client, superevent_id: str, cache_dir: Path, deadline: float) -> Path | None:
    """Download the best available skymap, or return None.

    Uses a cached copy when present so replays and the simulation do not
    re-fetch.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    for name in SKYMAP_PREFERENCE:
        cached = cache_dir / f"{superevent_id}_{name}"
        if cached.is_file() and cached.stat().st_size > 0:
            logger.info("Using cached skymap %s", cached)
            return cached

    try:
        available = set(client.files(superevent_id).json().keys())
    except Exception as exc:
        logger.warning("Could not list GraceDB files for %s: %s", superevent_id, exc)
        available = set()

    for name in SKYMAP_PREFERENCE:
        if available and name not in available:
            continue
        if time.monotonic() > deadline:
            logger.warning("Ran out of time before downloading a skymap for %s", superevent_id)
            return None
        target = cache_dir / f"{superevent_id}_{name}"
        try:
            payload = client.files(superevent_id, name).read()
        except Exception as exc:
            logger.debug("Skymap %s unavailable for %s: %s", name, superevent_id, exc)
            continue
        target.write_bytes(payload)
        logger.info("Downloaded %s (%.1f kB) to %s", name, len(payload) / 1024, target)
        return target

    logger.warning("No usable skymap found for %s (available: %s)", superevent_id, sorted(available))
    return None


def _read_probability(path: Path) -> tuple[np.ndarray, dict]:
    """Read a skymap as a RING probability map at `WORKING_NSIDE`.

    Returns
    -------
    prob : `numpy.ndarray`
        Probability per pixel, RING ordering, summing to 1.
    meta : `dict`
        Header metadata, including distance moments when present.
    """
    from ligo.skymap.io import read_sky_map

    # moc=False rasterizes a multi-order file to a flat map.
    prob, meta = read_sky_map(str(path), moc=False, nest=True)
    prob = np.asarray(prob, dtype=float)
    in_nside = hp.get_nside(prob)

    if in_nside != WORKING_NSIDE:
        # power=-2 conserves the summed probability across the resolution
        # change.
        prob = hp.ud_grade(prob, WORKING_NSIDE, order_in="NESTED", order_out="RING", power=-2)
    else:
        prob = hp.reorder(prob, n2r=True)

    total = prob.sum()
    if total <= 0:
        raise ValueError(f"Skymap {path} has zero total probability")
    return prob / total, dict(meta or {})


#: Alert notice files, most authoritative first. These are the *original* LVK
#: alert records, so one fetch recovers all the metadata the producer
#: discarded.
NOTICE_PREFERENCE = ("-update.json", "-initial.json", "-preliminary.json")


def _fetch_notice(client, superevent_id: str, enrichment: GwEnrichment) -> None:
    """Populate `enrichment` from the LVK alert notice JSON.

    The notice carries ``event.far``, ``event.classification`` and
    ``event.properties`` -- exactly the fields ``forward_alerts.py`` cuts on
    and
    then drops. ``p_astro.json`` is tried as a fallback because some
    superevents publish one instead.
    """
    import json

    for suffix in NOTICE_PREFERENCE:
        name = f"{superevent_id}{suffix}"
        try:
            payload = client.files(superevent_id, name).read()
        except Exception:
            continue
        try:
            notice = json.loads(payload)
        except Exception as exc:
            logger.debug("Could not parse %s: %s", name, exc)
            continue

        event = notice.get("event") or {}
        enrichment.notice_alert_type = notice.get("alert_type")
        if event.get("far") is not None:
            enrichment.far_hz = float(event["far"])
        classification = event.get("classification") or {}
        if classification:
            enrichment.classification = {str(k): float(v) for k, v in classification.items()}
        properties = event.get("properties") or {}
        if properties:
            enrichment.properties = {str(k): float(v) for k, v in properties.items()}
        enrichment.group = event.get("group")
        enrichment.pipeline = event.get("pipeline")
        enrichment.search = event.get("search")
        if event.get("significant") is not None:
            enrichment.significant = bool(event["significant"])
        logger.info("Read alert metadata for %s from %s", superevent_id, name)
        return

    # Fallbacks for superevents without a notice JSON attached.
    for name in ("p_astro.json", "bayestar.p_astro.json"):
        try:
            data = json.loads(client.files(superevent_id, name).read())
            enrichment.classification = {str(k): float(v) for k, v in data.items()}
            break
        except Exception:
            continue
    if not enrichment.properties:
        try:
            data = json.loads(client.files(superevent_id, "em_bright.json").read())
            enrichment.properties = {str(k): float(v) for k, v in data.items()}
        except Exception:
            pass


def _fetch_chirp_mass(client, superevent_id: str, enrichment: GwEnrichment) -> None:
    """Summarize ``mchirp_source.json`` if present.

    GraceDB serves this as ``{"bin_edges": [...], "probabilities": [...]}``
    with
    one more edge than probability; a bare two-element list is also accepted
    since that is the shape ``forward_alerts.py`` validates. The 44 Msun
    threshold is the one ``BBH_case_A`` cuts on.
    """
    import json

    try:
        data = json.loads(client.files(superevent_id, "mchirp_source.json").read())
    except Exception:
        return
    try:
        if isinstance(data, dict):
            edges = np.asarray(data["bin_edges"], dtype=float)
            probs = np.asarray(data["probabilities"], dtype=float)
        else:
            edges = np.asarray(data[0], dtype=float)
            probs = np.asarray(data[1], dtype=float)
        if edges.size != probs.size + 1:
            logger.debug("mchirp_source.json for %s has an unexpected shape", superevent_id)
            return
        total = probs.sum()
        if total <= 0:
            return
        probs = probs / total
        centers = 0.5 * (edges[:-1] + edges[1:])
        # searchsorted rather than interpolation: the distribution is often a
        # single populated bin, which makes the CDF flat and interpolation
        # ill-defined.
        median_bin = int(np.searchsorted(np.cumsum(probs), 0.5))
        median_bin = min(median_bin, centers.size - 1)
        enrichment.chirp_mass_median = float(centers[median_bin])
        enrichment.chirp_mass_prob_above_44 = float(probs[centers > 44.0].sum())
    except Exception as exc:
        logger.debug("Could not summarize mchirp_source.json for %s: %s", superevent_id, exc)


def enrich_gravitational_wave(
    trigger: Trigger,
    config: EnrichConfig,
) -> GwEnrichment | None:
    """Enrich a GW trigger in place from GraceDB.

    On success `trigger.localization` is replaced with a real probability map
    and `trigger.geometry` is recomputed from its 90% credible region, so every
    downstream percentage becomes a genuine probability. On any failure the
    trigger is left untouched and the returned enrichment carries ``error``.

    Parameters
    ----------
    trigger : `Trigger`
        Trigger to enrich. Mutated on success.
    config : `EnrichConfig`
        Enrichment settings, including the time budget.

    Returns
    -------
    enrichment : `GwEnrichment` or None
        None when the source is not a superevent id or enrichment is disabled.
    """
    if not config.gracedb:
        return None
    if not is_superevent_id(trigger.source):
        logger.debug("%s is not a superevent id; skipping GraceDB enrichment", trigger.source)
        return None

    deadline = time.monotonic() + config.timeout_s
    enrichment = GwEnrichment(
        superevent_id=trigger.source,
        gracedb_url=f"https://gracedb.ligo.org/superevents/{trigger.source}/view/",
    )

    try:
        client = _client(config)
    except Exception as exc:
        enrichment.error = f"GraceDB client unavailable: {exc}"
        logger.warning("%s", enrichment.error)
        return enrichment

    # Rich metadata from the alert notice: FAR, classification, properties.
    # Every step here is non-fatal -- a partial result is still worth posting.
    try:
        _fetch_notice(client, trigger.source, enrichment)
    except Exception as exc:
        logger.info("Could not read alert notice for %s: %s", trigger.source, exc)

    if enrichment.far_hz is None:
        try:
            far = client.superevent(trigger.source).json().get("far")
            if far is not None:
                enrichment.far_hz = float(far)
        except Exception as exc:
            logger.info("Could not read superevent record for %s: %s", trigger.source, exc)

    try:
        _fetch_chirp_mass(client, trigger.source, enrichment)
    except Exception as exc:
        logger.info("Could not read chirp mass for %s: %s", trigger.source, exc)

    skymap_path = None
    try:
        cache_dir = Path(config.cache_dir).expanduser()
        skymap_path = _download_skymap(client, trigger.source, cache_dir, deadline)
    except Exception as exc:
        enrichment.error = f"skymap download failed: {exc}"
        logger.warning("%s", enrichment.error)

    if skymap_path is None:
        if enrichment.error is None:
            enrichment.error = "no skymap available from GraceDB"
        return enrichment

    enrichment.skymap_path = str(skymap_path)

    try:
        prob, meta = _read_probability(skymap_path)
    except Exception as exc:
        enrichment.error = f"could not read skymap: {exc}"
        logger.warning("%s for %s", enrichment.error, trigger.source)
        return enrichment

    nside = hp.get_nside(prob)
    pixel_area = hp.nside2pixarea(nside, degrees=True)
    mask_50 = credible_mask(prob, 0.5)
    mask_90 = credible_mask(prob, 0.9)
    enrichment.area_50_deg2 = float(mask_50.sum() * pixel_area)
    enrichment.area_90_deg2 = float(mask_90.sum() * pixel_area)

    for key, attr in (("distmean", "distance_mean_mpc"), ("diststd", "distance_std_mpc")):
        value = meta.get(key)
        if value is not None and np.isfinite(value):
            setattr(enrichment, attr, float(value))

    trigger.localization = localization_from_probability(
        prob,
        provenance=(
            f"GraceDB {skymap_path.name.split('_', 1)[-1]} for {trigger.source}, "
            f"rasterized to nside={nside}"
        ),
        credible_level=0.9,
    )
    # Geometry now describes the real 90% credible region rather than the
    # producer's quantized 70% reward map.
    trigger.geometry = geometry_from_mask(mask_90, nside)
    trigger.enrichment = enrichment

    logger.info(
        "Enriched %s: 90%% area %.1f deg^2 (reward map implied %.0f), distance %s Mpc",
        trigger.source,
        enrichment.area_90_deg2,
        trigger.reward_map.sum() * hp.nside2pixarea(trigger.reward_map_nside, degrees=True),
        f"{enrichment.distance_mean_mpc:.0f}" if enrichment.distance_mean_mpc else "unknown",
    )
    return enrichment
