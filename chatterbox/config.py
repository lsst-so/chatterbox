"""Configuration for the chatterbox service.

Settings come from a YAML file, with a small number of environment variable
overrides for anything secret. Secrets are *never* stored in the YAML: the
config only names the environment variable to read them from.

Notes
-----
Kafka topic names are deliberately absent from the defaults. The upstream
``rubin-ToO-producer`` repository contains no real topic names either -- they
are purely deployment configuration -- so they must be supplied in the YAML.
"""

import logging
import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "Config",
    "SlackConfig",
    "IngestConfig",
    "DarkHoursConfig",
    "TemplateConfig",
    "EnrichConfig",
    "SimConfig",
    "PathsConfig",
    "LinksConfig",
    "SITES",
    "apply_site_defaults",
    "load_config",
    "apply_environment",
    "DEFAULT_CONFIG_PATHS",
]

logger = logging.getLogger(__name__)

#: What the top-level ``site`` setting supplies, per site.
#:
#: One deployment lives at one site, but the services it talks to are named
#: differently by each library: an EFD instance is ``summit_efd``, the matching
#: ConsDB site is ``summit``, and the RSP token is ``~/.lsst/summit_rsp``.
#: Setting all three by hand is three chances to end up polling one site's EFD
#: while pulling visit history from another -- which works, and is wrong.
#:
#: The ConsDB names are exactly the ones ``rubin_nights`` accepts
#: (``rubin_nights.connections.API_ENDPOINTS``). ``idf`` is deliberately not
#: here: an ``idf_efd`` exists, but ``rubin_nights`` has no ConsDB endpoint for
#: it, so set ``ingest.efd_name`` directly rather than implying a whole site.
SITES: dict[str, dict[str, str]] = {
    "summit": {
        "efd_name": "summit_efd",
        "opsim_site": "summit",
        "opsim_tokenfile": "~/.lsst/summit_rsp",
    },
    "base": {
        "efd_name": "base_efd",
        "opsim_site": "base",
        "opsim_tokenfile": "~/.lsst/base_rsp",
    },
    "usdf": {
        "efd_name": "usdf_efd",
        "opsim_site": "usdf",
        "opsim_tokenfile": "~/.lsst/usdf_rsp",
    },
    "usdf-dev": {
        "efd_name": "usdf_efd",
        "opsim_site": "usdf-dev",
        "opsim_tokenfile": "~/.lsst/usdf_rsp",
    },
}

#: Where each `SITES` entry lands, as ``(section, key)``.
_SITE_TARGETS = (
    ("ingest", "efd_name"),
    ("sim", "opsim_site"),
    ("sim", "opsim_tokenfile"),
)

DEFAULT_CONFIG_PATHS = (
    Path("config.yaml"),
    Path("~/.config/chatterbox/config.yaml").expanduser(),
)

#: Last data directory `apply_environment` diagnosed, so a bad path is reported
#: once rather than on every alert.
_validated_data_dir: Path | None = None


@dataclass
class SlackConfig:
    """Slack destination and credentials.

    Notes
    -----
    A bot token is required rather than an incoming webhook because incoming
    webhooks cannot upload files, and every ToO post carries plots.
    """

    bot_token_env: str = "SLACK_BOT_TOKEN"
    channel: str = "#too-alerts"
    #: Channel used when ``is_test`` is set on the alert. Falls back to
    #: ``channel`` when empty, in which case test alerts are still visibly
    #: marked in the message itself.
    test_channel: str = ""
    username: str = "chatterbox"
    icon_emoji: str = ":satellite:"
    #: Slack user/group IDs to mention on non-test alerts, e.g.
    #: ``["!subteam^S123"]``.
    mention: list[str] = field(default_factory=list)


@dataclass
class IngestConfig:
    """Where ToO alert records are read from.

    ``kind`` selects the transport:

    - ``efd``: poll the EFD for new ``too_alert`` records. This is the default:
      it needs no broker subscription, and on a summit or USDF host the
      credentials are already there.
    - ``kafka``: subscribe to the producer's output topic with hop-client.
    - ``files``: watch a directory that ``forward_alerts.py``'s ``FileSender``
      writes ``{source}.json`` into.
    - ``replay``: read explicit paths once, then stop (used by the CLI).
    """

    kind: str = "efd"
    #: hop URL, e.g. ``kafka://kafka.scimma.org/lsst.rubin-too-alerts``.
    kafka_url: str = ""
    kafka_group_id: str = "chatterbox"
    #: Directory watched by the ``files`` source.
    watch_dir: str = "~/.chatterbox/incoming"
    #: Seconds between directory scans.
    poll_interval_s: float = 2.0

    #: EFD instance: ``summit_efd``, ``usdf_efd``, ``idf_efd``, ``base_efd``.
    #: Empty lets an RSP host choose its own default, which only works where
    #: ``lsst.summit.utils`` is installed.
    efd_name: str = "summit_efd"
    #: InfluxDB database the alerts are written to. Deliberately not the
    #: default database, which holds the SAL topics.
    efd_database: str = "lsst.scimma"
    #: Measurement carrying the records.
    efd_topic: str = "lsst.scimma.too_alert"
    #: Seconds between EFD queries. Each one spans every pixel column of the
    #: reward map, so this is longer than the directory scan interval.
    efd_poll_interval_s: float = 10.0
    #: How far before startup the first query reaches. ``0`` means "only new
    #: alerts": chatterbox does not remember what it has posted, so a lookback
    #: makes a restart re-post anything inside it.
    efd_lookback_s: float = 0.0
    #: Consecutive failed queries tolerated before the service gives up and
    #: reports it. One failure is a blip; five in a row is an outage.
    efd_max_consecutive_errors: int = 5
    #: Process alerts flagged ``is_test``. When False they are dropped
    #: entirely.
    allow_tests: bool = True


@dataclass
class DarkHoursConfig:
    """Parameters for the accessible-dark-hours map.

    ``sun_alt_limit_deg`` of -12 and ``airmass_limit`` of 2 reproduce the
    definition used for the Slack report: hours per pixel with airmass < 2
    while the Sun is below -12 degrees.
    """

    nside: int = 64
    step_minutes: float = 5.0
    airmass_limit: float = 2.0
    sun_alt_limit_deg: float = -12.0
    #: Optionally exclude pixels close to the Moon. ``0`` disables the cut.
    moon_avoidance_deg: float = 0.0


@dataclass
class TemplateConfig:
    """Per-band LSST template coverage maps."""

    #: Directory the incremental templates tooling publishes its per-band
    #: HEALPix coverage maps to. This is the only source of template coverage;
    #: ``refresh-templates`` reads it and queries nothing.
    maps_dir: str = "/home/e/ebellm/u/workspace/incremental_templates/template_tools/output"
    #: Filename convention within `maps_dir`, e.g.
    #: ``template_coverage_healpix_y_nside64.fits``.
    map_pattern: str = "template_coverage_healpix_{band}_nside{nside}.fits"
    #: Resolution of the published maps; also appears in their filenames.
    nside: int = 64
    bands: list[str] = field(default_factory=lambda: ["u", "g", "r", "i", "z", "y"])
    #: Local cache of ``{band}.npy`` maps plus ``meta.json``, so the alert path
    #: never reads from `maps_dir` directly.
    cache_dir: str = "~/.chatterbox/templates"


@dataclass
class EnrichConfig:
    """Optional enrichment of the thin ToO record.

    The producer's output schema carries no probability density, distance,
    FAR or classification. For gravitational-wave alerts the ``source`` field
    is the GraceDB superevent id, so the real skymap can be fetched to recover
    them. This is best-effort and strictly time-bounded.
    """

    gracedb: bool = True
    gracedb_service_url: str = "https://gracedb.ligo.org/api/"
    timeout_s: float = 8.0
    #: Cache downloaded skymaps here so replays and the sim reuse them.
    cache_dir: str = "~/.chatterbox/skymaps"


@dataclass
class SimConfig:
    """Scheduler simulation launched as a background job."""

    enabled: bool = True
    #: Interpreter that runs the simulation subprocess. It needs healpy,
    #: rubin_scheduler and lsst_survey_sim. Empty means "the interpreter
    #: running chatterbox", which is usually right: if the bot can compute the
    #: almanac it can generally run the simulation too. Point this elsewhere
    #: only when the simulation genuinely needs a different environment.
    python: str = ""
    lsst_survey_sim: str = "~/Desktop/Repos/lsst_survey_sim"
    ts_config_scheduler: str = "~/Desktop/Repos/ts_config_scheduler"
    #: Checkout of ts_fbs_utils, which defines the ToO follow-up strategies and
    #: is imported by the scheduler config script. Empty relies on it being
    #: pip-installed. Its import root is ``<checkout>/python`` per the LSST
    #: layout; that is resolved automatically.
    ts_fbs_utils: str = ""
    #: Must be the tree containing ``sim_baseline``.
    rubin_sim_data: str = "~/RubinUtils/rubin_sim_data"

    #: Cached visit history the simulation starts from. It is fetched from
    #: ConsDB with ``lsst_survey_sim.fetch_previous_visits`` and refreshed
    #: automatically, so there is nothing to stage by hand.
    opsim_cache: str = "~/.chatterbox/work/opsim.parquet"
    #: Refresh the cache when it is older than this. 24 h means a ToO arriving
    #: on a night whose cache was not refreshed by cron refreshes it itself.
    #: ``0`` fetches fresh visits for every simulation.
    opsim_max_age_hours: float = 24.0
    #: RSP token for ConsDB. Empty uses the ``ACCESS_TOKEN`` environment
    #: variable, which is what ``fetch_previous_visits`` falls back to.
    opsim_tokenfile: str = "~/.lsst/usdf_rsp"
    #: ConsDB site; must match where the token came from.
    opsim_site: str = "usdf"
    #: Override the per-alert-class night count. ``0`` uses the class default.
    nights_override: int = 0
    #: Hard wall-clock cap on a simulation job.
    timeout_s: float = 7200.0
    #: Base URL that sim artifacts are published under, if any.
    artifact_base_url: str = ""

    #: Post per-night, per-band visit-count maps for the nights whose visits
    #: overlap the localization contour. These go to a new Slack thread of
    #: their own rather than into the alert's thread, which a 20-night run
    #: would otherwise flood.
    nightly_plots: bool = True
    #: Cap on figures, so a 50-night BBH run does not produce 50 uploads. The
    #: number left out is stated in the post rather than silently dropped.
    #: ``0`` means no cap.
    nightly_plots_max_nights: int = 10
    #: Resolution of the visit-count maps. These are for looking at, so a
    #: modest value keeps rendering quick.
    nightly_plots_nside: int = 256
    #: Post a single figure of coverage gained per night, per band, with the
    #: running total over it. Independent of `nightly_plots`: it is one image
    #: however long the run is, and it covers every night rather than only the
    #: ones that fit under the cap.
    nightly_coverage_plot: bool = True
    #: Filter carousel swap schedule, ``{"YYYY-MM-DD": ["g", "r", ...]}``.
    #:
    #: ``lsst_survey_sim.simulate_lsst.setup_band_scheduler`` hardcodes a
    #: schedule that ends 2026-03-25 and then falls back to picking the
    #: carousel from Moon illumination, which can silently drop requested ``u``
    #: visits. Supply a current schedule here to override it; leave empty to
    #: use the upstream default and have the choice reported in the result.
    band_swap_schedule: dict[str, list[str]] = field(default_factory=dict)
    band_swap_end_date: str = ""


@dataclass
class PathsConfig:
    """Filesystem locations for runtime state."""

    work_dir: str = "~/.chatterbox/work"


@dataclass
class LinksConfig:
    """Permalinks included in every post.

    No weather URL exists anywhere in the sibling repositories, so these are
    defaults to be edited rather than authoritative endpoints. ``extra`` lets
    additional ``label: url`` pairs be appended without a code change.
    """

    weather: str = "https://www.meteoblue.com/en/weather/week/cerro-pach%c3%b3n_chile_3897347"
    seeing: str = "https://noirlab.edu/science/observing-noirlab/weather-webcams/cerro-pachon"
    almanac: str = ""
    observatory_status: str = ""
    extra: dict[str, str] = field(default_factory=dict)


@dataclass
class Config:
    """Top-level chatterbox configuration."""

    #: Which Rubin site this instance runs at: ``summit``, ``base``, ``usdf``
    #: or ``usdf-dev``. Supplies `SITES` defaults for the settings that name a
    #: site in three different vocabularies -- the EFD instance, the ConsDB
    #: site and the RSP token file. Any of those stated explicitly in the
    #: config file still wins. Empty means "no site", leaving each setting at
    #: its own default, which is what an existing config file gets.
    site: str = ""
    slack: SlackConfig = field(default_factory=SlackConfig)
    ingest: IngestConfig = field(default_factory=IngestConfig)
    dark_hours: DarkHoursConfig = field(default_factory=DarkHoursConfig)
    templates: TemplateConfig = field(default_factory=TemplateConfig)
    enrich: EnrichConfig = field(default_factory=EnrichConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    links: LinksConfig = field(default_factory=LinksConfig)

    @property
    def slack_token(self) -> str | None:
        """Bot token from the environment, or None when unset.

        Returns
        -------
        token : `str` or None
            When None, callers should degrade to writing output locally rather
            than failing -- that is what makes the pipeline testable offline.
        """
        return os.environ.get(self.slack.bot_token_env) or None

    def work_path(self, *parts: str) -> Path:
        """Build a path under the work directory, creating its parents."""
        p = Path(self.paths.work_dir).expanduser().joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


def _from_mapping(cls: type, data: Any) -> Any:
    """Recursively build a nested dataclass from plain YAML mappings.

    Unknown keys are reported and ignored rather than silently dropped, since a
    typo in a config key would otherwise look like the setting had no effect.
    """
    if not is_dataclass(cls) or not isinstance(data, dict):
        return data
    known = {f.name: f for f in fields(cls)}
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        name = key.replace("-", "_")
        if name not in known:
            logger.warning("Ignoring unknown config key %r for %s", key, cls.__name__)
            continue
        kwargs[name] = _from_mapping(known[name].type, value)
    return cls(**kwargs)


def _was_stated(data: Any, section: str, key: str) -> bool:
    """Did the config file set ``section.key`` itself?

    Only settings the file is silent about are filled in from the site, so an
    explicit value is never overwritten by one inferred from somewhere else.
    The dashed spelling the loader accepts counts too.
    """
    if not isinstance(data, dict):
        return False
    block = data.get(section)
    if block is None:
        block = data.get(section.replace("_", "-"))
    if not isinstance(block, dict):
        return False
    return any(str(name).replace("-", "_") == key for name in block)


def apply_site_defaults(config: Config, stated: Any = None) -> Config:
    """Fill the site-dependent settings from the top-level ``site``.

    Parameters
    ----------
    config : `Config`
        Configuration to complete, in place.
    stated : `dict`, optional
        The raw YAML mapping it was built from, used to tell "the file set
        this" from "this is the field default". Without it every site-dependent
        setting is filled, which is what a caller constructing a `Config` by
        hand wants.

    Returns
    -------
    config : `Config`
        The same object, for convenience.

    Raises
    ------
    ValueError
        When ``site`` is not one of `SITES`. A typo here would otherwise be
        silent and leave the instance pointed at whichever site the defaults
        name, so it fails at startup instead.
    """
    site = (config.site or "").strip().lower()
    if not site:
        return config
    if site not in SITES:
        raise ValueError(
            f"Unknown site {config.site!r}; expected one of {', '.join(sorted(SITES))}. "
            "For an EFD with no matching ConsDB endpoint, such as idf_efd, set "
            "ingest.efd_name directly and leave site unset."
        )

    config.site = site
    defaults = SITES[site]
    applied, kept = [], []
    for section, key in _SITE_TARGETS:
        if _was_stated(stated, section, key):
            kept.append(f"{section}.{key}={getattr(getattr(config, section), key)!r}")
            continue
        setattr(getattr(config, section), key, defaults[key])
        applied.append(f"{section}.{key}={defaults[key]}")

    if applied:
        logger.info("site=%s supplies %s", site, ", ".join(applied))
    if kept:
        # Not a warning: overriding one setting for a site is a normal thing to
        # want. It is worth saying so, though, because a half-overridden site
        # is exactly the state this setting exists to make visible.
        logger.info("site=%s overridden by the config file for %s", site, ", ".join(kept))
    return config


def load_config(path: str | Path | None = None) -> Config:
    """Load configuration from YAML, falling back to defaults.

    Parameters
    ----------
    path : `str`, `pathlib.Path`, or None
        Explicit config file. When None, the first readable entry of
        `DEFAULT_CONFIG_PATHS` is used; if none exist, defaults are returned.

    Returns
    -------
    config : `Config`
    """
    candidates = [Path(path)] if path is not None else list(DEFAULT_CONFIG_PATHS)
    for candidate in candidates:
        candidate = candidate.expanduser()
        if not candidate.is_file():
            continue
        logger.info("Loading configuration from %s", candidate)
        with open(candidate) as f:
            data = yaml.safe_load(f) or {}
        # Field names use snake_case; accept the dashed spelling too so the
        # YAML can match the style used by forward_alerts.py's own config.
        config = _from_mapping(Config, data)
        # `data` is passed through so an explicit setting is not replaced by
        # one inferred from `site`.
        return apply_site_defaults(config, stated=data)

    if path is not None:
        raise FileNotFoundError(f"Config file not found: {path}")
    logger.info("No config file found; using defaults")
    return Config()


def apply_environment(config: Config) -> None:
    """Export the environment variables the Rubin libraries read.

    ``rubin_scheduler`` finds its data through ``RUBIN_SIM_DATA_DIR`` and
    silently falls back to ``$HOME/rubin_sim_data`` when it is unset, so
    ``sim.rubin_sim_data`` has to reach the process actually doing the work.
    The simulation subprocess gets it through its own environment; the alert
    path -- which needs the almanac for sunset, moonrise and the dark-hours map
    -- runs in *this* process, so it has to be set here.

    It also puts the configured ``lsst_survey_sim`` and ``ts_fbs_utils``
    checkouts on ``sys.path``. Both are normally used from a clone rather than
    pip-installed, and doing it here means every entry point -- the alert path,
    ``refresh-opsim``, ``doctor`` -- sees the same thing.

    Call this once after loading the configuration and before anything imports
    or calls into ``rubin_scheduler``. ``get_data_dir()`` reads the variable on
    every call, so setting it at runtime is enough.

    Parameters
    ----------
    config : `Config`
        Loaded configuration.

    Notes
    -----
    The configured value wins over an inherited ``RUBIN_SIM_DATA_DIR``: the
    config file is the more deliberate statement of intent here, and a stale
    shell variable quietly overriding it is exactly the confusion this
    function exists to prevent. The override is logged when it happens.
    """
    # Checkouts first: the strategy lookup and the ConsDB fetch both need them,
    # and neither is fatal if absent.
    from .deps import add_checkout

    added = [
        add_checkout(config.sim.lsst_survey_sim, "sim.lsst_survey_sim"),
        add_checkout(config.sim.ts_fbs_utils, "sim.ts_fbs_utils"),
    ]
    if any(root is not None for root in added):
        # The strategy lookup memoizes, so one that ran before the checkout
        # reached sys.path would otherwise stay cached as "missing".
        from .alerts.classes import _live_strategies

        _live_strategies.cache_clear()

    configured = (config.sim.rubin_sim_data or "").strip()
    if not configured:
        return

    resolved = Path(configured).expanduser()
    inherited = os.environ.get("RUBIN_SIM_DATA_DIR")
    if inherited and Path(inherited).expanduser() != resolved:
        logger.info(
            "Overriding inherited RUBIN_SIM_DATA_DIR=%s with sim.rubin_sim_data=%s",
            inherited,
            resolved,
        )
    os.environ["RUBIN_SIM_DATA_DIR"] = str(resolved)

    # Called from both the CLI and process_trigger, so only diagnose a given
    # path once; repeating the same error per alert is just noise.
    global _validated_data_dir
    if _validated_data_dir == resolved:
        return
    _validated_data_dir = resolved

    # Fail loudly here rather than several seconds into an alert, and name the
    # subdirectory the almanac actually needs.
    if not resolved.is_dir():
        logger.error(
            "sim.rubin_sim_data=%s does not exist; the almanac and dark-hours map "
            "will fail. Point it at a rubin_sim_data tree, or run "
            "'scheduler_download_data' to create one.",
            resolved,
        )
    elif not (resolved / "site_models").is_dir():
        logger.error(
            "sim.rubin_sim_data=%s has no 'site_models' subdirectory, which is "
            "where the almanac reads sunsets.npz from. Run "
            "'scheduler_download_data --dirs site_models'.",
            resolved,
        )
    else:
        logger.debug("RUBIN_SIM_DATA_DIR=%s", resolved)
