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
    "load_config",
    "DEFAULT_CONFIG_PATHS",
]

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATHS = (
    Path("config.yaml"),
    Path("~/.config/chatterbox/config.yaml").expanduser(),
)


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

    - ``kafka``: subscribe to the producer's output topic with hop-client.
    - ``files``: watch a directory that ``forward_alerts.py``'s ``FileSender``
      writes ``{source}.json`` into.
    - ``replay``: read explicit paths once, then stop (used by the CLI).
    """

    kind: str = "files"
    #: hop URL, e.g. ``kafka://kafka.scimma.org/lsst.rubin-too-alerts``.
    kafka_url: str = ""
    kafka_group_id: str = "chatterbox"
    #: Directory watched by the ``files`` source.
    watch_dir: str = "~/.chatterbox/incoming"
    #: Seconds between directory scans.
    poll_interval_s: float = 2.0
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
    #: Interpreter with rubin_scheduler + lsst_survey_sim available.
    python: str = "~/mamba/envs/scheduler_dev_env/bin/python"
    lsst_survey_sim: str = "~/Desktop/Repos/lsst_survey_sim"
    ts_config_scheduler: str = "~/Desktop/Repos/ts_config_scheduler"
    #: Must be the tree containing ``sim_baseline``.
    rubin_sim_data: str = "~/RubinUtils/rubin_sim_data"
    #: Pre-staged visit history required by ``setup_scheduler``.
    opsim_h5: str = "~/Desktop/Repos/lsst_survey_sim/notebooks2/opsim.h5"
    #: Override the per-alert-class night count. ``0`` uses the class default.
    nights_override: int = 0
    #: Hard wall-clock cap on a simulation job.
    timeout_s: float = 7200.0
    #: Base URL that sim artifacts are published under, if any.
    artifact_base_url: str = ""
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
        return _from_mapping(Config, data)

    if path is not None:
        raise FileNotFoundError(f"Config file not found: {path}")
    logger.info("No config file found; using defaults")
    return Config()
