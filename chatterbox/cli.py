"""Command-line interface.

Subcommands
-----------
``serve``
    Consume the configured source and post about each record.
``replay``
    Handle one or more record files. ``--dry-run`` renders everything and
    prints the message without posting; this is the main development loop.
``refresh-templates``
    Refresh the local cache of per-band template coverage maps.
``refresh-opsim``
    Refresh the cached visit history the simulation starts from.
``simulate``
    Run the scheduler simulation for a record synchronously and print coverage.
``test-post``
    Send a short message to confirm Slack credentials and channel access.
``doctor``
    Report which capabilities are available and how to fix the missing ones.
"""

import argparse
import logging
import sys
from pathlib import Path

from .config import Config, apply_environment, load_config

__all__ = ["main", "build_parser"]

logger = logging.getLogger("chatterbox")


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="chatterbox",
        description="Low-latency Slack reporting for Rubin multi-messenger targets of opportunity.",
    )
    parser.add_argument("-c", "--config", help="Path to config.yaml")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Log at DEBUG level",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_serve = sub.add_parser("serve", help="Consume the configured alert source")
    p_serve.add_argument("--no-sim", action="store_true", help="Do not launch simulations")

    p_replay = sub.add_parser("replay", help="Handle one or more ToO record files")
    p_replay.add_argument("paths", nargs="+", help="Record files (.json or .avro)")
    p_replay.add_argument(
        "--dry-run",
        action="store_true",
        help="Render and print the message without posting to Slack",
    )
    p_replay.add_argument("--no-sim", action="store_true", help="Do not launch a simulation")
    p_replay.add_argument(
        "--sim-wait",
        action="store_true",
        help="Wait for the simulation and print its coverage before exiting",
    )
    p_replay.add_argument("--out-dir", help="Where to write plots")

    p_templates = sub.add_parser(
        "refresh-templates",
        help="Refresh the local cache of per-band template coverage maps",
    )
    p_templates.add_argument(
        "--path",
        help="Directory of coverage maps (default: templates.maps_dir)",
    )
    p_templates.add_argument("--nside", type=int, help="Override the cache resolution")

    p_opsim = sub.add_parser(
        "refresh-opsim",
        help="Refresh the cached visit history the simulation starts from",
    )
    p_opsim.add_argument("--force", action="store_true", help="Refresh even if the cache is current")
    p_opsim.add_argument(
        "--day-obs",
        type=int,
        help="Fetch visits before this day_obs (default: tomorrow, i.e. everything so far)",
    )

    p_sim = sub.add_parser("simulate", help="Run the scheduler simulation for a record")
    p_sim.add_argument("path", help="Record file")
    p_sim.add_argument("--nights", type=int, help="Override the per-class night count")

    sub.add_parser("test-post", help="Post a test message to confirm Slack access")
    sub.add_parser("doctor", help="Report what works, what does not, and how to fix it")

    return parser


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # These are noisy at INFO and say nothing useful about our own work.
    #
    # Deliberately excludes "astropy": astropy installs its own Logger subclass
    # at import time and fails if a plain logger of that name already exists,
    # so calling getLogger("astropy") here would break importing astropy.
    for noisy in ("matplotlib", "urllib3", "numexpr", "PIL", "fsspec", "healpy"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _cmd_serve(args: argparse.Namespace, config: Config) -> int:
    from .app import run_service

    handled = run_service(config, run_sim=not args.no_sim)
    print(f"Handled {handled} record(s).")
    return 0


def _cmd_replay(args: argparse.Namespace, config: Config) -> int:
    from .app import process_trigger
    from .ingest.decode import load_record_file
    from .slackbot.client import SlackPoster, render_blocks_as_text

    poster = SlackPoster(config, dry_run=args.dry_run)
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else None

    failures = 0
    for path in args.paths:
        print(f"\n{'=' * 72}\n{path}\n{'=' * 72}")
        try:
            record = load_record_file(path)
            report = process_trigger(
                record,
                config,
                poster=poster,
                post=not args.dry_run,
                run_sim=not args.no_sim,
                sim_wait=args.sim_wait,
                out_dir=out_dir,
            )
        except Exception as exc:
            logger.error("Failed on %s: %s", path, exc, exc_info=args.verbose)
            failures += 1
            continue

        if args.dry_run:
            print(render_blocks_as_text(report.blocks))

        print(f"Stage 1 completed in {report.elapsed_s:.2f} s")
        for plot in report.plots:
            print(f"  plot: {plot}")
        for warning in report.warnings:
            print(f"  warning: {warning}")
        if report.posted is not None and report.posted.offline:
            print("  (offline: payload written instead of posted)")

    return 1 if failures else 0


def _cmd_refresh_templates(args: argparse.Namespace, config: Config) -> int:
    from .astro.templates import load_source_maps

    cfg = config.templates
    nside = args.nside or cfg.nside
    maps_dir = args.path or cfg.maps_dir

    try:
        coverage = load_source_maps(
            maps_dir,
            nside=nside,
            bands=tuple(cfg.bands),
            pattern=cfg.map_pattern,
        )
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    out = coverage.save(cfg.cache_dir)
    print(f"Cached template coverage for {','.join(coverage.bands)} from {maps_dir} to {out}")
    for band in coverage.bands:
        print(f"  {band}: {coverage.area_deg2(band):>9,.0f} deg^2")
    if coverage.missing_bands:
        print(f"  no map published for: {', '.join(coverage.missing_bands)}")
    return 0


def _cmd_refresh_opsim(args: argparse.Namespace, config: Config) -> int:
    from .sim.opsim import default_day_obs, ensure_opsim

    cfg = config.sim
    day_obs = args.day_obs or default_day_obs()
    try:
        _, cache = ensure_opsim(
            cfg.opsim_cache,
            day_obs=day_obs,
            tokenfile=cfg.opsim_tokenfile or None,
            site=cfg.opsim_site,
            max_age_hours=cfg.opsim_max_age_hours,
            force=args.force,
            lsst_survey_sim=cfg.lsst_survey_sim,
        )
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    print(f"Visit history at {cache.path}: {cache.describe()}")
    return 1 if cache.stale_reason else 0


def _cmd_doctor(args: argparse.Namespace, config: Config) -> int:
    from .doctor import diagnose, format_report

    checks = diagnose(config)
    print(format_report(checks))
    return 1 if any(not c.ok and c.fatal for c in checks) else 0


def _cmd_simulate(args: argparse.Namespace, config: Config) -> int:
    from .ingest.decode import load_record_file
    from .ingest.enrich_gracedb import enrich_gravitational_wave
    from .sim.runner import launch_simulation, load_sim_result

    record = load_record_file(args.path)
    from .ingest.decode import decode_record

    trigger = decode_record(record)
    # Enrich first so coverage is measured against a real probability map.
    enrich_gravitational_wave(trigger, config.enrich)

    job = launch_simulation(trigger, config, nights=args.nights, wait=True)
    if job is None:
        print("Simulation is disabled (sim.enabled is false).")
        return 1

    result = load_sim_result(job.job_dir)
    if result is None:
        print(f"No result written. Log tail:\n{job.tail_log(40)}", file=sys.stderr)
        return 1
    if not result.ok:
        print(f"Simulation {result.status}: {result.error}", file=sys.stderr)
        print(f"Log tail:\n{job.tail_log(30)}", file=sys.stderr)
        return 1

    print(f"\n{result.source} ({result.alert_type}), {result.nights} nights")
    print(f"  {result.too_visits:,} ToO visits of {result.total_visits:,} total")
    print(f"  fraction of localization {result.quantity} covered:")
    for band in ("u", "g", "r", "i", "z", "y"):
        print(f"    {band}: {100 * result.coverage.get(band, 0.0):6.2f}%")
    print(f"  any band: {100 * result.any_band:6.2f}%")
    print(f"  ran in {result.runtime_s / 60:.1f} min; output {result.visits_path}")
    return 0


def _cmd_test_post(args: argparse.Namespace, config: Config) -> int:
    from .slackbot.client import SlackPoster

    poster = SlackPoster(config)
    blocks = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": ":wave: *chatterbox test post* -- if you can read this, credentials and "
                "channel access are working.",
            },
        }
    ]
    posted = poster.post(blocks, "chatterbox test post", label="test_post")
    if posted.offline:
        print(
            f"Offline: {config.slack.bot_token_env} is not set, so nothing was sent. "
            f"Payload written under {poster.output_dir}."
        )
        return 1
    print(f"Posted to {posted.channel} (ts={posted.ts}).")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``chatterbox`` console script."""
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    try:
        config = load_config(args.config)
    except Exception as exc:
        print(f"Could not load configuration: {exc}", file=sys.stderr)
        return 2

    # Must happen before any subcommand imports or calls rubin_scheduler.
    apply_environment(config)

    handlers = {
        "serve": _cmd_serve,
        "replay": _cmd_replay,
        "refresh-templates": _cmd_refresh_templates,
        "refresh-opsim": _cmd_refresh_opsim,
        "simulate": _cmd_simulate,
        "test-post": _cmd_test_post,
        "doctor": _cmd_doctor,
    }
    return handlers[args.command](args, config)


if __name__ == "__main__":
    sys.exit(main())
