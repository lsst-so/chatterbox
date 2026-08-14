# chatterbox

Low-latency Slack reporting for Rubin multi-messenger targets of opportunity.

On receipt of a ToO trigger, chatterbox posts one message answering "should we
care, and can we even see it tonight?", then follows up in-thread with the
result of a scheduler simulation.

## What it posts

**Immediately** (measured ~8 s end to end, including a GraceDB skymap fetch):

- **Alert-class statistics** specific to the messenger — gravitational wave,
  high-energy neutrino, galactic supernova, or potentially hazardous asteroid —
  including the criteria that caused the producer to issue the alert, and for
  GW events the distance, FAR, classification and chirp mass recovered from
  GraceDB.
- **Sun and Moon statistics** for the relevant night: sunset/sunrise, the
  −12° and −18° crossings, moonrise/moonset, illumination, and the Moon's
  separation from the localization. In UTC and Chile local time.
- **Weather and site links** for Cerro Pachón, combined with the above.
- **Localization contour over accessible dark hours** — a full-sky map coloured
  by hours per pixel with airmass < 2 while the Sun is below −12°, with the
  localization drawn on top.
- **Localization contour over existing template coverage** — one panel per LSST
  band showing prior visit counts, with the fraction of the localization that
  already has templates in each band.
- **The planned follow-up**: epochs, bands, visit counts and exposure times,
  read live from `ts_fbs_utils` so it cannot drift from what the scheduler runs.

**In-thread when the simulation finishes** (~40 s for 1 night, tens of minutes
for a full 20–50 night run):

- **Expected fraction of the localization covered in each band**, plus a
  per-epoch breakdown, an "any band" total, and a cumulative-coverage plot.

## Install

`chatterbox` needs `rubin_scheduler` for the almanac, plus `healpy`,
`ligo.skymap` and the usual astronomy stack. On a machine that already has a
scheduler environment, the simplest route is a venv that inherits from it,
which leaves that environment untouched:

```bash
~/mamba/envs/scheduler_dev_env/bin/python -m venv --system-site-packages .venv
```

```bash
.venv/bin/pip install -e ".[slack,kafka,test]" --config-settings editable_mode=compat
```

Alternatively install everything from scratch with
`pip install -e ".[rubin,slack,kafka,test]"`.

`editable_mode=compat` matters in a `--system-site-packages` venv. Without it,
setuptools installs an import-hook `.pth` whose finder is silently never
registered in that layout, so `import chatterbox` fails everywhere except from
the repository root. Compat mode writes a plain path entry instead, which works.
A non-editable `pip install .` is unaffected, and `python -m chatterbox.cli`
always works from the repository root without installing anything.

Optional extras and what they buy:

| Extra   | Needed for                                                    |
| ------- | ------------------------------------------------------------- |
| `rubin` | the almanac, dark-hours map, plots, and the simulation         |
| `slack` | actually posting (`slack-sdk`); without it, output is local    |
| `kafka` | the Kafka ingest source (`hop-client`) and `.avro` records     |
| `test`  | `pytest`, `ruff`, `black`, `isort`                             |

## Configure

```bash
cp config.yaml.example config.yaml
```

Then set the bot token in the environment — it is never stored in the config:

```bash
export SLACK_BOT_TOKEN=xoxb-...
```

The token needs the `chat:write` and `files:write` scopes. A bot token is
required rather than an incoming webhook because webhooks cannot upload files,
and every post carries plots.

**Kafka topics are not in the defaults.** The upstream `rubin-ToO-producer`
repository contains no real topic names either — they are purely deployment
configuration — so `ingest.kafka_url` must be filled in for Kafka mode.
Authentication is delegated entirely to hop-client's own
`~/.config/hop/auth.toml`, exactly as the producer does it.

## Use

All commands below are shown as `python -m chatterbox.cli`, which works from the
repository root whether or not the package is installed. Once installed, the
`chatterbox` console script is equivalent.

Build the per-band template coverage cache first. This is the one thing that
must be done ahead of time, because it is far too slow for the alert path:

```bash
python -m chatterbox.cli refresh-templates
```

Run it nightly from cron. The cache's build time appears in every post, so
staleness is visible rather than silent. To build from a ConsDB export instead
of querying ConsDB, pass `--csv path/to/visits.csv`.

Check everything works without posting anything — this is the main development
loop, and it renders both plots and prints the message as text:

```bash
python -m chatterbox.cli replay tests/data/gw_case_b.json --dry-run --no-sim
```

Confirm Slack credentials and channel access:

```bash
python -m chatterbox.cli test-post
```

Run the service:

```bash
python -m chatterbox.cli serve
```

Run just the simulation for a record and print per-band coverage:

```bash
python -m chatterbox.cli simulate tests/data/gw_case_b.json --nights 3
```

## The one thing to understand: area vs probability

The producer's output record carries exactly nine fields. It computes FAR,
classification probabilities, distance, `p_astro` and the probability density,
cuts on them, logs them, and then **discards them**. What it publishes is a
*binary* 70% credible region at nside 32 — 3.36 deg² per pixel.

So from the record alone, "fraction of the localization covered" can only mean
*area*, not probability.

chatterbox handles this in two ways, and never blurs them:

1. **Enrichment.** For a gravitational-wave alert, `source` is the GraceDB
   superevent id, so the real multi-order skymap and the original alert notice
   are fetched from the public API. That recovers the true probability density,
   distance, FAR, classification, properties and chirp mass. This is
   best-effort and time-bounded; failure is not fatal.
2. **Labelling.** Every percentage is described as *probability* or *area*
   according to what actually backs it, driven by
   `Localization.is_probability` rather than by hand-written message text. When
   enrichment fails, the post says so explicitly.

Neutrino and supernova alerts have no equivalent lookup, so they stay
area-based.

## Layout

```
chatterbox/
  config.py       YAML + env settings
  models.py       Trigger, Localization, Geometry, GwEnrichment
  ingest/         record decoding, transports, GraceDB enrichment
  alerts/         alert_type -> messenger, criteria, follow-up strategy
  astro/          skymaps, almanac, dark hours, template coverage
  plots/          dark-hours map, template panels, coverage curve
  sim/            simulation driver, per-band coverage
  slackbot/       Block Kit construction, delivery
  app.py          two-stage orchestration
  cli.py          serve | replay | refresh-templates | simulate | test-post
scripts/
  make_fixture.py generate realistic record fixtures from real skymaps
```

`astro/skymap.py` vendors the `Skymap` and `KahanAdder` classes from
`rubin-ToO-producer`'s `forward_alerts.py` so that credible areas reported here
match the areas the producer cut on, and so tests can build faithful fixtures.

## Test

```bash
.venv/bin/python -m pytest
```

196 tests, no network access required. Notable checks:

- The dark-hours map is validated against `astropy`'s full `AltAz` transform —
  an independent code path from the fast approximation used in production —
  and agrees to about one minute per pixel.
- The vendored follow-up-strategy snapshot is compared against live
  `ts_fbs_utils`, so upstream drift fails a test rather than going unnoticed.
- Stage-1 latency is asserted, so a regression that makes the alert path slow
  is a test failure.
- Every `alert_type` the producer can emit is checked to have a messenger, a
  strategy, and a night count.

## Known limitations

**No producer emits asteroid alerts.** There is no PHA/SSO handler anywhere in
`rubin-ToO-producer` or its history; `SSO_night` and `SSO_twilight` exist only
as `ts_fbs_utils` strategy labels. Those classes are implemented and tested
against synthetic records, and are marked in the registry so a reader cannot
mistake them for live coverage.

**Updates and retractions cannot arrive on this path.** The producer's
`should_follow_up` rejects retraction messages before its duplicate check, so
`is_update` is rendered but will always be false in practice. Showing
retractions would require consuming the upstream messenger topics directly.

**The filter carousel schedule is stale upstream.** `lsst_survey_sim`'s
built-in swap schedule ends 2026-03-25, after which it falls back to choosing
bands from Moon illumination — which silently drops requested `u`, and often
`r` and `i`, from a ToO's scripted visits. Set `sim.band_swap_schedule` to a
current schedule; whichever path was used is reported in the threaded reply.

**Coverage geometry is approximate.** Both the template maps and the coverage
calculation approximate LSSTCam's footprint as a 1.75° disc, ignoring the real
focal plane and chip gaps. This matches the existing analysis notebooks and
keeps the two consistent with each other.

**An enriched skymap may disagree with the alert label.** The producer cuts on
the skymap that was current when the alert fired; GraceDB may since have served
an updated one. A 90% area that looks inconsistent with the stated criteria is
usually this, not an error.
