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
  band, drawn from the published binary coverage maps, with the fraction of the
  localization that already has templates in each band.
- **The planned follow-up**: epochs, bands, visit counts and exposure times,
  read live from `ts_fbs_utils` so it cannot drift from what the scheduler runs.

**In-thread when the simulation finishes** (~40 s for 1 night, tens of minutes
for a full 20–50 night run):

- **Expected fraction of the localization covered in each band**, plus a
  per-epoch breakdown, a night-by-night cumulative progression, and an
  "any band" total.

**And, in a thread of its own, every figure that simulation produced**, in this
order, with a numbered line naming each one:

1. **The cumulative-coverage curve** against time, per band, over the whole run.
2. **Coverage gained per night** — one bar per band per night, with the running
   any-band total stepped over the top, on an axis of **nights since the
   trigger**. This is the same total the threaded reply quotes, so the last
   point on the line can be checked against the text, and it covers *every*
   night of the run, including any the per-night maps below leave out. Set
   `sim.nightly_coverage_plot: false` to skip it.
3. **Per-night, per-band maps of simulated visit counts** with the localization
   contour drawn on top, one figure per night. Only nights with visits that
   overlap the contour are plotted; every visit counts towards that, not just
   the tagged ToO follow-up, since a nominal-cadence pass landing on the
   localization covers it just as well. Set `sim.nightly_plots: false` to stop
   producing these.

Plus the path the simulation ran in and the visit database it wrote. The figures
get their own thread rather than the alert's, which a 20-night run would
otherwise flood.

**And if chatterbox itself fails, it says so in the channel.** A ToO nobody
hears about is indistinguishable from no ToO at all, so every failure that would
otherwise be silent gets posted: a record that could not be decoded, a
simulation whose driver died without writing a result, a crash in the background
thread that posts it, and the monitoring stream itself going away. Each message
names what was being attempted, the event id when it is known, the exception,
and the tail of the relevant log. The failure path is guarded in turn, so a
broken channel cannot turn one failure into two.

## Install

`chatterbox` needs `rubin_scheduler` for the almanac, plus `healpy`,
`ligo.skymap` and the usual astronomy stack. On a machine that already has a
scheduler environment, the simplest route is a venv that inherits from it,
which leaves that environment untouched:

```bash
~/mamba/envs/scheduler_dev_env/bin/python -m venv --system-site-packages .venv
```

```bash
.venv/bin/pip install -e ".[slack,efd,kafka,test]" --config-settings editable_mode=compat
```

Alternatively install everything from scratch with
`pip install -e ".[rubin,slack,efd,kafka,test]"`.

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
| `efd`   | the EFD ingest source (`lsst-efd-client`), the default         |
| `kafka` | the Kafka ingest source (`hop-client`) and `.avro` records     |
| `test`  | `pytest`, `ruff`, `black`, `isort`                             |

The EFD client is opened directly as `EfdClient(efd_name, db_name="lsst.scimma")`,
exactly as the summit analysis notebooks do, so `ingest.efd_name` must name an
instance (the top-level `site` fills it in).

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

**Say which site this instance runs at**, in one line:

```yaml
site: summit     # summit | base | usdf | usdf-dev
```

The same site is named differently by each library, so one setting fills all
three rather than leaving three chances to disagree:

| `site` | `ingest.efd_name` | `sim.opsim_site` | `sim.opsim_tokenfile` |
| --- | --- | --- | --- |
| `summit` | `summit_efd` | `summit` | `~/.lsst/summit_rsp` |
| `base` | `base_efd` | `base` | `~/.lsst/base_rsp` |
| `usdf` | `usdf_efd` | `usdf` | `~/.lsst/usdf_rsp` |
| `usdf-dev` | `usdf_efd` | `usdf-dev` | `~/.lsst/usdf_rsp` |

Any of the three set explicitly still wins, and both what the site supplied and
what was overridden are logged at startup — a half-overridden site is exactly
the state worth seeing. An unknown name is an error at startup rather than a
silent no-op, since a typo would otherwise leave the instance quietly pointed at
whichever site the defaults name. Leave `site` empty and each setting keeps its
own default, which is what an existing config file gets.

There is no `idf`: an `idf_efd` exists, but `rubin_nights` has no ConsDB
endpoint for it, so set `ingest.efd_name: idf_efd` directly for that. The
ConsDB names come from `rubin_nights.connections.API_ENDPOINTS`.

`chatterbox doctor` prints the resolved site and all three values on one line.

**Kafka topics are not in the defaults.** The upstream `rubin-ToO-producer`
repository contains no real topic names either — they are purely deployment
configuration — so `ingest.kafka_url` must be filled in for Kafka mode.
Authentication is delegated entirely to hop-client's own
`~/.config/hop/auth.toml`, exactly as the producer does it.

## Where alerts come from

`ingest.kind` selects one of three monitoring paths. **The default is `efd`**:
the producer's records are archived in the EFD, so nothing has to be subscribed
to and, on a summit or USDF host, the credentials are already in place.

The alerts are the `too_alert` measurement in the **`lsst.scimma`** InfluxDB
database — *not* the default database an EFD client connects to, which holds the
SAL topics, so `chatterbox` sets `db_name` on the client explicitly. It then
polls `select_time_series` every `ingest.efd_poll_interval_s` for new records,
re-reading a short trailing window each time so nothing that arrives late is
missed (see the ingest-lag note below).

Two things about that reconstruction are worth knowing, because getting either
wrong produces a plausible-looking wrong answer rather than an error:

- InfluxDB has no arrays, so the reward map arrives as one column per pixel —
  `reward_map0` … `reward_map12287` at nside 32 — and the instrument array as
  `instrument0/1/2`. Those columns must be ordered **numerically**: sorted as
  text, `reward_map10` precedes `reward_map2` and the localization is scrambled
  into a mask that is still the right size and still looks like a sky map.
- A field InfluxDB has no value for comes back as `NaN`, and `NaN` casts to
  `True`, which would silently *add* sky to the credible region. Nulls are
  counted, logged and forced to False; a row whose pixels are *all* null is
  rejected rather than posted as an empty localization.

Set `ingest.efd_name` directly only to point at an instance the top-level
`site` does not cover, or to override it.

**Which EFD is being read is always stated.** The instance is resolved from the
client once it connects and appears in the startup log (`Watching the summit_efd
EFD for lsst.scimma.too_alert (database lsst.scimma) …`), on every record's
metadata, and in the failure posted to Slack, so "monitoring stopped" names the
site that stopped. If the client opens a different instance from the configured
one, that disagreement is logged rather than silently preferred.

`ingest.efd_lookback_s` is `0` by default, so only alerts arriving after startup
are posted. chatterbox keeps no record of what it has already sent, so anything
inside a longer lookback window is re-posted after a restart. To catch an alert
that landed just before you start the service — one you just injected, say —
without changing the default, run `serve --since UTC_TIME` or `serve --lookback
SECONDS` for that one invocation.

**A record's timestamp is not when it becomes queryable.** The EFD stamps each
record with the producer's *send* time (that is its InfluxDB time index), but
the relay does not make it readable until some seconds — occasionally minutes —
later. A plain forward-marching poll would sweep its window past that timestamp
before the row appears and lose the alert with no error and no post. So every
poll re-reads a trailing `ingest.efd_revisit_s` window (default `900`, i.e. 15
minutes) *behind* the watermark; a late row still lands inside a query window,
and the `(source, time)` de-dup means the overlap never posts anything twice.
Set it comfortably above the real ingest lag — measured at ~11 s on `base_efd`,
so the default has wide margin. The trade-off is the mirror of the lookback one:
because the de-dup only spans a single run, a **restart** can re-post alerts
from the last `efd_revisit_s`; lower it if that is noisy while testing, or raise
it if the relay is ever slower than the window.

A single failed query is retried; `ingest.efd_max_consecutive_errors` (5) in a
row is treated as an outage, posted to the channel, and stops the service rather
than leaving it quietly watching nothing.

The other two paths are unchanged: `kafka` subscribes to the producer's output
topic with hop-client, and `files` watches the directory its `FileSender` writes
`{source}.json` into.

## Use

All commands below are shown as `python -m chatterbox.cli`, which works from the
repository root whether or not the package is installed. Once installed, the
`chatterbox` console script is equivalent.

Refresh the two caches the bot reads from. Both are cron jobs:

```bash
python -m chatterbox.cli refresh-templates
```

```bash
python -m chatterbox.cli refresh-opsim
```

Template coverage is not computed by chatterbox. It is produced upstream by the
incremental templates tooling, which publishes one **binary** HEALPix FITS map
per band at `templates.maps_dir` — 1 where a template exists, 0 where it does
not — named `template_coverage_healpix_{band}_nside{nside}.fits`. This command
reads those files and caches them locally; it queries nothing.

Run it from cron so the cache tracks the published maps. The cache's timestamp
appears in every post, so staleness is visible rather than silent, and bands
with no published map are named explicitly rather than silently reading 0%.
Point at a different directory with `--path`.

`refresh-opsim` caches the visit history the simulation starts from, fetched
from ConsDB with `lsst_survey_sim`'s own `fetch_previous_visits`. That means
`sim.lsst_survey_sim` must point at a checkout (or the package must be
installed) even when you are only refreshing the cache and not simulating.

There is no opsim file to stage by hand: the cache refreshes when it is older than
`sim.opsim_max_age_hours` (24 h by default), or when it predates the night being
simulated — so a ToO arriving on a night cron has not covered refreshes it
itself before simulating. If ConsDB is unreachable and a cache exists, the
simulation uses it and says so in the threaded reply rather than failing.

Check everything works without posting anything — this is the main development
loop, and it renders both plots and prints the message as text:

```bash
python -m chatterbox.cli replay tests/data/gw_case_b.json --dry-run --no-sim
```

Check the environment — which capabilities work, which are degraded, and the
exact command to fix each one. Worth running first on any new host:

```bash
python -m chatterbox.cli doctor
```

Confirm Slack credentials and channel access:

```bash
python -m chatterbox.cli test-post
```

Run the service. It polls the EFD by default; see
[Where alerts come from](#where-alerts-come-from):

```bash
python -m chatterbox.cli serve
```

Run just the simulation for a record and print per-band coverage:

```bash
python -m chatterbox.cli simulate tests/data/gw_case_b.json --nights 3
```

The job directory it prints holds everything the simulation produced: the opsim
database of visits, the cumulative-coverage plot, the per-night coverage figure,
and the per-night visit-count figures, so the sim can be inspected without going
through Slack.

`simulate` posts nothing by default, since it is an inspection command. Add
`--post` to send the coverage and the figures to the channel — the same two
messages the service posts, minus the alert thread to hang the coverage under,
so it goes out standalone. `--dry-run` renders both payloads under
`<work_dir>/posts` instead, which is how to read a real run's messages before
letting them reach anyone:

```bash
python -m chatterbox.cli simulate tests/data/gw_case_b.json --nights 3 --post
```

Post a simulation that already finished, from its job directory:

```bash
python -m chatterbox.cli post-sim ~/.chatterbox/work/sim/S251112cm_20260814T192732
```

This is the recovery path when a simulation outlived whatever launched it. A
simulation runs in a **detached** process, so it survives the CLI exiting and
writes every figure — but the thread waiting to post it does not. `replay` and
`serve` therefore wait for a launched simulation before exiting (Ctrl-C leaves
it running and prints this command), and if a service is shut down mid-run it
logs the same command for the job it had to abandon.

## How nights are numbered

Two conventions have to agree here, and both hinge on the same boundary.

**A Rubin observing day runs 12:00 UTC to 12:00 UTC.** That is what keeps a
whole Chilean night under one number — local midnight falls in the middle of it,
not on the edge. `day_obs` for a trigger is therefore the date of
`event − 12 h`, taken from `rubin_nights.dayobs_utils` when it is importable so
it cannot drift from upstream. Using the trigger's UTC *calendar date* instead
splits the night in half: an alert arriving between 00:00 and 12:00 UTC — the
back half of the night at Cerro Pachón, when a fast response matters most —
would name the following evening and delay the entire simulated follow-up by a
day.

**`night` in a simulation output is `floor(mjd − mjd_start)`**, a plain day
count from the observatory's noon-UTC epoch — not an almanac sunset lookup. It
shares the 12:00 UTC boundary with `day_obs`, which is what makes the two line
up: the observing day the simulation starts on is the one the trigger falls in.
The number itself says nothing about the alert, though — a November 2025 event
lands on night 11 whatever the follow-up looks like.

So every night that reaches a reader is reported as **nights since the
trigger**: `+0` is the trigger night, `+1` the night after. The anchor is
computed with the same `floor(mjd − mjd_start)` the visits are labelled with, so
it cannot be off by one. The survey night stays visible in the figure titles and
in the per-night map filenames, so a figure can still be matched against the
visit database, and `result.json` records `trigger_night` and `first_sim_night`
alongside the relative keys.

Nights before `+0` should not appear. If one ever does — the run starting
somewhere other than the trigger's night — the driver logs a warning and the
figure marks the boundary rather than quietly folding those visits into the
first epoch, since a pre-trigger night is nominal cadence, not follow-up.

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
  ingest/         record decoding, transports (EFD, Kafka, files),
                  GraceDB enrichment
  alerts/         alert_type -> messenger, criteria, follow-up strategy
  astro/          skymaps, almanac, dark hours, template coverage
  plots/          dark-hours map, template panels, coverage curve,
                  per-night coverage, per-night visit counts
  sim/            simulation driver, per-band coverage
  slackbot/       Block Kit construction, delivery
  app.py          two-stage orchestration
  cli.py          serve | replay | refresh-templates | refresh-opsim |
                  simulate | post-sim | test-post | doctor
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

407 tests, no network access required. Notable checks:

- The dark-hours map is validated against `astropy`'s full `AltAz` transform —
  an independent code path from the fast approximation used in production —
  and agrees to about one minute per pixel.
- The vendored follow-up-strategy snapshot is compared against live
  `ts_fbs_utils`, so upstream drift fails a test rather than going unnoticed.
- Stage-1 latency is asserted, so a regression that makes the alert path slow
  is a test failure.
- Every `alert_type` the producer can emit is checked to have a messenger, a
  strategy, and a night count.
- The nightly plots select on the *credible region*, not on every non-zero
  pixel: a real BAYESTAR map has faint tails almost everywhere, and selecting
  on those would both match visits far outside the contour and make the
  overlap test crawl over the whole sky.
- The simulation's figures reach Slack even when the alert has no message to
  thread onto, and a failed coverage post does not take them down with it —
  both were ways for a run to produce PNGs and post nothing.
- A backgrounded simulation is joinable, and a service shut down mid-run names
  the `post-sim` command for the job it abandoned. The posting thread dying with
  its process was the reason a `replay` wrote every figure and posted none.
- Splitting coverage by night gives the same total as measuring it in one pass,
  so the last point on the per-night figure always agrees with the percentage
  quoted in the thread. A second night over the same field adds nothing.
- A flattened EFD row rebuilds into a record that the same decoder accepts, with
  the same area and centroid as the record it came from — asserted against a
  deliberately shuffled column order, since lexical sorting is wrong in a way
  that does not raise.
- Every silent-failure path posts: a dropped record, a simulation with no
  result, a crash in the posting thread, and a dead monitoring stream. The
  failure path itself is asserted never to raise.
- Every `site` entry is checked to name all three settings, so a partial one
  cannot leave a service pointed at another site, and an explicit setting is
  asserted to survive the site filling in the rest.
- `day_obs` is asserted against `rubin_nights`' own convention, and the trigger
  night against `floor(mjd − mjd_start)` — the expression `ModelObservatory`
  actually labels visits with. A separate test shows the two agree, which is
  what makes "night +0" the trigger night. A result with no anchor still
  renders, as survey nights.

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

**Simulated coverage geometry is approximate.** The per-band coverage
calculation approximates LSSTCam's footprint as a 1.75° disc around each
simulated pointing, ignoring the real focal plane and chip gaps. This matches
the existing analysis notebooks. Template coverage does not share this
approximation — those maps are read from the upstream tooling as published, and
a file that turns out not to be binary is flagged rather than reinterpreted.

**An enriched skymap may disagree with the alert label.** The producer cuts on
the skymap that was current when the alert fired; GraceDB may since have served
an updated one. A 90% area that looks inconsistent with the stated criteria is
usually this, not an error.
