# Roadmap

The goal is a library whose results are correct by default and easy to
understand. Two rounds of gap screening fixed about 90 correctness bugs, and
most of them fell into six recurring classes: counter resets, interval
closing, logging gaps, sample-count means, time zones and per-signal
grouping. Because the same classes keep recurring, the fix is in the
architecture rather than in individual detectors.

## W1: signal semantics core (started)

`SignalSpec`, `infer_specs`, `profile()` and the spec-driven primitives in
`ts_shape.signals`. Detectors call the primitives instead of re-deriving
signal semantics. See [Signal semantics and profiling](guides/signals.md).

## Consistency audit (done, October 2026)

An audit of wording, names, outputs and behavior. The rules are in [Naming and
output conventions](naming.md), and tests enforce them:

- every documented call binds against the real signature;
- detector outputs follow their declared shape, and empty results keep every column;
- one name per concept, checked by a lint over every signature, with deprecated aliases for old names;
- every public parameter is documented;
- inputs are checked the same way in every pack.

Also done in October 2026: full precision, value columns checked,
`best_and_worst_shifts` returns a DataFrame, `batch_transition_matrix` returns
a long table.

Open, all additive (new accepted forms, nothing removed):

- **Run state in one form.** `DowntimeEvents` reads string states
  (`"Running"`) or booleans, `MachineStateEvents` and `OEEEvents` read
  booleans. Accept either everywhere through `SignalSpec`.
- **One place for the signal uuid.** Some detectors take it in the
  constructor, others per method.
- **Durations in one form.** `OEEEvents` takes `ideal_cycle_time` in float
  seconds, `takt_time` takes an offset string. Accept both everywhere.

In a future major release the deprecated aliases (warning since 0.0.63) are
removed.

## W2: 1.0, correct by default, one API, one output

- Done (October 2026): the correctness flags (`handle_resets`,
  `interval_end`, `time_weighted`, `group_by_uuid`, `slope_per`, ...) default
  to the correct behavior. `max_gap` stays `None` (no cap), because no single
  value fits every signal; set it per signal or through a `SignalSpec`.
- Done: no silent data loss. Skipped inputs warn with `DataQualityWarning`;
  loaders take `errors="warn" | "raise"`.
- Move the detectors pack by pack onto the primitives, with defaults taken
  from the spec.
- One enforced output contract (point, interval, summary) and one keyword
  style for every detector.
- KPIs carry their basis (run seconds, planned seconds, excluded gaps,
  resets), so every number can be audited.

## W3: proven correctness (harness started)

- A registry-wide invariant harness (`tests/invariants`) runs every detector
  method on empty input, on tz-aware input, with an unrelated extra signal,
  and with nullable, pyarrow and text-number dtypes (same result required).
  It is pinned to a baseline: 330 methods, 0 failures (278 run every
  check; 52 are not applicable, e.g. a required parameter the harness has
  no input for).
- Property-based tests (hypothesis) cover the primitives, counters, intervals
  and window bounds.
- Reference tests, done: AIAG MSA Gauge R&R (average-and-range method,
  K1/K2/K3 and the `EV²/(n·r)` correction of AV); ISO 50001 / 50006 EnPI;
  ISO 22400 OEE and MTBF / MTTR; Western Electric / Nelson SPC rules 1 to 8;
  Cp, Cpk, Pp, Ppk, Cpm (Montgomery, Introduction to SQC); GUM measurement
  uncertainty (JCGM 100 §4.4.3); Weibull MLE (Lieblein–Zelen ball bearings,
  Lawless Ex. 5.2.2).
- The docs are tested: every `ts_shape` name in a code block must import, and
  CI builds the docs with `mkdocs build --strict`.

## W4: scale and lean install

- Done: the core install needs only pandas, numpy and scipy; cloud and SQL
  loaders are extras.
- Done: slow row loops vectorized (shift assignment, maintenance trend fits,
  routing, warm-up, stockout; 10 to 300 times faster on 200k rows).
- Done: 10 and 100 million rows. Detector construction copies, sorts and
  converts only what it must, per-run loops in 19 modules are numpy, and
  100M-row frames no longer run out of memory. `ChangePointEvents` defaults
  to `method="auto"`: exact PELT up to 20,000 samples, binary segmentation
  above, so change points are no longer quadratic on long signals.
- Done: a detector whose methods read only the signals its constructor
  names keeps only those rows; the input checks still warn from the whole
  frame. On 10M rows with five signals a one-signal detector adds about
  0.25 GB to its input instead of 0.4 GB, and runs in half the time.
- Done: `benchmarks/run.py` measures time and peak memory per detector and
  row count, and CI runs a scaling check (`pytest -m scale`).
- After 1.0: backend-agnostic primitives (Polars, DuckDB), TimescaleDB
  push-down.

## W5: understanding

- `profile()` as the entry point.
- A question-first cookbook ("Why did OEE drop?", "Which station is the
  bottleneck?", "Is my gauge capable?") on a bundled, realistic plant dataset.
- A generated detector reference: assumptions, input signal kinds and output
  shape.

## W6: ecosystem

- Grafana and TimescaleDB recipes.
- A command line (`ts-shape profile data.parquet`).
- Done: deprecation policy. Renamed names keep working with a warning and are
  removed only in a major release.

## Process analytics

Added: change-point detection (`ChangePointEvents`: exact PELT and binary
segmentation), multivariate SPC (`MultivariateSPCEvents`: Hotelling T² with
Phase I and II limits and contributions), reliability (`ReliabilityEvents`:
MTBF and MTTR per ISO 22400, censored Weibull life model).

Candidates:

| Feature | Value | Effort |
|---|---|---|
| Vibration: FFT spectral band energy | high | M |
| Tool wear vs. measurement drift; first-article and setup verification; part disposition | high | M each |
| MQTT or Kafka streaming loader | medium-high | M |
| Detector evaluation (precision and recall against labels) | medium | M |
| Scaling transforms | medium | S |
| Energy and production plot helpers in an optional `[viz]` extra | medium | S |

## Unmerged branches

Two large branches from earlier work were never merged. Both can be ported in
their own pull requests after reconciling them with the core:

- **`claude/reporting-python-library-2fhu1a`**: `ts_shape.reporting` (8.7k
  lines, 211 tests): facility calendar with IANA time zones, weighted KPI
  roll-up, ISA-95 hierarchy, benchmarking, scorecard and variance bridge, FX
  normalization, reporting coverage. To reconcile: build the calendar on
  `signals.windows` / `shift_windows` (already DST-correct) instead of a second
  implementation; drop its docs CI job (the strict docs build is in `ci.yml`);
  keep SECURITY.md and the issue templates.
- **`claude/global-production-reporting-w5f10n`**: `FleetProductionReport` and
  `FleetSpec` (11k lines): machine-status adapters (PackML, stack light, code
  maps), counter wrap vs. reset, part-done pulses, cycle-time unit inference,
  production context with late binding. To reconcile: map the status adapters
  onto `SignalSpec(kind="state")` and `OperatingStateEvents`; add counter wrap
  as a `reset_policy` (`"wrap"`) in `SignalSpec` and `utils.counters` instead
  of a parallel counter model; fold cycle-time unit inference into
  `infer_specs`.
