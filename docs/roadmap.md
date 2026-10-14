# Roadmap: the reference library for manufacturing time series

ts-shape is broad: more than 70 detectors in 8 packs. The goal is to make it
**trusted by default** and **understood immediately**. Two gap screens (see
the screening reports) fixed about 90 correctness bugs. The same six bug
classes kept coming back:
- counter resets;
- interval closing;
- logging gaps;
- sample-count means;
- timezones;
- per-signal grouping.

That points to the architecture, not to individual detectors. The roadmap
fixes the architecture.

## W1 — Signal semantics core ✅ (started)
`SignalSpec`, `infer_specs`, `profile()` and the spec-driven primitives
(`ts_shape.signals`). Detectors call the primitives instead of re-deriving
semantics row by row. See [Signal semantics & profiling](guides/signals.md).

## W2 — 1.0: correct by default, one API, one output
- Migrate the detectors pack by pack onto the primitives. Defaults come from
  the spec.
- The opt-in correctness flags (`handle_resets`, `interval_end`, `max_gap`,
  `time_weighted`, `group_by_uuid`) become the default behavior.
  Deprecation path: one minor release with `DeprecationWarning`, then 1.0.
- One enforced output contract (point / interval / summary) for every
  detector, and one API style with standard keyword names.
- KPIs carry their basis (run seconds, planned seconds, excluded gaps,
  resets), so every number can be audited.

## W3 — Proven correctness ✅ (harness started)
- A registry-wide invariant harness (`tests/invariants`) runs every detector
  method on empty input, on tz-aware input, and with an unrelated extra
  signal. It is pinned to a ratchet baseline.
- Property-based tests (hypothesis) cover the primitives.
- Reference-validated KPI tests:
  - AIAG MSA Gauge R&R ✅ (average-and-range worksheet: K1/K2/K3 and the
    `EV²/(n·r)` AV correction);
  - ISO 50001 / 50006 EnPI ✅ (opt-in regression baseline);
  - ISO 22400 OEE and MTBF / MTTR ✅ (reference examples);
  - next: AIAG SPC rules and Cp/Cpk.
- The docs are tested: every `ts_shape` name in a code block must import,
  and CI builds the docs with `mkdocs build --strict`.

## W4 — Scale and lean install
- Core install: pandas, numpy and scipy only. Cloud and SQL loaders become
  extras.
- Vectorize the remaining row loops.
- After 1.0: backend-agnostic primitives (Polars / DuckDB) and
  TimescaleDB push-down. Publish benchmarks.

## W5 — Understanding
- `profile()` as the entry point.
- A question-first cookbook ("Why did OEE drop?", "Which station is the
  bottleneck?", "Is my gauge capable?") on a bundled, realistic plant
  dataset.
- Auto-generated detector reference: assumptions, input signal kinds, and
  output shape.

## W6 — Ecosystem
- Grafana / TimescaleDB recipes.
- A CLI (`ts-shape profile data.parquet`).
- Semver with a deprecation policy from 1.0.

## Process analytics
Added:
- change-point detection (`ChangePointEvents`: exact PELT and binary segmentation);
- multivariate SPC (`MultivariateSPCEvents`: Hotelling T² with Phase I / II limits and contributions);
- reliability (`ReliabilityEvents`: MTBF / MTTR per ISO 22400 and a censored Weibull life model).

Candidates from past sessions, by value and effort:

| Feature | Value | Effort |
|---|---|---|
| Vibration: FFT spectral band energy | high | M |
| Tool wear vs. measurement drift; first-article / setup verification; part disposition | high | M each |
| MQTT / Kafka streaming loader | med-high | M |
| Detector evaluation (precision / recall against labels) | med | M |
| Scaling transforms | med | S |
| Energy / production plot helpers in an optional `[viz]` extra | med | S |

## Recoverable work on other branches
Two large branches from earlier sessions were never merged. Both are intact
and can be ported in their own PRs. Reconcile each with the core first:

- **`claude/reporting-python-library-2fhu1a`**: `ts_shape.reporting`
  (8.7k lines, 211 tests).
  - Contents:
    - facility calendar with IANA time zones;
    - weighted KPI roll-up;
    - ISA-95 hierarchy;
    - benchmarking, scorecard and variance bridge;
    - FX normalization;
    - reporting coverage.
  - Reconcile:
    - build the calendar on `signals.windows` / `shift_windows` (DST-correct
      already) instead of a second implementation;
    - drop its docs-CI job (the strict docs build is now in `ci.yml`);
    - keep SECURITY.md and the issue templates.
- **`claude/global-production-reporting-w5f10n`**: `FleetProductionReport`
  and `FleetSpec` (11k lines).
  - Contents:
    - machine-status adapters (PackML, stacklight, code maps);
    - counter wrap vs. reset;
    - part-done pulses;
    - cycle-time unit inference;
    - production context with late binding.
  - Reconcile:
    - map the status adapters onto `SignalSpec(kind="state")` and
      `OperatingStateEvents`;
    - add counter wrap as a `reset_policy` (`"wrap"`) in `SignalSpec` and
      `utils.counters`, instead of a parallel counter model;
    - fold cycle-time unit inference into `infer_specs`.
