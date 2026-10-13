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
  `time_weighted`, `group_by_uuid`) become the default behaviour.
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
- Next: reference-validated KPI tests:
  - ISO 22400 OEE, MTBF/MTTR;
  - AIAG SPC rules and Cp/Cpk;
  - AIAG MSA Gauge R&R;
  - ISO 50001 EnPI.

## W4 — Scale and lean install
- Core install: pandas, numpy and scipy only. Cloud and SQL loaders become
  extras.
- Vectorise the remaining row loops.
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
