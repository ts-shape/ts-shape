# Migration & Deprecations

A running list of renames and deprecated aliases, so existing code keeps working
while you migrate. Deprecated names still function but emit a
`DeprecationWarning`; they may be removed in a future major release. The
authoritative, dated record is the [Changelog](changelog.md) — this page is the
quick old → new lookup.

## Renamed classes and methods (old names deprecated, removed in 1.0)

Detector classes all end in `Events`. Event classes no longer use `calculate_` / `compute_` / `process_`: methods that find events start with `detect_`, and metrics are nouns (see [Naming & Output Conventions](naming.md)).

The old names still work. They emit a `DeprecationWarning`, and the event log accepts them too: for example, `to_event_log(df, "OEECalculator.calculate_oee")` is resolved to `OEEEvents.oee`.

| Old | New |
|---|---|
| `CycleTimeTracking` | `CycleTimeEvents` |
| `DowntimeTracking` | `DowntimeEvents` |
| `OperatorPerformanceTracking` | `OperatorPerformanceEvents` |
| `PartProductionTracking` | `PartProductionEvents` |
| `PerformanceLossTracking` | `PerformanceLossEvents` |
| `QualityTracking` | `QualityEvents` |
| `ReworkTracking` | `ReworkEvents` |
| `ScrapTracking` | `ScrapEvents` |
| `SetupTimeTracking` | `SetupTimeEvents` |
| `TargetTracking` | `TargetEvents` |
| `OEECalculator` (`calculate_availability`, `calculate_performance`, `calculate_quality`, `calculate_oee`) | `OEEEvents` (`availability`, `performance`, `quality`, `oee`) |
| `PeriodSummary` | `PeriodSummaryEvents` |
| `ShiftReporting` | `ShiftReportEvents` |
| `ShiftHandoverReport` | `ShiftHandoverEvents` |
| `ProcessStabilityIndex` | `ProcessStabilityIndexEvents` |
| `StatisticalProcessControlRuleBased` (`process`, `apply_rules_vectorized`, `calculate_control_limits`, `calculate_dynamic_control_limits`) | `StatisticalProcessControlEvents` (`detect_violations`, `detect_rule_violations`, `control_limits`, `dynamic_control_limits`) |
| `ToleranceDeviationEvents.process_and_group_data_with_events` / `.compute_capability_indices` | `.detect_deviations` / `.capability_indices` |
| `DataGapAnalysisEvents.find_gaps` | `.detect_gaps` |
| `MultiSensorValidationEvents.identify_outlier_sensor` | `.detect_outlier_sensor` |
| `LeadTimeAnalysisEvents.calculate_lead_times` | `.lead_times` |
| `EnvironmentalCompensationEvents.estimate_sensitivity` | `.sensitivity` |
| `DesignOfExperimentsEvents.compute_effects` | `.effects` |
| `*.alarm_duration_stats`, `*.batch_duration_stats`, `*.state_duration_stats` | `*_statistics` |
| `DutyCycleEvents.duty_cycle_per_window`, `RuntimeAccountingEvents.runtime_per_window` | `*_by_window` |

To find every remaining old name in your code, run your tests with `-W error::DeprecationWarning`.

## Every detector output carries its time bounds

Each detector returns the canonical columns of its shape (see [Naming & Output Conventions](naming.md)):
- **intervals:** `start`, `end`, `duration_seconds`, `uuid`, `source_uuid`;
- **summaries:** `start`, `end`, `duration_seconds`, also when they are keyed by `date`, `(date, shift)` or `period`;
- **points:** `systime`, `uuid`, `source_uuid`.

The new columns were added next to the old ones; none was removed or renamed. Two consequences:

- **Merging two summaries on `date` or `(date, shift)`** now finds `start`, `end` and `duration_seconds` on both sides, and pandas suffixes them `_x` / `_y`. Merge on the window instead, or select your columns first:

```python
quality = QualityTracking(df).daily_quality_summary("line1:ok", "line1:nok")
production = PartProductionTracking(df).daily_production_summary("line1:part", "line1:count")
daily = quality.merge(production, on=["start", "end", "duration_seconds"])
```

- **Night shifts are one row.** An overnight shift (e.g. 22:00 to 06:00) is attributed to the day it starts (`production_day=True` is now the default). Before, grouping by calendar date split one night across two rows, and each row mixed two different nights. Pass `production_day=False` to get the old grouping.

## `ShiftReporting.best_and_worst_shifts` will return a DataFrame

It still returns `{"best": DataFrame, "worst": DataFrame}`, but now emits a `FutureWarning`. Pass `as_frame=True` to get the 1.0 behavior already: one frame with a `rank_type` column (`best` / `worst`).

## Removed exporter aliases

The event-log exporters were renamed into a consistent `to_event_log_*` family.
The old aliases (`to_flat_df`, `to_ocel_tables`) have now been **removed** —
update any remaining imports:

| Removed | Use instead |
|---|---|
| `to_flat_df(...)` | `to_event_log_xes(...)` |
| `to_ocel_tables(...)` | `to_event_log_ocel(...)` |

```python
# Before
from ts_shape.eventlog import to_flat_df, to_ocel_tables

# After
from ts_shape.eventlog import to_event_log_xes, to_event_log_ocel
```

## `to_event_log_ocel` now returns five tables

To fully model OCEL 2.0, `to_event_log_ocel(log)` returns an `OCEL2Tables`
bundle (object-to-object relations and time-varying object attributes were
added) instead of a 3-tuple. Access the frames by attribute:

```python
# Before
events_df, objects_df, relations_df = to_event_log_ocel(log)

# After
tables = to_event_log_ocel(log)
tables.events, tables.objects, tables.relations          # as before
tables.o2o, tables.object_changes                        # new OCEL 2.0 tables
```

## `org:resource` in XES export

`to_event_log_xes` previously copied the case object id into `org:resource`.
Per the XES standard `org:resource` is the performing resource, so it is now
populated only from an `operator`-type object relation, and **omitted** when
the log has no operators. Supply one via `objects={"operator": ...}` if you
need it.

## Class aliases

| Alias (back-compat) | Canonical name | Notes |
|---|---|---|
| `OrderTraceabilityEvents` | `ValueTraceabilityEvents` | Same class; the alias tracks single-ID traceability that is no longer order-specific. New code should import `ValueTraceabilityEvents`. |

## Standardized detector output (May 2026)

Every public DataFrame-returning detector method now emits one of three
canonical shapes — `point`, `interval`, or `summary` — defined in
`src/ts_shape/events/_output.py`. If you previously read shape-specific column
names, switch to the canonical ones:

| Old column(s) | Canonical column |
|---|---|
| `start_time`, `gap_start`, `window_start`, `period_start`, `disturbance_start` | `start` |
| `end_time`, `gap_end`, `window_end`, `period_end`, `disturbance_end` | `end` |
| `severity_score`, `severity_level` | `severity` |
| `OEECalculator`: `date` | `start` + `end` + `duration_seconds` (86400) |

See the [Changelog](changelog.md) for the full dated history of breaking changes.
