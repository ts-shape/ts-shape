# Naming and output conventions

One concept, one name. This page is the rule set for every detector, and the checklist for reviewing a new one. Older spellings still work: they are deprecated aliases that emit a `DeprecationWarning` naming the replacement, and they will be removed in 1.0 (see [Migration & Deprecations](migration.md)).

## Classes

| Kind | Suffix | Examples |
|---|---|---|
| Emits events or KPI tables from signals | `Events` | `DowntimeEvents`, `OEEEvents`, `ReliabilityEvents` |
| Transforms a frame into a frame | role noun | `DoubleFilter`, `TimezoneShift`, `PowerIntegrator` |
| Loads data | `Loader` / `DataAccess` | `ParquetLoader` |
| Computes features | role noun | `NumericStatistics`, `CycleExtractor` |

## Methods

| Returns | Name | Examples |
|---|---|---|
| Events found in the data | `detect_<what>` | `detect_change_points`, `detect_micro_stops` |
| A metric or KPI table | the noun | `oee`, `mtbf_mttr`, `availability_trend` |
| A metric per window | `<noun>_by_window`, `<noun>_by_shift` | `enpi_by_window`, `downtime_by_shift` |
| Descriptive statistics | `<noun>_statistics` | `cycle_time_statistics` |

Event classes do not use the prefixes `calculate_`, `compute_`, `get_` or `process_`.

## Parameters

| Concept | Name | Not |
|---|---|---|
| The input frame | `dataframe` | `df`, `data` |
| Timestamp column | `time_column` (default `"systime"`) | `column_name`, `timestamp_column`, `systime_col` |
| Value column | `value_column`; per role `value_column_<role>` (`value_column_counter`, `value_column_state`) | `value_col`, `column`, `counter_column`, `<role>_value_column` |
| Signal identifier column | `uuid_column` | `uuid_col`, `id_column` |
| The one signal a detector analyzes | `signal_uuid`, or a role name when the role matters: `setpoint_uuid`, `run_state_uuid`, `counter_uuid` | `uuid`, `sensor_uuid` |
| Several signals | `signal_uuids` | `sensor_uuids`, `uuid_list`, `uuids` |
| Run / idle state signal | `run_state_uuid` | `run_uuid`, `state_uuid` |
| Event identifier of the output | `event_uuid`, default `"<pack>:<name>"` | free-form defaults such as `outlier_event` |
| Time bucket (offset string) | `window` | `freq`, `period`, `resample` |
| Rolling window in rows | `window_size` | an integer `window` |
| Shortest event kept | `min_duration` (offset string) | `min_hold`, `min_length` |
| Longest gap a value is held | `max_gap` (offset string) | |
| Sigma multiplier | `n_sigma` | `z_threshold`, `threshold_sigma` |
| An absolute level | `threshold` | |
| Bounds of an allowed range | `lower_limit`, `upper_limit` | `low`/`high`, `lower_bound`/`upper_bound`, `low_limit`/`high_limit` |
| Specification limits | `lower_spec`, `upper_spec` | `lsl`, `usl` |
| Two severity levels | `warning_threshold`, `critical_threshold` | `minor_threshold`, `moderate_threshold` |
| Count the state at the start of the data | `include_initial` | `include_first`, `count_initial_active` |
| Restrict the output | `only_<what>` | `<what>_only` |
| Counter resets | `handle_resets` | |

Signal, column and frame arguments come first. Every optional argument after them is keyword-only.

## Output columns

Every detector returns a `pandas.DataFrame` in one of four shapes (`ts_shape.events._output`):

| Shape | Required columns | Use for |
|---|---|---|
| point | `systime`, `uuid`, `source_uuid` | something that happened at an instant |
| interval | `start`, `end`, `duration_seconds`, `uuid`, `source_uuid` | something that lasted |
| summary | `start`, `end`, `duration_seconds` | a value per time window |
| static | — | a table not tied to time (statistics by reason, a fitted model) |

- **Time** columns are tz-aware whenever the input is. Calendar labels (`date`, `shift`, `period`) may be added next to `start` and `end`, never instead of them.
- **Units** go in the column name: `_seconds`, `_minutes`, `_hours`, `_kwh`. Percentages end in `_pct` and run 0–100. Fractions end in `_ratio` and run 0–1. A bare name such as `availability` does not say which.
- **Durations** are float seconds in `duration_seconds`. Timedelta columns may be added for convenience, but not instead.
- **Counts** use `<thing>_count`. Use `sample_count` for samples, not `n_samples`.
- **Severity** is one of `info`, `warn`, `critical` when a detector emits a categorical level. Numeric scores go in `severity_score`.
- **Empty results** keep the full column set, so code written against a normal result also works when nothing was found.

## Errors and warnings

| Situation | Behavior |
|---|---|
| A required column is missing | `ColumnNotFoundError` (a `ValueError`) naming the column |
| The requested signal uuid is not in a non-empty frame | `DataQualityWarning`, empty result with the full schema |
| Too little data for a statistic | empty result or NaN, plus `DataQualityWarning` |
| A deprecated name is used | `DeprecationWarning` naming the replacement |

## Wording

American spelling (analyze, normalize, behavior). Use:
- **signal** for a uuid's time series, not tag or series;
- **window** for a time bucket;
- **downtime** for unplanned stops, **idle** for planned ones;
- **part** for a produced unit;
- **NOK** for a rejected part;
- **setpoint** for a controller target, **target** for a production goal.
