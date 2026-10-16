# Naming and output conventions

One concept, one name. This page is the rule set for every detector, and the checklist for reviewing a new one. Older spellings still work: they are deprecated aliases that emit a `DeprecationWarning` naming the replacement, and they are removed only in a future major release (see [Migration & Deprecations](migration.md)).

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
| Value column | `value_column`; per role `value_column_<role>` (`value_column_counter`, `value_column_state`); several: `value_columns` | `column_name`, `value_col`, `value_cols`, `column`, `metric`, `counter_column`, `<role>_value_column` |
| Signal identifier column | `uuid_column` (also `cycle_uuid_column`, `metadata_uuid_column`, ...) | `uuid_col`, `*_uuid_col`, `id_column` |
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
| Counter resets | `handle_resets` (default `True`) | |

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
- **Durations** are float seconds in `duration_seconds`. Timedelta columns may be added for convenience, but not instead. Any other duration is reported in `<name>_seconds`. A column in another unit (`_minutes`, or the legacy `_s`) is a convenience only and sits right after its `_seconds` twin.
- **KPI percentages carry their basis:** a `_pct` computed from times or counts comes with the columns it was computed from (for example `run_seconds`, `planned_seconds`, `excluded_seconds`, `good_parts`, `total_parts`).
- **Summary methods return a DataFrame** (one row, or one row per item) with `start` / `end` / `duration_seconds` for the data the summary covers. A dict or list return is legacy, available only through `as_frame=False`.
- **Renamed columns** keep their legacy spelling as a deprecated twin until the next major release: the canonical column (same values, or the same value in seconds for a `_ms` column) sits right after its legacy twin, so a frame reads old, new. Window bounds are the exception: `start`, `end`, `duration_seconds` lead every summary, and the legacy label (`window_start`, `window_end`, `time`) stays where it was.
- **Counts** use `<thing>_count`. Use `sample_count` for samples, not `n_samples`.
- **Severity:** a detector's `severity` keeps its own labels (`minor`/`major`, `low`/`medium`/`high`, ...). Every result with a categorical `severity` also has `severity_level`, one of `info`, `warn`, `critical`, so results of different detectors can be filtered together. A numeric score (outlier detection) has no level.
- **Empty results** keep the full column set, so code written against a normal result also works when nothing was found.
- **Precision:** numbers are returned as computed. Detectors never round; rounding belongs in reports.
- **Defaults are correct:** a flag that fixes a known error (`handle_resets`, `interval_end="next_change"`, `time_weighted`, `group_by_uuid`) defaults to the correct behavior; the naive behavior is opt-in.

## Errors and warnings

| Situation | Behavior |
|---|---|
| The time column is missing (non-empty frame) | `ColumnNotFoundError` (a `ValueError`) naming the column and listing the columns |
| The uuid column is missing (non-empty frame) | `ColumnNotFoundError` |
| A value column argument names a column the frame does not have | `ColumnNotFoundError` naming the column |
| A signal uuid argument (`*_uuid`, `*_uuids`) is not in a non-empty frame | `DataQualityWarning` listing the available uuids; the result is empty. A few detectors (alarm, batch, degradation, line balancing, ...) also raise `ValueError` |
| Too little data for a statistic | empty result or NaN, never `inf` and never a placeholder such as `0.0` or `0 %`; a mode of no data is `None` |
| A code that a mapping table does not know | kept unchanged, with a `DataQualityWarning` counting them |
| Input rows are skipped (non-numeric values, unreadable files, malformed spec entries) | `DataQualityWarning` saying what was skipped; never silent |
| A deprecated name is used | `DeprecationWarning` naming the replacement |

The uuid and column checks are applied to every detector in `ts_shape.events` by its base class, so they behave the same in every pack. Transforms, features, signals, the event log and the context helpers use the same checks (`ts_shape.utils.frames`), and contract tests run every public function on empty, tz-aware, nullable and pyarrow input. They report the call you made, not the detector's internal calls. A missing *value* column (`value_column`, `value_column_<role>`, ...) raises `ColumnNotFoundError` naming the column when the method looks it up.

## Wording

American spelling (analyze, normalize, behavior). Use:
- **signal** for a uuid's time series, not tag or series;
- **window** for a time bucket;
- **downtime** for unplanned stops, **idle** for planned ones;
- **part** for a produced unit;
- **NOK** for a rejected part;
- **setpoint** for a controller target, **target** for a production goal.
