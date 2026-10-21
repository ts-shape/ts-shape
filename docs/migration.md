# Migration and deprecations

Renames, deprecated aliases and behavior changes. Deprecated
names still work and emit a `DeprecationWarning`; they are removed only in a
future major release. To find every deprecated name your code still uses, run
your tests with `-W error::DeprecationWarning`. The dated record is the
[Changelog](changelog.md).

## October 2026, round 16: DST-day windows, more chunkable methods

Each change is pinned by a hand-computed test in
`tests/reference/test_round16_dst.py`. `tests/invariants/test_dst_windows.py`
checks every summary method on both DST changes.

### Results that change (DST days on tz-aware data only)

On a DST change a daily window is 23 h (82800 s, spring) or 25 h (90000 s,
fall) long. These methods measured it as a fixed 24 h; they now use the real
window. Windows that do not cross a DST change are unchanged, bit for bit.

- Time shares, percentages and rates:
  - `ThresholdMonitoringEvents.time_above_threshold`: `time_above`,
    `time_above_seconds`, `pct_above` and `time_above_pct`, in both
    `interval_end` modes. A signal that is always above the threshold on
    2024-03-31 in Europe/Berlin reads 82800 s and 100 %. Before, it read
    86400 s, or 95.8 % when the data ended at midnight.
  - `ThresholdMonitoringEvents.threshold_exceedance_trend`:
    `total_duration(_seconds)`.
  - `MicroStopEvents.micro_stop_frequency`: `pct_of_window` and
    `share_of_window_pct`.
  - `DutyCycleEvents.duty_cycle_by_window(time_weighted=False)`: `on_time`
    and `off_time`.
  - `InventoryMonitoringEvents.consumption_rate`: the rate per hour.
  - `FlowMetricsEvents.throughput` and `flow_summary`:
    `throughput_per_hour`.
  - `DisturbanceRecoveryEvents.disturbance_frequency`: `pct_time_disturbed`
    and `time_disturbed_pct`.
- `end` is the next local midnight and `duration_seconds` the real length,
  instead of start + 24 h. Affected:
  - `FlowMetricsEvents.throughput`, `flow_summary` and `wip_over_time`
  - `BottleneckDetectionEvents.detect_bottleneck`
  - `AssetHealthIndexEvents.health_index`
  - `MultiSensorValidationEvents.detect_disagreement`
  - `SensorDriftEvents.detect_zero_drift`
  - `MaterialBalanceEvents.detect_balance_exceedance`
  - `LineBalancingEvents.station_cycle_times` and `balance_metrics`
  - `PerformanceLossEvents.slow_periods`
  - `ControlLoopHealthEvents.detect_oscillation`
- Membership follows the real day:
  - `SensorDriftEvents.detect_zero_drift`, `detect_span_drift` and
    `calibration_health` look up the reference over the real day. A 23 h
    day no longer includes reference samples from the next day's first
    hour.
  - `ControlLoopHealthEvents.loop_health_summary` counts oscillation
    episodes in the real window (`oscillation_count`).
  - `DutyCycleEvents.excessive_cycling` averages the intervals in the real
    window (`avg_on_duration` / `avg_off_duration`).
- `DisturbanceRecoveryEvents.disturbance_frequency` also assigns
  disturbances to windows by the real day.

### Chunked processing

- `ts_shape.stream.CONDITIONAL` (`"Class.method" -> Condition(argument,
  reason)`) lists methods that run chunked only with one argument:
  - `DataGapAnalysisEvents.detect_gaps` and `coverage_by_period` need
    `min_gap=...`.
  - `AnomalyClassificationEvents.detect_flatline` needs
    `interval_end="last_sample"`.
  - `ChangeoverEvents.changeover_window` works, but not with
    `until="stable_band"`.

  Without it, `NotChunkableError` names the argument to pass.
- Calendar chunks: `chunk="W"` (Monday to Monday; also `"W-<DAY>"`) and
  `chunk="MS"`, cut at local midnight.
  `PeriodSummaryEvents.weekly_summary` and `monthly_summary` are chunkable
  with `chunk="W"` and `chunk="MS"`.
- `MicroStopEvents.micro_stop_frequency` is chunkable. Empty windows between
  slices are filled in on merge.
- `iter_chunked` yields all `detect_gaps` rows, and the last
  `coverage_by_period` window, after the last slice. They need the whole
  run's sample count or median sampling step.
- Fix: chunked `ChangeoverEvents.detect_changeover` with `min_duration`,
  `changeover_window` with a `config` `min_duration`, and
  `MicroStopEvents.micro_stop_impact` now match the in-memory result when an
  event crosses a slice border.
- `DataGapAnalysisEvents.interpolation_candidates` is no longer chunkable.
  Its `safe_to_interpolate` uses the whole signal's standard deviation, so
  run it in memory.

## October 2026, round 15: chunked processing

New only; no existing result changes. See
[Chunked processing](guides/chunked-processing.md).

- `ts_shape.run_chunked(source, Target, "method", chunk="1D", **kwargs)` runs
  a detector slice by slice and returns the same frame as
  `Target(all_data, ...).method(...)`. Events and windows that cross a slice
  border are stitched: open intervals are carried into the next slice, and
  overnight shift windows belong to the day they start. `iter_chunked` yields
  finished rows slice by slice. `time_range_source(load, start, end)` loads
  one slice per call, for databases. A source can be any iterable of frames
  or `(name, frame)` pairs in time order, such as a loader's
  `stream_by_time_range`.
- `Pipeline.run_stream(source, chunk="1D")` runs a pipeline the same way.
  Transforms must be row-local. `keep_data=True` keeps the transformed data.
- `ts_shape.NotChunkableError` (a `ValueError`) is raised for methods that
  need the whole signal: whole-signal statistics, model fits, first-window
  baselines and static summaries. `ts_shape.stream.SUPPORT` lists the
  chunkable methods, each checked against the in-memory result by
  `tests/test_stream_equivalence.py`.
- On data in a time zone with DST, chunked runs reject windows of several
  hours that are not whole days (for example `"8h"`). Use a window that
  divides an hour, whole days, or UTC data.

## October 2026, round 14: audit leftovers

Each change is pinned by a hand-computed test in
`tests/reference/test_round14_*.py`.

### Results that change

- `DemandPatternEvents.demand_by_period` keeps periods without orders as zero
  demand (`empty_periods="zero"`, the new default): every period from the
  first to the last one with data is returned, empty ones with
  `total_demand = 0`, `peak_demand = 0`, `order_count = 0` and `avg_demand`
  NaN. New `order_count` column. `detect_demand_spikes` and
  `seasonality_summary` use the zero-filled totals, which lowers the spike
  baseline and the weekday averages (orders of 10 every other day: Monday
  average 10 → 5). `empty_periods="drop"` restores the former output.
  Cumulative counters must be differenced into per-order quantities first.
- `LeadTimeAnalysisEvents.lead_times` with order identifiers pairs orders and
  deliveries one-to-one, first in first out, within each identifier. One
  delivery could close several orders with the same identifier and later
  deliveries were ignored (two orders, deliveries on Jan 5 and Jan 6: 96 h and
  72 h → 96 h and 96 h). Unpaired orders and deliveries raise a
  `DataQualityWarning` on this path too. Multi-line orders need line-level
  identifiers. `lead_time_statistics` and `detect_lead_time_anomalies`
  follow.
- `DataGapAnalysisEvents.coverage_by_period`: the data covers
  `[first sample, last sample + median sampling step)`; time of the first and
  last window outside that span is missing (in `gap_seconds`; `gap_count`
  still counts internal gaps). 1 s data from 10:30 to 11:29:59 in hourly
  windows reads 50 % / 50 % (was 100 % / 100 %), as `data_completeness`.
- `SamplingResolutionEvents.detect_aliasing` takes the sampling rate from the
  timestamps of non-missing values (every other value NaN on a 1 s grid:
  1.0 Hz → 0.5 Hz; dominant frequency and `aliasing_risk` follow).
- `DegradationDetectionEvents` and `AssetHealthIndexEvents`: the baseline is
  the half-open `[first, first + window)`, as long as the rolling windows, so
  a stationary signal has a variance ratio of 1. `detect_variance_increase`
  does not flag samples inside the baseline window.
- `ControlLoopHealthEvents.detect_oscillation` and
  `AnomalyClassificationEvents.detect_oscillation` / `classify_anomalies`: a
  zero carries the last non-zero sign, so `1, 0, -1` is one crossing (was two)
  and touching zero is none. The control-loop period doubles where zeros sat
  between the signs.

### New and clarified

- `RateOfChangeEvents.rate_statistics`: `mean_abs_rate`, `std_abs_rate`,
  `max_abs_rate`, `min_abs_rate`. The `*_rate` columns are statistics of the
  absolute rate today and become statistics of the signed rate in the next
  major release; read `*_abs_rate` for the absolute rate.
- `CalibrationValidationEvents.calibration_curve`: a `power` column after
  `coefficient_index`. Coefficients are highest power first and describe
  measured = f(reference).
- Canonical twins: `OperatingStateEvents.state_duration_statistics`
  `count` → `interval_count`; `MicroStopEvents.micro_stop_frequency`
  `count` → `micro_stop_count`.
- `signals={uuid: ...}` arguments (`AssetHealthIndexEvents`,
  `StartupDetectionEvents.detect_startup_multi_signal`) get the
  uuid-not-found warning; `AssetHealthIndexEvents` warns once per signal whose
  values are in another `value_*` column, also when the index is computed
  from the others.
- `signals.primitives.windows` (docs): sub-daily windows are fixed lengths
  laid out like `resample`, so their wall-clock start moves after a DST
  change; use `shift_windows` for wall-clock shifts.

## October 2026, round 13: audit fixes

Fixes from an audit of the engineering, quality and maintenance packs. Each
one is pinned by a hand-computed test in `tests/reference/test_round13_*.py`.

### Engineering: time-weighted results

- `ControlLoopHealthEvents.error_integrals`: IAE, ISE and ITAE integrate the
  held error over each window, split at the window edges. Time that crossed a
  window edge was lost: a constant error of 1 sampled every 30 min gave IAE
  1800 per hour instead of 3600. ITAE measures time from the window start.
  A window covered by a held value but without samples is now reported.
- `ControlLoopHealthEvents.output_saturation`: the `pct_time_at_*` columns
  are shares of observed time, not of samples (33.3 % → 90.9 % on an output
  held at 100 for 50 of 55 minutes); `longest_saturation_seconds` is the
  held run length (a single sample held for 50 min was 0 s).
- `ThresholdMonitoringEvents.time_above_threshold` (`next_change`): windows
  covered by a held value are reported even without samples, and a signal
  that starts above the threshold counts one exceedance.
- `SteadyStateDetectionEvents.detect_steady_state`: the statistics use the
  steady samples only; the first sample after the steady period was
  included (20 zeros then 100: mean 5.0 → 0.0). `steady_state_statistics`
  returns NaN, not 0.0, when there is no steady period or no time span.
- `WarmUpCoolDownEvents.time_to_target`: overshoot is measured until the next
  ramp starts, not over the rest of the data.
- `RateOfChangeEvents.detect_rapid_change` and
  `StartupDetectionEvents.detect_startup_by_slope`: an interval starts at the
  sample before the first fast step (a single jump lasts one sample
  interval, not 0 s).
- `StartupDetectionEvents.assess_startup_quality`: the stability tail has at
  least two samples; a short startup ending flat scores 1.0 (was 0.0).

### Quality

- `ToleranceDeviationEvents`: `tolerance_lag` delays the limits, not the
  reported event times. All limit rows are used, also with `is_delta=False`,
  and a frame without an `is_delta` column works. `capability_indices` for a
  missing signal returns an empty frame with a warning instead of raising.
- `AnomalyClassificationEvents`: `detect_flatline` ends a run when the value
  leaves the tolerance around the run's first value (a slow ramp was reported
  as a stuck sensor). Windows are half-open and the last partial window is
  evaluated (a drift in the last 30 min of 90 min of data was missed).
- `SamplingResolutionEvents` and `ControlLoopHealthEvents`: sampling
  intervals are no longer truncated to whole milliseconds (1.5 ms sampling
  read as 1000 Hz; 0.5 ms sampling returned nothing).
- `DataGapAnalysisEvents`: a gap is an interval *longer* than `min_gap`, as
  documented. The default `min_gap=None` is twice the median sampling
  interval (as in `coverage_by_period`), so regular data has no gaps.
  `gap_summary` reports the real data span and NaN shortest/mean gap when
  there are no gaps.
- Calendar windows such as `"MS"` work in
  `CalibrationValidationEvents.detect_recalibration_need`,
  `calibration_stability` and
  `EnvironmentalCompensationEvents.detect_env_exceedance` (they raised).

### Maintenance, transforms, severity

- `AssetHealthIndexEvents.health_events`: a recovery to a better band is
  `direction="above"`.
- `IntegerCalc.power_column`: negative exponents and results beyond int64 are
  float (`2 ** 64` wrapped to 0).
- One severity table: the event log's `ts_shape:severity` equals the
  detector's `severity_level` column. Changed in the event log: tolerance
  `major` → critical (was warn), CUSUM `high` → warn (was critical),
  flow-constraint `severe` → critical (was missing). `minor` is `info` and
  threshold `alarm` is `critical` in `severity_level`.

### Robustness

- A signal whose values are in another `value_*` column than the one the
  method reads now emits a `DataQualityWarning` naming the column to pass
  (`value_column=...`). Results are unchanged.
- `MachineStateEvents` and `DutyCycleEvents`: a missing state (`NaN`,
  `pd.NA`) is unknown time, not idle/off; ratios use the known time. Averages
  over no intervals are NaN. `duty_cycle_by_window` and `cycle_count` extend
  to the constructor's `data_end`.
- `MicroStopEvents` and `DutyCycleEvents.on_off_intervals` accept a naive or
  string `data_end` on tz-aware data (local wall time; it raised).
- A `value_column` that does not exist raises `ColumnNotFoundError` in
  `PerformanceLossEvents` and in the state-based detectors (it fell back
  silently). `ProductionScheduleAdherenceEvents` raises
  `ColumnNotFoundError` (a `ValueError`) for missing schedule columns.
- Variance ratios and `shift_sigma` are NaN, not `inf`, when the reference
  spread is undefined (`ProcessWindowEvents`, `DisturbanceRecoveryEvents`,
  `ChangePointEvents`).
- `CrossSignalAnalytics` and `FeatureMatrixExporter` accept numbers stored as
  object dtype.

### Results that change (names and intervals)

- `OperatingStateEvents` intervals end at the next state's first sample
  (`interval_end="next_change"`, the library default), so consecutive states
  cover the signal without gaps. At 30 s sampling a 15-minute phase is 900 s
  (was 870 s). `interval_end="last_sample"` restores the former intervals.
- `MicroStopEvents.micro_stop_patterns(hour_grouping=False)` names shifts
  `shift_1` / `shift_2` / `shift_3` (was `morning` / `afternoon` / `night`;
  pass `shift_definitions=` for your own) and groups a shift that crosses
  midnight by its start date: two stops at 23:00 and 02:00 in a 22–06 shift
  are one group (`avg_count` 2.0, was two groups of 1.0).
- `LineThroughputEvents.cycle_quality_check`: `tolerance_pct` is a percent
  (default `10.0`) like every other `*_pct`; a value below 1 is still read as
  the former fraction, with a `DeprecationWarning`. `deviation_pct` is 0–100
  (was 0–1); the fraction is in the new `deviation_ratio`. `quality_flag`
  bins follow the tolerance (good ≤ tol, warning ≤ 2.5 × tol); at the
  default they are the former fixed bins.
- `GoldenBatchDeviationEvents` works with tz-aware timestamps (it raised).
- `Pipeline` steps accept deprecated keyword names (with the warning) instead
  of rejecting them as unknown.

### Renamed parameters

| Where | Old | New |
|---|---|---|
| `ChangeoverEvents.detect_changeover`, `SetpointChangeEvents.detect_setpoint_steps` | `min_hold` | `min_duration` |
| `ChangeoverEvents.changeover_window` config key | `min_hold` | `min_duration` |
| `SetpointChangeEvents.detect_setpoint_changes` | `min_hold` | `min_step_duration` (its `min_duration` is the ramp duration) |
| `OutlierDetectionEvents.detect_outliers_zscore`, `detect_outliers_mad` | `threshold` | `n_sigma` |
| `ToleranceDeviationEvents` (argument and attribute) | `tolerance_column`, `actual_column` | `value_column_tolerance`, `value_column_actual` |
| `ContextEnricher.enrich_with_tolerances` | `low_col`, `high_col` | `lower_limit_column`, `upper_limit_column` |
| `ContextEnricher.enrich_with_mapping` | `raw_value_col`, `mapped_value_col` | `value_column_raw`, `value_column_mapped` |
| `GoldenBatchDeviationEvents` | `reference_df` | `dataframe` |
| `CycleDataProcessor` (argument and attribute) | `cycles_df`, `values_df` | `dataframe_cycles`, `dataframe_values` |
| `PatternRecognition` methods | `uuid` | `signal_uuid` |
| `LineThroughputEvents.cycle_quality_check` | `tolerance_pct` as a fraction | `tolerance_pct` in percent, or `tolerance_ratio` |

### Renamed output columns (old column kept as a twin)

Reading the old column does not warn; it is removed in the next major release.

| Where | Old | New |
|---|---|---|
| `ToleranceDeviationEvents.capability_indices` | `Cp`, `Cpk`, `Pp`, `Ppk`, `Cpm` | `cp`, `cpk`, `pp`, `ppk`, `cpm` |
| `ContextEnricher.enrich_with_tolerances` | `low_limit`, `high_limit` | `lower_limit`, `upper_limit` |
| `DutyCycleEvents` | `on_time`, `off_time`, `avg_on_duration`, `avg_off_duration` | `*_seconds` |
| `MachineStateEvents.state_quality_metrics` | `avg_run_duration`, `avg_idle_duration`, `run_idle_ratio` | `*_seconds`, `run_idle_factor` |
| `MicroStopEvents` | `total_run_time`, `total_micro_stop_time`, `total_lost_time`, `avg_lost_time` | `*_seconds` |
| `OperatingStateEvents.state_duration_statistics` | `mean/std/min/max/total_duration` | `*_duration_seconds` |
| `ThresholdMonitoringEvents` | `time_above`, `total_duration` | `time_above_seconds`, `total_duration_seconds` |
| `SignalQualityEvents.sampling_regularity` | `mean/std/min/max_interval` | `*_interval_seconds` (and `gap_duration_seconds` beside `gap_duration`) |
| `CycleTimeEvents` | `count`, `moving_avg`, `avg/min/max_cycle_time` | `cycle_count`, `moving_avg_seconds`, `*_cycle_time_seconds` |
| `LineBalancingEvents` | `cycle_time_mean/median/std`, `bottleneck_cycle_time`, `n_stations` | `*_seconds`, `station_count` |
| `LineThroughputEvents` | `count`, `expected_time` | `part_count`, `expected_time_seconds` |
| `WarmUpCoolDownEvents.warmup_consistency` | `deviation_from_median_duration` | `deviation_from_median_duration_seconds` |
| `LeadTimeAnalysisEvents`, `SetupTimeEvents`, `BatchTrackingEvents` | `count` | `order_count`, `setup_count`, `batch_count` |
| `ProcessWindowEvents.windowed_statistics`, `OperatingRangeEvents.value_distribution` | `count` | `sample_count` |
| `ReliabilityEvents.weibull_fit` | `n_failures`, `n_censored` | `failure_count`, `censored_count` |
| `SamplingResolutionEvents` | `n_unique_values`, `resolution_ratio` | `unique_value_count`, `resolution_factor` |
| `ValueDistributionEvents.detect_mode_changes` | `n_modes_detected` | `mode_count` |
| `DesignOfExperimentsEvents.effects` | `n_runs` | `run_count` |
| `DegradationDetectionEvents`, `DisturbanceRecoveryEvents`, `ProcessWindowEvents` | `variance_ratio` | `variance_factor` |
| `FlowMetricsEvents.flow_summary` | `consistency_ratio` | `consistency_factor` |
| `StatisticalProcessControlEvents.detect_violations` | `triggered_rule`, `value_double` | `rule`, `value` (as `detect_rule_violations`) |

`docs/naming.md` now spells out the rules behind these: durations carry a
unit suffix, counts are `<thing>_count`, and an unbounded quotient is
`*_factor` (a `*_ratio` stays within 0–1).

## October 2026, round 12: oscillations, EnPI baseline, windowed features

### Results that change

- `SetpointChangeEvents.control_quality_metrics`: `oscillation_count` and
  `oscillation_freq_hz` count a crossing of the setpoint only once the error
  leaves a band around it on the other side. The band (`deadband=`) defaults
  to the settle band, `tol` or `settle_pct` of each step, so noise a settled
  response shows inside the band is no longer counted as oscillation. A step
  that swings three times and then sits in ±0.1 noise with `tol=0.5`:
  82 → 3. With the default `tol=0.0` nothing changes; `deadband=0` restores
  the former count. `overshoot_metrics` and `oscillation_frequency` take the
  same `deadband=` and default to `0` (unchanged).
- `EnergyPerformanceIndicatorEvents.enpi_vs_baseline(baseline="regression")`
  leaves baseline windows without output (zero or missing units) out of the
  `energy = a + b * units` fit and names them in a `DataQualityWarning`. A
  window with energy but no units (the counter's first window, a stop day)
  formerly pulled the fit: with a 600 kWh idle day before nine days on
  `200 + 2 * units` the intercept was ~330 instead of 200. Results change only
  when the baseline period has such windows.
- `ts_shape.utils.frames.window_offset` with a `Timedelta` (or
  `datetime.timedelta` / `numpy.timedelta64`) window returns a fixed
  duration on pandas 2.2 as on pandas 3: `Timedelta("2D")` → `<48 * Hours>`
  (pandas 2.2 formerly gave a two-calendar-day offset, 47 h across the spring
  DST change). Window ends of every method that takes a `Timedelta` window
  are affected on pandas 2.2 only. Offset strings like `"1D"` stay calendar
  days.

### New

- `DescriptiveFeatures.compute_windowed(window="1s", metrics=None)` (also
  `compute(output_format="windowed", window=...)`): one row per time window,
  `start`, `end`, `duration_seconds`, then `{uuid}__{feature}` for every
  signal. Every window from the first to the last timestamp is present; a
  signal without data in a window is NaN, its counts (`total_rows`,
  `true_count`, ...) are 0. `metrics=` selects features by name. See
  [Features for ML](guides/ml-features.md).

## October 2026, round 11: typed API, docs that run

No result changes except the fixes listed under *Fixes that change output*.

### New and wider imports

- `from ts_shape import X` is fully typed for mypy, pyright and IDEs (it was
  `object`). Deprecated classes, methods and attributes are reported by type
  checkers (mypy: `--enable-error-code deprecated`); runtime warnings are
  unchanged.
- New top-level exports: every transform and feature class, `ValueMapper`,
  `PipelineResult`, and `TsShapeWarning`, `DataQualityWarning`,
  `PerformanceWarning`, `LoaderConfigWarning`, `ColumnNotFoundError`,
  `LoaderError`. `ts_shape.events` re-exports every detector;
  `ts_shape.context` has `__all__`; `ts_shape.loader` also exports the Azure
  flexible-file, Databricks and energy API loaders and `ContextEnricher`.
- `list_detectors()` lists every top-level public class, now including the
  transform and feature classes, and no longer imports the modules.

### Import time

`from ts_shape import OutlierDetectionEvents` takes about 0.35 s (was
1.2–1.9 s): scipy and scikit-learn load only in the methods that use them.
`outlier_detection.IsolationForest` is no longer a module attribute
(`SKLEARN_AVAILABLE` is kept). The z-score is computed in numpy, identical to
current scipy; on scipy older than 1.18 the last bits can differ, nearly
constant data gives NaN scores, and scipy's "Precision loss" warning is no
longer emitted.

### PostgreSQL loaders

`TimescaleDBDataAccess` and `DatapointDB` build their URL with the psycopg2
driver named explicitly (works on SQLAlchemy 2.1) and escape the password.
New settings: `db_port` (default 5432) and `db_driver` (default
`"psycopg2"`), as keys of `TimescaleDBDataAccess`'s `db_config` and as
keyword arguments of `DatapointDB`, which also takes `db_name` (default
`"config_repository"`). Both
have a `.url`; without psycopg2 an ImportError names `ts-shape[postgres]`.
Loader `repr()` shows the settings with passwords, keys, tokens and
connection strings hidden.

### Fixes that change output

- `PatternRecognition.discover_motifs(top_k > 1)` no longer returns trivial
  matches: a later motif never starts within `exclusion_zone` (default
  `window_size // 2`) of an earlier one, so ranks 2+ are different shapes.
  `top_k=1` is unchanged.
- `StartupDetectionEvents.detect_startup_by_slope` works with repeated index
  labels (it raised `IndexError`).
- `detect_startup_multi_signal` returns its full column set when nothing is
  found, for `logic="any"` and `"all"` (it returned a frame with no columns).
- `DatapointDB` gives an empty frame for a device without data points (it
  raised `KeyError: 'uuid'`).

### Names follow docs/naming.md

Old names keep working until the next major release: a renamed keyword or attribute warns with a `DeprecationWarning` naming the replacement; a renamed output column stays as a twin with the same values. A canonical column sits right after its legacy twin; window bounds (`start`, `end`, `duration_seconds`) lead every summary.

### Renamed parameters and attributes

| Old | New | Where |
|---|---|---|
| `state_uuid` | `run_state_uuid` | `DowntimeEvents` (`downtime_by_shift`, `downtime_by_reason`, `top_downtime_reasons`, `availability_trend`), `SetupTimeEvents` (`setup_durations`, `setup_by_product`, `setup_statistics`, `setup_trend`), `ShiftHandoverEvents` (`generate_report`, `highlight_issues`), `IdleEnergyDetectionEvents` (`idle_energy_by_window`, `idle_energy_by_shift`, `idle_energy_trend`), `EnergyEfficiencyEvents.idle_energy_waste`, `LongDowntimeEvents`, `ReliabilityEvents` |
| `.state_uuid` (attribute) | `.run_state_uuid` | `LongDowntimeEvents`, `ReliabilityEvents` |
| `run_value_column` | `value_column_state` | `OEEEvents.performance` |
| `cut_value_column` | `value_column_cut` | `ContinuousProcessAlignmentEvents.segment_by_cut` |
| `id_value_column` (argument and attribute) | `value_column_id` | `MultiProcessTraceabilityEvents`, `RoutingTraceabilityEvents` |
| `handover_value_column` (argument and attribute) | `value_column_handover` | `MultiProcessTraceabilityEvents` |
| `routing_value_column` (argument and attribute) | `value_column_routing` | `RoutingTraceabilityEvents` |
| `df` | `dataframe` | `to_event_log`, `object_intervals`, `detect_objects`, `attach_objects`, `run_backtest`, `evaluate` and `LambdaDetector.evaluate` / `.to_event_log` (lambda rules), `EnergyConsumptionEvents.normalize`, `EnergyEfficiencyEvents.normalize`, `StatisticalProcessControlEvents.rule_1` .. `rule_8` |
| `data` | `dataframe` | `CycleDataProcessor.group_by_cycle_uuid` |
| `column` | `group_column` | `CycleDataProcessor.split_dataframes_by_group` |
| `id_column` | `uuid_column` | `EnergyConsumptionEvents.normalize`, `EnergyEfficiencyEvents.normalize` |
| `expected_freq` | `expected_interval` | `SignalQualityEvents.detect_missing_data`, `data_completeness` |
| `expected_frequency` | `expected_frequency_hz` | `SamplingResolutionEvents.detect_aliasing` (a frequency in Hz, not an interval) |
| `uuid_a` / `uuid_b` | `signal_uuid_a` / `signal_uuid_b` | `SignalCorrelationEvents.rolling_correlation`, `correlation_breakdown`, `lag_correlation` |
| `period1` / `period2` | `period_a` / `period_b` | `PeriodSummaryEvents.compare_periods` |
| `parquet_only` | `only_parquet` | `AzureBlobParquetLoader.list_structure`, `DatabricksUnityParquetLoader.list_structure` |

### Renamed output columns (old column kept as a twin)

| Old | New | Where |
|---|---|---|
| `window_start` only | `start`, `end`, `duration_seconds` lead; `window_start` kept | `SamplingResolutionEvents` (all four methods), `EnvironmentalCompensationEvents.detect_correlation` / `sensitivity` / `detect_env_exceedance` (also keeps `window_end`), `MeasurementUncertaintyEvents.type_a_uncertainty` / `combined_uncertainty`, `CalibrationValidationEvents.detect_recalibration_need` / `calibration_stability`, `TimeToThresholdEvents.time_to_threshold_windows`. The event-log taxonomy now treats them as summaries (they were static), so their events carry the window's time instead of the default timestamp. |
| `time` | `start`, `end` (= `time`), `duration_seconds` (of `duration`) | `RateOfChangeEvents.detect_step_changes` |
| `mean_interval_ms`, `std_interval_ms`, `max_gap_ms` | `mean_interval_seconds`, `std_interval_seconds`, `max_gap_seconds` | `SamplingResolutionEvents.sampling_jitter` |
| `n_samples` | `sample_count` | `CapabilityTrendingEvents.capability_over_time`, `MeasurementUncertaintyEvents.type_a_uncertainty`, `ValueDistributionEvents.detect_bimodal` / `normality_windows` / `percentile_tracking` |
| `pct_of_total` | `share_of_total_pct` | `DowntimeEvents.downtime_by_reason` / `top_downtime_reasons`, `QualityEvents.nok_by_reason`, `ReworkEvents.rework_by_reason`, `ScrapEvents.scrap_by_reason`, `MaterialBalanceEvents.contribution_breakdown` |
| `pct_of_best` | `share_of_best_pct` | `OperatorPerformanceEvents.operator_comparison`, `ProcessStabilityIndexEvents.stability_comparison` |
| `pct_of_window` | `share_of_window_pct` | `MicroStopEvents.micro_stop_frequency` |
| `pct_of_available_time` | `share_of_available_time_pct` | `SetupTimeEvents.setup_statistics` |
| `pct_above` | `time_above_pct` | `ThresholdMonitoringEvents.time_above_threshold` |
| `pct_time_at_high` / `pct_time_at_low` / `pct_time_saturated` | `time_at_high_pct` / `time_at_low_pct` / `time_saturated_pct` | `ControlLoopHealthEvents.output_saturation` |
| `pct_saturated` | `saturated_pct` | `ControlLoopHealthEvents.loop_health_summary` |
| `pct_time_disturbed` | `time_disturbed_pct` | `DisturbanceRecoveryEvents.disturbance_frequency` |
| `pct_GRR` / `pct_PV` / `pct_GRR_tolerance` | `grr_pct` / `pv_pct` / `grr_tolerance_pct` | `GaugeRepeatabilityEvents.gauge_rr_summary` |
| `pct_contribution` | `contribution_pct` | `MeasurementUncertaintyEvents.uncertainty_budget` |
| `availability`, `performance`, `oee_score` | `availability_ratio`, `performance_ratio`, `oee_ratio` | `LineThroughputEvents.throughput_oee` |
| `period1_value`, `period2_value` | `period_a_value`, `period_b_value` | `PeriodSummaryEvents.compare_periods` |

## October 2026, round 10: fast at 10 and 100 million rows

Results are unchanged except where listed below; event ids from
`to_event_log` are byte-identical.

- `ChangePointEvents.detect_change_points` and `segment_statistics` default to
  `method="auto"`: exact PELT up to 20,000 samples (results unchanged), binary
  segmentation above that (approximate; results on long series can differ and
  the slowness warning no longer fires). Pass `method="pelt"` for the exact
  search on long series.
- Rolling window moments (degradation and health scores) and matrix-profile
  distances (motif and discord discovery) differ from before by at most 1e-9
  relative; motif and discord positions are unchanged.
- Setpoint KPIs no longer fail on an actual signal with duplicate index
  labels.
- A missing uuid no longer matches the text `"nan"` or `"None"` when
  selecting signals.

## October 2026, round 9: KPIs match their textbook definitions

Several KPIs gave wrong numbers. They now follow the published definitions
(ISO 22400-2, Montgomery *Introduction to Statistical Quality Control*,
Ogata *Modern Control Engineering*, ISO 50006, Nakajima), and each one is
pinned by a hand-computed reference test in `tests/reference/`. **Results
change**; no code change is needed unless you depended on the old values.
Where an option keeps the old behavior, it is named below.

### Time and parts are booked to the period they belong to

Durations used to be booked to the shift/hour/day of the reading that started
them, and counter increments to the reading that reported them. Both are now
split at period boundaries (wall-clock, DST-safe), via the new
`ts_shape.utils.time_accounting` helpers.

Affected: `DowntimeEvents.downtime_by_shift` / `availability_trend`,
`ShiftHandoverEvents`, `RuntimeAccountingEvents.runtime_by_window` /
`operating_hours_meter`, `MicroStopEvents.micro_stop_impact`, `OEEEvents`
(availability, performance, daily OEE) and `QualityTrackingEvents.nok_by_shift`.

| Probe | Old | New |
|---|---|---|
| Stop 07:00–15:00, shift_1 (06–14) availability | 0 % (480 min down) | 12.5 % (60 up / 420 down) |
| Same stop, `availability_trend(window="1h")`, 07:00 hour | 480 min down | 60 min down (one row per hour) |
| Night shift 22–06 running at 100 %, OEE day 1 / day 2 | 12.5 % / 0 % | 100 % / 100 % |
| 30 parts counted 13:50–14:10, shift_1 / shift_2 | 0 / 30 | 15 / 15 |

- A counter increment is spread over `(previous reading, reading]` in
  proportion to time; a reset counts in the period of the reading that shows
  it. `handle_resets=False` keeps the per-period `max(0, last - first)`.
- Part counts in OEE (`actual_parts`, `total_parts`, `reject_parts`,
  `good_parts`) and shift-handover quantities are floats now.
- Shift and trend outputs list every period that holds observed time, not only
  periods that contain a reading.
- A day without state rows has NaN availability, performance and OEE (was:
  the counter span as run time, availability 0).
- Performance or availability above 100 % is no longer clipped; the value is
  kept and a `DataQualityWarning` names the period.
- `micro_stop_frequency` still counts a stop in the window it starts in.

### Production

- `PerformanceLossEvents` counts parts (counter increments), not counter rows;
  cycle time = run time / parts. Counter +2 per minute for 60 min at a 30 s
  ideal cycle: performance 50 % → 100 %, loss 1770 s → 0 s. A new
  `run_seconds` column carries the basis.
- `PeriodSummaryEvents.from_daily_data` recomputes KPIs from summed bases
  (quality = Σgood/Σtotal, availability = Σrun/Σplanned, OEE = A×P×Q,
  MTTR = Σrepair/Σfailures, EnPI = Σenergy/Σunits, ...). Ratios without a
  basis column are weighted means (plain mean with a `DataQualityWarning`);
  numeric columns with no rule are dropped with a warning instead of summed.
  A new `aggregations=` argument overrides the rule per column. Quality of
  900/1000 and 0/10: 45.0 % → 89.1 %; MTTR 1 h and 3 h: 4.0 → 2.0.
- `MicroStopEvents.micro_stop_impact` gains `performance_loss_pct` (minor
  stops are a performance loss); `availability_loss_pct` stays with the same
  value.

### Setpoint response

- `time_to_settle` is the time after which the response stays in the band
  until the end of the window (last entry, not first). The window now also ends
  at the next setpoint change. New `settled_within_window` column;
  `t_settle_seconds` is NaN when the response does not settle. `settled` means
  settled within the window with a final in-band stretch of at least `hold`.
  Underdamped step ζ=0.2, ωn=0.1, 2 % band: 18 s → 197 s.
- `overshoot_pct` / `undershoot_pct` are percent (0–100) like every other
  `*_pct` column: 0.5266 → 52.66. The fraction is in the new
  `overshoot_ratio` / `undershoot_ratio` columns.

### Statistical process control and capability

`StatisticalProcessControlEvents(..., sigma_method="moving_range")` (default)
estimates σ as MR̄/1.128 (individuals chart); `sigma_method="std"` brings back
the sample standard deviation.

| Probe | Old | New |
|---|---|---|
| Reference 1..20: control limits | −7.25 / 28.25 | 7.84 / 13.16 |

- The rule 1–8 zones and `detect_cusum_shifts` use the same σ̂.
- `dynamic_control_limits(method="moving_range")` tests each point against the
  `window_size` points before it (center = their mean, σ̂ = their MR̄/1.128);
  the first `window_size` points have NaN limits.
- `dynamic_control_limits(method="ewma")` is an EWMA chart:
  z_i = λx_i + (1−λ)z_{i−1}, limits μ0 ± L·σ̂·√(λ/(2−λ)·[1−(1−λ)^{2i}]).
  `ewma` is z_i, `mean` is μ0; new `ewma_lambda=` (default 2/(window_size+1))
  and `n_sigma=` (L) set the new `lcl` / `ucl` columns. Both methods add a
  `value` column.
- Cp/Cpk use the within σ (MR̄/1.128) and Pp/Ppk the overall σ (AIAG), in
  `CapabilityTrendingEvents` (`sigma_method=` on the constructor) and
  `ToleranceDeviationEvents.capability_indices` (`sigma_method=` on the
  method). `sigma_method="std"` makes Cp/Cpk equal Pp/Ppk again. New
  `std_within` column. Per-window Pp/Ppk use the window's own σ (they used the
  whole signal's). Example x = 10.0, 10.4, 10.2, 10.6, 9.8, specs 9–11:
  Cp 1.054 → 0.836, Pp unchanged 1.054.

### Energy and supply chain

- The EnPI rolling baseline, `EnergyEfficiencyEvents.sec_trend` and
  `efficiency_trend` `rolling_avg_efficiency` are Σnumerator/Σdenominator over
  the window (ISO 50006), not the mean of ratios: EnPI 1 and 3 over 130 kWh /
  110 units: 2.0 → 1.182.
- `InventoryMonitoringEvents.stockout_prediction` uses the refill-aware
  consumption rate (`handle_refills=True`, new); a refill no longer cancels
  consumption. `handle_refills=False` keeps first-minus-last.

### Documented, unchanged

- `SignalCorrelationEvents.lag_correlation`: positive `lag_periods` means `uuid_b`
  leads `uuid_a` (b(t) = a(t−5) gives −5). The docstring now says so.

## October 2026, round 8: legacy code removed

### Requirements

ts-shape now needs **pandas 2.2.2 or later** and **numpy 2.0 or later** (was
pandas 2.1, numpy 1.26). A CI job tests the oldest supported versions, and
results are the same on pandas 2.2 and pandas 3: daily windows are calendar
days (23 h / 25 h on a DST change) on both.

`setup.py`, `MANIFEST.in` and the `requirements*.txt` files are gone;
`pyproject.toml` is the only place dependencies are declared.

### Deprecation warnings you can see

Deprecation warnings point at your own line now. Before, warnings from
detector methods pointed inside ts-shape, and Python's default filters hid
them, so a deprecated name could be used without a hint. Run your tests with
`-W error::DeprecationWarning` once to find them.

Newly deprecated (old name still works and warns):

Loader time ranges are `start` / `end` everywhere; positional calls are
unchanged.

| Old | New |
|---|---|
| recipe spec key `duration_s` | `duration_seconds` |
| `ShiftHandoverEvents.from_shift_data(downtime_df=...)` with only `downtime_minutes` | a `downtime_seconds` column |
| `CycleDataProcessor.cycle_uuid_col` / `.systime_col` (attributes) | `.cycle_uuid_column` / `.time_column` |
| `RuntimeAccountingEvents.run_uuid` (attribute) | `.run_state_uuid` |
| `MultiSensorValidationEvents.sensor_uuids` (attribute) | `.signal_uuids` |
| `start_timestamp=` / `end_timestamp=` (Azure Blob, Databricks Unity and Spark parquet loaders, `TimescaleDBDataAccess`, `S3ProxyDataAccess`) | `start=` / `end=` |
| `start_time=` / `end_time=` (`ParquetLoader.load_by_time_range`, `load_files_by_time_range_and_uuids`) | `start=` / `end=` |
| `TimescaleDBDataAccess.start_timestamp` / `.end_timestamp`, same on `S3ProxyDataAccess` (attributes) | `.start` / `.end` |

### Behavior changes

- **Time-named numeric columns stay numbers.** A class without its time
  column used to convert every column whose name contained "time" or "date"
  (`cycle_time_mean`, `update_rate`) to datetimes. Only datetime or text
  columns are used now, with a `DataQualityWarning`.
- **Mixed time units.** `SegmentProcessor.apply_ranges` and
  `ContinuousProcessAlignmentEvents.segment_by_cut` compare times in one
  unit; ranges in seconds against data in nanoseconds matched no rows.
- **Hour folders are UTC in every loader** (Azure, Databricks Unity,
  Parquet): a tz-aware bound is converted to UTC, a naive bound is UTC.
  Before, some loaders used local wall-clock hours.
- `CycleDataProcessor.split_dataframes_by_group` raises
  `ColumnNotFoundError` for a frame without the group column (it was logged
  and skipped). `CriticalParameterRankingEvents.rank` raises
  `ColumnNotFoundError` for a missing outcome column.
- Problems that were only logged are `DataQualityWarning`s: degenerate SPC
  limits, a singular MSPC covariance, ignored or overlapping segment ranges,
  invalid or overlapping cycles, empty feature tables. Text in numeric
  columns is reported in pattern recognition, feature export, segment
  statistics, cycles and continuous-process alignment.
- `StatisticalProcessControlEvents.detect_cusum_shifts` counts a repeated
  sample (same time, same value) once, with a `DataQualityWarning`. A
  duplicated row was added to the running sum twice and could report a shift
  on its own.
- `ProfileComparison.cluster(n_clusters=...)` always returns the requested
  number of clusters (SciPy's `maxclust` cut could return fewer).
- New exports: `ts_shape.events.production` lists `FlowMetricsEvents`,
  `LineBalancingEvents`, `RuntimeAccountingEvents`; `ts_shape.features` lists
  `PatternRecognition`, `DescriptiveFeatures`, `TimeWindowedFeatureTable`.
- The API reference no longer has pages for private modules, and package
  pages have clean URLs (`.../events/production/` instead of
  `.../events/production/__init__/`).

## October 2026, round 7: every function robust and consistent

Every public function, not only the detectors, now handles empty frames,
timezones, nullable and pyarrow dtypes, text time columns and missing
columns the same way. Contract tests check this for every public callable.

### Renamed parameters

The old names work and warn.

| Old | New | Where |
|---|---|---|
| `column_name` (a value column) | `value_column` | filters, `IntegerCalc`, `LambdaProcessor.apply_function`, `UnitConverter.convert_column`, `Numeric/Boolean/StringStatistics` |
| `column_name` (the time column) | `time_column` | `DateTimeFilter`, `TimestampStatistics`, `ValueMapper` |
| `freq` | `window` | `PowerIntegrator`, `DataHarmonizer.resample_to_uniform` / `merge_multi_signals`, `TimeGroupedStatistics` |
| `df`, `uuid_col`, `value_cols`, `group_col` | `dataframe`, `uuid_column`, `value_columns`, `group_column` | `FeatureMatrixExporter.to_feature_matrix` |
| `cycle_uuid_col`, `systime_col` | `cycle_uuid_column`, `time_column` | `CycleDataProcessor` |
| `metric` | `value_column` | `CycleDataProcessor.compare_cycles`, `identify_golden_cycles` |
| `metadata_uuid_col`, `tolerance_uuid_col`, `mapping_uuid_col` | `*_uuid_column` | `ContextEnricher` |

`DateTimeFilter` takes the bound right after the frame:
`filter_after_date(df, "2024-01-02", time_column="systime")`. The old order
still works and warns.

### Results that were wrong

- `CrossSignalAnalytics.lead_lag` reported the leader backwards. A positive
  `optimal_lag` now means `signal_a` leads. All its methods sort by time and
  use only rows where both signals have a value.
- `PatternRecognition` sorts by time, keeps the timezone, and needs `uuid=`
  when the frame holds several signals. `normalize=False` now really uses
  the raw distance.
- `BooleanFilter` edges are found per uuid in time order; a missing value
  holds the previous state. `StringFilter.detect_changes_in_string` always
  reports the first row of each uuid, also for `string` dtype.
- `DataHarmonizer.fill_gaps(max_gap=)` measures the gap between the valid
  samples around it; signals without values are kept as NaN columns.
- `LambdaProcessor.apply_function` and
  `TimezoneShift.revert_to_original_timezone` no longer change your frame.
- `FeatureMatrixExporter` uses the sample standard deviation (ddof=1) and
  aggregates only `value_*` columns by default.
- `ReliabilityEvents`, `InventoryMonitoringEvents.stockout_prediction`:
  NaN instead of `inf`. `LeadTimeAnalysisEvents.lead_times` pairs each order
  with the first later delivery, never a negative lead time.
- A zero-length step (duplicate timestamp) is no data gap and starts no
  flatline. Duplicated rows no longer duplicate outlier and tolerance events.
- `object_intervals(min_duration=)` no longer loses an id that comes back
  after a short blip.

### Event ids

`to_event_log` builds `ocel:eid` from the detector, `source_uuid`, the
timestamps, the activity and the row's values, no longer from the row
position. The same detector on two machines no longer collides, and `concat`
no longer drops events. Event ids from earlier releases change.

Also in `to_event_log`: a string in `objects=` that is not a column is used
as the object id for every row; a NaN in an activity template gives
`unknown` with a warning; interval rows with `end < start` are dropped with a
warning; the static shape uses `timestamp=` (default 1970-01-01 UTC), not
the current time.

### No data, unknown codes

- Percentages of no data (`true_percentage`, `uppercase_percentage`, ...)
  are NaN, not 0. A mode of no data is `None`.
- `CrossSignalAnalytics` with too little data gives NaN, not 0.0 / 1.0.
- `ValueMapper` keeps codes the mapping table does not know, with a
  `DataQualityWarning` (they were NaN). Duplicate keys raise `ValueError`.
- `ContextEnricher` calls build on each other; `get_enriched_dataframe()`
  returns the cumulative result. Duplicate keys and colliding column names
  raise `ValueError`.
- `ToleranceDeviationEvents.capability_indices` and `MultivariateSPCEvents`
  with too little data give NaN / an empty result with a warning instead of
  raising.

### Errors

A missing column raises `ColumnNotFoundError` (a `ValueError`) in every
public function; it was often a `KeyError`.

### Loaders, signals, rules

- `DatabricksSparkParquetLoader.build_paths` includes the hour of a
  mid-hour start and builds folder hours in UTC. `ParquetLoader` matches
  uuids to whole file names. `EnergyAPILoader` returns UTC times.
- `signals.align` handles mixed time units and names the time column after
  `time_column`; `per_window` handles booleans with missing values.
  `SignalSpec` validates its fields.
- Lambda rule expressions reject `and`, `or`, `not`, `in` and chained
  comparisons at compile time (use `&`, `|`, `~`, `.isin`). Interval rules
  end at the next sample after the last true one, like `signals.intervals`.
- `Pipeline` passes arrays and frames as step arguments and never changes
  your input frame.

### New

- `max_gap` on `detect_flatline`, `threshold_with_hysteresis` and
  `on_off_intervals`.
- `DataHarmonizer.detect_gaps` adds `start`, `end`, `duration_seconds`.
- `TimeGroupedStatistics.calculate_custom_func(group_by_uuid=True)`.

## October 2026, round 6: consistent outputs

### Summary methods return a DataFrame

Like `best_and_worst_shifts` (round 5), these methods take `as_frame: bool = True`. The default returns a DataFrame whose columns are the old keys, plus `start` / `end` / `duration_seconds` for the data the summary covers. `as_frame=False` returns the old dict or list and emits a `DeprecationWarning`; that form will be removed in the next major release. To keep old code working unchanged, pass `as_frame=False`. Alternatively, read a one-row result with `.iloc[0]`, which supports the same `["key"]` access.

| Method | Default DataFrame |
|---|---|
| `SteadyStateDetectionEvents.steady_state_statistics` | one row; the dict keys plus `uuid`, `source_uuid` |
| `MachineStateEvents.state_quality_metrics` | one row; the dict keys plus `uuid`, `source_uuid` |
| `TargetEvents.target_hit_rate` | one row; `start` / `end` span the analysed days |
| `ToleranceDeviationEvents.capability_indices` | one row; `Cpm` is always a column (NaN without `target_value`), plus `sample_count` |
| `BottleneckDetectionEvents.throughput_constraint_summary` | one row per station: `bottleneck_count`, `bottleneck_pct`, `avg_utilization_pct`, `most_frequent_bottleneck` |
| `FlowConstraintEvents.flow_constraint_analytics` | one row per `constraint_type` (`blocked`, `starved`): `event_count`, `total_duration_seconds`, `avg_duration_seconds`, `minor_count`, `moderate_count`, `severe_count`, `time_alignment_quality`. The event lists come from `blocked_events` / `starved_events`. |
| `ShiftHandoverEvents.highlight_issues` | one row per issue, with the report row's shift window and `date` |

### KPI percentages carry their basis

New columns; none were removed:

| Method | New columns |
|---|---|
| `OEEEvents.oee` | `run_seconds`, `planned_seconds`, `excluded_seconds`, `actual_parts`, `ideal_parts`, `total_parts`, `reject_parts`, `good_parts` (NaN without quality counters), `resets` |
| `OEEEvents.availability` | `excluded_seconds` (time `max_gap` removed as no data) |
| `LineThroughputEvents.throughput_oee` | `run_seconds`, `planned_seconds`, `resets` (with `handle_resets=True`) |
| `DowntimeEvents.downtime_by_shift` / `availability_trend` | `excluded_seconds` |
| `PerformanceLossEvents` (all three methods) | `run_seconds` |

### Durations in seconds

Each `*_minutes` or `*_s` duration column now has a `*_seconds` twin right after it: `total_seconds`, `downtime_seconds`, `uptime_seconds`, `avg_seconds` (downtime), `avg_cycle_time_seconds`, `target_cycle_time_seconds`, `loss_seconds` (performance loss), `avg_seconds`, `min_seconds`, `max_seconds`, `median_seconds`, `std_seconds`, `avg_setup_seconds`, `total_setup_seconds` (setup time), `downtime_seconds` (shift handover, also in `OUTPUT_COLUMNS`). The old columns stay as a convenience. Their removal would be a breaking change and is left for a future major release; new code should read the `*_seconds` columns. In recipe-phase specs `duration_seconds` is the new spelling of the `duration_s` criterion; `duration_s` still works and, since round 8, warns.

### Run state as string or boolean

`DowntimeEvents` and `ShiftHandoverEvents` read either a string state signal (`running_value="Running"` in `value_string`) or a boolean run signal (`value_bool`, True = running). The boolean signal is used when the state rows carry no `value_string`, or when `running_value` / `stopped_value` is a bool. Both forms give the same downtime. Before this change, a boolean run signal was compared with `"Running"` and therefore reported 100 % downtime.

### Fixes

`FlowConstraintEvents.flow_constraint_analytics` applies its `warning_threshold` / `critical_threshold` to the severity counts and to the event lists. Before, it accepted them but classified with the 5 s / 30 s defaults.

### Same numbers for any data source, totals that add up

- **Dtypes no longer change results.** Value columns with pandas nullable,
  pyarrow or text-number dtypes are converted once when a detector is built.
  Text that is not a number is reported with a `DataQualityWarning` at
  construction, not when a method runs. Counter totals on nullable or pyarrow
  data were too high (the first reading counted as production): they now
  match NumPy data.
- **Group totals add up to the counter total.** Per shift, day, operator,
  part, reason, batch and schedule period, the part made between the last
  reading of one group and the first reading of the next now counts in the
  later group (`utils.counters.counter_by_group`). Quantities can grow by one
  increment per group boundary; `resets` can grow by one when a reset falls on
  a group boundary. Per-operator/part/reason totals no longer double count
  when those groups alternate. `shift_production(date=...)` includes the
  increment into the day's first reading. `batch_yield` books production
  between two batches to the earlier batch. `handle_resets=False` is
  unchanged.
- **`FlowConstraintEvents`** intervals end at the next change
  (`interval_end="next_change"`); pass `interval_end="last_sample"` for the
  previous, shorter durations.
- **`MultiSensorValidationEvents.detect_outlier_sensor`** needs three sensors
  to name an outlier; with two it skips the window and warns.
- **`ToleranceDeviationEvents.capability_indices`** on empty data returns an
  empty frame with the full schema; `CapabilityTrendingEvents` uses the spec
  limits in force in each window.
- **Energy methods with `window="W"` / `"ME"`** report the real week or month
  (`end` was `start` + 1 day).
- **Shift labels:** a row with a missing timestamp gets the shift `"unknown"`.
- **Degenerate inputs:** duplicate timestamps give `NaN` rates (with a
  warning) instead of `inf`; `detect_oscillation(window<=0)` raises.

### Cycles, segments and objects

These were verified against hand-built scenarios; the fixes change results:

- **Cycles** (`CycleExtractor`):
  - back-to-back cycles sharing a timestamp come out correctly;
  - a second start while a cycle is open aborts it (`is_complete=False`) instead of
    producing overlapping cycles;
  - a held start pulse is one start;
  - value/state-change cycles read only `start_uuid`;
  - a cycle already running at the first sample has `is_complete=False`;
  - `cycle_uuid` is deterministic (same data, same id).
- **Cycle data** (`CycleDataProcessor`): every method uses `start <= t < end`. A
  sample exactly at a cycle end belongs to the next cycle, not to both; open cycles
  get no samples. `identify_golden_cycles(method="target_value")` needs `target=`.
- **Segments** (`SegmentExtractor` / `SegmentProcessor`):
  - the last segment runs to the end of the data (it had zero length);
  - `apply_ranges` cuts half-open, so boundary samples are no longer duplicated;
  - the first sample of an empty-id gap is no longer given to the previous order.
- **`segment_by_cut`:** material is assigned to the piece the next cut finishes. Before,
  it was one piece off and piece 1 got two pieces' worth. Material after the last
  cut has `piece_id` `<NA>`.
- **Objects** (`object_intervals` / `detect_objects`):
  - an empty or NaN reading ends a presence;
  - a change-encoded id stays present to the end of the data;
  - at a hand-over instant only the incoming object gets the event;
  - a child is `part_of` only the parent it overlaps longest;
  - `min_duration` measures the presence, not first-to-last sample;
  - use `to_event_log_ocel(log, pm4py=True)` for pm4py export.
- **`production_dates`:** correct after a DST change (it was one day off).

### Severity levels and small fixes

- **`severity_level`.** Every result with a categorical `severity` also has
  `severity_level` (`info` / `warn` / `critical`), right after `severity`. The
  `severity` labels are unchanged. Code that selects columns by position sees one
  more column.
- **`TimeWindowedFeatureTable`:** `freq` is renamed `window`. `freq` still works and
  warns.
- **`CriticalParameterRankingEvents()`:** the long-format frame is optional and
  `rank()` never used it, so a time column is no longer needed.
- **`ContinuousProcessAlignmentEvents.lag_profile`:** a window without speed
  readings has NaN speed and lag. Before, it reported the lag at `min_speed`:
  thousands of seconds.

## October 2026, round 5: correct by default

The correct behavior is now the default, and results are returned at full
precision. No name was removed, so your code keeps running, but **results
change** wherever your data has counter resets, change-encoded signals, uneven
sampling or several signals in one frame. Each old behavior is one argument
away.

### Defaults that changed

| Parameter | Old default | New default | What it fixes | Where |
|---|---|---|---|---|
| `handle_resets` | `False` | `True` | A counter that resets (shift start, rollover) no longer makes a window report too little, often 0. Results carry a `resets` column. | production (OEE, parts, quality, scrap, rework, operator, target, period summary, shift report and handover, batch yield, line throughput, process alignment), energy (EnPI, efficiency, consumption, carbon), `utils.counters`, `TimeGroupedStatistics` (`'diff'`) |
| `interval_end` | `"last_sample"` | `"next_change"` | An interval lasts until the signal changes. A change-encoded (deadband) signal's interval is no longer cut to zero length. | machine state, duty cycle, micro stops, long downtime, steady state, threshold monitoring, degradation, vibration, recipe phases, inventory, signal quality, flatline, correlation breakdown, `SegmentExtractor`, `utils.intervals` |
| `time_weighted` | `False` | `True` | Shares and averages are weighted by time, not by sample count, so irregular sampling no longer biases them. | bottleneck detection, duty cycle by window, operating range, material balance, idle energy, efficiency idle waste |
| `group_by_uuid` | `False` | `True` | Statistics are computed per signal instead of pooling unrelated signals. | `OutlierDetectionEvents`, `TimeGroupedStatistics`, `BooleanFilter` edges, `StringFilter.detect_changes_in_string` |
| `slope_per` | `"sample"` | `"second"` | Drift slopes are per second, so irregular sampling no longer distorts them; `min_slope` is per second too. | `AnomalyClassificationEvents.detect_drift`, `classify_anomalies` |
| `missing_as_zero` | `True` | `False` | A missing flow signal makes the balance NaN instead of a false imbalance. | `MaterialBalanceEvents` |
| `handle_refills` | `False` | `True` | A refill no longer turns into negative consumption. | `InventoryMonitoringEvents.consumption_rate` |
| `include_severity` | `False` | `True` | Violations carry `triggered_rule` and `severity`. | `StatisticalProcessControlEvents.detect_violations` |
| `regex` | `True` | `False` | Substring filters match literally (`"A.1"` no longer matches `"AB1"`). | `StringFilter.filter_string_contains`, `StringStatistics.contains_substring_count` |
| `trim_to_range` | `False` | `True` | Loaders return only rows inside the requested time range. | Azure Blob and Databricks Unity parquet loaders |
| `as_frame` | `False` | `True` | `best_and_worst_shifts` returns one DataFrame with a `rank_type` column; `as_frame=False` (a dict) is deprecated. | `ShiftReportEvents` |
| `layout` (new) | — | `"long"` | `batch_transition_matrix` returns `from_state`, `to_state`, `transition_count`, and an empty result keeps that schema. `layout="wide"` gives the old matrix. | `BatchTrackingEvents` |

To compare old and new numbers, pass the old default explicitly, for example `oee(..., handle_resets=False)`.

### Other result changes

- **Full precision.** Detectors no longer round results (they used 0 to 8 decimals). Round when you display: `df.round(2)`.
- **`machine_running_pct` is a percentage (0–100).** In `IdleEnergyDetectionEvents.idle_energy_by_window` and `EnergyEfficiencyEvents.idle_energy_waste` it held a 0–1 ratio. The ratio is `machine_running_ratio`.
- **Flatlines:** with `"next_change"`, a single sample counts as a flatline only when the gap after it is longer than the signal's usual sampling interval.
- **`ValueDistributionEvents.detect_bimodal`** returns a unimodal row for a constant signal (it was empty or had a `None` mode).
- **`CriticalParameterRankingEvents.rank`** raises `ValueError` for an unknown candidate column (it was skipped).
- **`TimezoneShift.list_available_timezones`** returns a sorted list (it was a set).
- **`make_timeseries`** no longer rounds `value_double`.

### Nothing is skipped silently

- **Loaders:** an unreadable blob or file used to disappear from the result. Multi-object Azure Blob and Databricks Unity methods take `errors="warn"` (default: skip it and name it in one `DataQualityWarning`) or `errors="raise"` (`LoaderError`). Series metadata that exists but cannot be read raises `LoaderError`.
- **Non-numeric values** in numeric inputs are still treated as missing, but a `DataQualityWarning` now says how many there were.
- **A missing value column** (`value_column`, `value_column_<role>`, ...) raises `ColumnNotFoundError` naming it and listing the columns, instead of a bare `KeyError`.
- Malformed recipe-phase specs, untestable normality windows, failed fits and event-log object bindings to a missing column now warn instead of being dropped.
- `lambda_rules.load_yaml` without PyYAML raises `ImportError` pointing to `ts-shape[yaml]`.

### Newly deprecated

| Old | New |
|---|---|
| `OrderTraceabilityEvents` | `ValueTraceabilityEvents` |
| `ToleranceDeviationEvents(tolerance_uuid=...)` alone | `upper_tolerance_uuid` and/or `lower_tolerance_uuid` (either one alone is a one-sided spec) |
| `RoutingTraceabilityEvents(station_map=...)` | `state_map` |
| `eventlog.adapters.adapt(legacy=...)` | `frame=` |
| integer `window` in `SignalCorrelationEvents` | `window_size` |
| `best_and_worst_shifts(as_frame=False)` | the default DataFrame |

### New

- `OutlierDetectionEvents(signal_uuids=[...])` restricts the analysis to some signals.
- `pip install ts-shape[parquet]` installs pyarrow (also pulled in by `[s3]` and `[azure]`); `[all]` now installs every optional feature.

## Renamed parameters (old keywords deprecated)

Parameter names follow [Naming & Output Conventions](naming.md). The old keyword still works, but emits a `DeprecationWarning`.

| Old | New | Where |
|---|---|---|
| `column_name` (the time column) | `time_column` | every detector constructor (`Base`) |
| `freq`, `period`, `resample` | `window` | data-gap coverage, value distribution, demand patterns, period summary, correlation and root-cause methods |
| integer `window` (a row count) | `window_size` | `SignalCorrelationEvents.rolling_correlation` / `correlation_breakdown`, `StatisticalProcessControlEvents.dynamic_control_limits` |
| `z_threshold`, `threshold_sigma` | `n_sigma` | anomaly classification, anomaly correlation, disturbance recovery |
| `low`/`high`, `lower`/`upper`, `low_limit`/`high_limit`, `lower_bound`/`upper_bound` | `lower_limit` / `upper_limit` | threshold hysteresis, operating range, control-loop saturation, time-to-threshold |
| `minor_threshold` / `moderate_threshold` | `warning_threshold` / `critical_threshold` | `FlowConstraintEvents.flow_constraint_analytics` |
| `include_first`, `count_initial_active` | `include_initial` | reliability (Weibull), alarm frequency |
| `significant_only`, `on_crossing_only` | `only_significant`, `only_crossings` | root-cause ranking, asset-health events |
| `sensor_uuids` | `signal_uuids` | `MultiSensorValidationEvents` |
| `run_uuid` | `run_state_uuid` | `RuntimeAccountingEvents` |
| `counter_column`, `energy_column`, `state_column` | `value_column_counter`, `value_column_energy`, `value_column_state` | energy pack |

Optional arguments are keyword-only. Passing them positionally still works, but warns, for example `detect_outliers_zscore(3.0)` instead of `detect_outliers_zscore(threshold=3.0)`.

The default `event_uuid` values now all follow `"<pack>:<name>"`:
- `outlier_event` → `quality:outlier`
- `startup_event` → `eng:startup`
- `setpoint_change_event` → `eng:setpoint_change`
- `FlowMetricsEvents`: `prod:flow` → `prod:flow_metrics` (it used to collide with `FlowConstraintEvents`)

If you filter results by these values, update the filters.

## Renamed classes and methods (old names deprecated)

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

## Behavior fixes that change results

These change output because the old output was wrong or contradicted the documentation. Each one has a regression test.

- **SPC rules now follow Nelson (1984) / Western Electric** (`StatisticalProcessControlEvents`):
  - rule 8 flags 8 points in a row with **none** within 1σ (it used to flag 8 points *within* 1σ);
  - rule 3 is 6 points steadily rising or falling (was 7), and rule 4 is 14 points alternating up and down (was 15);
  - rules 5 and 6 count points beyond the next zone too, per side;
  - rule 2 measures sides against the chart center line;
  - the rules no longer add columns to your frame, and the vectorized and per-rule paths flag the same points.
- **`ToleranceDeviationEvents.detect_deviations`** keeps the measured value and reports single-sample deviations. A missing `actual_uuid` raises instead of returning nothing. `capability_indices` adds `Cpm` when `target_value` is given.
- **`LineThroughputEvents.takt_adherence`:**
  - a cycle exactly equal to takt is no longer a violation;
  - a decrease in a numeric counter (a reset) is no longer a cycle.
- **`MachineStateEvents.state_quality_metrics`:**
  - follows the signal's spec like `detect_run_idle`;
  - `run_idle_ratio` is NaN, not 0.0, when there is no idle time.
- **`ShiftHandoverEvents.highlight_issues`** without `report_df` or the four uuids raises instead of returning `[]`.
- **`PeriodSummaryEvents.from_daily_data`** honors anchored weeks (`'W-FRI'` …).
- **`GoldenBatchDeviationEvents.phase_breakdown`** compares each phase with the same phase of the golden batch.
- **`EnvironmentalCompensationEvents`:**
  - constructing without `env_uuid` raises;
  - `compensated_signal`'s `uuid` is the event uuid.
- **`SetpointChangeEvents`:** two setpoint changes at the same instant no longer multiply the KPI rows.
- **`ProcessWindowEvents`** uses the strict `>` its docstrings describe.
- **Hour-folder loaders** no longer drop the hour that contains the range end. Databricks loaders apply a single `start` or `end` bound.

Parameters that had no effect now warn when passed: `RateOfChangeEvents.detect_rapid_change(window=...)`, `MeasurementUncertaintyEvents(part_uuid=...)`, `ContinuousProcessAlignmentEvents(ref_uuid=...)`. When both `settle_pct` and `tol` are passed to a setpoint KPI, a warning says `settle_pct` wins. When both `state_map` and `station_map` are passed, a warning says `state_map` wins.

## Every detector output carries its time bounds

Each detector returns the canonical columns of its shape (see [Naming & Output Conventions](naming.md)):
- **intervals:** `start`, `end`, `duration_seconds`, `uuid`, `source_uuid`;
- **summaries:** `start`, `end`, `duration_seconds`, also when they are keyed by `date`, `(date, shift)` or `period`;
- **points:** `systime`, `uuid`, `source_uuid`.

The new columns were added next to the old ones; none was removed or renamed. Two consequences:

- **Merging two summaries on `date` or `(date, shift)`** now finds `start`, `end` and `duration_seconds` on both sides, and pandas suffixes them `_x` / `_y`. Merge on the window instead, or select your columns first:

```python
import numpy as np
import pandas as pd
from ts_shape import PartProductionEvents, QualityEvents

t = pd.date_range("2024-01-01", periods=96, freq="h")
df = pd.concat([
    pd.DataFrame({"systime": t, "uuid": "line1:ok", "value_integer": np.arange(96) * 9}),
    pd.DataFrame({"systime": t, "uuid": "line1:nok", "value_integer": np.arange(96)}),
    pd.DataFrame({"systime": t, "uuid": "line1:part", "value_string": "P-100"}),
    pd.DataFrame({"systime": t, "uuid": "line1:count", "value_integer": np.arange(96) * 10}),
], ignore_index=True)

quality = QualityEvents(df).daily_quality_summary("line1:ok", "line1:nok")
production = PartProductionEvents(df).daily_production_summary("line1:part", "line1:count")
daily = quality.merge(production, on=["start", "end", "duration_seconds"])
```

- **Night shifts are one row.** An overnight shift (e.g. 22:00 to 06:00) is attributed to the day it starts (`production_day=True` is now the default). Before, grouping by calendar date split one night across two rows, and each row mixed two different nights. Pass `production_day=False` to get the old grouping.

## Removed exporter aliases

The event-log exporters were renamed into a consistent `to_event_log_*` family.
The old aliases (`to_flat_df`, `to_ocel_tables`) have now been **removed** —
update any remaining imports:

| Removed | Use instead |
|---|---|
| `to_flat_df(...)` | `to_event_log_xes(...)` |
| `to_ocel_tables(...)` | `to_event_log_ocel(...)` |

<!-- no-run -->

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

<!-- no-run -->

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
