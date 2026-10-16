<!-- Round 11 (naming consistency) rows for docs/migration.md; merge into the round-11 section. -->

Old names keep working until the next major release: a renamed keyword or attribute warns with a `DeprecationWarning` naming the replacement; a renamed output column stays as a twin with the same values. A canonical column sits right after its legacy twin; window bounds (`start`, `end`, `duration_seconds`) lead every summary.

### Parameters and attributes

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

### Output columns

| Old | New | Where |
|---|---|---|
| `window_start` only | `start`, `end`, `duration_seconds` lead; `window_start` kept | `SamplingResolutionEvents` (all four methods), `EnvironmentalCompensationEvents.detect_correlation` / `sensitivity` / `detect_env_exceedance` (also keeps `window_end`), `MeasurementUncertaintyEvents.type_a_uncertainty` / `combined_uncertainty`, `CalibrationValidationEvents.detect_recalibration_need` / `calibration_stability`, `TimeToThresholdEvents.time_to_threshold_windows`. The event-log taxonomy now treats them as summaries (they were static), so their events carry the window's time instead of the default timestamp. |
| `time` | `start`, `end` (= `time`), `duration_seconds` (of `duration`) | `RateOfChangeEvents.detect_step_changes` |
| `mean_interval_ms`, `std_interval_ms`, `max_gap_ms` | `mean_interval_seconds`, `std_interval_seconds`, `max_gap_seconds` | `SamplingResolutionEvents.sampling_jitter` |
| `n_samples` | `sample_count` | `CapabilityTrendingEvents.capability_over_time`, `MeasurementUncertaintyEvents.type_a_uncertainty`, `ValueDistributionEvents.detect_bimodal` / `normality_windows` / `percentile_tracking` |
| `pct_<x>` | `<x>_pct` | `pct_of_total` (`DowntimeEvents.downtime_by_reason` / `top_downtime_reasons`, `QualityEvents.nok_by_reason`, `ReworkEvents.rework_by_reason`, `ScrapEvents.scrap_by_reason`, `MaterialBalanceEvents.contribution_breakdown`), `pct_of_best` (`OperatorPerformanceEvents.operator_comparison`, `ProcessStabilityIndexEvents.stability_comparison`), `pct_of_window` (`MicroStopEvents.micro_stop_frequency`), `pct_of_available_time` (`SetupTimeEvents.setup_statistics`), `pct_above` (`ThresholdMonitoringEvents.time_above_threshold`), `pct_time_at_high` / `pct_time_at_low` / `pct_time_saturated` (`ControlLoopHealthEvents.output_saturation`), `pct_saturated` (`ControlLoopHealthEvents.loop_health_summary`), `pct_time_disturbed` (`DisturbanceRecoveryEvents.disturbance_frequency`), `pct_GRR` / `pct_PV` / `pct_GRR_tolerance` (`GaugeRepeatabilityEvents.gauge_rr_summary`), `pct_contribution` (`MeasurementUncertaintyEvents.uncertainty_budget`) |
| `availability`, `performance`, `oee_score` | `availability_ratio`, `performance_ratio`, `oee_ratio` | `LineThroughputEvents.throughput_oee` |
| `period1_value`, `period2_value` | `period_a_value`, `period_b_value` | `PeriodSummaryEvents.compare_periods` |
