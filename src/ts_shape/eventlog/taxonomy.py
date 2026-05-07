"""Per-method label rules — the source of truth for ``ocel:activity`` names.

Every public DataFrame-returning method on every detector class under
``ts_shape.events.*`` must have an entry here. The coverage test
``tests/eventlog/test_adapter_coverage.py`` enforces this.

A :class:`LabelRule` describes:

* ``template`` — the ``ocel:activity`` value. May contain ``{field}``
  placeholders that get substituted from legacy DataFrame columns.
* ``pack`` — one of ``quality``, ``production``, ``engineering``,
  ``maintenance``, ``supplychain``, ``energy``, ``correlation``.
* ``shape`` — how to interpret each legacy row:
  ``"point"``      → single-timestamp event (`systime` or first datetime),
  ``"interval"``   → has ``start``/``end`` columns,
  ``"summary"``    → a windowed/aggregate row (uses end column or any datetime),
  ``"static"``     → no time semantics; uses ``ts_shape:fallback_now``.
* ``produces_objects`` — object types the adapter *auto-extracts* from
  the legacy DataFrame's standard columns (e.g. ``source_uuid -> asset``).
  Callers can always attach additional contextual object types (batch,
  shift, operator, ...) via the ``objects=`` argument to
  :func:`~ts_shape.eventlog.to_event_log`. Empty tuple means "no
  auto-extraction; objects only appear if the caller binds them".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class LabelRule:
    template: str
    pack: str
    shape: str = "point"
    produces_objects: tuple[str, ...] = ("asset",)
    severity_field: str | None = None
    value_field: str | None = None
    drop_fields: tuple[str, ...] = ()
    # Standard attribute extension: maps fixed keys (see
    # ts_shape.eventlog.schema.STANDARD_ATTR_KEYS) to either a legacy column
    # name to rename, or a literal scalar to broadcast. Keys outside that
    # tuple raise in the coverage test.
    standard_attrs: Mapping[str, object] = field(default_factory=dict)


# Aliases used while building the registry below to keep entries terse.
_Q = "quality"
_PR = "production"
_E = "engineering"
_M = "maintenance"
_SC = "supplychain"
_EN = "energy"
_CO = "correlation"

P = "point"
I = "interval"
S = "summary"
ST = "static"

ASSET = ("asset",)
NONE: tuple[str, ...] = ()


def _r(template: str, pack: str, shape: str = P,
       objs: tuple[str, ...] = ASSET, **kw) -> LabelRule:
    return LabelRule(template=template, pack=pack, shape=shape,
                     produces_objects=objs, **kw)


# ----------------------------------------------------------------------------
# REGISTRY: (ClassName, method_name) -> LabelRule
# ----------------------------------------------------------------------------

REGISTRY: dict[tuple[str, str], LabelRule] = {

    # ---- correlation -------------------------------------------------------
    ("AnomalyCorrelationEvents", "cascade_detection"):
        _r("correlation.anomaly.cascade", _CO, P, objs=("signal",)),
    ("AnomalyCorrelationEvents", "coincident_anomalies"):
        _r("correlation.anomaly.coincident", _CO, P, objs=("signal",)),
    ("AnomalyCorrelationEvents", "root_cause_ranking"):
        _r("correlation.anomaly.root_cause_rank", _CO, ST, objs=("signal",)),
    ("SignalCorrelationEvents", "correlation_breakdown"):
        _r("correlation.signal.breakdown", _CO, ST, objs=("signal",)),
    ("SignalCorrelationEvents", "lag_correlation"):
        _r("correlation.signal.lag", _CO, ST, objs=("signal",)),
    ("SignalCorrelationEvents", "rolling_correlation"):
        _r("correlation.signal.rolling", _CO, S, objs=("signal",)),

    # ---- energy ------------------------------------------------------------
    ("CarbonIntensityEvents", "carbon_intensity_per_unit"):
        _r("energy.carbon.intensity_per_unit", _EN, S),
    ("CarbonIntensityEvents", "emission_factor_audit"):
        _r("energy.carbon.factor_audit", _EN, S),
    ("CarbonIntensityEvents", "emissions_by_window"):
        _r("energy.carbon.emissions_by_window", _EN, S),
    ("CarbonIntensityEvents", "total_emissions_by_window"):
        _r("energy.carbon.total_emissions_by_window", _EN, S),
    ("EnergyConsumptionEvents", "consumption_baseline_deviation"):
        _r("energy.consumption.baseline_deviation", _EN, S),
    ("EnergyConsumptionEvents", "consumption_by_window"):
        _r("energy.consumption.by_window", _EN, S),
    ("EnergyConsumptionEvents", "energy_per_unit"):
        _r("energy.consumption.per_unit", _EN, S),
    ("EnergyConsumptionEvents", "normalize"):
        _r("energy.consumption.normalize", _EN, S),
    ("EnergyConsumptionEvents", "peak_demand_detection"):
        _r("energy.consumption.peak_demand", _EN, P),
    ("EnergyEfficiencyEvents", "efficiency_comparison"):
        _r("energy.efficiency.comparison", _EN, S),
    ("EnergyEfficiencyEvents", "efficiency_trend"):
        _r("energy.efficiency.trend", _EN, S),
    ("EnergyEfficiencyEvents", "idle_energy_waste"):
        _r("energy.efficiency.idle_waste", _EN, S),
    ("EnergyEfficiencyEvents", "normalize"):
        _r("energy.efficiency.normalize", _EN, S),
    ("EnergyEfficiencyEvents", "specific_energy_consumption"):
        _r("energy.efficiency.specific_consumption", _EN, S),
    ("EnergyPerformanceIndicatorEvents", "enpi_by_hierarchy"):
        _r("energy.enpi.by_hierarchy", _EN, S),
    ("EnergyPerformanceIndicatorEvents", "enpi_by_window"):
        _r("energy.enpi.by_window", _EN, S),
    ("EnergyPerformanceIndicatorEvents", "enpi_vs_baseline"):
        _r("energy.enpi.vs_baseline", _EN, S),
    ("IdleEnergyDetectionEvents", "idle_energy_by_shift"):
        _r("energy.idle.by_shift", _EN, S, objs=("asset", "shift")),
    ("IdleEnergyDetectionEvents", "idle_energy_by_window"):
        _r("energy.idle.by_window", _EN, S),
    ("IdleEnergyDetectionEvents", "idle_energy_trend"):
        _r("energy.idle.trend", _EN, S),

    # ---- engineering -------------------------------------------------------
    ("ControlLoopHealthEvents", "detect_oscillation"):
        _r("engineering.control_loop.oscillation", _E, I),
    ("ControlLoopHealthEvents", "error_integrals"):
        _r("engineering.control_loop.error_integrals", _E, S),
    ("ControlLoopHealthEvents", "loop_health_summary"):
        _r("engineering.control_loop.health_summary", _E, S),
    ("ControlLoopHealthEvents", "output_saturation"):
        _r("engineering.control_loop.output_saturation", _E, I),
    ("DisturbanceRecoveryEvents", "before_after_comparison"):
        _r("engineering.disturbance.before_after", _E, S),
    ("DisturbanceRecoveryEvents", "detect_disturbances"):
        _r("engineering.disturbance.detected", _E, P),
    ("DisturbanceRecoveryEvents", "disturbance_frequency"):
        _r("engineering.disturbance.frequency", _E, S),
    ("DisturbanceRecoveryEvents", "recovery_time"):
        _r("engineering.disturbance.recovery_time", _E, I),
    ("MaterialBalanceEvents", "balance_check"):
        _r("engineering.material_balance.check", _E, S),
    ("MaterialBalanceEvents", "contribution_breakdown"):
        _r("engineering.material_balance.contribution", _E, S),
    ("MaterialBalanceEvents", "detect_balance_exceedance"):
        _r("engineering.material_balance.exceedance", _E, P),
    ("MaterialBalanceEvents", "imbalance_trend"):
        _r("engineering.material_balance.trend", _E, S),
    ("OperatingRangeEvents", "detect_regime_change"):
        _r("engineering.operating_range.regime_change", _E, P),
    ("OperatingRangeEvents", "operating_envelope"):
        _r("engineering.operating_range.envelope", _E, S),
    ("OperatingRangeEvents", "time_in_range"):
        _r("engineering.operating_range.time_in_range", _E, S),
    ("OperatingRangeEvents", "value_distribution"):
        _r("engineering.operating_range.distribution", _E, S),
    ("ProcessStabilityIndex", "score_trend"):
        _r("engineering.stability.score_trend", _E, S),
    ("ProcessStabilityIndex", "stability_comparison"):
        _r("engineering.stability.comparison", _E, S),
    ("ProcessStabilityIndex", "stability_score"):
        _r("engineering.stability.score", _E, S),
    ("ProcessStabilityIndex", "worst_periods"):
        _r("engineering.stability.worst_periods", _E, S),
    ("ProcessWindowEvents", "detect_mean_shift"):
        _r("engineering.process_window.mean_shift", _E, P),
    ("ProcessWindowEvents", "detect_variance_change"):
        _r("engineering.process_window.variance_change", _E, P),
    ("ProcessWindowEvents", "window_comparison"):
        _r("engineering.process_window.comparison", _E, S),
    ("ProcessWindowEvents", "windowed_statistics"):
        _r("engineering.process_window.statistics", _E, S),
    ("RateOfChangeEvents", "detect_rapid_change"):
        _r("engineering.rate_of_change.rapid", _E, P),
    ("RateOfChangeEvents", "detect_step_changes"):
        _r("engineering.rate_of_change.step", _E, P),
    ("RateOfChangeEvents", "rate_statistics"):
        _r("engineering.rate_of_change.statistics", _E, S),
    ("SetpointChangeEvents", "control_quality_metrics"):
        _r("engineering.setpoint.control_quality", _E, S),
    ("SetpointChangeEvents", "decay_rate"):
        _r("engineering.setpoint.decay_rate", _E, S),
    ("SetpointChangeEvents", "detect_setpoint_changes"):
        _r("engineering.setpoint.change", _E, P),
    ("SetpointChangeEvents", "detect_setpoint_ramps"):
        _r("engineering.setpoint.ramp", _E, I),
    ("SetpointChangeEvents", "detect_setpoint_steps"):
        _r("engineering.setpoint.step_{change_type}", _E, I),
    ("SetpointChangeEvents", "oscillation_frequency"):
        _r("engineering.setpoint.oscillation_frequency", _E, S),
    ("SetpointChangeEvents", "overshoot_metrics"):
        _r("engineering.setpoint.overshoot_metrics", _E, S),
    ("SetpointChangeEvents", "rise_time"):
        _r("engineering.setpoint.rise_time", _E, S),
    ("SetpointChangeEvents", "time_to_settle"):
        _r("engineering.setpoint.time_to_settle", _E, S),
    ("SetpointChangeEvents", "time_to_settle_derivative"):
        _r("engineering.setpoint.time_to_settle_derivative", _E, S),
    ("SignalComparisonEvents", "correlation_windows"):
        _r("engineering.signal_comparison.correlation_windows", _E, S, objs=("signal",)),
    ("SignalComparisonEvents", "detect_divergence"):
        _r("engineering.signal_comparison.divergence", _E, P, objs=("signal",)),
    ("SignalComparisonEvents", "deviation_statistics"):
        _r("engineering.signal_comparison.deviation_statistics", _E, S, objs=("signal",)),
    ("SignalComparisonEvents", "tracking_error_trend"):
        _r("engineering.signal_comparison.tracking_error_trend", _E, S, objs=("signal",)),
    ("StartupDetectionEvents", "assess_startup_quality"):
        _r("engineering.startup.quality_assessment", _E, S),
    ("StartupDetectionEvents", "detect_failed_startups"):
        _r("engineering.startup.failed", _E, P),
    ("StartupDetectionEvents", "detect_startup_adaptive"):
        _r("engineering.startup.adaptive", _E, I),
    ("StartupDetectionEvents", "detect_startup_by_slope"):
        _r("engineering.startup.by_slope", _E, I),
    ("StartupDetectionEvents", "detect_startup_by_threshold"):
        _r("engineering.startup.by_threshold", _E, I),
    ("StartupDetectionEvents", "detect_startup_multi_signal"):
        _r("engineering.startup.multi_signal", _E, I),
    ("StartupDetectionEvents", "track_startup_phases"):
        _r("engineering.startup.phase_{phase}", _E, I),
    ("SteadyStateDetectionEvents", "detect_steady_state"):
        _r("engineering.steady_state.detected", _E, I),
    ("SteadyStateDetectionEvents", "detect_transient_periods"):
        _r("engineering.steady_state.transient", _E, I),
    ("SteadyStateDetectionEvents", "steady_state_value_bands"):
        _r("engineering.steady_state.value_bands", _E, S),
    ("ThresholdMonitoringEvents", "multi_level_threshold"):
        _r("engineering.threshold.multi_level", _E, P),
    ("ThresholdMonitoringEvents", "threshold_exceedance_trend"):
        _r("engineering.threshold.exceedance_trend", _E, S),
    ("ThresholdMonitoringEvents", "threshold_with_hysteresis"):
        _r("engineering.threshold.hysteresis", _E, P),
    ("ThresholdMonitoringEvents", "time_above_threshold"):
        _r("engineering.threshold.time_above", _E, S),
    ("WarmUpCoolDownEvents", "detect_cooldown"):
        _r("engineering.thermal.cooldown", _E, I),
    ("WarmUpCoolDownEvents", "detect_warmup"):
        _r("engineering.thermal.warmup", _E, I),
    ("WarmUpCoolDownEvents", "time_to_target"):
        _r("engineering.thermal.time_to_target", _E, S),
    ("WarmUpCoolDownEvents", "warmup_consistency"):
        _r("engineering.thermal.warmup_consistency", _E, S),

    # ---- maintenance -------------------------------------------------------
    ("DegradationDetectionEvents", "detect_level_shift"):
        _r("maintenance.degradation.level_shift", _M, P),
    ("DegradationDetectionEvents", "detect_trend_degradation"):
        _r("maintenance.degradation.trend", _M, I),
    ("DegradationDetectionEvents", "detect_variance_increase"):
        _r("maintenance.degradation.variance_increase", _M, I),
    ("DegradationDetectionEvents", "health_score"):
        _r("maintenance.health.score_window", _M, S),
    ("FailurePredictionEvents", "detect_exceedance_pattern"):
        _r("maintenance.failure.exceedance_pattern", _M, P),
    ("FailurePredictionEvents", "remaining_useful_life"):
        _r("maintenance.failure.remaining_useful_life", _M, S),
    ("FailurePredictionEvents", "time_to_threshold"):
        _r("maintenance.failure.time_to_threshold", _M, S),
    ("VibrationAnalysisEvents", "bearing_health_indicators"):
        _r("maintenance.vibration.bearing_health", _M, S),
    ("VibrationAnalysisEvents", "detect_amplitude_growth"):
        _r("maintenance.vibration.amplitude_growth", _M, P),
    ("VibrationAnalysisEvents", "detect_rms_exceedance"):
        _r("maintenance.vibration.rms_exceedance", _M, P),

    # ---- production --------------------------------------------------------
    ("AlarmManagementEvents", "alarm_duration_stats"):
        _r("production.alarm.duration_stats", _PR, S),
    ("AlarmManagementEvents", "alarm_frequency"):
        _r("production.alarm.frequency", _PR, S),
    ("AlarmManagementEvents", "chattering_detection"):
        _r("production.alarm.chattering", _PR, I),
    ("AlarmManagementEvents", "standing_alarms"):
        _r("production.alarm.standing", _PR, I),
    ("BatchTrackingEvents", "batch_duration_stats"):
        _r("production.batch.duration_stats", _PR, S, objs=("asset", "batch")),
    ("BatchTrackingEvents", "batch_transition_matrix"):
        _r("production.batch.transition_matrix", _PR, ST, objs=("asset", "batch")),
    ("BatchTrackingEvents", "batch_yield"):
        _r("production.batch.yield", _PR, S, objs=("asset", "batch")),
    ("BatchTrackingEvents", "detect_batches"):
        _r("production.batch.detected", _PR, I, objs=("asset", "batch")),
    ("BottleneckDetectionEvents", "detect_bottleneck"):
        _r("production.bottleneck.detected", _PR, P, objs=("asset", "station")),
    ("BottleneckDetectionEvents", "shifting_bottleneck"):
        _r("production.bottleneck.shifting", _PR, S, objs=("asset", "station")),
    ("BottleneckDetectionEvents", "station_utilization"):
        _r("production.bottleneck.station_utilization", _PR, S, objs=("asset", "station")),
    ("ChangeoverEvents", "changeover_quality_metrics"):
        _r("production.changeover.quality_metrics", _PR, S),
    ("ChangeoverEvents", "changeover_window"):
        _r("production.changeover.window", _PR, I),
    ("ChangeoverEvents", "detect_changeover"):
        _r("production.changeover.detected", _PR, I),
    ("ContinuousProcessAlignmentEvents", "align_to_reference"):
        _r("production.alignment.to_reference", _PR, S, objs=("asset", "station")),
    ("ContinuousProcessAlignmentEvents", "alignment_quality"):
        _r("production.alignment.quality", _PR, S, objs=("asset", "station")),
    ("ContinuousProcessAlignmentEvents", "lag_profile"):
        _r("production.alignment.lag_profile", _PR, S, objs=("asset", "station")),
    ("ContinuousProcessAlignmentEvents", "segment_by_cut"):
        _r("production.alignment.segment_by_cut", _PR, I, objs=("asset", "station")),
    ("CycleTimeTracking", "cycle_time_by_part"):
        _r("production.cycle_time.by_part", _PR, S, objs=("asset", "part")),
    ("CycleTimeTracking", "cycle_time_statistics"):
        _r("production.cycle_time.statistics", _PR, S),
    ("CycleTimeTracking", "cycle_time_trend"):
        _r("production.cycle_time.trend", _PR, S),
    ("CycleTimeTracking", "detect_slow_cycles"):
        _r("production.cycle_time.slow", _PR, P, objs=("asset", "cycle")),
    ("CycleTimeTracking", "hourly_cycle_time_summary"):
        _r("production.cycle_time.hourly_summary", _PR, S),
    ("DowntimeTracking", "availability_trend"):
        _r("production.downtime.availability_trend", _PR, S),
    ("DowntimeTracking", "downtime_by_reason"):
        _r("production.downtime.by_reason", _PR, S),
    ("DowntimeTracking", "downtime_by_shift"):
        _r("production.downtime.by_shift", _PR, S, objs=("asset", "shift")),
    ("DowntimeTracking", "top_downtime_reasons"):
        _r("production.downtime.top_reasons", _PR, ST),
    ("DutyCycleEvents", "cycle_count"):
        _r("production.duty_cycle.count", _PR, S),
    ("DutyCycleEvents", "duty_cycle_per_window"):
        _r("production.duty_cycle.per_window", _PR, S),
    ("DutyCycleEvents", "excessive_cycling"):
        _r("production.duty_cycle.excessive", _PR, P),
    ("DutyCycleEvents", "on_off_intervals"):
        _r("production.duty_cycle.on_off", _PR, I),
    ("FlowConstraintEvents", "blocked_events"):
        _r("production.flow.blocked", _PR, I, objs=("asset", "station")),
    ("FlowConstraintEvents", "starved_events"):
        _r("production.flow.starved", _PR, I, objs=("asset", "station")),
    ("LineThroughputEvents", "count_parts"):
        _r("production.throughput.count_parts", _PR, S),
    ("LineThroughputEvents", "cycle_quality_check"):
        _r("production.throughput.cycle_quality_check", _PR, P, objs=("asset", "cycle")),
    ("LineThroughputEvents", "takt_adherence"):
        _r("production.throughput.takt_adherence", _PR, S),
    ("LineThroughputEvents", "throughput_oee"):
        _r("production.throughput.oee", _PR, S),
    ("LineThroughputEvents", "throughput_trends"):
        _r("production.throughput.trends", _PR, S),
    ("LongDowntimeEvents", "count_events_between_gaps"):
        _r("production.long_downtime.events_between_gaps", _PR, S),
    ("LongDowntimeEvents", "detect_long_downtime"):
        _r("production.long_downtime.detected", _PR, I),
    ("MachineStateEvents", "detect_rapid_transitions"):
        _r("production.machine_state.rapid_transitions", _PR, P),
    ("MachineStateEvents", "detect_run_idle"):
        _r("production.machine_state.{state}", _PR, I),
    ("MachineStateEvents", "transition_events"):
        _r("production.machine_state.transition_{transition}", _PR, P),
    ("MicroStopEvents", "detect_micro_stops"):
        _r("production.micro_stop.detected", _PR, I),
    ("MicroStopEvents", "micro_stop_frequency"):
        _r("production.micro_stop.frequency", _PR, S),
    ("MicroStopEvents", "micro_stop_impact"):
        _r("production.micro_stop.impact", _PR, S),
    ("MicroStopEvents", "micro_stop_patterns"):
        _r("production.micro_stop.patterns", _PR, S),
    ("MultiProcessTraceabilityEvents", "build_timeline"):
        _r("production.traceability.timeline", _PR, P, objs=("asset", "serial", "station")),
    ("MultiProcessTraceabilityEvents", "handover_log"):
        _r("production.traceability.handover", _PR, P, objs=("asset", "serial", "station")),
    ("MultiProcessTraceabilityEvents", "lead_time"):
        _r("production.traceability.lead_time", _PR, S, objs=("serial",)),
    ("MultiProcessTraceabilityEvents", "parallel_activity"):
        _r("production.traceability.parallel_activity", _PR, I, objs=("serial", "station")),
    ("MultiProcessTraceabilityEvents", "routing_paths"):
        _r("production.traceability.routing_paths", _PR, ST, objs=("serial",)),
    ("MultiProcessTraceabilityEvents", "station_statistics"):
        _r("production.traceability.station_statistics", _PR, S, objs=("station",)),
    ("OEECalculator", "calculate_availability"):
        _r("production.oee.availability", _PR, S),
    ("OEECalculator", "calculate_oee"):
        _r("production.oee.total", _PR, S),
    ("OEECalculator", "calculate_performance"):
        _r("production.oee.performance", _PR, S),
    ("OEECalculator", "calculate_quality"):
        _r("production.oee.quality", _PR, S),
    ("OperatorPerformanceTracking", "operator_comparison"):
        _r("production.operator.comparison", _PR, S, objs=("operator",)),
    ("OperatorPerformanceTracking", "operator_efficiency"):
        _r("production.operator.efficiency", _PR, S, objs=("operator",)),
    ("OperatorPerformanceTracking", "production_by_operator"):
        _r("production.operator.production", _PR, S, objs=("operator",)),
    ("OperatorPerformanceTracking", "quality_by_operator"):
        _r("production.operator.quality", _PR, S, objs=("operator",)),
    ("OrderTraceabilityEvents", "build_timeline"):
        _r("production.order.timeline", _PR, P, objs=("asset", "work_order", "station")),
    ("OrderTraceabilityEvents", "current_status"):
        _r("production.order.current_status", _PR, ST, objs=("work_order",)),
    ("OrderTraceabilityEvents", "lead_time"):
        _r("production.order.lead_time", _PR, S, objs=("work_order",)),
    ("OrderTraceabilityEvents", "station_dwell_statistics"):
        _r("production.order.station_dwell_statistics", _PR, S, objs=("station",)),
    ("ValueTraceabilityEvents", "build_timeline"):
        _r("production.value_trace.timeline", _PR, P, objs=("asset", "serial", "station")),
    ("ValueTraceabilityEvents", "current_status"):
        _r("production.value_trace.current_status", _PR, ST, objs=("serial",)),
    ("ValueTraceabilityEvents", "lead_time"):
        _r("production.value_trace.lead_time", _PR, S, objs=("serial",)),
    ("ValueTraceabilityEvents", "station_dwell_statistics"):
        _r("production.value_trace.station_dwell_statistics", _PR, S, objs=("station",)),
    ("PartProductionTracking", "daily_production_summary"):
        _r("production.part.daily_summary", _PR, S, objs=("asset", "part")),
    ("PartProductionTracking", "production_by_part"):
        _r("production.part.production", _PR, S, objs=("asset", "part")),
    ("PartProductionTracking", "production_totals"):
        _r("production.part.totals", _PR, S, objs=("asset", "part")),
    ("PerformanceLossTracking", "performance_by_shift"):
        _r("production.performance.by_shift", _PR, S, objs=("asset", "shift")),
    ("PerformanceLossTracking", "performance_trend"):
        _r("production.performance.trend", _PR, S),
    ("PerformanceLossTracking", "slow_periods"):
        _r("production.performance.slow_period", _PR, I),
    ("PeriodSummary", "compare_periods"):
        _r("production.period.compare", _PR, S),
    ("PeriodSummary", "from_daily_data"):
        _r("production.period.from_daily", _PR, S),
    ("PeriodSummary", "monthly_summary"):
        _r("production.period.monthly_summary", _PR, S),
    ("PeriodSummary", "weekly_summary"):
        _r("production.period.weekly_summary", _PR, S),
    ("QualityTracking", "daily_quality_summary"):
        _r("production.quality.daily_summary", _PR, S),
    ("QualityTracking", "nok_by_reason"):
        _r("production.quality.nok_by_reason", _PR, S),
    ("QualityTracking", "nok_by_shift"):
        _r("production.quality.nok_by_shift", _PR, S, objs=("asset", "shift")),
    ("QualityTracking", "quality_by_part"):
        _r("production.quality.by_part", _PR, S, objs=("asset", "part")),
    ("ReworkTracking", "rework_by_reason"):
        _r("production.rework.by_reason", _PR, S),
    ("ReworkTracking", "rework_by_shift"):
        _r("production.rework.by_shift", _PR, S, objs=("asset", "shift")),
    ("ReworkTracking", "rework_cost"):
        _r("production.rework.cost", _PR, S),
    ("ReworkTracking", "rework_rate"):
        _r("production.rework.rate", _PR, S),
    ("ReworkTracking", "rework_trend"):
        _r("production.rework.trend", _PR, S),
    ("RoutingTraceabilityEvents", "build_routing_timeline"):
        _r("production.routing.timeline", _PR, P, objs=("asset", "serial", "station")),
    ("RoutingTraceabilityEvents", "lead_time"):
        _r("production.routing.lead_time", _PR, S, objs=("serial",)),
    ("RoutingTraceabilityEvents", "routing_paths"):
        _r("production.routing.paths", _PR, ST, objs=("serial",)),
    ("RoutingTraceabilityEvents", "station_statistics"):
        _r("production.routing.station_statistics", _PR, S, objs=("station",)),
    ("ScrapTracking", "scrap_by_reason"):
        _r("production.scrap.by_reason", _PR, S),
    ("ScrapTracking", "scrap_by_shift"):
        _r("production.scrap.by_shift", _PR, S, objs=("asset", "shift")),
    ("ScrapTracking", "scrap_cost"):
        _r("production.scrap.cost", _PR, S),
    ("ScrapTracking", "scrap_trend"):
        _r("production.scrap.trend", _PR, S),
    ("SetupTimeTracking", "setup_by_product"):
        _r("production.setup.by_product", _PR, S, objs=("asset", "part")),
    ("SetupTimeTracking", "setup_durations"):
        _r("production.setup.durations", _PR, I),
    ("SetupTimeTracking", "setup_statistics"):
        _r("production.setup.statistics", _PR, S),
    ("SetupTimeTracking", "setup_trend"):
        _r("production.setup.trend", _PR, S),
    ("ShiftHandoverReport", "from_shift_data"):
        _r("production.shift.handover_from_data", _PR, S, objs=("asset", "shift")),
    ("ShiftHandoverReport", "generate_report"):
        _r("production.shift.handover_report", _PR, S, objs=("asset", "shift")),
    ("ShiftReporting", "best_and_worst_shifts"):
        _r("production.shift.best_and_worst", _PR, S, objs=("asset", "shift")),
    ("ShiftReporting", "shift_comparison"):
        _r("production.shift.comparison", _PR, S, objs=("asset", "shift")),
    ("ShiftReporting", "shift_production"):
        _r("production.shift.production", _PR, S, objs=("asset", "shift")),
    ("ShiftReporting", "shift_targets"):
        _r("production.shift.targets", _PR, S, objs=("asset", "shift")),
    ("TargetTracking", "compare_to_target"):
        _r("production.target.compare", _PR, S),
    ("TargetTracking", "target_achievement_summary"):
        _r("production.target.achievement_summary", _PR, S),

    # ---- quality -----------------------------------------------------------
    ("AnomalyClassificationEvents", "classify_anomalies"):
        _r("quality.anomaly.classified_{anomaly_class}", _Q, P),
    ("AnomalyClassificationEvents", "detect_drift"):
        _r("quality.anomaly.drift", _Q, P),
    ("AnomalyClassificationEvents", "detect_flatline"):
        _r("quality.anomaly.flatline", _Q, I),
    ("AnomalyClassificationEvents", "detect_oscillation"):
        _r("quality.anomaly.oscillation", _Q, I),
    ("CapabilityTrendingEvents", "capability_forecast"):
        _r("quality.capability.forecast", _Q, S),
    ("CapabilityTrendingEvents", "capability_over_time"):
        _r("quality.capability.over_time", _Q, S),
    ("CapabilityTrendingEvents", "detect_capability_drop"):
        _r("quality.capability.drop", _Q, P),
    ("CapabilityTrendingEvents", "yield_estimate"):
        _r("quality.capability.yield_estimate", _Q, S),
    ("DataGapAnalysisEvents", "coverage_by_period"):
        _r("quality.data_gap.coverage_by_period", _Q, S, objs=("signal",)),
    ("DataGapAnalysisEvents", "find_gaps"):
        _r("quality.data_gap.gap", _Q, I, objs=("signal",)),
    ("DataGapAnalysisEvents", "gap_summary"):
        _r("quality.data_gap.summary", _Q, S, objs=("signal",)),
    ("DataGapAnalysisEvents", "interpolation_candidates"):
        _r("quality.data_gap.interpolation_candidate", _Q, P, objs=("signal",)),
    ("GaugeRepeatabilityEvents", "gauge_rr_summary"):
        _r("quality.gauge_rr.summary", _Q, ST, objs=("tool",)),
    ("GaugeRepeatabilityEvents", "measurement_bias"):
        _r("quality.gauge_rr.bias", _Q, ST, objs=("tool",)),
    ("GaugeRepeatabilityEvents", "repeatability"):
        _r("quality.gauge_rr.repeatability", _Q, ST, objs=("tool",)),
    ("GaugeRepeatabilityEvents", "reproducibility"):
        _r("quality.gauge_rr.reproducibility", _Q, ST, objs=("tool",)),
    ("MultiSensorValidationEvents", "consensus_score"):
        _r("quality.multi_sensor.consensus_score", _Q, S, objs=("sensor",)),
    ("MultiSensorValidationEvents", "detect_disagreement"):
        _r("quality.multi_sensor.disagreement", _Q, P, objs=("sensor",)),
    ("MultiSensorValidationEvents", "identify_outlier_sensor"):
        _r("quality.multi_sensor.outlier_sensor", _Q, ST, objs=("sensor",)),
    ("MultiSensorValidationEvents", "pairwise_bias"):
        _r("quality.multi_sensor.pairwise_bias", _Q, ST, objs=("sensor",)),
    ("OutlierDetectionEvents", "detect_outliers_iqr"):
        _r("quality.outlier.iqr", _Q, P, severity_field="severity_score"),
    ("OutlierDetectionEvents", "detect_outliers_isolation_forest"):
        _r("quality.outlier.isolation_forest", _Q, P, severity_field="severity_score"),
    ("OutlierDetectionEvents", "detect_outliers_mad"):
        _r("quality.outlier.mad", _Q, P, severity_field="severity_score"),
    ("OutlierDetectionEvents", "detect_outliers_zscore"):
        _r("quality.outlier.zscore", _Q, P, severity_field="severity_score"),
    ("SensorDriftEvents", "calibration_health"):
        _r("quality.sensor_drift.calibration_health", _Q, S, objs=("sensor",)),
    ("SensorDriftEvents", "detect_span_drift"):
        _r("quality.sensor_drift.span_drift", _Q, P, objs=("sensor",)),
    ("SensorDriftEvents", "detect_zero_drift"):
        _r("quality.sensor_drift.zero_drift", _Q, P, objs=("sensor",)),
    ("SensorDriftEvents", "drift_trend"):
        _r("quality.sensor_drift.trend", _Q, S, objs=("sensor",)),
    ("SignalQualityEvents", "data_completeness"):
        _r("quality.signal.completeness", _Q, S, objs=("signal",)),
    ("SignalQualityEvents", "detect_missing_data"):
        _r("quality.signal.missing", _Q, I, objs=("signal",)),
    ("SignalQualityEvents", "detect_out_of_range"):
        _r("quality.signal.out_of_range", _Q, P, objs=("signal",)),
    ("SignalQualityEvents", "sampling_regularity"):
        _r("quality.signal.sampling_regularity", _Q, S, objs=("signal",)),
    ("StatisticalProcessControlRuleBased", "apply_rules_vectorized"):
        _r("quality.spc.rule_violation", _Q, P),
    ("StatisticalProcessControlRuleBased", "calculate_control_limits"):
        _r("quality.spc.control_limits", _Q, ST),
    ("StatisticalProcessControlRuleBased", "calculate_dynamic_control_limits"):
        _r("quality.spc.control_limits_dynamic", _Q, S),
    ("StatisticalProcessControlRuleBased", "detect_cusum_shifts"):
        _r("quality.spc.cusum_shift", _Q, P),
    ("StatisticalProcessControlRuleBased", "interpret_violations"):
        _r("quality.spc.violation_interpretation", _Q, P),
    ("StatisticalProcessControlRuleBased", "process"):
        _r("quality.spc.rule_violation", _Q, P),
    ("StatisticalProcessControlRuleBased", "rule_1"):
        _r("quality.spc.rule_1", _Q, P),
    ("StatisticalProcessControlRuleBased", "rule_2"):
        _r("quality.spc.rule_2", _Q, P),
    ("StatisticalProcessControlRuleBased", "rule_3"):
        _r("quality.spc.rule_3", _Q, P),
    ("StatisticalProcessControlRuleBased", "rule_4"):
        _r("quality.spc.rule_4", _Q, P),
    ("StatisticalProcessControlRuleBased", "rule_5"):
        _r("quality.spc.rule_5", _Q, P),
    ("StatisticalProcessControlRuleBased", "rule_6"):
        _r("quality.spc.rule_6", _Q, P),
    ("StatisticalProcessControlRuleBased", "rule_7"):
        _r("quality.spc.rule_7", _Q, P),
    ("StatisticalProcessControlRuleBased", "rule_8"):
        _r("quality.spc.rule_8", _Q, P),
    ("ToleranceDeviationEvents", "process_and_group_data_with_events"):
        _r("quality.tolerance.deviation", _Q, P, severity_field="severity"),
    ("ValueDistributionEvents", "detect_bimodal"):
        _r("quality.distribution.bimodal", _Q, S),
    ("ValueDistributionEvents", "detect_mode_changes"):
        _r("quality.distribution.mode_change", _Q, P),
    ("ValueDistributionEvents", "normality_windows"):
        _r("quality.distribution.normality", _Q, S),
    ("ValueDistributionEvents", "percentile_tracking"):
        _r("quality.distribution.percentile", _Q, S),

    # ---- supplychain -------------------------------------------------------
    ("DemandPatternEvents", "demand_by_period"):
        _r("supplychain.demand.by_period", _SC, S, objs=("material",)),
    ("DemandPatternEvents", "detect_demand_spikes"):
        _r("supplychain.demand.spike", _SC, P, objs=("material",)),
    ("DemandPatternEvents", "seasonality_summary"):
        _r("supplychain.demand.seasonality", _SC, ST, objs=("material",)),
    ("InventoryMonitoringEvents", "consumption_rate"):
        _r("supplychain.inventory.consumption_rate", _SC, S, objs=("material",)),
    ("InventoryMonitoringEvents", "detect_low_stock"):
        _r("supplychain.inventory.low_stock", _SC, P, objs=("material",)),
    ("InventoryMonitoringEvents", "reorder_point_breach"):
        _r("supplychain.inventory.reorder_point_breach", _SC, P, objs=("material",)),
    ("InventoryMonitoringEvents", "stockout_prediction"):
        _r("supplychain.inventory.stockout_prediction", _SC, S, objs=("material",)),
    ("LeadTimeAnalysisEvents", "calculate_lead_times"):
        _r("supplychain.lead_time.calculated", _SC, I, objs=("work_order",)),
    ("LeadTimeAnalysisEvents", "detect_lead_time_anomalies"):
        _r("supplychain.lead_time.anomaly", _SC, P, objs=("work_order",)),
    ("LeadTimeAnalysisEvents", "lead_time_statistics"):
        _r("supplychain.lead_time.statistics", _SC, S, objs=("work_order",)),
}


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def get(class_name: str, method_name: str) -> LabelRule | None:
    return REGISTRY.get((class_name, method_name))


def render_activity(rule: LabelRule, row: Mapping[str, object]) -> str:
    """Substitute ``{field}`` placeholders in ``rule.template`` from ``row``.

    Missing fields render as ``unknown`` rather than raising — adapters
    should not crash on partial input data.
    """
    template = rule.template
    if "{" not in template:
        return template
    out = template
    # Minimal templating: scan for {name} occurrences.
    i = 0
    parts: list[str] = []
    while i < len(template):
        j = template.find("{", i)
        if j == -1:
            parts.append(template[i:])
            break
        parts.append(template[i:j])
        k = template.find("}", j + 1)
        if k == -1:
            parts.append(template[j:])
            break
        field_name = template[j + 1:k]
        val = row.get(field_name) if hasattr(row, "get") else None  # type: ignore[arg-type]
        if val is None:
            try:
                val = row[field_name]  # type: ignore[index]
            except (KeyError, TypeError):
                val = "unknown"
        parts.append(str(val))
        i = k + 1
    return "".join(parts)
