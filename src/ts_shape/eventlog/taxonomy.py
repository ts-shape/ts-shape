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
from collections.abc import Mapping


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
I = "interval"  # noqa: E741 — one-letter shape alias used in the REGISTRY literal below
S = "summary"
ST = "static"

ASSET = ("asset",)
NONE: tuple[str, ...] = ()


def _r(
    template: str, pack: str, shape: str = P, objs: tuple[str, ...] = ASSET, **kw
) -> LabelRule:
    return LabelRule(
        template=template, pack=pack, shape=shape, produces_objects=objs, **kw
    )


# ----------------------------------------------------------------------------
# REGISTRY: (ClassName, method_name) -> LabelRule
# ----------------------------------------------------------------------------

REGISTRY: dict[tuple[str, str], LabelRule] = {
    # ---- correlation -------------------------------------------------------
    ("AnomalyCorrelationEvents", "cascade_detection"): _r(
        "correlation.anomaly.cascade",
        _CO,
        P,
        objs=("signal",),
        standard_attrs={
            "ts_shape:method": "cascade",
            "ts_shape:direction": "leader_to_follower",
        },
    ),
    ("AnomalyCorrelationEvents", "coincident_anomalies"): _r(
        "correlation.anomaly.coincident",
        _CO,
        P,
        objs=("signal",),
        standard_attrs={
            "ts_shape:method": "coincidence",
            "ts_shape:sample_count": "anomaly_count",
        },
    ),
    ("AnomalyCorrelationEvents", "root_cause_ranking"): _r(
        "correlation.anomaly.root_cause_rank",
        _CO,
        ST,
        objs=("signal",),
        standard_attrs={
            "ts_shape:method": "root_cause_rank",
            "ts_shape:confidence": "leader_ratio",
            "ts_shape:sample_count": "leader_count",
        },
    ),
    ("SignalCorrelationEvents", "correlation_breakdown"): _r(
        "correlation.signal.breakdown",
        _CO,
        ST,
        objs=("signal",),
        standard_attrs={
            "ts_shape:method": "breakdown",
            "ts_shape:confidence": "min_correlation",
        },
    ),
    ("SignalCorrelationEvents", "lag_correlation"): _r(
        "correlation.signal.lag",
        _CO,
        ST,
        objs=("signal",),
        standard_attrs={
            "ts_shape:method": "lag",
            "ts_shape:confidence": "correlation",
        },
    ),
    ("SignalCorrelationEvents", "rolling_correlation"): _r(
        "correlation.signal.rolling",
        _CO,
        S,
        objs=("signal",),
        standard_attrs={
            "ts_shape:method": "rolling_correlation",
            "ts_shape:confidence": "correlation",
        },
    ),
    # ---- energy ------------------------------------------------------------
    ("CarbonIntensityEvents", "carbon_intensity_per_unit"): _r(
        "energy.carbon.intensity_per_unit",
        _EN,
        S,
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("CarbonIntensityEvents", "emission_factor_audit"): _r(
        "energy.carbon.factor_audit",
        _EN,
        S,
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("CarbonIntensityEvents", "emissions_by_window"): _r(
        "energy.carbon.emissions_by_window",
        _EN,
        S,
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("CarbonIntensityEvents", "total_emissions_by_window"): _r(
        "energy.carbon.total_emissions_by_window",
        _EN,
        S,
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("EnergyConsumptionEvents", "consumption_baseline_deviation"): _r(
        "energy.consumption.baseline_deviation",
        _EN,
        S,
        standard_attrs={
            "ts_shape:method": "baseline_deviation",
            "ts_shape:direction": "deviation",
            "ts_shape:baseline": "baseline",
            "ts_shape:deviation_pct": "deviation_pct",
        },
    ),
    ("EnergyConsumptionEvents", "consumption_by_window"): _r(
        "energy.consumption.by_window",
        _EN,
        S,
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("EnergyConsumptionEvents", "energy_per_unit"): _r(
        "energy.consumption.per_unit",
        _EN,
        S,
        standard_attrs={"ts_shape:sample_count": "units"},
    ),
    ("EnergyConsumptionEvents", "normalize"): _r(
        "energy.consumption.normalize",
        _EN,
        S,
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("EnergyConsumptionEvents", "peak_demand_detection"): _r(
        "energy.consumption.peak_demand",
        _EN,
        P,
        standard_attrs={
            "ts_shape:method": "peak_demand",
            "ts_shape:direction": "above",
            "ts_shape:threshold_high": "threshold",
        },
    ),
    ("EnergyEfficiencyEvents", "efficiency_comparison"): _r(
        "energy.efficiency.comparison",
        _EN,
        S,
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("EnergyEfficiencyEvents", "efficiency_trend"): _r(
        "energy.efficiency.trend",
        _EN,
        S,
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("EnergyEfficiencyEvents", "idle_energy_waste"): _r(
        "energy.efficiency.idle_waste",
        _EN,
        S,
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("EnergyEfficiencyEvents", "normalize"): _r(
        "energy.efficiency.normalize",
        _EN,
        S,
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("EnergyEfficiencyEvents", "specific_energy_consumption"): _r(
        "energy.efficiency.specific_consumption",
        _EN,
        S,
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("EnergyPerformanceIndicatorEvents", "enpi_by_hierarchy"): _r(
        "energy.enpi.by_hierarchy",
        _EN,
        S,
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("EnergyPerformanceIndicatorEvents", "enpi_by_window"): _r(
        "energy.enpi.by_window", _EN, S, standard_attrs={"ts_shape:sample_count": None}
    ),
    ("EnergyPerformanceIndicatorEvents", "enpi_vs_baseline"): _r(
        "energy.enpi.vs_baseline",
        _EN,
        S,
        standard_attrs={
            "ts_shape:method": "enpi_vs_baseline",
            "ts_shape:direction": "deviation",
            "ts_shape:baseline": "baseline_enpi",
            "ts_shape:deviation_pct": "deviation_pct",
        },
    ),
    ("IdleEnergyDetectionEvents", "idle_energy_by_shift"): _r(
        "energy.idle.by_shift",
        _EN,
        S,
        objs=("asset", "shift"),
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("IdleEnergyDetectionEvents", "idle_energy_by_window"): _r(
        "energy.idle.by_window", _EN, S, standard_attrs={"ts_shape:sample_count": None}
    ),
    ("IdleEnergyDetectionEvents", "idle_energy_trend"): _r(
        "energy.idle.trend", _EN, S, standard_attrs={"ts_shape:sample_count": None}
    ),
    # ---- engineering -------------------------------------------------------
    ("ControlLoopHealthEvents", "detect_oscillation"): _r(
        "engineering.control_loop.oscillation",
        _E,
        I,
        standard_attrs={"ts_shape:lifecycle_state": "oscillating"},
    ),
    ("ControlLoopHealthEvents", "error_integrals"): _r(
        "engineering.control_loop.error_integrals",
        _E,
        S,
        standard_attrs={"ts_shape:sample_count": "sample_count"},
    ),
    ("ControlLoopHealthEvents", "loop_health_summary"): _r(
        "engineering.control_loop.health_summary",
        _E,
        S,
        standard_attrs={
            "ts_shape:sample_count": None,
            "ts_shape:outcome": "health_grade",
        },
    ),
    ("ControlLoopHealthEvents", "output_saturation"): _r(
        "engineering.control_loop.output_saturation",
        _E,
        I,
        standard_attrs={"ts_shape:lifecycle_state": "saturated"},
    ),
    ("DisturbanceRecoveryEvents", "before_after_comparison"): _r(
        "engineering.disturbance.before_after",
        _E,
        S,
        standard_attrs={"ts_shape:sample_count": None, "ts_shape:baseline": "pre_mean"},
    ),
    ("DisturbanceRecoveryEvents", "detect_disturbances"): _r(
        "engineering.disturbance.detected",
        _E,
        P,
        standard_attrs={
            "ts_shape:method": "disturbance",
            "ts_shape:direction": "direction",
            "ts_shape:deviation": "peak_deviation",
        },
    ),
    ("DisturbanceRecoveryEvents", "disturbance_frequency"): _r(
        "engineering.disturbance.frequency",
        _E,
        S,
        standard_attrs={"ts_shape:sample_count": "disturbance_count"},
    ),
    ("DisturbanceRecoveryEvents", "recovery_time"): _r(
        "engineering.disturbance.recovery_time",
        _E,
        I,
        standard_attrs={
            "ts_shape:lifecycle_state": "recovering",
            "ts_shape:baseline": "pre_disturbance_mean",
        },
    ),
    ("MaterialBalanceEvents", "balance_check"): _r(
        "engineering.material_balance.check",
        _E,
        S,
        standard_attrs={"ts_shape:sample_count": None, "ts_shape:outcome": "balanced"},
    ),
    ("MaterialBalanceEvents", "contribution_breakdown"): _r(
        "engineering.material_balance.contribution",
        _E,
        S,
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("MaterialBalanceEvents", "detect_balance_exceedance"): _r(
        "engineering.material_balance.exceedance",
        _E,
        P,
        standard_attrs={
            "ts_shape:method": "balance_exceedance",
            "ts_shape:direction": "outside",
            "ts_shape:deviation_pct": "max_imbalance_pct",
        },
    ),
    ("MaterialBalanceEvents", "imbalance_trend"): _r(
        "engineering.material_balance.trend",
        _E,
        S,
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("OperatingRangeEvents", "detect_regime_change"): _r(
        "engineering.operating_range.regime_change",
        _E,
        P,
        standard_attrs={
            "ts_shape:outcome": "regime_change",
            "ts_shape:baseline": "prev_mean",
            "ts_shape:deviation": "shift_magnitude",
        },
    ),
    ("OperatingRangeEvents", "operating_envelope"): _r(
        "engineering.operating_range.envelope",
        _E,
        S,
        standard_attrs={
            "ts_shape:sample_count": None,
            "ts_shape:threshold_low": "min_value",
            "ts_shape:threshold_high": "max_value",
        },
    ),
    ("OperatingRangeEvents", "time_in_range"): _r(
        "engineering.operating_range.time_in_range",
        _E,
        S,
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("OperatingRangeEvents", "value_distribution"): _r(
        "engineering.operating_range.distribution",
        _E,
        S,
        standard_attrs={"ts_shape:sample_count": "count"},
    ),
    ("ProcessStabilityIndex", "score_trend"): _r(
        "engineering.stability.score_trend",
        _E,
        S,
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("ProcessStabilityIndex", "stability_comparison"): _r(
        "engineering.stability.comparison",
        _E,
        S,
        standard_attrs={
            "ts_shape:sample_count": None,
            "ts_shape:baseline": "best_score",
        },
    ),
    ("ProcessStabilityIndex", "stability_score"): _r(
        "engineering.stability.score",
        _E,
        S,
        standard_attrs={"ts_shape:sample_count": None, "ts_shape:outcome": "grade"},
    ),
    ("ProcessStabilityIndex", "worst_periods"): _r(
        "engineering.stability.worst_periods",
        _E,
        S,
        standard_attrs={
            "ts_shape:sample_count": None,
            "ts_shape:outcome": "primary_issue",
        },
    ),
    ("ProcessWindowEvents", "detect_mean_shift"): _r(
        "engineering.process_window.mean_shift",
        _E,
        P,
        standard_attrs={
            "ts_shape:method": "mean_shift",
            "ts_shape:direction": "shift",
            "ts_shape:baseline": "prev_mean",
            "ts_shape:deviation": "shift_sigma",
        },
    ),
    ("ProcessWindowEvents", "detect_variance_change"): _r(
        "engineering.process_window.variance_change",
        _E,
        P,
        standard_attrs={
            "ts_shape:method": "variance_change",
            "ts_shape:direction": "change",
            "ts_shape:baseline": "prev_std",
        },
    ),
    ("ProcessWindowEvents", "window_comparison"): _r(
        "engineering.process_window.comparison",
        _E,
        S,
        standard_attrs={
            "ts_shape:sample_count": None,
            "ts_shape:outcome": "is_anomalous",
        },
    ),
    ("ProcessWindowEvents", "windowed_statistics"): _r(
        "engineering.process_window.statistics",
        _E,
        S,
        standard_attrs={"ts_shape:sample_count": "count"},
    ),
    ("RateOfChangeEvents", "detect_rapid_change"): _r(
        "engineering.rate_of_change.rapid",
        _E,
        P,
        standard_attrs={
            "ts_shape:method": "rapid_change",
            "ts_shape:direction": "direction",
        },
    ),
    ("RateOfChangeEvents", "detect_step_changes"): _r(
        "engineering.rate_of_change.step",
        _E,
        P,
        standard_attrs={
            "ts_shape:method": "step_change",
            "ts_shape:direction": "step",
            "ts_shape:deviation": "delta",
        },
    ),
    ("RateOfChangeEvents", "rate_statistics"): _r(
        "engineering.rate_of_change.statistics",
        _E,
        S,
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("SetpointChangeEvents", "control_quality_metrics"): _r(
        "engineering.setpoint.control_quality",
        _E,
        S,
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("SetpointChangeEvents", "decay_rate"): _r(
        "engineering.setpoint.decay_rate",
        _E,
        S,
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("SetpointChangeEvents", "detect_setpoint_changes"): _r(
        "engineering.setpoint.change",
        _E,
        P,
        standard_attrs={"ts_shape:outcome": "change_type"},
    ),
    ("SetpointChangeEvents", "detect_setpoint_ramps"): _r(
        "engineering.setpoint.ramp",
        _E,
        I,
        standard_attrs={"ts_shape:lifecycle_state": "ramping"},
    ),
    ("SetpointChangeEvents", "detect_setpoint_steps"): _r(
        "engineering.setpoint.step_{change_type}",
        _E,
        I,
        standard_attrs={"ts_shape:lifecycle_state": "stepping"},
    ),
    ("SetpointChangeEvents", "oscillation_frequency"): _r(
        "engineering.setpoint.oscillation_frequency",
        _E,
        S,
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("SetpointChangeEvents", "overshoot_metrics"): _r(
        "engineering.setpoint.overshoot_metrics",
        _E,
        S,
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("SetpointChangeEvents", "rise_time"): _r(
        "engineering.setpoint.rise_time",
        _E,
        S,
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("SetpointChangeEvents", "time_to_settle"): _r(
        "engineering.setpoint.time_to_settle",
        _E,
        S,
        standard_attrs={"ts_shape:sample_count": None, "ts_shape:outcome": "settled"},
    ),
    ("SetpointChangeEvents", "time_to_settle_derivative"): _r(
        "engineering.setpoint.time_to_settle_derivative",
        _E,
        S,
        standard_attrs={"ts_shape:sample_count": None, "ts_shape:outcome": "settled"},
    ),
    ("SignalComparisonEvents", "correlation_windows"): _r(
        "engineering.signal_comparison.correlation_windows",
        _E,
        S,
        objs=("signal",),
        standard_attrs={
            "ts_shape:method": "correlation",
            "ts_shape:confidence": "correlation",
        },
    ),
    ("SignalComparisonEvents", "detect_divergence"): _r(
        "engineering.signal_comparison.divergence",
        _E,
        P,
        objs=("signal",),
        standard_attrs={
            "ts_shape:method": "divergence",
            "ts_shape:direction": "direction",
            "ts_shape:deviation": "max_deviation",
        },
    ),
    ("SignalComparisonEvents", "deviation_statistics"): _r(
        "engineering.signal_comparison.deviation_statistics",
        _E,
        S,
        objs=("signal",),
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("SignalComparisonEvents", "tracking_error_trend"): _r(
        "engineering.signal_comparison.tracking_error_trend",
        _E,
        S,
        objs=("signal",),
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("StartupDetectionEvents", "assess_startup_quality"): _r(
        "engineering.startup.quality_assessment",
        _E,
        S,
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("StartupDetectionEvents", "detect_failed_startups"): _r(
        "engineering.startup.failed",
        _E,
        P,
        standard_attrs={"ts_shape:outcome": "failure_reason"},
    ),
    ("StartupDetectionEvents", "detect_startup_adaptive"): _r(
        "engineering.startup.adaptive",
        _E,
        I,
        standard_attrs={"ts_shape:lifecycle_state": "starting"},
    ),
    ("StartupDetectionEvents", "detect_startup_by_slope"): _r(
        "engineering.startup.by_slope",
        _E,
        I,
        standard_attrs={"ts_shape:lifecycle_state": "starting"},
    ),
    ("StartupDetectionEvents", "detect_startup_by_threshold"): _r(
        "engineering.startup.by_threshold",
        _E,
        I,
        standard_attrs={"ts_shape:lifecycle_state": "starting"},
    ),
    ("StartupDetectionEvents", "detect_startup_multi_signal"): _r(
        "engineering.startup.multi_signal",
        _E,
        I,
        standard_attrs={"ts_shape:lifecycle_state": "starting"},
    ),
    ("StartupDetectionEvents", "track_startup_phases"): _r(
        "engineering.startup.phase_{phase}",
        _E,
        I,
        standard_attrs={"ts_shape:lifecycle_state": "phase"},
    ),
    ("SteadyStateDetectionEvents", "detect_steady_state"): _r(
        "engineering.steady_state.detected",
        _E,
        I,
        standard_attrs={"ts_shape:lifecycle_state": "steady"},
    ),
    ("SteadyStateDetectionEvents", "detect_transient_periods"): _r(
        "engineering.steady_state.transient",
        _E,
        I,
        standard_attrs={"ts_shape:lifecycle_state": "transient"},
    ),
    ("SteadyStateDetectionEvents", "steady_state_value_bands"): _r(
        "engineering.steady_state.value_bands",
        _E,
        S,
        standard_attrs={
            "ts_shape:sample_count": None,
            "ts_shape:threshold_low": "lower_band",
            "ts_shape:threshold_high": "upper_band",
        },
    ),
    ("ThresholdMonitoringEvents", "multi_level_threshold"): _r(
        "engineering.threshold.multi_level",
        _E,
        P,
        standard_attrs={
            "ts_shape:method": "multi_level_threshold",
            "ts_shape:direction": "above",
        },
    ),
    ("ThresholdMonitoringEvents", "threshold_exceedance_trend"): _r(
        "engineering.threshold.exceedance_trend",
        _E,
        S,
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("ThresholdMonitoringEvents", "threshold_with_hysteresis"): _r(
        "engineering.threshold.hysteresis",
        _E,
        P,
        standard_attrs={
            "ts_shape:method": "hysteresis",
            "ts_shape:direction": "outside",
        },
    ),
    ("ThresholdMonitoringEvents", "time_above_threshold"): _r(
        "engineering.threshold.time_above",
        _E,
        S,
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("WarmUpCoolDownEvents", "detect_cooldown"): _r(
        "engineering.thermal.cooldown",
        _E,
        I,
        standard_attrs={"ts_shape:lifecycle_state": "cooldown"},
    ),
    ("WarmUpCoolDownEvents", "detect_warmup"): _r(
        "engineering.thermal.warmup",
        _E,
        I,
        standard_attrs={"ts_shape:lifecycle_state": "warmup"},
    ),
    ("WarmUpCoolDownEvents", "time_to_target"): _r(
        "engineering.thermal.time_to_target",
        _E,
        S,
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("WarmUpCoolDownEvents", "warmup_consistency"): _r(
        "engineering.thermal.warmup_consistency",
        _E,
        S,
        standard_attrs={"ts_shape:sample_count": None},
    ),
    # ---- maintenance -------------------------------------------------------
    ("DegradationDetectionEvents", "detect_level_shift"): _r(
        "maintenance.degradation.level_shift",
        _M,
        P,
        standard_attrs={
            "ts_shape:method": "level_shift",
            "ts_shape:direction": "shift",
            "ts_shape:baseline": "prev_mean",
            "ts_shape:deviation": "shift_magnitude",
        },
    ),
    ("DegradationDetectionEvents", "detect_trend_degradation"): _r(
        "maintenance.degradation.trend",
        _M,
        I,
        standard_attrs={
            "ts_shape:lifecycle_state": "degrading",
            "ts_shape:direction": "trend",
        },
    ),
    ("DegradationDetectionEvents", "detect_variance_increase"): _r(
        "maintenance.degradation.variance_increase",
        _M,
        I,
        standard_attrs={
            "ts_shape:lifecycle_state": "variance_increase",
            "ts_shape:baseline": "baseline_variance",
        },
    ),
    ("DegradationDetectionEvents", "health_score"): _r(
        "maintenance.health.score_window",
        _M,
        S,
        standard_attrs={
            "ts_shape:sample_count": None,  # health_score has no count column
        },
    ),
    ("FailurePredictionEvents", "detect_exceedance_pattern"): _r(
        "maintenance.failure.exceedance_pattern",
        _M,
        P,
        standard_attrs={
            "ts_shape:method": "exceedance_pattern",
            "ts_shape:direction": "above",
            "ts_shape:sample_count": "warning_count",
        },
    ),
    ("FailurePredictionEvents", "remaining_useful_life"): _r(
        "maintenance.failure.remaining_useful_life",
        _M,
        S,
        standard_attrs={
            "ts_shape:method": "remaining_useful_life",
            "ts_shape:confidence": "confidence",
            "ts_shape:threshold_low": "rul_seconds",
        },
    ),
    ("FailurePredictionEvents", "time_to_threshold"): _r(
        "maintenance.failure.time_to_threshold",
        _M,
        S,
        standard_attrs={
            "ts_shape:method": "time_to_threshold",
            "ts_shape:confidence": 1.0,
            "ts_shape:threshold_low": "estimated_time_seconds",
        },
    ),
    ("VibrationAnalysisEvents", "bearing_health_indicators"): _r(
        "maintenance.vibration.bearing_health",
        _M,
        S,
        standard_attrs={
            "ts_shape:sample_count": None,
        },
    ),
    ("VibrationAnalysisEvents", "detect_amplitude_growth"): _r(
        "maintenance.vibration.amplitude_growth",
        _M,
        P,
        standard_attrs={
            "ts_shape:method": "amplitude_growth",
            "ts_shape:direction": "up",
            "ts_shape:baseline": "baseline_amplitude",
            "ts_shape:deviation_pct": "growth_pct",
        },
    ),
    ("VibrationAnalysisEvents", "detect_rms_exceedance"): _r(
        "maintenance.vibration.rms_exceedance",
        _M,
        P,
        standard_attrs={
            "ts_shape:method": "rms_exceedance",
            "ts_shape:direction": "above",
            "ts_shape:baseline": "baseline_rms",
        },
    ),
    # ---- production --------------------------------------------------------
    ("AlarmManagementEvents", "alarm_duration_stats"): _r(
        "production.alarm.duration_stats",
        _PR,
        S,
        standard_attrs={"ts_shape:sample_count": "count"},
    ),
    ("AlarmManagementEvents", "alarm_frequency"): _r(
        "production.alarm.frequency",
        _PR,
        S,
        standard_attrs={"ts_shape:sample_count": "count"},
    ),
    ("AlarmManagementEvents", "chattering_detection"): _r(
        "production.alarm.chattering",
        _PR,
        I,
        standard_attrs={"ts_shape:lifecycle_state": "chattering"},
    ),
    ("AlarmManagementEvents", "standing_alarms"): _r(
        "production.alarm.standing",
        _PR,
        I,
        standard_attrs={"ts_shape:lifecycle_state": "standing"},
    ),
    ("BatchTrackingEvents", "batch_duration_stats"): _r(
        "production.batch.duration_stats",
        _PR,
        S,
        objs=("asset", "batch"),
        standard_attrs={"ts_shape:sample_count": "count"},
    ),
    ("BatchTrackingEvents", "batch_transition_matrix"): _r(
        "production.batch.transition_matrix",
        _PR,
        ST,
        objs=("asset", "batch"),
        standard_attrs={"ts_shape:method": "transition_matrix"},
    ),
    ("BatchTrackingEvents", "batch_yield"): _r(
        "production.batch.yield",
        _PR,
        S,
        objs=("asset", "batch"),
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("BatchTrackingEvents", "detect_batches"): _r(
        "production.batch.detected",
        _PR,
        I,
        objs=("asset", "batch"),
        standard_attrs={"ts_shape:lifecycle_state": "running"},
    ),
    ("BottleneckDetectionEvents", "detect_bottleneck"): _r(
        "production.bottleneck.detected",
        _PR,
        P,
        objs=("asset", "station"),
        standard_attrs={"ts_shape:outcome": "bottleneck_station"},
    ),
    ("BottleneckDetectionEvents", "shifting_bottleneck"): _r(
        "production.bottleneck.shifting",
        _PR,
        S,
        objs=("asset", "station"),
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("BottleneckDetectionEvents", "station_utilization"): _r(
        "production.bottleneck.station_utilization",
        _PR,
        S,
        objs=("asset", "station"),
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("ChangeoverEvents", "changeover_quality_metrics"): _r(
        "production.changeover.quality_metrics",
        _PR,
        S,
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("ChangeoverEvents", "changeover_window"): _r(
        "production.changeover.window",
        _PR,
        I,
        standard_attrs={"ts_shape:lifecycle_state": "changeover"},
    ),
    ("ChangeoverEvents", "detect_changeover"): _r(
        "production.changeover.detected",
        _PR,
        I,
        standard_attrs={"ts_shape:lifecycle_state": "changeover"},
    ),
    ("ContinuousProcessAlignmentEvents", "align_to_reference"): _r(
        "production.alignment.to_reference",
        _PR,
        S,
        objs=("asset", "station"),
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("ContinuousProcessAlignmentEvents", "alignment_quality"): _r(
        "production.alignment.quality",
        _PR,
        S,
        objs=("asset", "station"),
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("ContinuousProcessAlignmentEvents", "lag_profile"): _r(
        "production.alignment.lag_profile",
        _PR,
        S,
        objs=("asset", "station"),
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("ContinuousProcessAlignmentEvents", "segment_by_cut"): _r(
        "production.alignment.segment_by_cut",
        _PR,
        I,
        objs=("asset", "station"),
        standard_attrs={"ts_shape:lifecycle_state": "segment"},
    ),
    ("CycleTimeTracking", "cycle_time_by_part"): _r(
        "production.cycle_time.by_part",
        _PR,
        S,
        objs=("asset", "part"),
        standard_attrs={"ts_shape:sample_count": "count"},
    ),
    ("CycleTimeTracking", "cycle_time_statistics"): _r(
        "production.cycle_time.statistics",
        _PR,
        S,
        standard_attrs={"ts_shape:sample_count": "count"},
    ),
    ("CycleTimeTracking", "cycle_time_trend"): _r(
        "production.cycle_time.trend",
        _PR,
        S,
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("CycleTimeTracking", "detect_slow_cycles"): _r(
        "production.cycle_time.slow",
        _PR,
        P,
        objs=("asset", "cycle"),
        standard_attrs={
            "ts_shape:method": "slow_cycle",
            "ts_shape:direction": "above",
        },
    ),
    ("CycleTimeTracking", "hourly_cycle_time_summary"): _r(
        "production.cycle_time.hourly_summary",
        _PR,
        S,
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("DowntimeTracking", "availability_trend"): _r(
        "production.downtime.availability_trend",
        _PR,
        S,
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("DowntimeTracking", "downtime_by_reason"): _r(
        "production.downtime.by_reason",
        _PR,
        S,
        standard_attrs={
            "ts_shape:outcome": "reason",
            "ts_shape:sample_count": "occurrences",
        },
    ),
    ("DowntimeTracking", "downtime_by_shift"): _r(
        "production.downtime.by_shift",
        _PR,
        S,
        objs=("asset", "shift"),
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("DowntimeTracking", "top_downtime_reasons"): _r(
        "production.downtime.top_reasons",
        _PR,
        ST,
        standard_attrs={"ts_shape:method": "top_reasons"},
    ),
    ("DutyCycleEvents", "cycle_count"): _r(
        "production.duty_cycle.count",
        _PR,
        S,
        standard_attrs={"ts_shape:sample_count": "count"},
    ),
    ("DutyCycleEvents", "duty_cycle_per_window"): _r(
        "production.duty_cycle.per_window",
        _PR,
        S,
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("DutyCycleEvents", "excessive_cycling"): _r(
        "production.duty_cycle.excessive",
        _PR,
        P,
        standard_attrs={
            "ts_shape:method": "excessive_cycling",
            "ts_shape:direction": "above",
        },
    ),
    ("DutyCycleEvents", "on_off_intervals"): _r(
        "production.duty_cycle.on_off",
        _PR,
        I,
        standard_attrs={"ts_shape:lifecycle_state": "state"},
    ),
    ("FlowConstraintEvents", "blocked_events"): _r(
        "production.flow.blocked",
        _PR,
        I,
        objs=("asset", "station"),
        standard_attrs={"ts_shape:lifecycle_state": "blocked"},
    ),
    ("FlowConstraintEvents", "starved_events"): _r(
        "production.flow.starved",
        _PR,
        I,
        objs=("asset", "station"),
        standard_attrs={"ts_shape:lifecycle_state": "starved"},
    ),
    ("FlowMetricsEvents", "flow_summary"): _r(
        "production.flow.summary",
        _PR,
        S,
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("FlowMetricsEvents", "lead_time"): _r(
        "production.flow.lead_time",
        _PR,
        P,
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("FlowMetricsEvents", "throughput"): _r(
        "production.flow.throughput",
        _PR,
        S,
        standard_attrs={"ts_shape:sample_count": "units_out"},
    ),
    ("FlowMetricsEvents", "wip_over_time"): _r(
        "production.flow.wip",
        _PR,
        S,
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("LineBalancingEvents", "balance_metrics"): _r(
        "production.line_balance.metrics",
        _PR,
        S,
        objs=("asset", "station"),
        standard_attrs={"ts_shape:sample_count": "n_stations"},
    ),
    ("LineBalancingEvents", "station_cycle_times"): _r(
        "production.line_balance.station_cycle_times",
        _PR,
        S,
        objs=("asset", "station"),
        standard_attrs={"ts_shape:sample_count": "cycle_count"},
    ),
    ("LineBalancingEvents", "yamazumi"): _r(
        "production.line_balance.yamazumi",
        _PR,
        ST,
        objs=("asset", "station"),
        standard_attrs={"ts_shape:method": "yamazumi"},
    ),
    ("LineThroughputEvents", "count_parts"): _r(
        "production.throughput.count_parts",
        _PR,
        S,
        standard_attrs={"ts_shape:sample_count": "count"},
    ),
    ("LineThroughputEvents", "cycle_quality_check"): _r(
        "production.throughput.cycle_quality_check",
        _PR,
        P,
        objs=("asset", "cycle"),
        standard_attrs={"ts_shape:outcome": "quality_status"},
    ),
    ("LineThroughputEvents", "takt_adherence"): _r(
        "production.throughput.takt_adherence",
        _PR,
        S,
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("LineThroughputEvents", "throughput_oee"): _r(
        "production.throughput.oee",
        _PR,
        S,
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("LineThroughputEvents", "throughput_trends"): _r(
        "production.throughput.trends",
        _PR,
        S,
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("LongDowntimeEvents", "count_events_between_gaps"): _r(
        "production.long_downtime.events_between_gaps",
        _PR,
        S,
        standard_attrs={"ts_shape:sample_count": "count"},
    ),
    ("LongDowntimeEvents", "detect_long_downtime"): _r(
        "production.long_downtime.detected",
        _PR,
        I,
        standard_attrs={"ts_shape:lifecycle_state": "down"},
    ),
    ("MachineStateEvents", "detect_rapid_transitions"): _r(
        "production.machine_state.rapid_transitions",
        _PR,
        P,
        standard_attrs={
            "ts_shape:method": "rapid_transitions",
            "ts_shape:direction": "above",
        },
    ),
    ("MachineStateEvents", "detect_run_idle"): _r(
        "production.machine_state.{state}",
        _PR,
        I,
        standard_attrs={"ts_shape:lifecycle_state": "state"},
    ),
    ("MachineStateEvents", "transition_events"): _r(
        "production.machine_state.transition_{transition}",
        _PR,
        P,
        standard_attrs={"ts_shape:outcome": "transition"},
    ),
    ("MicroStopEvents", "detect_micro_stops"): _r(
        "production.micro_stop.detected",
        _PR,
        I,
        standard_attrs={"ts_shape:lifecycle_state": "micro_stopped"},
    ),
    ("MicroStopEvents", "micro_stop_frequency"): _r(
        "production.micro_stop.frequency",
        _PR,
        S,
        standard_attrs={"ts_shape:sample_count": "count"},
    ),
    ("MicroStopEvents", "micro_stop_impact"): _r(
        "production.micro_stop.impact",
        _PR,
        S,
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("MicroStopEvents", "micro_stop_patterns"): _r(
        "production.micro_stop.patterns",
        _PR,
        S,
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("MultiProcessTraceabilityEvents", "build_timeline"): _r(
        "production.traceability.timeline",
        _PR,
        P,
        objs=("asset", "serial", "station"),
        standard_attrs={
            "ts_shape:lifecycle_state": "station_name",
            "ts_shape:direction": "station_sequence",
        },
    ),
    ("MultiProcessTraceabilityEvents", "handover_log"): _r(
        "production.traceability.handover",
        _PR,
        P,
        objs=("asset", "serial", "station"),
        standard_attrs={
            "ts_shape:lifecycle_state": "handover",
            "ts_shape:direction": "station_sequence",
        },
    ),
    ("MultiProcessTraceabilityEvents", "lead_time"): _r(
        "production.traceability.lead_time",
        _PR,
        S,
        objs=("serial",),
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("MultiProcessTraceabilityEvents", "parallel_activity"): _r(
        "production.traceability.parallel_activity",
        _PR,
        I,
        objs=("serial", "station"),
        standard_attrs={
            "ts_shape:lifecycle_state": "parallel",
            "ts_shape:direction": "station_name",
        },
    ),
    ("MultiProcessTraceabilityEvents", "routing_paths"): _r(
        "production.traceability.routing_paths",
        _PR,
        ST,
        objs=("serial",),
        standard_attrs={"ts_shape:method": "routing_paths"},
    ),
    ("MultiProcessTraceabilityEvents", "station_statistics"): _r(
        "production.traceability.station_statistics",
        _PR,
        S,
        objs=("station",),
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("OEECalculator", "calculate_availability"): _r(
        "production.oee.availability",
        _PR,
        S,
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("OEECalculator", "calculate_oee"): _r(
        "production.oee.total", _PR, S, standard_attrs={"ts_shape:sample_count": None}
    ),
    ("OEECalculator", "calculate_performance"): _r(
        "production.oee.performance",
        _PR,
        S,
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("OEECalculator", "calculate_quality"): _r(
        "production.oee.quality", _PR, S, standard_attrs={"ts_shape:sample_count": None}
    ),
    ("OperatorPerformanceTracking", "operator_comparison"): _r(
        "production.operator.comparison",
        _PR,
        S,
        objs=("operator",),
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("OperatorPerformanceTracking", "operator_efficiency"): _r(
        "production.operator.efficiency",
        _PR,
        S,
        objs=("operator",),
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("OperatorPerformanceTracking", "production_by_operator"): _r(
        "production.operator.production",
        _PR,
        S,
        objs=("operator",),
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("OperatorPerformanceTracking", "quality_by_operator"): _r(
        "production.operator.quality",
        _PR,
        S,
        objs=("operator",),
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("OrderTraceabilityEvents", "build_timeline"): _r(
        "production.order.timeline",
        _PR,
        P,
        objs=("asset", "work_order", "station"),
        standard_attrs={
            "ts_shape:lifecycle_state": "station_name",
            "ts_shape:direction": "station_sequence",
        },
    ),
    ("OrderTraceabilityEvents", "current_status"): _r(
        "production.order.current_status",
        _PR,
        ST,
        objs=("work_order",),
        standard_attrs={"ts_shape:method": "current_status"},
    ),
    ("OrderTraceabilityEvents", "lead_time"): _r(
        "production.order.lead_time",
        _PR,
        S,
        objs=("work_order",),
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("OrderTraceabilityEvents", "station_dwell_statistics"): _r(
        "production.order.station_dwell_statistics",
        _PR,
        S,
        objs=("station",),
        standard_attrs={"ts_shape:sample_count": "count"},
    ),
    ("ValueTraceabilityEvents", "build_timeline"): _r(
        "production.value_trace.timeline",
        _PR,
        P,
        objs=("asset", "serial", "station"),
        standard_attrs={
            "ts_shape:lifecycle_state": "station_name",
            "ts_shape:direction": "station_sequence",
        },
    ),
    ("ValueTraceabilityEvents", "current_status"): _r(
        "production.value_trace.current_status",
        _PR,
        ST,
        objs=("serial",),
        standard_attrs={"ts_shape:method": "current_status"},
    ),
    ("ValueTraceabilityEvents", "lead_time"): _r(
        "production.value_trace.lead_time",
        _PR,
        S,
        objs=("serial",),
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("ValueTraceabilityEvents", "station_dwell_statistics"): _r(
        "production.value_trace.station_dwell_statistics",
        _PR,
        S,
        objs=("station",),
        standard_attrs={"ts_shape:sample_count": "count"},
    ),
    ("PartProductionTracking", "daily_production_summary"): _r(
        "production.part.daily_summary",
        _PR,
        S,
        objs=("asset", "part"),
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("PartProductionTracking", "production_by_part"): _r(
        "production.part.production",
        _PR,
        S,
        objs=("asset", "part"),
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("PartProductionTracking", "production_totals"): _r(
        "production.part.totals",
        _PR,
        S,
        objs=("asset", "part"),
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("PerformanceLossTracking", "performance_by_shift"): _r(
        "production.performance.by_shift",
        _PR,
        S,
        objs=("asset", "shift"),
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("PerformanceLossTracking", "performance_trend"): _r(
        "production.performance.trend",
        _PR,
        S,
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("PerformanceLossTracking", "slow_periods"): _r(
        "production.performance.slow_period",
        _PR,
        I,
        standard_attrs={"ts_shape:lifecycle_state": "slow"},
    ),
    ("PeriodSummary", "compare_periods"): _r(
        "production.period.compare",
        _PR,
        S,
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("PeriodSummary", "from_daily_data"): _r(
        "production.period.from_daily",
        _PR,
        S,
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("PeriodSummary", "monthly_summary"): _r(
        "production.period.monthly_summary",
        _PR,
        S,
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("PeriodSummary", "weekly_summary"): _r(
        "production.period.weekly_summary",
        _PR,
        S,
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("QualityTracking", "daily_quality_summary"): _r(
        "production.quality.daily_summary",
        _PR,
        S,
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("QualityTracking", "nok_by_reason"): _r(
        "production.quality.nok_by_reason",
        _PR,
        S,
        standard_attrs={"ts_shape:outcome": "reason", "ts_shape:sample_count": "count"},
    ),
    ("QualityTracking", "nok_by_shift"): _r(
        "production.quality.nok_by_shift",
        _PR,
        S,
        objs=("asset", "shift"),
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("QualityTracking", "quality_by_part"): _r(
        "production.quality.by_part",
        _PR,
        S,
        objs=("asset", "part"),
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("ReworkTracking", "rework_by_reason"): _r(
        "production.rework.by_reason",
        _PR,
        S,
        standard_attrs={"ts_shape:outcome": "reason", "ts_shape:sample_count": "count"},
    ),
    ("ReworkTracking", "rework_by_shift"): _r(
        "production.rework.by_shift",
        _PR,
        S,
        objs=("asset", "shift"),
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("ReworkTracking", "rework_cost"): _r(
        "production.rework.cost", _PR, S, standard_attrs={"ts_shape:sample_count": None}
    ),
    ("ReworkTracking", "rework_rate"): _r(
        "production.rework.rate", _PR, S, standard_attrs={"ts_shape:sample_count": None}
    ),
    ("ReworkTracking", "rework_trend"): _r(
        "production.rework.trend",
        _PR,
        S,
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("RoutingTraceabilityEvents", "build_routing_timeline"): _r(
        "production.routing.timeline",
        _PR,
        P,
        objs=("asset", "serial", "station"),
        standard_attrs={
            "ts_shape:lifecycle_state": "station_name",
            "ts_shape:direction": "station_sequence",
        },
    ),
    ("RoutingTraceabilityEvents", "lead_time"): _r(
        "production.routing.lead_time",
        _PR,
        S,
        objs=("serial",),
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("RoutingTraceabilityEvents", "routing_paths"): _r(
        "production.routing.paths",
        _PR,
        ST,
        objs=("serial",),
        standard_attrs={"ts_shape:method": "routing_paths"},
    ),
    ("RoutingTraceabilityEvents", "station_statistics"): _r(
        "production.routing.station_statistics",
        _PR,
        S,
        objs=("station",),
        standard_attrs={"ts_shape:sample_count": "sample_count"},
    ),
    ("RuntimeAccountingEvents", "operating_hours_meter"): _r(
        "production.runtime.hours_meter",
        _PR,
        S,
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("RuntimeAccountingEvents", "runtime_per_window"): _r(
        "production.runtime.per_window",
        _PR,
        S,
        standard_attrs={"ts_shape:sample_count": "start_count"},
    ),
    ("RuntimeAccountingEvents", "runtime_summary"): _r(
        "production.runtime.summary",
        _PR,
        S,
        standard_attrs={"ts_shape:sample_count": "start_count"},
    ),
    ("ScrapTracking", "scrap_by_reason"): _r(
        "production.scrap.by_reason",
        _PR,
        S,
        standard_attrs={"ts_shape:outcome": "reason", "ts_shape:sample_count": "count"},
    ),
    ("ScrapTracking", "scrap_by_shift"): _r(
        "production.scrap.by_shift",
        _PR,
        S,
        objs=("asset", "shift"),
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("ScrapTracking", "scrap_cost"): _r(
        "production.scrap.cost", _PR, S, standard_attrs={"ts_shape:sample_count": None}
    ),
    ("ScrapTracking", "scrap_trend"): _r(
        "production.scrap.trend", _PR, S, standard_attrs={"ts_shape:sample_count": None}
    ),
    ("SetupTimeTracking", "setup_by_product"): _r(
        "production.setup.by_product",
        _PR,
        S,
        objs=("asset", "part"),
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("SetupTimeTracking", "setup_durations"): _r(
        "production.setup.durations",
        _PR,
        I,
        standard_attrs={"ts_shape:lifecycle_state": "setup"},
    ),
    ("SetupTimeTracking", "setup_statistics"): _r(
        "production.setup.statistics",
        _PR,
        S,
        standard_attrs={"ts_shape:sample_count": "count"},
    ),
    ("SetupTimeTracking", "setup_trend"): _r(
        "production.setup.trend", _PR, S, standard_attrs={"ts_shape:sample_count": None}
    ),
    ("ShiftHandoverReport", "from_shift_data"): _r(
        "production.shift.handover_from_data",
        _PR,
        S,
        objs=("asset", "shift"),
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("ShiftHandoverReport", "generate_report"): _r(
        "production.shift.handover_report",
        _PR,
        S,
        objs=("asset", "shift"),
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("ShiftReporting", "best_and_worst_shifts"): _r(
        "production.shift.best_and_worst",
        _PR,
        S,
        objs=("asset", "shift"),
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("ShiftReporting", "shift_comparison"): _r(
        "production.shift.comparison",
        _PR,
        S,
        objs=("asset", "shift"),
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("ShiftReporting", "shift_production"): _r(
        "production.shift.production",
        _PR,
        S,
        objs=("asset", "shift"),
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("ShiftReporting", "shift_targets"): _r(
        "production.shift.targets",
        _PR,
        S,
        objs=("asset", "shift"),
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("TargetTracking", "compare_to_target"): _r(
        "production.target.compare",
        _PR,
        S,
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("TargetTracking", "target_achievement_summary"): _r(
        "production.target.achievement_summary",
        _PR,
        S,
        standard_attrs={"ts_shape:sample_count": None},
    ),
    # ---- quality -----------------------------------------------------------
    ("AnomalyClassificationEvents", "classify_anomalies"): _r(
        "quality.anomaly.classified_{anomaly_class}",
        _Q,
        P,
        standard_attrs={"ts_shape:outcome": "anomaly_type"},
    ),
    ("AnomalyClassificationEvents", "detect_drift"): _r(
        "quality.anomaly.drift",
        _Q,
        P,
        standard_attrs={
            "ts_shape:method": "drift",
            "ts_shape:direction": "direction",
        },
    ),
    ("AnomalyClassificationEvents", "detect_flatline"): _r(
        "quality.anomaly.flatline",
        _Q,
        I,
        standard_attrs={"ts_shape:lifecycle_state": "flatline"},
    ),
    ("AnomalyClassificationEvents", "detect_oscillation"): _r(
        "quality.anomaly.oscillation",
        _Q,
        I,
        standard_attrs={"ts_shape:lifecycle_state": "oscillation"},
    ),
    ("CapabilityTrendingEvents", "capability_forecast"): _r(
        "quality.capability.forecast",
        _Q,
        S,
        standard_attrs={
            "ts_shape:method": "capability_forecast",
            "ts_shape:confidence": 1.0,
            "ts_shape:threshold_low": "windows_to_threshold",
        },
    ),
    ("CapabilityTrendingEvents", "capability_over_time"): _r(
        "quality.capability.over_time",
        _Q,
        S,
        standard_attrs={"ts_shape:sample_count": "n_samples"},
    ),
    ("CapabilityTrendingEvents", "detect_capability_drop"): _r(
        "quality.capability.drop",
        _Q,
        P,
        standard_attrs={
            "ts_shape:method": "capability_drop",
            "ts_shape:direction": "down",
            "ts_shape:baseline": "prev_avg_cpk",
            "ts_shape:deviation_pct": "drop_pct",
        },
    ),
    ("CapabilityTrendingEvents", "yield_estimate"): _r(
        "quality.capability.yield_estimate",
        _Q,
        S,
        standard_attrs={
            "ts_shape:method": "yield_estimate",
            "ts_shape:confidence": 1.0,
        },
    ),
    ("DataGapAnalysisEvents", "coverage_by_period"): _r(
        "quality.data_gap.coverage_by_period",
        _Q,
        S,
        objs=("signal",),
        standard_attrs={"ts_shape:sample_count": "sample_count"},
    ),
    ("DataGapAnalysisEvents", "find_gaps"): _r(
        "quality.data_gap.gap",
        _Q,
        I,
        objs=("signal",),
        standard_attrs={"ts_shape:lifecycle_state": "gap"},
    ),
    ("DataGapAnalysisEvents", "gap_summary"): _r(
        "quality.data_gap.summary",
        _Q,
        S,
        objs=("signal",),
        standard_attrs={"ts_shape:sample_count": "total_gaps"},
    ),
    ("DataGapAnalysisEvents", "interpolation_candidates"): _r(
        "quality.data_gap.interpolation_candidate",
        _Q,
        P,
        objs=("signal",),
        standard_attrs={"ts_shape:outcome": "safe_to_interpolate"},
    ),
    ("GaugeRepeatabilityEvents", "gauge_rr_summary"): _r(
        "quality.gauge_rr.summary",
        _Q,
        ST,
        objs=("tool",),
        standard_attrs={"ts_shape:method": "gauge_rr"},
    ),
    ("GaugeRepeatabilityEvents", "measurement_bias"): _r(
        "quality.gauge_rr.bias",
        _Q,
        ST,
        objs=("tool",),
        standard_attrs={"ts_shape:method": "bias"},
    ),
    ("GaugeRepeatabilityEvents", "repeatability"): _r(
        "quality.gauge_rr.repeatability",
        _Q,
        ST,
        objs=("tool",),
        standard_attrs={"ts_shape:method": "repeatability"},
    ),
    ("GaugeRepeatabilityEvents", "reproducibility"): _r(
        "quality.gauge_rr.reproducibility",
        _Q,
        ST,
        objs=("tool",),
        standard_attrs={"ts_shape:method": "reproducibility"},
    ),
    ("MultiSensorValidationEvents", "consensus_score"): _r(
        "quality.multi_sensor.consensus_score",
        _Q,
        S,
        objs=("sensor",),
        standard_attrs={
            "ts_shape:sample_count": None,
            "ts_shape:confidence": "consensus_score",
        },
    ),
    ("MultiSensorValidationEvents", "detect_disagreement"): _r(
        "quality.multi_sensor.disagreement",
        _Q,
        P,
        objs=("sensor",),
        standard_attrs={
            "ts_shape:method": "disagreement",
            "ts_shape:direction": "spread",
            "ts_shape:deviation": "max_spread",
        },
    ),
    ("MultiSensorValidationEvents", "identify_outlier_sensor"): _r(
        "quality.multi_sensor.outlier_sensor",
        _Q,
        ST,
        objs=("sensor",),
        standard_attrs={"ts_shape:method": "outlier_sensor"},
    ),
    ("MultiSensorValidationEvents", "pairwise_bias"): _r(
        "quality.multi_sensor.pairwise_bias",
        _Q,
        ST,
        objs=("sensor",),
        standard_attrs={"ts_shape:method": "pairwise_bias"},
    ),
    ("OutlierDetectionEvents", "detect_outliers_iqr"): _r(
        "quality.outlier.iqr",
        _Q,
        P,
        severity_field="severity",
        standard_attrs={
            "ts_shape:method": "iqr",
            "ts_shape:direction": "outside",
        },
    ),
    ("OutlierDetectionEvents", "detect_outliers_isolation_forest"): _r(
        "quality.outlier.isolation_forest",
        _Q,
        P,
        severity_field="severity",
        standard_attrs={
            "ts_shape:method": "isolation_forest",
            "ts_shape:direction": "outside",
            "ts_shape:confidence": "severity",
        },
    ),
    ("OutlierDetectionEvents", "detect_outliers_mad"): _r(
        "quality.outlier.mad",
        _Q,
        P,
        severity_field="severity",
        standard_attrs={
            "ts_shape:method": "mad",
            "ts_shape:direction": "outside",
        },
    ),
    ("OutlierDetectionEvents", "detect_outliers_zscore"): _r(
        "quality.outlier.zscore",
        _Q,
        P,
        severity_field="severity",
        standard_attrs={
            "ts_shape:method": "zscore",
            "ts_shape:direction": "outside",
        },
    ),
    ("SensorDriftEvents", "calibration_health"): _r(
        "quality.sensor_drift.calibration_health",
        _Q,
        S,
        objs=("sensor",),
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("SensorDriftEvents", "detect_span_drift"): _r(
        "quality.sensor_drift.span_drift",
        _Q,
        P,
        objs=("sensor",),
        standard_attrs={
            "ts_shape:method": "span_drift",
            "ts_shape:direction": "drift",
            "ts_shape:deviation_pct": "sensitivity_change_pct",
        },
    ),
    ("SensorDriftEvents", "detect_zero_drift"): _r(
        "quality.sensor_drift.zero_drift",
        _Q,
        P,
        objs=("sensor",),
        standard_attrs={
            "ts_shape:method": "zero_drift",
            "ts_shape:direction": "drift",
            "ts_shape:deviation": "mean_offset",
        },
    ),
    ("SensorDriftEvents", "drift_trend"): _r(
        "quality.sensor_drift.trend",
        _Q,
        S,
        objs=("sensor",),
        standard_attrs={
            "ts_shape:sample_count": None,
            "ts_shape:confidence": "r_squared",
        },
    ),
    ("SignalQualityEvents", "data_completeness"): _r(
        "quality.signal.completeness",
        _Q,
        S,
        objs=("signal",),
        standard_attrs={"ts_shape:sample_count": "actual_count"},
    ),
    ("SignalQualityEvents", "detect_missing_data"): _r(
        "quality.signal.missing",
        _Q,
        I,
        objs=("signal",),
        standard_attrs={"ts_shape:lifecycle_state": "missing"},
    ),
    ("SignalQualityEvents", "detect_out_of_range"): _r(
        "quality.signal.out_of_range",
        _Q,
        P,
        objs=("signal",),
        standard_attrs={
            "ts_shape:method": "out_of_range",
            "ts_shape:direction": "direction",
            "ts_shape:threshold_low": "min_observed",
            "ts_shape:threshold_high": "max_observed",
        },
    ),
    ("SignalQualityEvents", "sampling_regularity"): _r(
        "quality.signal.sampling_regularity",
        _Q,
        S,
        objs=("signal",),
        standard_attrs={
            "ts_shape:sample_count": None,
            "ts_shape:confidence": "regularity_score",
        },
    ),
    ("StatisticalProcessControlRuleBased", "apply_rules_vectorized"): _r(
        "quality.spc.rule_violation",
        _Q,
        P,
        standard_attrs={
            "ts_shape:method": "western_electric",
            "ts_shape:direction": "outside",
        },
    ),
    ("StatisticalProcessControlRuleBased", "calculate_control_limits"): _r(
        "quality.spc.control_limits",
        _Q,
        ST,
        standard_attrs={
            "ts_shape:method": "control_limits",
            "ts_shape:baseline": "mean",
            "ts_shape:threshold_low": "3sigma_lower",
            "ts_shape:threshold_high": "3sigma_upper",
        },
    ),
    ("StatisticalProcessControlRuleBased", "calculate_dynamic_control_limits"): _r(
        "quality.spc.control_limits_dynamic",
        _Q,
        S,
        standard_attrs={
            "ts_shape:sample_count": None,
            "ts_shape:baseline": "mean",
            "ts_shape:threshold_low": "3sigma_lower",
            "ts_shape:threshold_high": "3sigma_upper",
        },
    ),
    ("StatisticalProcessControlRuleBased", "detect_cusum_shifts"): _r(
        "quality.spc.cusum_shift",
        _Q,
        P,
        standard_attrs={
            "ts_shape:method": "cusum",
            "ts_shape:direction": "shift",
        },
    ),
    ("StatisticalProcessControlRuleBased", "interpret_violations"): _r(
        "quality.spc.violation_interpretation",
        _Q,
        P,
        standard_attrs={"ts_shape:outcome": "rule"},
    ),
    ("StatisticalProcessControlRuleBased", "process"): _r(
        "quality.spc.rule_violation",
        _Q,
        P,
        standard_attrs={
            "ts_shape:method": "western_electric",
            "ts_shape:direction": "outside",
        },
    ),
    ("StatisticalProcessControlRuleBased", "rule_1"): _r(
        "quality.spc.rule_1",
        _Q,
        P,
        standard_attrs={
            "ts_shape:method": "western_electric_rule_1",
            "ts_shape:direction": "outside",
        },
    ),
    ("StatisticalProcessControlRuleBased", "rule_2"): _r(
        "quality.spc.rule_2",
        _Q,
        P,
        standard_attrs={
            "ts_shape:method": "western_electric_rule_2",
            "ts_shape:direction": "outside",
        },
    ),
    ("StatisticalProcessControlRuleBased", "rule_3"): _r(
        "quality.spc.rule_3",
        _Q,
        P,
        standard_attrs={
            "ts_shape:method": "western_electric_rule_3",
            "ts_shape:direction": "outside",
        },
    ),
    ("StatisticalProcessControlRuleBased", "rule_4"): _r(
        "quality.spc.rule_4",
        _Q,
        P,
        standard_attrs={
            "ts_shape:method": "western_electric_rule_4",
            "ts_shape:direction": "outside",
        },
    ),
    ("StatisticalProcessControlRuleBased", "rule_5"): _r(
        "quality.spc.rule_5",
        _Q,
        P,
        standard_attrs={
            "ts_shape:method": "western_electric_rule_5",
            "ts_shape:direction": "outside",
        },
    ),
    ("StatisticalProcessControlRuleBased", "rule_6"): _r(
        "quality.spc.rule_6",
        _Q,
        P,
        standard_attrs={
            "ts_shape:method": "western_electric_rule_6",
            "ts_shape:direction": "outside",
        },
    ),
    ("StatisticalProcessControlRuleBased", "rule_7"): _r(
        "quality.spc.rule_7",
        _Q,
        P,
        standard_attrs={
            "ts_shape:method": "western_electric_rule_7",
            "ts_shape:direction": "outside",
        },
    ),
    ("StatisticalProcessControlRuleBased", "rule_8"): _r(
        "quality.spc.rule_8",
        _Q,
        P,
        standard_attrs={
            "ts_shape:method": "western_electric_rule_8",
            "ts_shape:direction": "outside",
        },
    ),
    ("ToleranceDeviationEvents", "process_and_group_data_with_events"): _r(
        "quality.tolerance.deviation",
        _Q,
        P,
        severity_field="severity",
        standard_attrs={
            "ts_shape:method": "tolerance",
            "ts_shape:direction": "outside",
            "ts_shape:threshold_low": "lower_tolerance",
            "ts_shape:threshold_high": "upper_tolerance",
            "ts_shape:deviation": "deviation_abs",
            "ts_shape:deviation_pct": "deviation_pct",
        },
    ),
    ("ValueDistributionEvents", "detect_bimodal"): _r(
        "quality.distribution.bimodal",
        _Q,
        S,
        standard_attrs={"ts_shape:sample_count": "n_samples"},
    ),
    ("ValueDistributionEvents", "detect_mode_changes"): _r(
        "quality.distribution.mode_change",
        _Q,
        P,
        standard_attrs={
            "ts_shape:method": "mode_change",
            "ts_shape:direction": "shift",
        },
    ),
    ("ValueDistributionEvents", "normality_windows"): _r(
        "quality.distribution.normality",
        _Q,
        S,
        standard_attrs={
            "ts_shape:sample_count": "n_samples",
            "ts_shape:confidence": "p_value",
        },
    ),
    ("ValueDistributionEvents", "percentile_tracking"): _r(
        "quality.distribution.percentile",
        _Q,
        S,
        standard_attrs={"ts_shape:sample_count": "n_samples"},
    ),
    # ---- supplychain -------------------------------------------------------
    ("DemandPatternEvents", "demand_by_period"): _r(
        "supplychain.demand.by_period",
        _SC,
        S,
        objs=("material",),
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("DemandPatternEvents", "detect_demand_spikes"): _r(
        "supplychain.demand.spike",
        _SC,
        P,
        objs=("material",),
        standard_attrs={
            "ts_shape:method": "spike",
            "ts_shape:direction": "up",
            "ts_shape:baseline": "baseline_mean",
            "ts_shape:deviation": "spike_magnitude",
        },
    ),
    ("DemandPatternEvents", "seasonality_summary"): _r(
        "supplychain.demand.seasonality",
        _SC,
        ST,
        objs=("material",),
        standard_attrs={"ts_shape:method": "seasonality"},
    ),
    ("InventoryMonitoringEvents", "consumption_rate"): _r(
        "supplychain.inventory.consumption_rate",
        _SC,
        S,
        objs=("material",),
        standard_attrs={"ts_shape:sample_count": None},
    ),
    ("InventoryMonitoringEvents", "detect_low_stock"): _r(
        "supplychain.inventory.low_stock",
        _SC,
        P,
        objs=("material",),
        standard_attrs={
            "ts_shape:method": "low_stock",
            "ts_shape:direction": "below",
            "ts_shape:deviation": "deficit",
        },
    ),
    ("InventoryMonitoringEvents", "reorder_point_breach"): _r(
        "supplychain.inventory.reorder_point_breach",
        _SC,
        P,
        objs=("material",),
        standard_attrs={
            "ts_shape:method": "reorder_point_breach",
            "ts_shape:direction": "below",
            "ts_shape:deviation": "deficit",
        },
    ),
    ("InventoryMonitoringEvents", "stockout_prediction"): _r(
        "supplychain.inventory.stockout_prediction",
        _SC,
        S,
        objs=("material",),
        standard_attrs={
            "ts_shape:method": "stockout_prediction",
            "ts_shape:confidence": 1.0,
            "ts_shape:threshold_low": "estimated_stockout_time_hours",
        },
    ),
    ("LeadTimeAnalysisEvents", "calculate_lead_times"): _r(
        "supplychain.lead_time.calculated",
        _SC,
        I,
        objs=("work_order",),
        standard_attrs={"ts_shape:lifecycle_state": "delivered"},
    ),
    ("LeadTimeAnalysisEvents", "detect_lead_time_anomalies"): _r(
        "supplychain.lead_time.anomaly",
        _SC,
        P,
        objs=("work_order",),
        standard_attrs={
            "ts_shape:method": "lead_time_anomaly",
            "ts_shape:direction": "deviation",
            "ts_shape:confidence": "z_score",
        },
    ),
    ("LeadTimeAnalysisEvents", "lead_time_statistics"): _r(
        "supplychain.lead_time.statistics",
        _SC,
        S,
        objs=("work_order",),
        standard_attrs={"ts_shape:sample_count": "count"},
    ),
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
        field_name = template[j + 1 : k]
        val = row.get(field_name) if hasattr(row, "get") else None  # type: ignore[arg-type]
        if val is None:
            try:
                val = row[field_name]  # type: ignore[index]
            except (KeyError, TypeError):
                val = "unknown"
        parts.append(str(val))
        i = k + 1
    return "".join(parts)
