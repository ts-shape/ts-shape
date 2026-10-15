"""Events

Extract events from shaped timeseries across quality, maintenance,
production, engineering, energy, correlation, supply chain and development
domains.

Subpackages:
- quality: Outlier detection, SPC, tolerance deviation
- production: Machine state, throughput, changeover, OEE, alarms, batches, shifts
- engineering: Setpoint changes, startup detection
- maintenance: Degradation detection, failure prediction, vibration analysis
- supplychain: Inventory monitoring, lead time analysis, demand patterns
- energy: Consumption analysis, efficiency tracking
- correlation: Signal correlation, anomaly correlation
- development: Design of experiments, design space, golden batch, recipe
  phase adherence, critical parameter ranking

Every detector class is also importable from this package (and from
``ts_shape``), e.g. ``from ts_shape.events import OutlierDetectionEvents``.
"""

# This module re-exports the deprecated class aliases on purpose; using them
# anywhere else is still reported by mypy's "deprecated" check.
# mypy: disable-error-code="deprecated"

import importlib as _importlib
from typing import TYPE_CHECKING

# Explicit imports for type checkers and IDEs only (so ``from ts_shape import X``
# is typed as X, not object); at runtime __getattr__ below imports lazily.
# tests/typing/test_lazy_exports.py keeps this block in sync with the map.
if TYPE_CHECKING:
    from ts_shape.events.correlation.anomaly_correlation import (
        AnomalyCorrelationEvents as AnomalyCorrelationEvents,
    )
    from ts_shape.events.correlation.root_cause_correlation import (
        RootCauseCorrelationEvents as RootCauseCorrelationEvents,
    )
    from ts_shape.events.correlation.signal_correlation import (
        SignalCorrelationEvents as SignalCorrelationEvents,
    )
    from ts_shape.events.development.critical_parameter_ranking import (
        CriticalParameterRankingEvents as CriticalParameterRankingEvents,
    )
    from ts_shape.events.development.design_of_experiments import (
        DesignOfExperimentsEvents as DesignOfExperimentsEvents,
    )
    from ts_shape.events.development.design_space import (
        DesignSpaceEvents as DesignSpaceEvents,
    )
    from ts_shape.events.development.golden_batch import (
        GoldenBatchDeviationEvents as GoldenBatchDeviationEvents,
    )
    from ts_shape.events.development.recipe_phase_adherence import (
        RecipePhaseAdherenceEvents as RecipePhaseAdherenceEvents,
    )
    from ts_shape.events.energy.carbon_intensity import (
        CarbonIntensityEvents as CarbonIntensityEvents,
    )
    from ts_shape.events.energy.consumption_analysis import (
        EnergyConsumptionEvents as EnergyConsumptionEvents,
    )
    from ts_shape.events.energy.efficiency_tracking import (
        EnergyEfficiencyEvents as EnergyEfficiencyEvents,
    )
    from ts_shape.events.energy.energy_performance_indicator import (
        EnergyPerformanceIndicatorEvents as EnergyPerformanceIndicatorEvents,
    )
    from ts_shape.events.energy.idle_energy_detection import (
        IdleEnergyDetectionEvents as IdleEnergyDetectionEvents,
    )
    from ts_shape.events.engineering.control_loop_health import (
        ControlLoopHealthEvents as ControlLoopHealthEvents,
    )
    from ts_shape.events.engineering.disturbance_recovery import (
        DisturbanceRecoveryEvents as DisturbanceRecoveryEvents,
    )
    from ts_shape.events.engineering.material_balance import (
        MaterialBalanceEvents as MaterialBalanceEvents,
    )
    from ts_shape.events.engineering.operating_range import (
        OperatingRangeEvents as OperatingRangeEvents,
    )
    from ts_shape.events.engineering.process_stability_index import (
        ProcessStabilityIndex as ProcessStabilityIndex,
    )
    from ts_shape.events.engineering.process_stability_index import (
        ProcessStabilityIndexEvents as ProcessStabilityIndexEvents,
    )
    from ts_shape.events.engineering.process_window import (
        ProcessWindowEvents as ProcessWindowEvents,
    )
    from ts_shape.events.engineering.rate_of_change import (
        RateOfChangeEvents as RateOfChangeEvents,
    )
    from ts_shape.events.engineering.setpoint_events import (
        SetpointChangeEvents as SetpointChangeEvents,
    )
    from ts_shape.events.engineering.signal_comparison import (
        SignalComparisonEvents as SignalComparisonEvents,
    )
    from ts_shape.events.engineering.startup_events import (
        StartupDetectionEvents as StartupDetectionEvents,
    )
    from ts_shape.events.engineering.steady_state_detection import (
        SteadyStateDetectionEvents as SteadyStateDetectionEvents,
    )
    from ts_shape.events.engineering.threshold_monitoring import (
        ThresholdMonitoringEvents as ThresholdMonitoringEvents,
    )
    from ts_shape.events.engineering.time_to_threshold import (
        TimeToThresholdEvents as TimeToThresholdEvents,
    )
    from ts_shape.events.engineering.warmup_analysis import (
        WarmUpCoolDownEvents as WarmUpCoolDownEvents,
    )
    from ts_shape.events.maintenance.asset_health_index import (
        AssetHealthIndexEvents as AssetHealthIndexEvents,
    )
    from ts_shape.events.maintenance.degradation_detection import (
        DegradationDetectionEvents as DegradationDetectionEvents,
    )
    from ts_shape.events.maintenance.failure_prediction import (
        FailurePredictionEvents as FailurePredictionEvents,
    )
    from ts_shape.events.maintenance.reliability import (
        ReliabilityEvents as ReliabilityEvents,
    )
    from ts_shape.events.maintenance.vibration_analysis import (
        VibrationAnalysisEvents as VibrationAnalysisEvents,
    )
    from ts_shape.events.production.alarm_management import (
        AlarmManagementEvents as AlarmManagementEvents,
    )
    from ts_shape.events.production.batch_tracking import (
        BatchTrackingEvents as BatchTrackingEvents,
    )
    from ts_shape.events.production.bottleneck_detection import (
        BottleneckDetectionEvents as BottleneckDetectionEvents,
    )
    from ts_shape.events.production.changeover import (
        ChangeoverEvents as ChangeoverEvents,
    )
    from ts_shape.events.production.continuous_process_alignment import (
        ContinuousProcessAlignmentEvents as ContinuousProcessAlignmentEvents,
    )
    from ts_shape.events.production.cycle_time_tracking import (
        CycleTimeEvents as CycleTimeEvents,
    )
    from ts_shape.events.production.cycle_time_tracking import (
        CycleTimeTracking as CycleTimeTracking,
    )
    from ts_shape.events.production.downtime_tracking import (
        DowntimeEvents as DowntimeEvents,
    )
    from ts_shape.events.production.downtime_tracking import (
        DowntimeTracking as DowntimeTracking,
    )
    from ts_shape.events.production.duty_cycle import DutyCycleEvents as DutyCycleEvents
    from ts_shape.events.production.flow_constraints import (
        FlowConstraintEvents as FlowConstraintEvents,
    )
    from ts_shape.events.production.flow_metrics import (
        FlowMetricsEvents as FlowMetricsEvents,
    )
    from ts_shape.events.production.line_balancing import (
        LineBalancingEvents as LineBalancingEvents,
    )
    from ts_shape.events.production.line_throughput import (
        LineThroughputEvents as LineThroughputEvents,
    )
    from ts_shape.events.production.long_downtime_events import (
        LongDowntimeEvents as LongDowntimeEvents,
    )
    from ts_shape.events.production.machine_state import (
        MachineStateEvents as MachineStateEvents,
    )
    from ts_shape.events.production.micro_stop_detection import (
        MicroStopEvents as MicroStopEvents,
    )
    from ts_shape.events.production.multi_process_traceability import (
        MultiProcessTraceabilityEvents as MultiProcessTraceabilityEvents,
    )
    from ts_shape.events.production.oee_calculator import OEECalculator as OEECalculator
    from ts_shape.events.production.oee_calculator import OEEEvents as OEEEvents
    from ts_shape.events.production.operating_state import (
        OperatingStateEvents as OperatingStateEvents,
    )
    from ts_shape.events.production.operator_performance import (
        OperatorPerformanceEvents as OperatorPerformanceEvents,
    )
    from ts_shape.events.production.operator_performance import (
        OperatorPerformanceTracking as OperatorPerformanceTracking,
    )
    from ts_shape.events.production.order_traceability import (
        OrderTraceabilityEvents as OrderTraceabilityEvents,
    )
    from ts_shape.events.production.order_traceability import (
        ValueTraceabilityEvents as ValueTraceabilityEvents,
    )
    from ts_shape.events.production.part_tracking import (
        PartProductionEvents as PartProductionEvents,
    )
    from ts_shape.events.production.part_tracking import (
        PartProductionTracking as PartProductionTracking,
    )
    from ts_shape.events.production.performance_loss import (
        PerformanceLossEvents as PerformanceLossEvents,
    )
    from ts_shape.events.production.performance_loss import (
        PerformanceLossTracking as PerformanceLossTracking,
    )
    from ts_shape.events.production.period_summary import PeriodSummary as PeriodSummary
    from ts_shape.events.production.period_summary import (
        PeriodSummaryEvents as PeriodSummaryEvents,
    )
    from ts_shape.events.production.quality_tracking import (
        QualityEvents as QualityEvents,
    )
    from ts_shape.events.production.quality_tracking import (
        QualityTracking as QualityTracking,
    )
    from ts_shape.events.production.rework_tracking import ReworkEvents as ReworkEvents
    from ts_shape.events.production.rework_tracking import (
        ReworkTracking as ReworkTracking,
    )
    from ts_shape.events.production.routing_traceability import (
        RoutingTraceabilityEvents as RoutingTraceabilityEvents,
    )
    from ts_shape.events.production.runtime_accounting import (
        RuntimeAccountingEvents as RuntimeAccountingEvents,
    )
    from ts_shape.events.production.schedule_adherence import (
        ProductionScheduleAdherenceEvents as ProductionScheduleAdherenceEvents,
    )
    from ts_shape.events.production.scrap_tracking import ScrapEvents as ScrapEvents
    from ts_shape.events.production.scrap_tracking import ScrapTracking as ScrapTracking
    from ts_shape.events.production.setup_time_tracking import (
        SetupTimeEvents as SetupTimeEvents,
    )
    from ts_shape.events.production.setup_time_tracking import (
        SetupTimeTracking as SetupTimeTracking,
    )
    from ts_shape.events.production.shift_handover import (
        ShiftHandoverEvents as ShiftHandoverEvents,
    )
    from ts_shape.events.production.shift_handover import (
        ShiftHandoverReport as ShiftHandoverReport,
    )
    from ts_shape.events.production.shift_reporting import (
        ShiftReportEvents as ShiftReportEvents,
    )
    from ts_shape.events.production.shift_reporting import (
        ShiftReporting as ShiftReporting,
    )
    from ts_shape.events.production.target_tracking import TargetEvents as TargetEvents
    from ts_shape.events.production.target_tracking import (
        TargetTracking as TargetTracking,
    )
    from ts_shape.events.quality.anomaly_classification import (
        AnomalyClassificationEvents as AnomalyClassificationEvents,
    )
    from ts_shape.events.quality.calibration_validation import (
        CalibrationValidationEvents as CalibrationValidationEvents,
    )
    from ts_shape.events.quality.capability_trending import (
        CapabilityTrendingEvents as CapabilityTrendingEvents,
    )
    from ts_shape.events.quality.change_point import (
        ChangePointEvents as ChangePointEvents,
    )
    from ts_shape.events.quality.data_gap_analysis import (
        DataGapAnalysisEvents as DataGapAnalysisEvents,
    )
    from ts_shape.events.quality.environmental_compensation import (
        EnvironmentalCompensationEvents as EnvironmentalCompensationEvents,
    )
    from ts_shape.events.quality.gauge_repeatability import (
        GaugeRepeatabilityEvents as GaugeRepeatabilityEvents,
    )
    from ts_shape.events.quality.measurement_uncertainty import (
        MeasurementUncertaintyEvents as MeasurementUncertaintyEvents,
    )
    from ts_shape.events.quality.multi_sensor_validation import (
        MultiSensorValidationEvents as MultiSensorValidationEvents,
    )
    from ts_shape.events.quality.multivariate_spc import (
        MultivariateSPCEvents as MultivariateSPCEvents,
    )
    from ts_shape.events.quality.outlier_detection import (
        OutlierDetectionEvents as OutlierDetectionEvents,
    )
    from ts_shape.events.quality.sampling_resolution import (
        SamplingResolutionEvents as SamplingResolutionEvents,
    )
    from ts_shape.events.quality.sensor_drift import (
        SensorDriftEvents as SensorDriftEvents,
    )
    from ts_shape.events.quality.signal_quality import (
        SignalQualityEvents as SignalQualityEvents,
    )
    from ts_shape.events.quality.statistical_process_control import (
        StatisticalProcessControlEvents as StatisticalProcessControlEvents,
    )
    from ts_shape.events.quality.statistical_process_control import (
        StatisticalProcessControlRuleBased as StatisticalProcessControlRuleBased,
    )
    from ts_shape.events.quality.tolerance_deviation import (
        ToleranceDeviationEvents as ToleranceDeviationEvents,
    )
    from ts_shape.events.quality.value_distribution import (
        ValueDistributionEvents as ValueDistributionEvents,
    )
    from ts_shape.events.supplychain.demand_pattern import (
        DemandPatternEvents as DemandPatternEvents,
    )
    from ts_shape.events.supplychain.inventory_monitoring import (
        InventoryMonitoringEvents as InventoryMonitoringEvents,
    )
    from ts_shape.events.supplychain.lead_time_analysis import (
        LeadTimeAnalysisEvents as LeadTimeAnalysisEvents,
    )

from ts_shape import _LAZY

# Classes are imported on first access, so ``import ts_shape.events`` stays
# cheap and a pack's optional dependencies are only needed when it is used.
_CLASSES = {
    name: module for name, module in _LAZY.items() if module.startswith(__name__ + ".")
}

__all__ = sorted(_CLASSES)


def __getattr__(name: str) -> object:
    if name in _CLASSES:
        return getattr(_importlib.import_module(_CLASSES[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_CLASSES))
