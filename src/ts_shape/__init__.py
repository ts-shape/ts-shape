"""ts-shape: Shape your timeseries data.

Common classes are re-exported at the top level so you can write::

    from ts_shape import OutlierDetectionEvents, OEECalculator, to_event_log

Imports are lazy (PEP 562): ``import ts_shape`` stays cheap and a
submodule is only imported the first time one of its names is accessed.
Deep imports (``from ts_shape.events.quality.outlier_detection import
OutlierDetectionEvents``) keep working unchanged.
"""

import importlib
import logging
from importlib.metadata import PackageNotFoundError, version
from logging import NullHandler

# Library best practice: add NullHandler so that consuming applications
# control logging output.  See https://docs.python.org/3/howto/logging.html
logging.getLogger(__name__).addHandler(NullHandler())

try:
    __version__ = version("ts_shape")
except PackageNotFoundError:  # running from a source tree, not installed
    __version__ = "0.0.0+unknown"


# Public name -> module that defines it. Lazily resolved by __getattr__.
# Keep this in sync with the event-detector packs, the eventlog API, and
# the loaders; tests/test_public_api.py guards it against drift.
_LAZY: dict[str, str] = {
    # -- events.quality --------------------------------------------------
    "AnomalyClassificationEvents": "ts_shape.events.quality.anomaly_classification",
    "CapabilityTrendingEvents": "ts_shape.events.quality.capability_trending",
    "DataGapAnalysisEvents": "ts_shape.events.quality.data_gap_analysis",
    "GaugeRepeatabilityEvents": "ts_shape.events.quality.gauge_repeatability",
    "MultiSensorValidationEvents": "ts_shape.events.quality.multi_sensor_validation",
    "OutlierDetectionEvents": "ts_shape.events.quality.outlier_detection",
    "SensorDriftEvents": "ts_shape.events.quality.sensor_drift",
    "SignalQualityEvents": "ts_shape.events.quality.signal_quality",
    "StatisticalProcessControlRuleBased": "ts_shape.events.quality.statistical_process_control",
    "ToleranceDeviationEvents": "ts_shape.events.quality.tolerance_deviation",
    "ValueDistributionEvents": "ts_shape.events.quality.value_distribution",
    # -- events.production ----------------------------------------------
    "AlarmManagementEvents": "ts_shape.events.production.alarm_management",
    "BatchTrackingEvents": "ts_shape.events.production.batch_tracking",
    "BottleneckDetectionEvents": "ts_shape.events.production.bottleneck_detection",
    "ChangeoverEvents": "ts_shape.events.production.changeover",
    "ContinuousProcessAlignmentEvents": "ts_shape.events.production.continuous_process_alignment",
    "CycleTimeTracking": "ts_shape.events.production.cycle_time_tracking",
    "DowntimeTracking": "ts_shape.events.production.downtime_tracking",
    "DutyCycleEvents": "ts_shape.events.production.duty_cycle",
    "FlowConstraintEvents": "ts_shape.events.production.flow_constraints",
    "FlowMetricsEvents": "ts_shape.events.production.flow_metrics",
    "LineBalancingEvents": "ts_shape.events.production.line_balancing",
    "LineThroughputEvents": "ts_shape.events.production.line_throughput",
    "LongDowntimeEvents": "ts_shape.events.production.long_downtime_events",
    "MachineStateEvents": "ts_shape.events.production.machine_state",
    "MicroStopEvents": "ts_shape.events.production.micro_stop_detection",
    "MultiProcessTraceabilityEvents": "ts_shape.events.production.multi_process_traceability",
    "OEECalculator": "ts_shape.events.production.oee_calculator",
    "OperatorPerformanceTracking": "ts_shape.events.production.operator_performance",
    "ValueTraceabilityEvents": "ts_shape.events.production.order_traceability",
    # OrderTraceabilityEvents is a backwards-compatible alias of the above.
    "OrderTraceabilityEvents": "ts_shape.events.production.order_traceability",
    "PartProductionTracking": "ts_shape.events.production.part_tracking",
    "PerformanceLossTracking": "ts_shape.events.production.performance_loss",
    "PeriodSummary": "ts_shape.events.production.period_summary",
    "QualityTracking": "ts_shape.events.production.quality_tracking",
    "ReworkTracking": "ts_shape.events.production.rework_tracking",
    "RoutingTraceabilityEvents": "ts_shape.events.production.routing_traceability",
    "RuntimeAccountingEvents": "ts_shape.events.production.runtime_accounting",
    "ScrapTracking": "ts_shape.events.production.scrap_tracking",
    "SetupTimeTracking": "ts_shape.events.production.setup_time_tracking",
    "ShiftHandoverReport": "ts_shape.events.production.shift_handover",
    "ShiftReporting": "ts_shape.events.production.shift_reporting",
    "TargetTracking": "ts_shape.events.production.target_tracking",
    # -- events.engineering ---------------------------------------------
    "ControlLoopHealthEvents": "ts_shape.events.engineering.control_loop_health",
    "DisturbanceRecoveryEvents": "ts_shape.events.engineering.disturbance_recovery",
    "MaterialBalanceEvents": "ts_shape.events.engineering.material_balance",
    "OperatingRangeEvents": "ts_shape.events.engineering.operating_range",
    "ProcessStabilityIndex": "ts_shape.events.engineering.process_stability_index",
    "ProcessWindowEvents": "ts_shape.events.engineering.process_window",
    "RateOfChangeEvents": "ts_shape.events.engineering.rate_of_change",
    "SetpointChangeEvents": "ts_shape.events.engineering.setpoint_events",
    "SignalComparisonEvents": "ts_shape.events.engineering.signal_comparison",
    "StartupDetectionEvents": "ts_shape.events.engineering.startup_events",
    "SteadyStateDetectionEvents": "ts_shape.events.engineering.steady_state_detection",
    "ThresholdMonitoringEvents": "ts_shape.events.engineering.threshold_monitoring",
    "WarmUpCoolDownEvents": "ts_shape.events.engineering.warmup_analysis",
    # -- events.maintenance ---------------------------------------------
    "DegradationDetectionEvents": "ts_shape.events.maintenance.degradation_detection",
    "FailurePredictionEvents": "ts_shape.events.maintenance.failure_prediction",
    "VibrationAnalysisEvents": "ts_shape.events.maintenance.vibration_analysis",
    # -- events.development (product & process R&D) ----------------------
    "CriticalParameterRankingEvents": "ts_shape.events.development.critical_parameter_ranking",
    "DesignOfExperimentsEvents": "ts_shape.events.development.design_of_experiments",
    "DesignSpaceEvents": "ts_shape.events.development.design_space",
    "GoldenBatchDeviationEvents": "ts_shape.events.development.golden_batch",
    "RecipePhaseAdherenceEvents": "ts_shape.events.development.recipe_phase_adherence",
    # -- events.energy --------------------------------------------------
    "CarbonIntensityEvents": "ts_shape.events.energy.carbon_intensity",
    "EnergyConsumptionEvents": "ts_shape.events.energy.consumption_analysis",
    "EnergyEfficiencyEvents": "ts_shape.events.energy.efficiency_tracking",
    "EnergyPerformanceIndicatorEvents": "ts_shape.events.energy.energy_performance_indicator",
    "IdleEnergyDetectionEvents": "ts_shape.events.energy.idle_energy_detection",
    # -- events.correlation ---------------------------------------------
    "AnomalyCorrelationEvents": "ts_shape.events.correlation.anomaly_correlation",
    "SignalCorrelationEvents": "ts_shape.events.correlation.signal_correlation",
    # -- events.supplychain ---------------------------------------------
    "DemandPatternEvents": "ts_shape.events.supplychain.demand_pattern",
    "InventoryMonitoringEvents": "ts_shape.events.supplychain.inventory_monitoring",
    "LeadTimeAnalysisEvents": "ts_shape.events.supplychain.lead_time_analysis",
    # -- loaders --------------------------------------------------------
    "ParquetLoader": "ts_shape.loader.timeseries.parquet_loader",
    "S3ProxyDataAccess": "ts_shape.loader.timeseries.s3proxy_parquet_loader",
    "AzureBlobParquetLoader": "ts_shape.loader.timeseries.azure_blob_loader",
    "AzureBlobFlexibleFileLoader": "ts_shape.loader.timeseries.azure_blob_loader",
    "AzureBlobEnergyLoader": "ts_shape.loader.timeseries.azure_blob_energy_loader",
    "DatabricksUnityParquetLoader": "ts_shape.loader.timeseries.databricks_unity_parquet_loader",
    "DatabricksUnityEnergyLoader": "ts_shape.loader.timeseries.databricks_unity_energy_loader",
    "EnergyAPILoader": "ts_shape.loader.timeseries.energy_api_loader",
    "TimescaleDBDataAccess": "ts_shape.loader.timeseries.timescale_loader",
    "MetadataJsonLoader": "ts_shape.loader.metadata.metadata_json_loader",
    "DatapointAPI": "ts_shape.loader.metadata.metadata_api_loader",
    "DatapointDB": "ts_shape.loader.metadata.metadata_db_loader",
    "DataIntegratorHybrid": "ts_shape.loader.combine.integrator",
    "ContextEnricher": "ts_shape.loader.context.context_enricher",
    # -- top-level utilities --------------------------------------------
    "list_detectors": "ts_shape.catalog",
    "make_timeseries": "ts_shape.datasets",
    "make_id_signal": "ts_shape.datasets",
    "UnitConverter": "ts_shape.transform.calculator.unit_conversion",
    "Pipeline": "ts_shape.pipeline",
}

# The eventlog package re-exports its whole public surface from its own
# __init__, so every eventlog name maps to that one module.
_EVENTLOG_NAMES = (
    "BacktestResult",
    "EventLog",
    "OCEL2Tables",
    "ObjectSpec",
    "attach_objects",
    "detect_objects",
    "object_intervals",
    "object_specs_from_metadata",
    "align_columns",
    "LabelRule",
    "LambdaDetector",
    "REGISTRY",
    "RuleSpec",
    "TriggerSpec",
    "UnsafeExpression",
    "compile_expression",
    "concat",
    "load_dicts",
    "load_yaml",
    "register_adapter",
    "register_lambda_rule",
    "register_object_type",
    "run_backtest",
    "to_event_log",
    "to_event_log_ocel",
    "to_event_log_xes",
    "unregister_lambda_rule",
    "validate",
    "OCEL_ACTIVITY",
    "OCEL_EID",
    "OCEL_OID",
    "OCEL_QUALIFIER",
    "OCEL_TIMESTAMP",
    "OCEL_TYPE",
    "TS_DETECTOR",
    "TS_DURATION_S",
    "TS_PACK",
    "TS_SEVERITY",
    "TS_START_TIMESTAMP",
    "TS_VALUE",
    "XES_ACTIVITY",
    "XES_CASE",
    "XES_LIFECYCLE",
    "XES_RESOURCE",
    "XES_TIMESTAMP",
)
for _name in _EVENTLOG_NAMES:
    _LAZY[_name] = "ts_shape.eventlog"
del _name

__all__ = sorted(_LAZY)


def __getattr__(name: str) -> object:
    """Lazily resolve a re-exported name to its defining module (PEP 562)."""
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(target)
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))
