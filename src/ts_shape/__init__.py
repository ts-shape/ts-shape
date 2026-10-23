"""ts-shape: Shape your timeseries data.

Common classes are re-exported at the top level so you can write::

    from ts_shape import OutlierDetectionEvents, OEEEvents, to_event_log

Imports are lazy (PEP 562): ``import ts_shape`` stays cheap and a
submodule is only imported the first time one of its names is accessed.
Deep imports (``from ts_shape.events.quality.outlier_detection import
OutlierDetectionEvents``) keep working unchanged.
"""

# This module re-exports the deprecated class aliases on purpose; using them
# anywhere else is still reported by mypy's "deprecated" check.
# mypy: disable-error-code="deprecated"

import importlib
import logging
from importlib.metadata import PackageNotFoundError, version
from logging import NullHandler
from typing import TYPE_CHECKING

# Explicit imports for type checkers and IDEs only (so ``from ts_shape import X``
# is typed as X, not object); at runtime __getattr__ below imports lazily.
# tests/typing/test_lazy_exports.py keeps this block in sync with the map.
if TYPE_CHECKING:
    from ts_shape.catalog import list_detectors as list_detectors
    from ts_shape.context.value_mapping import ValueMapper as ValueMapper
    from ts_shape.datasets import make_id_signal as make_id_signal
    from ts_shape.datasets import make_object_signals as make_object_signals
    from ts_shape.datasets import make_timeseries as make_timeseries
    from ts_shape.errors import ColumnNotFoundError as ColumnNotFoundError
    from ts_shape.errors import DataQualityWarning as DataQualityWarning
    from ts_shape.errors import LoaderConfigWarning as LoaderConfigWarning
    from ts_shape.errors import LoaderError as LoaderError
    from ts_shape.errors import NotChunkableError as NotChunkableError
    from ts_shape.errors import ObjectNotFoundError as ObjectNotFoundError
    from ts_shape.errors import PerformanceWarning as PerformanceWarning
    from ts_shape.errors import TsShapeWarning as TsShapeWarning
    from ts_shape.eventlog import OCEL_ACTIVITY as OCEL_ACTIVITY
    from ts_shape.eventlog import OCEL_EID as OCEL_EID
    from ts_shape.eventlog import OCEL_OID as OCEL_OID
    from ts_shape.eventlog import OCEL_QUALIFIER as OCEL_QUALIFIER
    from ts_shape.eventlog import OCEL_TIMESTAMP as OCEL_TIMESTAMP
    from ts_shape.eventlog import OCEL_TYPE as OCEL_TYPE
    from ts_shape.eventlog import REGISTRY as REGISTRY
    from ts_shape.eventlog import TS_DETECTOR as TS_DETECTOR
    from ts_shape.eventlog import TS_DURATION_S as TS_DURATION_S
    from ts_shape.eventlog import TS_PACK as TS_PACK
    from ts_shape.eventlog import TS_SEVERITY as TS_SEVERITY
    from ts_shape.eventlog import TS_START_TIMESTAMP as TS_START_TIMESTAMP
    from ts_shape.eventlog import TS_VALUE as TS_VALUE
    from ts_shape.eventlog import XES_ACTIVITY as XES_ACTIVITY
    from ts_shape.eventlog import XES_CASE as XES_CASE
    from ts_shape.eventlog import XES_LIFECYCLE as XES_LIFECYCLE
    from ts_shape.eventlog import XES_RESOURCE as XES_RESOURCE
    from ts_shape.eventlog import XES_TIMESTAMP as XES_TIMESTAMP
    from ts_shape.eventlog import BacktestResult as BacktestResult
    from ts_shape.eventlog import CarrierLoops as CarrierLoops
    from ts_shape.eventlog import EventLog as EventLog
    from ts_shape.eventlog import LabelRule as LabelRule
    from ts_shape.eventlog import LambdaDetector as LambdaDetector
    from ts_shape.eventlog import ObjectKPIs as ObjectKPIs
    from ts_shape.eventlog import ObjectSpec as ObjectSpec
    from ts_shape.eventlog import OCEL2Tables as OCEL2Tables
    from ts_shape.eventlog import RuleSpec as RuleSpec
    from ts_shape.eventlog import TriggerSpec as TriggerSpec
    from ts_shape.eventlog import UnsafeExpression as UnsafeExpression
    from ts_shape.eventlog import align_columns as align_columns
    from ts_shape.eventlog import attach_counts as attach_counts
    from ts_shape.eventlog import attach_objects as attach_objects
    from ts_shape.eventlog import carrier_loops as carrier_loops
    from ts_shape.eventlog import compile_expression as compile_expression
    from ts_shape.eventlog import concat as concat
    from ts_shape.eventlog import detect_objects as detect_objects
    from ts_shape.eventlog import load_dicts as load_dicts
    from ts_shape.eventlog import load_yaml as load_yaml
    from ts_shape.eventlog import object_intervals as object_intervals
    from ts_shape.eventlog import object_kpis as object_kpis
    from ts_shape.eventlog import (
        object_specs_from_metadata as object_specs_from_metadata,
    )
    from ts_shape.eventlog import presence_events as presence_events
    from ts_shape.eventlog import register_adapter as register_adapter
    from ts_shape.eventlog import register_lambda_rule as register_lambda_rule
    from ts_shape.eventlog import register_object_type as register_object_type
    from ts_shape.eventlog import run_backtest as run_backtest
    from ts_shape.eventlog import to_event_log as to_event_log
    from ts_shape.eventlog import to_event_log_ocel as to_event_log_ocel
    from ts_shape.eventlog import to_event_log_xes as to_event_log_xes
    from ts_shape.eventlog import unregister_lambda_rule as unregister_lambda_rule
    from ts_shape.eventlog import validate as validate
    from ts_shape.eventlog import wip_over_time as wip_over_time
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
    from ts_shape.features.cross_signal import (
        CrossSignalAnalytics as CrossSignalAnalytics,
    )
    from ts_shape.features.cycles.cycle_processor import (
        CycleDataProcessor as CycleDataProcessor,
    )
    from ts_shape.features.cycles.cycles_extractor import (
        CycleExtractor as CycleExtractor,
    )
    from ts_shape.features.export import FeatureMatrixExporter as FeatureMatrixExporter
    from ts_shape.features.pattern_recognition import (
        PatternRecognition as PatternRecognition,
    )
    from ts_shape.features.segment_analysis.profile_comparison import (
        ProfileComparison as ProfileComparison,
    )
    from ts_shape.features.segment_analysis.segment_extractor import (
        SegmentExtractor as SegmentExtractor,
    )
    from ts_shape.features.segment_analysis.segment_processor import (
        SegmentProcessor as SegmentProcessor,
    )
    from ts_shape.features.segment_analysis.time_windowed_features import (
        TimeWindowedFeatureTable as TimeWindowedFeatureTable,
    )
    from ts_shape.features.stats.boolean_stats import (
        BooleanStatistics as BooleanStatistics,
    )
    from ts_shape.features.stats.feature_table import (
        DescriptiveFeatures as DescriptiveFeatures,
    )
    from ts_shape.features.stats.numeric_stats import (
        NumericStatistics as NumericStatistics,
    )
    from ts_shape.features.stats.string_stats import (
        StringStatistics as StringStatistics,
    )
    from ts_shape.features.stats.timestamp_stats import (
        TimestampStatistics as TimestampStatistics,
    )
    from ts_shape.features.time_stats.time_stats_numeric import (
        TimeGroupedStatistics as TimeGroupedStatistics,
    )
    from ts_shape.loader.combine.integrator import (
        DataIntegratorHybrid as DataIntegratorHybrid,
    )
    from ts_shape.loader.context.context_enricher import (
        ContextEnricher as ContextEnricher,
    )
    from ts_shape.loader.metadata.metadata_api_loader import (
        DatapointAPI as DatapointAPI,
    )
    from ts_shape.loader.metadata.metadata_db_loader import DatapointDB as DatapointDB
    from ts_shape.loader.metadata.metadata_json_loader import (
        MetadataJsonLoader as MetadataJsonLoader,
    )
    from ts_shape.loader.timeseries.azure_blob_energy_loader import (
        AzureBlobEnergyLoader as AzureBlobEnergyLoader,
    )
    from ts_shape.loader.timeseries.azure_blob_loader import (
        AzureBlobFlexibleFileLoader as AzureBlobFlexibleFileLoader,
    )
    from ts_shape.loader.timeseries.azure_blob_loader import (
        AzureBlobParquetLoader as AzureBlobParquetLoader,
    )
    from ts_shape.loader.timeseries.databricks_spark_parquet_loader import (
        DatabricksSparkParquetLoader as DatabricksSparkParquetLoader,
    )
    from ts_shape.loader.timeseries.databricks_unity_energy_loader import (
        DatabricksUnityEnergyLoader as DatabricksUnityEnergyLoader,
    )
    from ts_shape.loader.timeseries.databricks_unity_parquet_loader import (
        DatabricksUnityParquetLoader as DatabricksUnityParquetLoader,
    )
    from ts_shape.loader.timeseries.energy_api_loader import (
        EnergyAPILoader as EnergyAPILoader,
    )
    from ts_shape.loader.timeseries.parquet_loader import ParquetLoader as ParquetLoader
    from ts_shape.loader.timeseries.s3proxy_parquet_loader import (
        S3ProxyDataAccess as S3ProxyDataAccess,
    )
    from ts_shape.loader.timeseries.timescale_loader import (
        TimescaleDBDataAccess as TimescaleDBDataAccess,
    )
    from ts_shape.pipeline import Pipeline as Pipeline
    from ts_shape.pipeline import PipelineResult as PipelineResult
    from ts_shape.profiling import ProfileReport as ProfileReport
    from ts_shape.profiling import profile as profile
    from ts_shape.signals import ObjectReport as ObjectReport
    from ts_shape.signals import SignalSpec as SignalSpec
    from ts_shape.signals import SignalSpecs as SignalSpecs
    from ts_shape.signals import identify_objects as identify_objects
    from ts_shape.signals import infer_specs as infer_specs
    from ts_shape.signals import (
        register_identifier_pattern as register_identifier_pattern,
    )
    from ts_shape.signals import register_identifier_type as register_identifier_type
    from ts_shape.stream import iter_chunked as iter_chunked
    from ts_shape.stream import run_chunked as run_chunked
    from ts_shape.stream import time_range_source as time_range_source
    from ts_shape.transform.calculator.numeric_calc import IntegerCalc as IntegerCalc
    from ts_shape.transform.calculator.power_integration import (
        PowerIntegrator as PowerIntegrator,
    )
    from ts_shape.transform.calculator.unit_conversion import (
        UnitConverter as UnitConverter,
    )
    from ts_shape.transform.filter.boolean_filter import BooleanFilter as BooleanFilter
    from ts_shape.transform.filter.boolean_filter import IsDeltaFilter as IsDeltaFilter
    from ts_shape.transform.filter.custom_filter import CustomFilter as CustomFilter
    from ts_shape.transform.filter.datetime_filter import (
        DateTimeFilter as DateTimeFilter,
    )
    from ts_shape.transform.filter.numeric_filter import DoubleFilter as DoubleFilter
    from ts_shape.transform.filter.numeric_filter import IntegerFilter as IntegerFilter
    from ts_shape.transform.filter.string_filter import StringFilter as StringFilter
    from ts_shape.transform.functions.lambda_func import (
        LambdaProcessor as LambdaProcessor,
    )
    from ts_shape.transform.harmonization import DataHarmonizer as DataHarmonizer
    from ts_shape.transform.time_functions.timestamp_converter import (
        TimestampConverter as TimestampConverter,
    )
    from ts_shape.transform.time_functions.timezone_shift import (
        TimezoneShift as TimezoneShift,
    )

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
    "CycleTimeTracking": "ts_shape.events.production.cycle_time_tracking",  # deprecated alias of CycleTimeEvents
    "DowntimeTracking": "ts_shape.events.production.downtime_tracking",  # deprecated alias of DowntimeEvents
    "OperatorPerformanceTracking": "ts_shape.events.production.operator_performance",  # deprecated alias of OperatorPerformanceEvents
    "PartProductionTracking": "ts_shape.events.production.part_tracking",  # deprecated alias of PartProductionEvents
    "PerformanceLossTracking": "ts_shape.events.production.performance_loss",  # deprecated alias of PerformanceLossEvents
    "QualityTracking": "ts_shape.events.production.quality_tracking",  # deprecated alias of QualityEvents
    "ReworkTracking": "ts_shape.events.production.rework_tracking",  # deprecated alias of ReworkEvents
    "ScrapTracking": "ts_shape.events.production.scrap_tracking",  # deprecated alias of ScrapEvents
    "SetupTimeTracking": "ts_shape.events.production.setup_time_tracking",  # deprecated alias of SetupTimeEvents
    "TargetTracking": "ts_shape.events.production.target_tracking",  # deprecated alias of TargetEvents
    "OEECalculator": "ts_shape.events.production.oee_calculator",  # deprecated alias of OEEEvents
    "PeriodSummary": "ts_shape.events.production.period_summary",  # deprecated alias of PeriodSummaryEvents
    "ShiftReporting": "ts_shape.events.production.shift_reporting",  # deprecated alias of ShiftReportEvents
    "ShiftHandoverReport": "ts_shape.events.production.shift_handover",  # deprecated alias of ShiftHandoverEvents
    "ProcessStabilityIndex": "ts_shape.events.engineering.process_stability_index",  # deprecated alias of ProcessStabilityIndexEvents
    "StatisticalProcessControlRuleBased": "ts_shape.events.quality.statistical_process_control",  # deprecated alias of StatisticalProcessControlEvents
    # -- events.quality --------------------------------------------------
    "AnomalyClassificationEvents": "ts_shape.events.quality.anomaly_classification",
    "CapabilityTrendingEvents": "ts_shape.events.quality.capability_trending",
    "ChangePointEvents": "ts_shape.events.quality.change_point",
    "DataGapAnalysisEvents": "ts_shape.events.quality.data_gap_analysis",
    "GaugeRepeatabilityEvents": "ts_shape.events.quality.gauge_repeatability",
    "CalibrationValidationEvents": "ts_shape.events.quality.calibration_validation",
    "EnvironmentalCompensationEvents": "ts_shape.events.quality.environmental_compensation",
    "MeasurementUncertaintyEvents": "ts_shape.events.quality.measurement_uncertainty",
    "SamplingResolutionEvents": "ts_shape.events.quality.sampling_resolution",
    "MultiSensorValidationEvents": "ts_shape.events.quality.multi_sensor_validation",
    "MultivariateSPCEvents": "ts_shape.events.quality.multivariate_spc",
    "OutlierDetectionEvents": "ts_shape.events.quality.outlier_detection",
    "SensorDriftEvents": "ts_shape.events.quality.sensor_drift",
    "SignalQualityEvents": "ts_shape.events.quality.signal_quality",
    "StatisticalProcessControlEvents": "ts_shape.events.quality.statistical_process_control",
    "ToleranceDeviationEvents": "ts_shape.events.quality.tolerance_deviation",
    "ValueDistributionEvents": "ts_shape.events.quality.value_distribution",
    # -- events.production ----------------------------------------------
    "AlarmManagementEvents": "ts_shape.events.production.alarm_management",
    "BatchTrackingEvents": "ts_shape.events.production.batch_tracking",
    "BottleneckDetectionEvents": "ts_shape.events.production.bottleneck_detection",
    "ChangeoverEvents": "ts_shape.events.production.changeover",
    "ContinuousProcessAlignmentEvents": "ts_shape.events.production.continuous_process_alignment",
    "CycleTimeEvents": "ts_shape.events.production.cycle_time_tracking",
    "DowntimeEvents": "ts_shape.events.production.downtime_tracking",
    "DutyCycleEvents": "ts_shape.events.production.duty_cycle",
    "FlowConstraintEvents": "ts_shape.events.production.flow_constraints",
    "FlowMetricsEvents": "ts_shape.events.production.flow_metrics",
    "LineBalancingEvents": "ts_shape.events.production.line_balancing",
    "LineThroughputEvents": "ts_shape.events.production.line_throughput",
    "LongDowntimeEvents": "ts_shape.events.production.long_downtime_events",
    "MachineStateEvents": "ts_shape.events.production.machine_state",
    "OperatingStateEvents": "ts_shape.events.production.operating_state",
    "MicroStopEvents": "ts_shape.events.production.micro_stop_detection",
    "MultiProcessTraceabilityEvents": "ts_shape.events.production.multi_process_traceability",
    "OEEEvents": "ts_shape.events.production.oee_calculator",
    "OperatorPerformanceEvents": "ts_shape.events.production.operator_performance",
    "ValueTraceabilityEvents": "ts_shape.events.production.order_traceability",
    "OrderTraceabilityEvents": "ts_shape.events.production.order_traceability",  # deprecated alias of ValueTraceabilityEvents
    "PartProductionEvents": "ts_shape.events.production.part_tracking",
    "PerformanceLossEvents": "ts_shape.events.production.performance_loss",
    "PeriodSummaryEvents": "ts_shape.events.production.period_summary",
    "ProductionScheduleAdherenceEvents": "ts_shape.events.production.schedule_adherence",
    "QualityEvents": "ts_shape.events.production.quality_tracking",
    "ReworkEvents": "ts_shape.events.production.rework_tracking",
    "RoutingTraceabilityEvents": "ts_shape.events.production.routing_traceability",
    "RuntimeAccountingEvents": "ts_shape.events.production.runtime_accounting",
    "ScrapEvents": "ts_shape.events.production.scrap_tracking",
    "SetupTimeEvents": "ts_shape.events.production.setup_time_tracking",
    "ShiftHandoverEvents": "ts_shape.events.production.shift_handover",
    "ShiftReportEvents": "ts_shape.events.production.shift_reporting",
    "TargetEvents": "ts_shape.events.production.target_tracking",
    # -- events.engineering ---------------------------------------------
    "ControlLoopHealthEvents": "ts_shape.events.engineering.control_loop_health",
    "DisturbanceRecoveryEvents": "ts_shape.events.engineering.disturbance_recovery",
    "MaterialBalanceEvents": "ts_shape.events.engineering.material_balance",
    "OperatingRangeEvents": "ts_shape.events.engineering.operating_range",
    "ProcessStabilityIndexEvents": "ts_shape.events.engineering.process_stability_index",
    "ProcessWindowEvents": "ts_shape.events.engineering.process_window",
    "RateOfChangeEvents": "ts_shape.events.engineering.rate_of_change",
    "TimeToThresholdEvents": "ts_shape.events.engineering.time_to_threshold",
    "SetpointChangeEvents": "ts_shape.events.engineering.setpoint_events",
    "SignalComparisonEvents": "ts_shape.events.engineering.signal_comparison",
    "StartupDetectionEvents": "ts_shape.events.engineering.startup_events",
    "SteadyStateDetectionEvents": "ts_shape.events.engineering.steady_state_detection",
    "ThresholdMonitoringEvents": "ts_shape.events.engineering.threshold_monitoring",
    "WarmUpCoolDownEvents": "ts_shape.events.engineering.warmup_analysis",
    # -- events.maintenance ---------------------------------------------
    "AssetHealthIndexEvents": "ts_shape.events.maintenance.asset_health_index",
    "ReliabilityEvents": "ts_shape.events.maintenance.reliability",
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
    "RootCauseCorrelationEvents": "ts_shape.events.correlation.root_cause_correlation",
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
    "DatabricksSparkParquetLoader": "ts_shape.loader.timeseries.databricks_spark_parquet_loader",
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
    "profile": "ts_shape.profiling",
    "ProfileReport": "ts_shape.profiling",
    "infer_specs": "ts_shape.signals",
    "identify_objects": "ts_shape.signals",
    "ObjectReport": "ts_shape.signals",
    "register_identifier_type": "ts_shape.signals",
    "register_identifier_pattern": "ts_shape.signals",
    "SignalSpec": "ts_shape.signals",
    "SignalSpecs": "ts_shape.signals",
    "make_id_signal": "ts_shape.datasets",
    "make_object_signals": "ts_shape.datasets",
    "Pipeline": "ts_shape.pipeline",
    "PipelineResult": "ts_shape.pipeline",
    "iter_chunked": "ts_shape.stream",
    "run_chunked": "ts_shape.stream",
    "time_range_source": "ts_shape.stream",
    "ValueMapper": "ts_shape.context.value_mapping",
    # -- transform ------------------------------------------------------
    "BooleanFilter": "ts_shape.transform.filter.boolean_filter",
    "CustomFilter": "ts_shape.transform.filter.custom_filter",
    "DataHarmonizer": "ts_shape.transform.harmonization",
    "DateTimeFilter": "ts_shape.transform.filter.datetime_filter",
    "DoubleFilter": "ts_shape.transform.filter.numeric_filter",
    "IntegerCalc": "ts_shape.transform.calculator.numeric_calc",
    "IntegerFilter": "ts_shape.transform.filter.numeric_filter",
    "IsDeltaFilter": "ts_shape.transform.filter.boolean_filter",
    "LambdaProcessor": "ts_shape.transform.functions.lambda_func",
    "PowerIntegrator": "ts_shape.transform.calculator.power_integration",
    "StringFilter": "ts_shape.transform.filter.string_filter",
    "TimestampConverter": "ts_shape.transform.time_functions.timestamp_converter",
    "TimezoneShift": "ts_shape.transform.time_functions.timezone_shift",
    "UnitConverter": "ts_shape.transform.calculator.unit_conversion",
    # -- features -------------------------------------------------------
    "BooleanStatistics": "ts_shape.features.stats.boolean_stats",
    "CrossSignalAnalytics": "ts_shape.features.cross_signal",
    "CycleDataProcessor": "ts_shape.features.cycles.cycle_processor",
    "CycleExtractor": "ts_shape.features.cycles.cycles_extractor",
    "DescriptiveFeatures": "ts_shape.features.stats.feature_table",
    "FeatureMatrixExporter": "ts_shape.features.export",
    "NumericStatistics": "ts_shape.features.stats.numeric_stats",
    "PatternRecognition": "ts_shape.features.pattern_recognition",
    "ProfileComparison": "ts_shape.features.segment_analysis.profile_comparison",
    "SegmentExtractor": "ts_shape.features.segment_analysis.segment_extractor",
    "SegmentProcessor": "ts_shape.features.segment_analysis.segment_processor",
    "StringStatistics": "ts_shape.features.stats.string_stats",
    "TimeGroupedStatistics": "ts_shape.features.time_stats.time_stats_numeric",
    "TimeWindowedFeatureTable": (
        "ts_shape.features.segment_analysis.time_windowed_features"
    ),
    "TimestampStatistics": "ts_shape.features.stats.timestamp_stats",
}

# Warnings and exceptions, kept out of ``_LAZY`` so that ``list_detectors()``
# (which catalogs ``_LAZY``) does not list them as classes to use.
_ERRORS: dict[str, str] = {
    name: "ts_shape.errors"
    for name in (
        "ColumnNotFoundError",
        "DataQualityWarning",
        "LoaderConfigWarning",
        "LoaderError",
        "NotChunkableError",
        "ObjectNotFoundError",
        "PerformanceWarning",
        "TsShapeWarning",
    )
}

# The eventlog package re-exports its whole public surface from its own
# __init__, so every eventlog name maps to that one module.
_EVENTLOG_NAMES = (
    "BacktestResult",
    "CarrierLoops",
    "ObjectKPIs",
    "attach_counts",
    "carrier_loops",
    "object_kpis",
    "presence_events",
    "wip_over_time",
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

__all__ = sorted({*_LAZY, *_ERRORS})


def __getattr__(name: str) -> object:
    """Lazily resolve a re-exported name to its defining module (PEP 562)."""
    target = _LAZY.get(name) or _ERRORS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(target)
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY) | set(_ERRORS))
