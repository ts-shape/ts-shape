# Examples

Runnable demo scripts covering every ts-shape module. Each example generates synthetic data and demonstrates the full API.

---

## By Domain

| Example | Modules Demonstrated | Run Command |
|---------|---------------------|-------------|
| [Quality Events](quality-events.md) | `OutlierDetectionEvents`, `StatisticalProcessControlRuleBased`, `ToleranceDeviationEvents` | `python examples/quality_events_demo.py` |
| [Production Events](production-events.md) | `MachineStateEvents`, `LineThroughputEvents`, `ChangeoverEvents`, `FlowConstraintEvents` | `python examples/production_events_demo.py` |
| [Production Tracking](production-tracking.md) | `PartProductionTracking`, `CycleTimeTracking`, `DowntimeTracking`, `QualityTracking`, `ShiftReporting` | `python examples/production_tracking_demo.py` |
| [Engineering Events](engineering-events.md) | `SetpointChangeEvents` (advanced usage) | `python examples/setpoint_events_advanced_usage.py` |
| [Maintenance Events](maintenance-events.md) | `DegradationDetectionEvents`, `FailurePredictionEvents`, `VibrationAnalysisEvents` | `python examples/maintenance_events_demo.py` |
| [Energy Events](energy-events.md) | `EnergyConsumptionEvents`, `EnergyEfficiencyEvents` | `python examples/energy_events_demo.py` |
| [Correlation Events](correlation-events.md) | `SignalCorrelationEvents`, `AnomalyCorrelationEvents` | `python examples/correlation_events_demo.py` |
| [Supply Chain Events](supplychain-events.md) | `InventoryMonitoringEvents`, `LeadTimeAnalysisEvents`, `DemandPatternEvents` | `python examples/supplychain_events_demo.py` |

## By Pipeline Stage

| Example | Modules Demonstrated | Run Command |
|---------|---------------------|-------------|
| [Loader Usage](loader-usage.md) | `ParquetLoader`, `MetadataJsonLoader`, `DataIntegratorHybrid` | `python examples/loader_usage_demo.py` |
| [Transform Operations](transform-operations.md) | `NumericFilter`, `StringFilter`, `DateTimeFilter`, `NumericCalc` | `python examples/transform_operations_demo.py` |
| [Statistics & Features](statistics-features.md) | `NumericStatistics`, `TimeGroupedStatistics`, `FeatureTable` | `python examples/features_statistics_demo.py` |
| [Cycle Extractor](cycle-extractor.md) | `CycleExtractor` (6 detection methods) | `python examples/cycle_extractor_enhancements_demo.py` |
| [Context Enricher](context-enricher.md) | `ContextEnricher`, `ValueMapping` | `python examples/context_enricher_demo.py` |
| [Statistics (Basic)](statistics-basic.md) | `NumericStatistics`, `StringStatistics`, `BooleanStatistics` | `python examples/statistics_demo.py` |
| [Event Log (XES & OCEL)](eventlog.md) | `to_event_log`, `concat`, `to_event_log_xes`, `to_event_log_ocel` | `python examples/eventlog_demo.py` |
