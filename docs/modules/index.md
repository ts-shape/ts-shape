# Module Reference

One page per detector class: what it computes, a short runnable example, the methods with their key output columns, and the behavior worth knowing. Every page links to the matching guide and to the [API reference](../reference/index.md), which lists all parameters.

| Pack | Covers | Start at |
|---|---|---|
| Production | machine state, downtime, OEE, throughput, cycle times, changeovers, alarms, batches, traceability, shift and period reports | [MachineStateEvents](production/machine-state.md) |
| Quality | outliers, SPC rules, tolerances and capability, sensor drift, signal quality and gaps, gauge R&R, change points, multivariate SPC | [OutlierDetectionEvents](quality/outlier-detection.md) |
| Engineering | setpoint response, startups, thresholds, rate of change, steady state, control loops, material balance, stability | [SetpointChangeEvents](engineering/setpoint-events.md) |
| Maintenance | degradation, remaining useful life, vibration, MTBF/MTTR and Weibull | [DegradationDetectionEvents](maintenance/degradation-detection.md) |
| Energy | kW to kWh integration, consumption, peak demand, idle energy, efficiency | [PowerIntegrator](energy/power-integration.md) |
| Correlation | rolling and lagged signal correlation, coincident anomalies and cascades | [SignalCorrelationEvents](correlation/signal-correlation.md) |
| Supply chain | inventory levels and stockouts, lead times, demand patterns | [InventoryMonitoringEvents](supplychain/inventory-monitoring.md) |
| Development | critical parameters, DOE runs and effects, design space, golden batch, recipe phases | [DesignOfExperimentsEvents](development/design-of-experiments.md) |
