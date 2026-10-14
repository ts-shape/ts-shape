# Process engineering

How to measure the response to setpoint changes, detect startups and find
steady states, ranges and disturbances. The classes live in
`ts_shape.events.engineering`. The end of the page lists the maintenance,
energy, correlation, supply-chain and development packs.

## Example data

An oven setpoint that steps from 20 to 180 and then to 160, and an oven
temperature that follows with a first-order lag (time constant 60 s), sampled
every 5 seconds.

```python
import numpy as np
import pandas as pd

t = pd.date_range("2024-06-03 06:00", periods=3 * 3600 // 5, freq="5s", tz="UTC")
setpoint = np.where(t < t[0] + pd.Timedelta("1h"), 20.0, np.where(t < t[0] + pd.Timedelta("2h"), 180.0, 160.0))
actual = np.empty(len(t))
actual[0] = 20.0
for i in range(1, len(t)):
    actual[i] = actual[i - 1] + (setpoint[i] - actual[i - 1]) * 5 / 60
actual += np.random.default_rng(0).normal(0, 0.2, len(t))

df = pd.concat([
    pd.DataFrame({"systime": t, "uuid": "oven:setpoint", "value_double": setpoint}),
    pd.DataFrame({"systime": t, "uuid": "oven:temp", "value_double": actual}),
], ignore_index=True)
```

## Setpoint changes and control quality

```python
from ts_shape.events.engineering.setpoint_events import SetpointChangeEvents

setpoint = SetpointChangeEvents(df, setpoint_uuid="oven:setpoint")
steps = setpoint.detect_setpoint_steps(min_delta=5.0)      # change_type, magnitude, prev_level, new_level

settling = setpoint.time_to_settle("oven:temp", tol=1.0, hold="30s", lookahead="30min")
quality = setpoint.control_quality_metrics("oven:temp", tol=1.0, hold="30s", lookahead="30min")
quality[["start", "t_settle_seconds", "rise_time_seconds", "overshoot_pct", "steady_state_error"]]
```

- `time_to_settle`: the time until the actual value stays within `tol` of the
  new setpoint for at least `hold`. `settle_pct` gives the band as a fraction
  of the step instead.
- `control_quality_metrics` returns, per setpoint change, settling time, rise
  time (10 % to 90 %), overshoot and undershoot, oscillation count and
  frequency, decay rate and steady-state error. Each is also available as its
  own method (`rise_time`, `overshoot_metrics`, `oscillation_frequency`,
  `decay_rate`).
- `detect_setpoint_ramps(min_rate=...)` finds gradual changes.

## Startups

```python
from ts_shape.events.engineering.startup_events import StartupDetectionEvents

startup = StartupDetectionEvents(df, target_uuid="oven:temp")
by_threshold = startup.detect_startup_by_threshold(threshold=100.0, min_above="2min")
by_slope = startup.detect_startup_by_slope(min_slope=0.5, min_duration="30s")
```

`hysteresis=(upper, lower)` on `detect_startup_by_threshold` avoids repeated
detections around the threshold. The class also detects startups from several
signals at once (`detect_startup_multi_signal`), against an adaptive baseline
(`detect_startup_adaptive`), finds failed startups (`detect_failed_startups`)
and tracks startup phases (`track_startup_phases`).

To check whether a startup came early or late against a shift start, compare
the `start` column with your planned time in pandas:

```python
planned = pd.Timestamp("2024-06-03 07:00", tz="UTC")
by_threshold["minutes_after_plan"] = (by_threshold["start"] - planned).dt.total_seconds() / 60
```

## Other engineering detectors

| Question | Class | Page |
|---|---|---|
| Is the value above a limit, with hysteresis and several levels? | `ThresholdMonitoringEvents` | [Threshold monitoring](../modules/engineering/threshold-monitoring.md) |
| Does the value change too fast, or jump? | `RateOfChangeEvents` | [Rate of change](../modules/engineering/rate-of-change.md) |
| When is the process steady, when transient? | `SteadyStateDetectionEvents` | [Steady state](../modules/engineering/steady-state.md) |
| Does the actual value diverge from a reference signal? | `SignalComparisonEvents` | [Signal comparison](../modules/engineering/signal-comparison.md) |
| What is the operating envelope, when does the regime change? | `OperatingRangeEvents` | [Operating range](../modules/engineering/operating-range.md) |
| How long do warm-up and cool-down take? | `WarmUpCoolDownEvents` | [Warm-up analysis](../modules/engineering/warmup-analysis.md) |
| How long until a value reaches a threshold? | `TimeToThresholdEvents` | [Time to threshold](../modules/engineering/time-to-threshold.md) |
| Windowed statistics per shift or batch? | `ProcessWindowEvents` | [Process windows](../modules/engineering/process-windows.md) |
| Is a control loop oscillating or saturated? | `ControlLoopHealthEvents` | [Control loop health](../modules/engineering/control-loop-health.md) |
| How fast does the process recover from an upset? | `DisturbanceRecoveryEvents` | [Disturbance recovery](../modules/engineering/disturbance-recovery.md) |
| Do inputs and outputs balance? | `MaterialBalanceEvents` | [Material balance](../modules/engineering/material-balance.md) |
| One stability score for a signal? | `ProcessStabilityIndexEvents` | [Process stability](../modules/engineering/process-stability.md) |

## Other packs

| Pack | Classes | Pages |
|---|---|---|
| `events.maintenance` | `DegradationDetectionEvents`, `FailurePredictionEvents` (remaining useful life), `VibrationAnalysisEvents`, `ReliabilityEvents` (MTBF, MTTR, Weibull), `AssetHealthIndexEvents` | [Degradation](../modules/maintenance/degradation-detection.md), [Failure prediction](../modules/maintenance/failure-prediction.md), [Vibration](../modules/maintenance/vibration-analysis.md), [Reliability](../modules/maintenance/reliability.md) |
| `events.energy` | `EnergyConsumptionEvents`, `EnergyEfficiencyEvents`, `EnergyPerformanceIndicatorEvents` (EnPI), `IdleEnergyDetectionEvents`, `CarbonIntensityEvents`; `PowerIntegrator` (`ts_shape.transform.calculator`) integrates power to energy | [Power integration](../modules/energy/power-integration.md), [Consumption](../modules/energy/consumption-analysis.md), [Efficiency](../modules/energy/efficiency-tracking.md) |
| `events.correlation` | `SignalCorrelationEvents`, `AnomalyCorrelationEvents`, `RootCauseCorrelationEvents` | [Signal correlation](../modules/correlation/signal-correlation.md), [Anomaly correlation](../modules/correlation/anomaly-correlation.md) |
| `events.supplychain` | `InventoryMonitoringEvents`, `LeadTimeAnalysisEvents`, `DemandPatternEvents` | [Inventory](../modules/supplychain/inventory-monitoring.md), [Lead time](../modules/supplychain/lead-time-analysis.md), [Demand patterns](../modules/supplychain/demand-patterns.md) |
| `events.development` | `DesignOfExperimentsEvents`, `DesignSpaceEvents`, `GoldenBatchDeviationEvents`, `RecipePhaseAdherenceEvents`, `CriticalParameterRankingEvents` | [Design of experiments](../modules/development/design-of-experiments.md), [Design space](../modules/development/design-space.md), [Golden batch](../modules/development/golden-batch.md), [Recipe phases](../modules/development/recipe-phase-adherence.md), [Parameter ranking](../modules/development/critical-parameter-ranking.md) |
