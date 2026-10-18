# Quality and SPC

How to find outliers, apply control-chart rules and compute process
capability. The classes live in `ts_shape.events.quality`.

## Example data

A measured shaft diameter that drifts upward and has a few spikes, an
in-control reference run for the control limits, and the specification
limits as two signals.

```python
import pandas as pd
import ts_shape

reference = ts_shape.make_timeseries(["shaft:diameter_ref"], n_points=200, baseline=20.0, noise=0.01, seed=1)
measured = ts_shape.make_timeseries(
    ["shaft:diameter"], n_points=500, baseline=20.0, noise=0.01, drift=0.03, n_outliers=3, start="2025-01-02"
)
spec_limits = pd.DataFrame({
    "systime": pd.Timestamp("2025-01-01"),
    "uuid": ["shaft:usl", "shaft:lsl"],
    "value_double": [20.05, 19.95],
})
df = pd.concat([reference, measured, spec_limits], ignore_index=True)
```

## Outliers

```python
from ts_shape.events.quality.outlier_detection import OutlierDetectionEvents

detector = OutlierDetectionEvents(df, value_column="value_double", signal_uuids=["shaft:diameter"])
zscore = detector.detect_outliers_zscore(n_sigma=3.0)
mad = detector.detect_outliers_mad(n_sigma=3.5)
iqr = detector.detect_outliers_iqr(threshold=(1.5, 1.5))
```

| Method | Suited for |
|---|---|
| `detect_outliers_zscore` | roughly normal data; mean and standard deviation are themselves affected by outliers |
| `detect_outliers_iqr` | skewed data |
| `detect_outliers_mad` | noisy or heavy-tailed data; median-based, the most robust of the three |
| `detect_outliers_isolation_forest` | multivariate or irregular patterns; needs `ts-shape[ml]` |

Statistics are computed per signal (`group_by_uuid=True`). Outliers closer than
`time_threshold` (default 5 minutes) are grouped into one event. Each row has
a numeric `severity` score.

## Control charts

`StatisticalProcessControlEvents` takes the center line and sigma from a
reference signal (`tolerance_uuid`, here the in-control run) and checks the
actual signal against the eight Nelson / Western Electric rules. Sigma is the
individuals-chart estimate MR̄ / 1.128 of consecutive reference values
(Montgomery, *Introduction to Statistical Quality Control*, §6.4); pass
`sigma_method="std"` for the sample standard deviation.

```python
from ts_shape.events.quality.statistical_process_control import StatisticalProcessControlEvents

spc = StatisticalProcessControlEvents(
    df, "value_double", tolerance_uuid="shaft:diameter_ref", actual_uuid="shaft:diameter", event_uuid="quality:spc"
)
limits = spc.control_limits()                   # mean and 1, 2, 3 sigma bands
violations = spc.detect_violations()            # one row per violating sample, with rule and severity
rules_1_2 = spc.detect_rule_violations(selected_rules=["rule_1", "rule_2"])
explained = spc.interpret_violations(rules_1_2) # adds interpretation, meaning, recommendation
shifts = spc.detect_cusum_shifts(k=0.5, h=5.0)  # small sustained shifts of the mean
```

`dynamic_control_limits(method="moving_range", window_size=20)` is a rolling
individuals chart (each point against the 20 points before it), and
`method="ewma"` an EWMA chart (column `ewma` against `lcl` / `ucl`). For several correlated variables use
`MultivariateSPCEvents` (Hotelling T²), and for a change of level without
reference data `ChangePointEvents`.

## Specification limits and capability

```python
from ts_shape.events.quality.tolerance_deviation import ToleranceDeviationEvents

tolerance = ToleranceDeviationEvents(
    df,
    value_column_tolerance="value_double",
    value_column_actual="value_double",
    actual_uuid="shaft:diameter",
    upper_tolerance_uuid="shaft:usl",
    lower_tolerance_uuid="shaft:lsl",
    event_uuid="quality:tolerance",
)
out_of_spec = tolerance.detect_deviations()                   # deviation_abs, deviation_pct, severity
capability = tolerance.capability_indices(target_value=20.0)  # Cp, Cpk, Pp, Ppk, Cpm in one row
```

The limits are signals, so they may change over time; each measurement is
compared with the limit in force at its timestamp. Give only one of the two
limits for a one-sided specification. Cp and Cpk use the within (short-term)
sigma MR̄ / 1.128 of consecutive measurements, Pp and Ppk the overall sample
standard deviation (AIAG SPC manual); `sigma_method="std"` makes Cp/Cpk equal
Pp/Ppk.

## Other quality detectors

| Question | Class | Page |
|---|---|---|
| Which kind of anomaly is it (spike, drift, oscillation, flatline)? | `AnomalyClassificationEvents` | [Anomaly classification](../modules/quality/anomaly-classification.md) |
| Is the sensor data usable (gaps, sampling, stuck values)? | `SignalQualityEvents`, `DataGapAnalysisEvents` | [Signal quality](../modules/quality/signal-quality.md), [Data gaps](../modules/quality/data-gap-analysis.md) |
| Is a sensor drifting or out of calibration? | `SensorDriftEvents`, `CalibrationValidationEvents` | [Sensor drift](../modules/quality/sensor-drift.md), [Calibration](../modules/quality/calibration-validation.md) |
| Do redundant sensors agree? | `MultiSensorValidationEvents` | [Multi-sensor validation](../modules/quality/multi-sensor-validation.md) |
| Is the gauge capable (Gauge R&R)? | `GaugeRepeatabilityEvents` | [Gauge R&R](../modules/quality/gauge-rr.md) |
| How does Cpk develop over time? | `CapabilityTrendingEvents` | [Capability trending](../modules/quality/capability-trending.md) |
| When did the process level change? | `ChangePointEvents` | [Change points](../modules/quality/change-point.md) |
| Several correlated variables at once? | `MultivariateSPCEvents` | [Multivariate SPC](../modules/quality/multivariate-spc.md) |
| Measurement uncertainty (GUM)? | `MeasurementUncertaintyEvents` | [Measurement uncertainty](../modules/quality/measurement-uncertainty.md) |
| Distribution shape, bimodality? | `ValueDistributionEvents` | [Value distribution](../modules/quality/value-distribution.md) |
| Is the sampling rate or resolution sufficient? | `SamplingResolutionEvents` | [Sampling and resolution](../modules/quality/sampling-resolution.md) |
| Correct a measurement for ambient conditions? | `EnvironmentalCompensationEvents` | [Environmental compensation](../modules/quality/environmental-compensation.md) |
