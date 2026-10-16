# Maintenance

How to watch an asset wear out and plan the intervention: trends and health
scores from condition signals, the time left until a limit, and MTBF, MTTR and
Weibull statistics from the failure history. The classes live in
`ts_shape.events.maintenance`.

## Example data

A press bearing over three days, one sample every 10 minutes: vibration that
rises steadily, a temperature that stays put.

```python
import pandas as pd
import ts_shape

kw = dict(n_points=432, freq="10min", start="2024-03-01")
df = pd.concat([
    ts_shape.make_timeseries(["press:vibration"], baseline=2.0, noise=0.05, drift=3.0, **kw),
    ts_shape.make_timeseries(["press:bearing_temp"], baseline=55.0, noise=0.3, **kw),
], ignore_index=True)
```

## Is the signal degrading?

`DegradationDetectionEvents` compares one signal with its own opening
period: trends, growing variance, level shifts and a 0-100 health score.

```python
from ts_shape.events.maintenance.degradation_detection import DegradationDetectionEvents

vib = DegradationDetectionEvents(df, "press:vibration")
trend = vib.detect_trend_degradation(window="6h", min_slope=0.0, direction="increasing")
health = vib.health_score(window="6h", baseline_window="12h")
print(health[["systime", "health_score"]].iloc[[0, -1]])
```

The baseline is the start of the data, so start at a known-good period (after a
repair or a calibration).

## One score for the asset

`AssetHealthIndexEvents` fuses several signals, each with the failure mode it
watches and a weight, into one score per window and projects when it reaches a
critical level:

```python
from ts_shape.events.maintenance.asset_health_index import AssetHealthIndexEvents

asset = AssetHealthIndexEvents(
    df, asset_id="press-01",
    signals={"press:vibration": "bearing", "press:bearing_temp": "thermal"},
)
index = asset.health_index(window="6h", baseline_window="12h")
forecast = asset.maintenance_forecast(window="6h", baseline_window="12h", critical_threshold=60.0)
print(forecast[["current_health", "decline_per_day", "recommended_maintenance_days"]].round(1))
```

## How long until the limit?

`FailurePredictionEvents` extrapolates the signal itself to a failure
threshold you know from the data sheet or from experience:

```python
from ts_shape.events.maintenance.failure_prediction import FailurePredictionEvents

fp = FailurePredictionEvents(df, "press:vibration")
tte = fp.time_to_threshold(6.0, direction="increasing")
print(tte[["systime", "current_value", "estimated_time_seconds"]].tail(1))
```

For vibration spectra (RMS, peak, crest factor, bearing frequencies) see
[Vibration analysis](../modules/maintenance/vibration-analysis.md).

## Failure history

`ReliabilityEvents` reads a status signal with run and failure states and
reports MTBF, MTTR and availability, a Weibull fit of the times between
failures, and the probability of surviving the next hours:

```python
import numpy as np
from ts_shape.events.maintenance.reliability import ReliabilityEvents

rng = np.random.default_rng(3)
up_h, down_h = rng.weibull(2.0, 12) * 150, rng.uniform(1, 6, 12)  # 12 run/repair cycles
starts = np.cumsum(np.column_stack([up_h, down_h]).ravel())
status = pd.DataFrame({
    "systime": pd.Timestamp("2024-01-01") + pd.to_timedelta(np.r_[0, starts[:-1]], unit="h"),
    "uuid": "press:status", "value_string": ["RUN", "BREAKDOWN"] * 12,
})
rel = ReliabilityEvents(status, "press:status", failure_states=("BREAKDOWN",),
                        value_column="value_string", max_gap="30D")
print(rel.mtbf_mttr()[["failures", "mtbf_hours", "mttr_hours", "availability_pct"]].round(1))
fit = rel.weibull_fit()   # beta > 1 ("wear_out"): failures increase with age
```

## Which class answers which question

| Question | Class | Page |
|---|---|---|
| Is one signal trending, getting noisier or shifting? | `DegradationDetectionEvents` | [Degradation detection](../modules/maintenance/degradation-detection.md) |
| One health score from several signals, and when to intervene? | `AssetHealthIndexEvents` | [Asset health index](../modules/maintenance/asset-health-index.md) |
| How long until a failure threshold? | `FailurePredictionEvents` | [Failure prediction](../modules/maintenance/failure-prediction.md) |
| Vibration levels and bearing defect frequencies? | `VibrationAnalysisEvents` | [Vibration analysis](../modules/maintenance/vibration-analysis.md) |
| MTBF, MTTR, Weibull, survival probability? | `ReliabilityEvents` | [Reliability](../modules/maintenance/reliability.md) |
| Which signal moved before the breakdowns? | `RootCauseCorrelationEvents` | [Root cause correlation](../modules/correlation/root-cause-correlation.md) |
| Hours since service, starts per day? | `RuntimeAccountingEvents` | [Runtime accounting](../modules/production/runtime-accounting.md) |
