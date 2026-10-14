# FailurePredictionEvents

Remaining useful life by linear extrapolation of the recent trend to a failure threshold, escalating warning/critical exceedances per window, and time to a threshold from the current rate of change.

`ts_shape.events.maintenance.failure_prediction` · [Engineering guide](../../guides/engineering.md) · [API reference](../../reference/ts_shape/events/maintenance/failure_prediction.md)

## Example

```python
import ts_shape
from ts_shape.events.maintenance.failure_prediction import FailurePredictionEvents

df = ts_shape.make_timeseries(["bearing_temp"], n_points=4320, freq="1min", baseline=60.0, noise=0.5, drift=25.0)

fp = FailurePredictionEvents(df, "bearing_temp")
rul = fp.remaining_useful_life(degradation_rate=0.0001, failure_threshold=95.0)
print(rul[["systime", "current_value", "rul_hours", "confidence"]].tail())
pattern = fp.detect_exceedance_pattern(80.0, 82.0, window="1D")
tte = fp.time_to_threshold(95.0, direction="increasing")
```

## Methods

| Method | Returns |
|---|---|
| `remaining_useful_life(degradation_rate, failure_threshold)` | per sample: `current_value`, `rul_seconds`, `rul_hours`, `confidence` |
| `detect_exceedance_pattern(warning_threshold, critical_threshold, window='1h')` | per window: `warning_count`, `critical_count`, `escalation_detected` |
| `time_to_threshold(threshold, direction='increasing')` | per sample: `current_value`, `rate_of_change`, `estimated_time_seconds` |

## Notes

- `degradation_rate` (units per second) is the fallback slope when no local slope can be computed; its sign gives the failure direction (positive: failure when the value rises to the threshold).
- `rul_*` is 0 once the threshold is reached and None when no failure is projected (flat trend or a trend moving away from the threshold).
- Exceedances compare the absolute value with the thresholds; `escalation_detected` marks increasing counts.
- On a signal without a real trend the RUL estimate is meaningless; confirm the trend first, e.g. with [DegradationDetectionEvents](degradation-detection.md).
- For a forecast with a fit quality and named thresholds see [TimeToThresholdEvents](../engineering/time-to-threshold.md).
