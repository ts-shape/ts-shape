# MultiSensorValidationEvents

Cross-checks redundant sensors that measure the same variable: windows where they disagree, pairwise bias, a consensus score, and which sensor deviates from the group.

`ts_shape.events.quality.multi_sensor_validation` · [Quality guide](../../guides/quality.md) · [API reference](../../reference/ts_shape/events/quality/multi_sensor_validation.md)

## Example

```python
import ts_shape
from ts_shape.events.quality.multi_sensor_validation import MultiSensorValidationEvents

sensors = ["temp_1", "temp_2", "temp_3"]
df = ts_shape.make_timeseries(sensors, n_points=1440, freq="1min", noise=0.1, seed=None)
df.loc[df["uuid"] == "temp_3", "value_double"] += 0.8  # temp_3 reads high

msv = MultiSensorValidationEvents(df, signal_uuids=sensors)
disagreements = msv.detect_disagreement(threshold=0.5, window="5min")
bias = msv.pairwise_bias(window="1h")
print(msv.detect_outlier_sensor(window="1h", method="median").head())
```

## Methods

| Method | Returns (per window) |
|---|---|
| `detect_disagreement(threshold, window='5min')` | windows where max − min across sensors exceeds `threshold`: `max_spread`, `sensor_high`, `sensor_low`, `duration` |
| `pairwise_bias(window='1h')` | one row per sensor pair: `sensor_a`, `sensor_b`, `bias`, `abs_bias` |
| `consensus_score(window='1h')` | `consensus_mean`, `spread_std`, `consensus_score` (1.0 = perfect agreement, 0.0 = high disagreement) |
| `detect_outlier_sensor(window='1h', method='median')` | `outlier_sensor`, `deviation`, `other_sensors_mean` |

## Notes

- Telling which sensor is off needs at least three sensors; windows with fewer reporting are skipped with a `DataQualityWarning`. On ties the first sensor in `signal_uuids` wins.
- `method` is `"median"` or `"mean"` (the group center).
- Take the `detect_disagreement` threshold from the largest spread seen in a known-good period plus a margin.
