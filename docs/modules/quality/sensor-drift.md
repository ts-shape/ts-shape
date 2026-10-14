# SensorDriftEvents

Calibration drift of an inline sensor per window: zero-point offset, span (gain) changes, the trend of the window mean or spread, and a 0-100 calibration health score, against a reference value, a reference signal or the first window.

`ts_shape.events.quality.sensor_drift` · [Quality guide](../../guides/quality.md) · [API reference](../../reference/ts_shape/events/quality/sensor_drift.md)

## Example

```python
import ts_shape
from ts_shape.events.quality.sensor_drift import SensorDriftEvents

df = ts_shape.make_timeseries(["ph_probe_01"], n_points=4320, freq="1min", baseline=7.0, noise=0.02, drift=0.15)

drift = SensorDriftEvents(df, signal_uuid="ph_probe_01", reference_value=7.0)  # pH 7 buffer
zero = drift.detect_zero_drift(window="8h", threshold=0.05)
span = drift.detect_span_drift(window="8h")
health = drift.calibration_health(window="8h", tolerance=0.1)
print(zero[["start", "mean_offset", "drift_rate", "severity"]])
```

## Methods

All arguments are keyword-only.

| Method | Returns (per window) |
|---|---|
| `detect_zero_drift(window='8h', threshold=None)` | `mean_offset`, `drift_rate`, `severity` |
| `detect_span_drift(window='8h')` | `sensitivity` (window mean / reference), `sensitivity_change_pct` (vs the first window) |
| `drift_trend(window='1D', metric='mean')` | trend of the window `mean` or `std`: `value`, `slope`, `r_squared`, `direction` |
| `calibration_health(window='8h', tolerance=None)` | `bias`, `precision` (window std), `health_score` |

## Notes

- The reference is `reference_value` (a constant) or `reference_uuid` (a signal in the same frame, averaged per window). Without one, `detect_zero_drift` measures the offset from the first window's mean, and `detect_span_drift` needs a reference.
- `drift_rate` is the offset change from the previous window. `severity` is `critical` above `threshold`, `high` above 66 %, `medium` above 33 %, else `low`; without `threshold` it is 3 × the first window's standard deviation.
- With `tolerance`, `health_score` = 100 − up to 50 for bias (relative to `tolerance`) − up to 50 for precision (relative to `tolerance / 3`). Without it, the score is relative to the window's own spread.
- Windows with fewer than 2 samples are skipped.
