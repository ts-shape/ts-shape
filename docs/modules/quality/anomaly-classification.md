# AnomalyClassificationEvents

Finds anomalous windows in a numeric signal and labels them spike, drift, oscillation, flatline or level shift; each type also has its own detector.

`ts_shape.events.quality.anomaly_classification` · [Quality guide](../../guides/quality.md) · [API reference](../../reference/ts_shape/events/quality/anomaly_classification.md)

## Example

```python
import ts_shape
from ts_shape.events.quality.anomaly_classification import AnomalyClassificationEvents

df = ts_shape.make_timeseries(["oven:temp"], n_points=2000, freq="10s", drift=15.0, n_outliers=5)

ac = AnomalyClassificationEvents(df, signal_uuid="oven:temp")
print(ac.classify_anomalies(window="10min", n_sigma=3.0)[["start", "anomaly_type", "severity", "details"]])
flatlines = ac.detect_flatline(min_duration="1min", tolerance=1e-6)
drifts = ac.detect_drift(window="1h", min_slope=0.0005)
```

## Methods

All arguments are keyword-only.

| Method | Returns |
|---|---|
| `classify_anomalies(window='10min', n_sigma=3.0, slope_per='second')` | anomalous windows: `start`, `end`, `anomaly_type`, `severity` (`warning`/`critical`), `details` |
| `detect_flatline(min_duration='1min', tolerance=1e-6, interval_end='next_change', max_gap=None)` | stuck intervals: `start`, `end`, `duration`, `stuck_value` |
| `detect_oscillation(window='1min', min_crossings=6)` | windows with at least `min_crossings` zero-crossings of the detrended signal: `crossing_count`, `amplitude` |
| `detect_drift(window='1h', min_slope=0.01, slope_per='second')` | non-overlapping windows whose least-squares slope exceeds `min_slope`: `slope`, `direction` |

## Notes

- A flatline run continues while each step is within `tolerance`. With `interval_end="next_change"` (default) it ends at the next differing sample, so a deadband historian (one row, then nothing until the value moves) still reports a stuck sensor; a single-sample run counts only when the gap after it is longer than the median sampling interval. `"last_sample"` ends the run at its own last sample.
- Samples at the same timestamp add no duration: a duplicated sample does not start a flatline.
- `max_gap` (e.g. `"15min"`): a gap between two samples longer than this is "no data"; the run before it ends at its last sample and a new run starts after it. `None` (default) lets a run span any gap.
- `slope_per="second"` (default) fits against elapsed time, so slopes are in signal units per second whatever the sampling rate. `slope_per="sample"` fits against the sample index.
- In `classify_anomalies` a window is a drift when its slope exceeds 0.1 × the signal's overall std per sample (per median sampling interval with `slope_per="second"`). A spike is a window with 1 to max(3, 10 % of samples) z-scores above `n_sigma`; it is critical when the largest z-score exceeds 2 × `n_sigma`.
- Value column defaults to `value_double`.
