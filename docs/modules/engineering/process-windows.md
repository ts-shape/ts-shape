# ProcessWindowEvents

Descriptive statistics of a signal per time window, and the windows whose mean or spread changed against the previous window or stands out from all windows.

`ts_shape.events.engineering.process_window` · [Engineering guide](../../guides/engineering.md) · [API reference](../../reference/ts_shape/events/engineering/process_window.md)

## Example

```python
import ts_shape
from ts_shape.events.engineering.process_window import ProcessWindowEvents

df = ts_shape.make_timeseries(["mixer_torque"], n_points=1440, freq="1min", baseline=40.0, noise=0.5)
df.loc[700:760, "value_double"] += 3.0

pw = ProcessWindowEvents(df, "mixer_torque")
stats = pw.windowed_statistics(window="1h")
print(pw.detect_mean_shift(window="1h", sensitivity=2.0))
variance = pw.detect_variance_change(window="1h", ratio_threshold=2.0)
outliers = pw.window_comparison(window="1h")
```

## Methods

All arguments are keyword-only.

| Method | Returns |
|---|---|
| `windowed_statistics(window='1h')` | per window: `sample_count`, `mean`, `std`, `min`, `max`, `median`, `p25`, `p75`, `range` |
| `detect_mean_shift(window='1h', sensitivity=2.0)` | `prev_mean`, `current_mean`, `shift_sigma` |
| `detect_variance_change(window='1h', ratio_threshold=2.0)` | `prev_std`, `current_std`, `variance_factor` |
| `window_comparison(window='1h')` | per window: `mean`, `z_score_vs_global`, `is_anomalous` |

## Notes

- A mean shift is flagged when |current mean − previous mean| > `sensitivity` × the previous window's standard deviation.
- `std` is the sample standard deviation (ddof=1): NaN for a window with one value. Such windows take no part in mean-shift or variance-change detection.
- A variance change is flagged when current std / previous std is above `ratio_threshold` or below its reciprocal. A change from a constant window (std 0) to a varying one is flagged with `variance_factor` NaN: the ratio is never `inf`.
- In both, `start` is the previous window's start and `end` the current window's start.
- `is_anomalous`: the window mean lies more than 2 standard deviations of all window means from their average.
