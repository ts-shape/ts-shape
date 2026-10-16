# SignalComparisonEvents

Compares an actual signal with a reference signal (setpoint vs actual, sensor A vs sensor B): divergence intervals, deviation statistics per window, the tracking-error trend and per-window correlation.

`ts_shape.events.engineering.signal_comparison` · [Engineering guide](../../guides/engineering.md) · [API reference](../../reference/ts_shape/events/engineering/signal_comparison.md)

## Example

```python
import pandas as pd
import ts_shape
from ts_shape.events.engineering.signal_comparison import SignalComparisonEvents

df = pd.concat([
    ts_shape.make_timeseries(["pressure_sp"], n_points=1440, freq="1min", baseline=8.0, noise=0.0),
    ts_shape.make_timeseries(["pressure_pv"], n_points=1440, freq="1min", baseline=8.0, noise=0.2, drift=3.0),
], ignore_index=True)

cmp = SignalComparisonEvents(df, reference_uuid="pressure_sp")
result = cmp.detect_divergence("pressure_pv", tolerance=2.0, min_duration="10min")
print(result[["start", "end", "max_deviation", "direction"]])
stats = cmp.deviation_statistics("pressure_pv", window="1h")
trend = cmp.tracking_error_trend("pressure_pv", window="6h")
```

## Methods

| Method | Returns |
|---|---|
| `detect_divergence(actual_uuid, tolerance, min_duration='1min')` | intervals with \|actual − reference\| > `tolerance`: `max_deviation`, `mean_deviation`, `direction` |
| `deviation_statistics(actual_uuid, window='1h')` | per window: `mae`, `max_error`, `rmse`, `bias` |
| `tracking_error_trend(actual_uuid, window='1D')` | per window: `mae`, `trend_slope`, `trend_direction` |
| `correlation_windows(actual_uuid, window='1h')` | per window: `correlation` (Pearson), `sample_count` |

## Notes

- Deviation is actual − reference, so a positive `bias` means the actual reads high.
- Two signals can drift apart slowly while staying inside `tolerance`; a falling `correlation` or rising `mae` trend shows it.
- `correlation_windows` skips windows with fewer than two aligned samples.
