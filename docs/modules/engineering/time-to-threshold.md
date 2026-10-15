# TimeToThresholdEvents

Forecasts when a trending signal (e.g. filter differential pressure, wear) will cross a threshold, by fitting a straight line to the recent data.

`ts_shape.events.engineering.time_to_threshold` · [Engineering guide](../../guides/engineering.md) · [API reference](../../reference/ts_shape/events/engineering/time_to_threshold.md)

## Example

```python
import ts_shape
from ts_shape.events.engineering.time_to_threshold import TimeToThresholdEvents

df = ts_shape.make_timeseries(["filter4:diff_pressure"], n_points=1440, freq="1min",
                              baseline=1.0, noise=0.02, drift=0.9)

ttt = TimeToThresholdEvents(df, "filter4:diff_pressure")
result = ttt.time_to_threshold(2.5, lookback="6h")
print(result[["current_value", "estimated_seconds", "confidence", "direction"]])
history = ttt.time_to_threshold_windows(2.5, window="1h", lookback="6h")
forecast = ttt.crossing_forecast({"warning": 2.0, "change filter": 2.5}, lookback="6h")
```

## Methods

| Method | Returns |
|---|---|
| `time_to_threshold(threshold, lookback='1h', direction='auto')` | one row: `current_value`, `slope_per_second`, `threshold`, `estimated_seconds`, `estimated_time`, `confidence`, `direction` |
| `time_to_threshold_windows(threshold, window='1h', lookback='1h')` | per window: `window_start`, `current_value`, `slope_per_second`, `estimated_seconds`, `confidence`, `direction` |
| `remaining_useful_range(lower_limit, upper_limit, lookback='1h')` | one row: `nearest_bound` (`upper`/`lower`/`none`), `nearest_bound_value`, `estimated_seconds`, `estimated_time`, `confidence` |
| `crossing_forecast(thresholds, lookback='1h')` | per named threshold: `threshold_value`, `estimated_seconds`, `estimated_time`, `reachable` |

## Notes

- A least-squares line is fitted to the last `lookback` of data (ending at the last sample), and the estimate is `(threshold − current_value) / slope` from the last measured value.
- `confidence` is the R² of the fit (0 to 1); a low value means the trend is mostly noise.
- When the trend points away from the threshold, or is flat, `estimated_seconds` and `estimated_time` are empty (`reachable=False` in `crossing_forecast`). `direction="auto"` only labels the result `rising`/`falling`.
- `remaining_useful_range` targets the upper bound on a rising trend and the lower bound on a falling one; if the signal is already beyond it, it returns 0 seconds.
- `time_to_threshold_windows` fits, for each window, the `lookback` ending at the window's end, so you can see how the forecast evolved.
- Linear extrapolation does not capture accelerating degradation; re-run it rather than trusting one long-range forecast.
