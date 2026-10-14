# TimeToThresholdEvents

> When will this signal hit X at the current rate? Linear-trend extrapolation to a threshold, to the edge of an operating band, or to several named limits.

**Module:** `ts_shape.events.engineering.time_to_threshold`  
**Guide:** [Process Engineering](../../guides/engineering.md)

---

## When to Use

Use for early warnings on slowly trending signals: a filter differential pressure rising toward its change limit, a bearing temperature creeping up, a tank level falling toward its low alarm, a coolant concentration drifting out of its band. [Threshold Monitoring](threshold-monitoring.md) tells you when a limit *was* crossed; this class estimates when it *will* be, if the recent trend continues.

---

## Quick Example

```python
from ts_shape.events.engineering.time_to_threshold import TimeToThresholdEvents

ttt = TimeToThresholdEvents(df, "filter4:diff_pressure")

ttt.time_to_threshold(threshold=2.5, lookback="6h")
ttt.time_to_threshold_windows(threshold=2.5, window="1h", lookback="6h")

ttt.remaining_useful_range(lower_limit=0.2, upper_limit=2.5, lookback="6h")
ttt.crossing_forecast({"warning": 2.0, "change filter": 2.5}, lookback="6h")
```

---

## Key Methods

| Method | Shape | Returns |
|---|---|---|
| `time_to_threshold(threshold, lookback, direction)` | static | one row: `current_value`, `slope_per_second`, `threshold`, `estimated_seconds`, `estimated_time`, `confidence`, `direction` |
| `time_to_threshold_windows(threshold, window, lookback)` | summary | per window: `window_start`, `current_value`, `slope_per_second`, `estimated_seconds`, `confidence`, `direction` |
| `remaining_useful_range(lower_limit, upper_limit, lookback)` | static | one row: `current_value`, `slope_per_second`, `nearest_bound` (`upper`/`lower`/`none`), `nearest_bound_value`, `estimated_seconds`, `estimated_time`, `confidence` |
| `crossing_forecast(thresholds, lookback)` | static | per named threshold: `threshold_name`, `threshold_value`, `estimated_seconds`, `estimated_time`, `reachable` |

## Notes

- A least-squares line is fitted to the last `lookback` of data (ending at the last sample). The estimate is extrapolated from the last measured value with that slope: `(threshold − current_value) / slope`.
- `confidence` is the R² of the fit (0 to 1). A low value means the trend is mostly noise and the estimate is unreliable.
- When the trend points away from the threshold, or is flat, `estimated_seconds` and `estimated_time` are empty (`reachable=False` in `crossing_forecast`). `direction="auto"` only labels the result `rising`/`falling` from the slope sign; it does not change the estimate.
- `remaining_useful_range` targets the upper bound on a rising trend and the lower bound on a falling one. If the signal is already beyond that bound, it returns 0 seconds.
- `time_to_threshold_windows` fits, for each window, the `lookback` period ending at the window's end, so you can see how the forecast evolved.
- Linear extrapolation does not capture accelerating degradation; re-run it regularly rather than trusting one long-range forecast.
