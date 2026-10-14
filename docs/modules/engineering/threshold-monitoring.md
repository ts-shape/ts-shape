# ThresholdMonitoringEvents

Intervals above (or below) named threshold levels, alarm intervals with hysteresis, time above a threshold per window, and the exceedance trend.

`ts_shape.events.engineering.threshold_monitoring` · [Engineering guide](../../guides/engineering.md) · [API reference](../../reference/ts_shape/events/engineering/threshold_monitoring.md)

## Example

```python
import ts_shape
from ts_shape.events.engineering.threshold_monitoring import ThresholdMonitoringEvents

df = ts_shape.make_timeseries(["reactor_pressure"], n_points=2880, freq="30s", baseline=80.0, noise=4.0, drift=12.0)

tm = ThresholdMonitoringEvents(df, "reactor_pressure")
levels = tm.multi_level_threshold({"warning": 85.0, "alarm": 92.0, "critical": 98.0}, direction="above")
alarms = tm.threshold_with_hysteresis(upper_limit=90.0, lower_limit=85.0)
print(tm.time_above_threshold(85.0, window="1h").tail())
```

## Methods

| Method | Returns |
|---|---|
| `multi_level_threshold(levels, direction='above')` | intervals per level: `start`, `end`, `duration`, `level`, `peak_value` |
| `threshold_with_hysteresis(upper_limit, lower_limit, max_gap=None)` | alarm intervals: `start`, `end`, `duration`, `peak_value`. With `max_gap` (e.g. `"15min"`) an alarm ends at its last sample before a longer gap; after the gap the signal must reach `upper_limit` again |
| `time_above_threshold(threshold, window='1h')` | per window: `time_above`, `pct_above`, `exceedance_count` |
| `threshold_exceedance_trend(threshold, window='1D')` | per window: `exceedance_count`, `total_duration`, `trend_direction` |

## Notes

- `levels` maps names to values, e.g. `{"warning": 80, "alarm": 90}`; `direction` is `"above"` or `"below"`.
- Hysteresis: the alarm starts when the signal crosses `upper_limit` and ends when it drops below `lower_limit`, which avoids chattering near a single limit.
- With `interval_end="next_change"` (default) intervals end at the next sample and `time_above_threshold` holds each value until the next sample, so it measures real time. `"last_sample"` uses the share of samples × window length, which is right only for uniform sampling.
