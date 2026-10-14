# DegradationDetectionEvents

Slow degradation in a sensor signal: sustained trends, growing variance, permanent level shifts, and a composite 0-100 health score against a baseline period.

`ts_shape.events.maintenance.degradation_detection` · [Engineering guide](../../guides/engineering.md) · [API reference](../../reference/ts_shape/events/maintenance/degradation_detection.md)

## Example

```python
import ts_shape
from ts_shape.events.maintenance.degradation_detection import DegradationDetectionEvents

df = ts_shape.make_timeseries(["bearing_temp"], n_points=4320, freq="1min", baseline=60.0, noise=0.5, drift=8.0)

dd = DegradationDetectionEvents(df, "bearing_temp")
trends = dd.detect_trend_degradation(window="6h", min_slope=0.00002, direction="increasing")
shifts = dd.detect_level_shift(3.0, hold="10min")
health = dd.health_score(window="1h", baseline_window="24h")
print(health[["systime", "health_score"]].tail())
```

## Methods

| Method | Returns |
|---|---|
| `detect_trend_degradation(window='1h', min_slope=0.0, direction='decreasing')` | intervals where the rolling regression slope exceeds `min_slope`: `avg_slope`, `total_change`, `duration_seconds` |
| `detect_variance_increase(window='1h', threshold_factor=2.0)` | intervals where rolling variance / baseline variance exceeds `threshold_factor`: `baseline_variance`, `current_variance`, `ratio` |
| `detect_level_shift(min_shift, hold='5min')` | point events: `systime`, `shift_magnitude`, `prev_mean`, `new_mean` |
| `health_score(window='1h', baseline_window='24h')` | per sample: `health_score`, `mean_drift_pct`, `variance_ratio`, `trend_slope` |

## Notes

- Slopes are in signal units per second; `direction` is `"increasing"` or `"decreasing"` (default).
- The baseline is the start of the data: the first `window` for `detect_variance_increase`, the first `baseline_window` for `health_score`. Start the input at a known-good period, or the baseline produces false alarms.
- A level shift must hold at least `hold` and be at least `min_shift` in size.
- Interval methods end an interval at the next sample after the condition stops holding (`interval_end="next_change"`, default) or at the last flagged sample (`"last_sample"`).
