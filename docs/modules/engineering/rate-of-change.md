# RateOfChangeEvents

Intervals where a signal changes faster than a rate limit, rate statistics per window, and sudden step jumps.

`ts_shape.events.engineering.rate_of_change` · [Engineering guide](../../guides/engineering.md) · [API reference](../../reference/ts_shape/events/engineering/rate_of_change.md)

## Example

```python
import ts_shape
from ts_shape.events.engineering.rate_of_change import RateOfChangeEvents

df = ts_shape.make_timeseries(["flow_rate"], n_points=3600, freq="1s", baseline=50.0, noise=0.1)
df.loc[1800:, "value_double"] += 20.0  # step

roc = RateOfChangeEvents(df, "flow_rate")
rapid = roc.detect_rapid_change(threshold=0.5)
stats = roc.rate_statistics(window="15min")
result = roc.detect_step_changes(min_delta=10.0, max_duration="5s")
print(result[["systime", "value_before", "value_after", "delta"]])
```

## Methods

| Method | Returns |
|---|---|
| `detect_rapid_change(threshold)` | intervals with \|rate\| ≥ `threshold`: `start`, `end`, `max_rate`, `direction` |
| `rate_statistics(window='1h')` | per window: `mean_rate`, `std_rate`, `max_rate`, `min_rate` |
| `detect_step_changes(min_delta, max_duration='5s')` | jumps of at least `min_delta` within `max_duration`: `time` (after the jump), `value_before`, `value_after`, `delta`, `duration` |

## Notes

- Rates are in signal units per second, computed between consecutive samples. Each run of consecutive samples at or above `threshold` is one interval; filter on `duration_seconds` to drop short ones.
- `detect_rapid_change(window=...)` is deprecated and ignored.
