# OperatingRangeEvents

The operating envelope of a signal per window (min, percentiles, max), shifts of the operating point, time spent in, below and above a range, and a value histogram.

`ts_shape.events.engineering.operating_range` · [Engineering guide](../../guides/engineering.md) · [API reference](../../reference/ts_shape/events/engineering/operating_range.md)

## Example

```python
import ts_shape
from ts_shape.events.engineering.operating_range import OperatingRangeEvents

df = ts_shape.make_timeseries(["extruder_speed"], n_points=1440, freq="1min", baseline=200.0, noise=8.0)
df.loc[900:, "value_double"] += 25.0  # new operating point

op = OperatingRangeEvents(df, "extruder_speed")
envelope = op.operating_envelope(window="1h")
result = op.detect_regime_change(window="1h", shift_threshold=2.0)
print(result[["start", "prev_mean", "new_mean", "shift_magnitude"]])
in_range = op.time_in_range(180.0, 220.0, window="1h")
histogram = op.value_distribution(n_bins=10)
```

## Methods

| Method | Returns |
|---|---|
| `operating_envelope(window='1h')` | per window: `min_value`, `p5`, `mean_value`, `p95`, `max_value`, `range_width` |
| `detect_regime_change(window='1h', shift_threshold=2.0)` | per shift: `start` (previous window), `end` (new window), `prev_mean`, `new_mean`, `shift_magnitude` |
| `time_in_range(lower_limit, upper_limit, window='1h', time_weighted=True)` | per window: `time_in_range_pct`, `time_below_pct`, `time_above_pct` |
| `value_distribution(n_bins=10)` | equal-width bins between min and max: `bin_lower`, `bin_upper`, `count`, `pct`, `cumulative_pct` |

## Notes

- A regime change is a window mean that differs from the previous window's by more than `shift_threshold` × the two windows' average standard deviation. Windows with fewer than two samples are skipped.
- `time_in_range` limits are inclusive. With `time_weighted=True` (default) each value is held until the next sample, so a window without samples that holds an earlier value is reported too; `time_weighted=False` gives the share of samples.
