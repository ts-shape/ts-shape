# PerformanceLossEvents

Speed losses: performance against a target cycle time per shift, per window and over time, from a part counter or cycle trigger.

`ts_shape.events.production.performance_loss` · [Production guide](../../guides/production.md) · [API reference](../../reference/ts_shape/events/production/performance_loss.md)

## Example

```python
import numpy as np
import pandas as pd
from ts_shape.events.production.performance_loss import PerformanceLossEvents

rng = np.random.default_rng(0)
t = pd.Timestamp("2024-01-01 06:00") + pd.to_timedelta(rng.normal(5.0, 0.4, 5000).cumsum(), unit="s")
df = pd.DataFrame({"systime": t, "uuid": "line1:parts", "value_integer": np.arange(1, 5001)})

perf = PerformanceLossEvents(df)
result = perf.performance_by_shift(cycle_uuid="line1:parts", target_cycle_time=4.5, max_cycle_gap="1min")
print(result[["date", "shift", "actual_parts", "performance_pct", "loss_minutes"]])
slow = perf.slow_periods(cycle_uuid="line1:parts", target_cycle_time=4.5, threshold_pct=90.0)
trend = perf.performance_trend(cycle_uuid="line1:parts", target_cycle_time=4.5, window="1h")
```

## Methods

| Method | Returns |
|---|---|
| `performance_by_shift(cycle_uuid, target_cycle_time, max_cycle_gap=None)` | per date and shift: `actual_parts`, `avg_cycle_time_seconds`, `target_cycle_time_seconds`, `performance_pct`, `loss_seconds`, `run_seconds` |
| `slow_periods(cycle_uuid, target_cycle_time, threshold_pct=90.0, window='1h')` | windows with `performance_pct` below `threshold_pct`: same metrics |
| `performance_trend(cycle_uuid, target_cycle_time, window='1D')` | per `period`: same metrics |

## Notes

- Performance = `target_cycle_time` × `actual_parts` / `run_seconds` (ISO 22400-2), and `avg_cycle_time_seconds` = `run_seconds` / `actual_parts`. `target_cycle_time` is in seconds.
- A numeric signal (`value_integer` by default) is a part counter: `actual_parts` is the sum of its reset-aware increments, so a counter that steps by 2 counts 2 parts, and the first increase after the first reading counts. `run_seconds` is the summed time between readings that report production. A non-numeric signal counts every row as one cycle.
- Set `max_cycle_gap` (e.g. `"5min"`): longer intervals are dropped as stops. Without it, breaks and downtime count as very slow cycles and show up as a large speed loss.
- One `target_cycle_time` per call; for a line running several part numbers, filter the data to one part per call.
- `loss_minutes` and the `*_s` cycle-time columns duplicate the `*_seconds` columns. Default shifts are 06-14, 14-22, 22-06; a night shift belongs to its start date (`production_day=True`).
