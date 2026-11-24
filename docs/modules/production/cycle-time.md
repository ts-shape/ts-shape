---
tags: [cycle-time]
---

# CycleTimeEvents

Cycle times per part number from a part-number string signal and a cycle-complete trigger (boolean or integer). A cycle time is the time between consecutive completions.

`ts_shape.events.production.cycle_time_tracking` · [Production guide](../../guides/production.md) · [API reference](../../reference/ts_shape/events/production/cycle_time_tracking.md)

## Example

```python
import pandas as pd
import ts_shape
from ts_shape.events.production.cycle_time_tracking import CycleTimeEvents

df = pd.concat([
    ts_shape.make_id_signal("line1:part_number", ["PN-100", "PN-200"], hold=100),
    ts_shape.make_timeseries(["line1:cycle_done"], n_points=200, value_column="value_bool"),
], ignore_index=True)

ct = CycleTimeEvents(df)
ids = dict(part_id_uuid="line1:part_number", cycle_trigger_uuid="line1:cycle_done")
result = ct.cycle_time_statistics(**ids, max_cycle_gap="5min")
print(result[["part_number", "cycle_count", "min_seconds", "avg_seconds", "max_seconds"]])
slow = ct.detect_slow_cycles(**ids, threshold_factor=1.5)
trend = ct.cycle_time_trend(**ids, part_number="PN-100", window_size=20)
```

## Methods

| Method | Returns |
|---|---|
| `cycle_time_by_part(...)` | one row per cycle: `systime` (completion), `part_number`, `cycle_time_seconds` |
| `cycle_time_statistics(...)` | per part: `cycle_count`, `min_seconds`, `avg_seconds`, `max_seconds`, `std_seconds`, `median_seconds`, `p10_seconds` (best demonstrated), `p90_seconds`, `mad_seconds` |
| `detect_slow_cycles(..., threshold_factor=1.5)` | cycles slower than `threshold_factor` × the part's median: adds `median_seconds`, `deviation_factor`, `is_slow` |
| `cycle_time_trend(..., part_number, window_size=20, stable_band_ratio=0.01)` | one part: `cycle_time_seconds`, `moving_avg`, `trend` (`improving`/`stable`/`degrading`; a change of the moving average within ±1 % of the part median is `stable`) |
| `hourly_cycle_time_summary(...)` | per hour and part: `cycles_completed`, `avg_`/`min_`/`max_cycle_time_seconds` |
| `detect_unusual_cycles(..., n_sigma=3.0)` | cycles outside robust two-sided bounds on the log cycle time per part: `median_seconds`, `lower_limit_seconds`, `upper_limit_seconds`, `z_score`, `cycle_class` (`short`/`long`), `direction` |

All methods take `part_id_uuid` and `cycle_trigger_uuid`.

## Notes

- Set `max_cycle_gap` (e.g. `"5min"`): longer intervals between completions are dropped as downtime. Without it, a break or jam between two completions becomes one very long cycle that inflates max/avg and is flagged as slow.
- The slow-cycle threshold is relative to each part's median, so mixed part numbers need no per-part limits.
- `detect_slow_cycles` uses a fixed factor and looks at slow cycles only; `detect_unusual_cycles` finds short and long cycles with statistical, scale-free bounds.
- The trend band is relative to the part's median (`stable_band_ratio`), so the same setting fits a 5 s and a 300 s cycle. Before, it was a fixed ±0.5 s.
- These methods need the cycle trigger and part signals named. To profile a PLC without naming any signal (cycle found automatically, pauses kept as their own class, segments, contribution register, influencers) use [CycleProfileEvents](../profiling/cycle-profiling.md).
