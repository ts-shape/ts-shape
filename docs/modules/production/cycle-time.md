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
print(result[["part_number", "count", "min_seconds", "avg_seconds", "max_seconds"]])
slow = ct.detect_slow_cycles(**ids, threshold_factor=1.5)
trend = ct.cycle_time_trend(**ids, part_number="PN-100", window_size=20)
```

## Methods

| Method | Returns |
|---|---|
| `cycle_time_by_part(...)` | one row per cycle: `systime` (completion), `part_number`, `cycle_time_seconds` |
| `cycle_time_statistics(...)` | per part: `count`, `min_seconds`, `avg_seconds`, `max_seconds`, `std_seconds`, `median_seconds` |
| `detect_slow_cycles(..., threshold_factor=1.5)` | cycles slower than `threshold_factor` × the part's median: adds `median_seconds`, `deviation_factor`, `is_slow` |
| `cycle_time_trend(..., part_number, window_size=20)` | one part: `cycle_time_seconds`, `moving_avg`, `trend` (`improving`/`stable`/`degrading`) |
| `hourly_cycle_time_summary(...)` | per hour and part: `cycles_completed`, `avg_`/`min_`/`max_cycle_time` |

All methods take `part_id_uuid` and `cycle_trigger_uuid`.

## Notes

- Set `max_cycle_gap` (e.g. `"5min"`): longer intervals between completions are dropped as downtime. Without it, a break or jam between two completions becomes one very long cycle that inflates max/avg and is flagged as slow.
- The slow-cycle threshold is relative to each part's median, so mixed part numbers need no per-part limits.
