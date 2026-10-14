# ReworkEvents

Rework counts per shift, per reason and over time, the rework rate against total production, and rework cost per part number, from a cumulative rework counter.

`ts_shape.events.production.rework_tracking` · [Production guide](../../guides/production.md) · [API reference](../../reference/ts_shape/events/production/rework_tracking.md)

## Example

```python
import numpy as np
import pandas as pd
import ts_shape
from ts_shape.events.production.rework_tracking import ReworkEvents

t = pd.date_range("2024-01-01 06:00", periods=960, freq="1min")
df = pd.concat([
    pd.DataFrame({"systime": t, "uuid": "line1:rework", "value_integer": np.arange(960) // 40}),
    pd.DataFrame({"systime": t, "uuid": "line1:parts", "value_integer": np.arange(960)}),
    ts_shape.make_id_signal("line1:part_id", ["PN-100", "PN-200"], hold=480, freq="1min", start="2024-01-01 06:00"),
], ignore_index=True)

rw = ReworkEvents(df)
print(rw.rework_rate(rework_uuid="line1:rework", total_production_uuid="line1:parts"))
cost = rw.rework_cost(rework_uuid="line1:rework", part_id_uuid="line1:part_id",
                      rework_costs={"PN-100": 12.50, "PN-200": 8.75})
```

## Methods

| Method | Returns |
|---|---|
| `rework_by_shift(rework_uuid)` | `date`, `shift`, `rework_count` |
| `rework_by_reason(rework_uuid, reason_uuid)` | `reason`, `rework_count`, `pct_of_total` |
| `rework_rate(rework_uuid, total_production_uuid)` | per shift: `total_produced`, `rework_count`, `rework_rate_pct` |
| `rework_cost(rework_uuid, part_id_uuid, rework_costs)` | per part: `rework_count`, `cost_per_rework`, `total_cost` |
| `rework_trend(rework_uuid, window='1D')` | `period`, `rework_count` |

## Notes

- With `handle_resets=True` (default) a decreasing counter is a reset and the rework after it still counts. Increments are computed once over the whole signal and booked to the shift, reason or part of the reading that reports them, so groups add up to the counter total. All methods except `rework_rate` add a `resets` column. `handle_resets=False` uses `max(0, last - first)` per group.
- `rework_costs` maps part numbers (values of the `part_id_uuid` signal) to the cost of one rework.
- Default shifts are 06-14, 14-22, 22-06; a night shift belongs to its start date (`production_day=True`).
