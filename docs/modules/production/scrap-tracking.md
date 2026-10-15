# ScrapEvents

Scrap quantity (weight or count) per shift, per reason and over time, and scrap cost per part or material, from a cumulative scrap signal (`value_double` by default).

`ts_shape.events.production.scrap_tracking` · [Production guide](../../guides/production.md) · [API reference](../../reference/ts_shape/events/production/scrap_tracking.md)

## Example

```python
import numpy as np
import pandas as pd
import ts_shape
from ts_shape.events.production.scrap_tracking import ScrapEvents

t = pd.date_range("2024-01-01 06:00", periods=960, freq="1min")
df = pd.concat([
    pd.DataFrame({"systime": t, "uuid": "line1:scrap_kg", "value_double": np.arange(960) * 0.05}),
    ts_shape.make_id_signal("line1:part_id", ["PN-100", "PN-200"], hold=480, freq="1min", start="2024-01-01 06:00"),
], ignore_index=True)

sc = ScrapEvents(df)
result = sc.scrap_by_shift(scrap_uuid="line1:scrap_kg")
print(result[["date", "shift", "scrap_quantity"]])
cost = sc.scrap_cost(scrap_uuid="line1:scrap_kg", part_id_uuid="line1:part_id",
                     material_costs={"PN-100": 3.20, "PN-200": 5.80})
```

## Methods

| Method | Returns |
|---|---|
| `scrap_by_shift(scrap_uuid)` | `date`, `shift`, `scrap_quantity` |
| `scrap_by_reason(scrap_uuid, reason_uuid)` | `reason`, `scrap_quantity`, `pct_of_total` |
| `scrap_cost(scrap_uuid, part_id_uuid, material_costs)` | per part: `scrap_quantity`, `cost_per_unit`, `total_cost` |
| `scrap_trend(scrap_uuid, window='1D')` | `period`, `scrap_quantity` |

## Notes

- With `handle_resets=True` (default) a decreasing signal is a reset and the scrap after it still counts; every method adds a `resets` column. Increments are booked to the shift, reason or part of the reading that reports them, so groups add up to the signal total. `handle_resets=False` uses `max(0, last - first)` per group.
- `material_costs` keys are the values of the `part_id_uuid` signal; the cost is per unit of the scrap signal (e.g. per kg).
- Default shifts are 06-14, 14-22, 22-06; a night shift belongs to its start date (`production_day=True`).
