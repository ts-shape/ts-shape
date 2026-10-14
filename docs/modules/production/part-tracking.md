# PartProductionEvents

Quantities per part number per window, per day and in total, from a part-number string signal and a production counter (`value_integer`).

`ts_shape.events.production.part_tracking` · [Production guide](../../guides/production.md) · [API reference](../../reference/ts_shape/events/production/part_tracking.md)

## Example

```python
import numpy as np
import pandas as pd
import ts_shape
from ts_shape.events.production.part_tracking import PartProductionEvents

t = pd.date_range("2024-01-01 06:00", periods=480, freq="1min")
df = pd.concat([
    ts_shape.make_id_signal("part_id", ["PN-100", "PN-200", "PN-100"], hold=160, freq="1min", start="2024-01-01 06:00"),
    pd.DataFrame({"systime": t, "uuid": "part_counter", "value_integer": np.arange(480) % 200}),  # resets
], ignore_index=True)

pt = PartProductionEvents(df)
print(pt.production_by_part(part_id_uuid="part_id", counter_uuid="part_counter", window="1h").head())
totals = pt.production_totals(part_id_uuid="part_id", counter_uuid="part_counter")
resets = pt.detect_resets(part_id_uuid="part_id", counter_uuid="part_counter")
```

## Methods

| Method | Returns |
|---|---|
| `production_by_part(part_id_uuid, counter_uuid, window='1h')` | per window and part: `start`, `part_number`, `quantity`, `first_count`, `last_count` (+ `resets`) |
| `daily_production_summary(part_id_uuid, counter_uuid)` | per date and part: `total_quantity`, `hours_active` (+ `resets`) |
| `production_totals(part_id_uuid, counter_uuid, start_date=None, end_date=None)` | per part: `total_quantity`, `days_produced` (+ `resets`) |
| `detect_resets(part_id_uuid, counter_uuid)` | one row per counter drop: time, `part_number`, `count_before`, `count_after`, `drop` |

## Notes

- Counters often reset at a shift change, part change or controller restart. With `handle_resets=True` (default) the quantity is the sum of per-reading increments, a drop counts as a reset that contributes the new counter value, and `resets` counts resets per group.
- `handle_resets=False` reports `max(0, last_count - first_count)` per group, so a window with a reset loses its production.
