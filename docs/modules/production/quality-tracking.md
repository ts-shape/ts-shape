# QualityEvents

NOK counts, NOK rate and first-pass yield per shift, per day and per part number, and a defect-reason Pareto, from cumulative OK/NOK counters.

`ts_shape.events.production.quality_tracking` · [Production guide](../../guides/production.md) · [API reference](../../reference/ts_shape/events/production/quality_tracking.md)

## Example

```python
import numpy as np
import pandas as pd
import ts_shape
from ts_shape.events.production.quality_tracking import QualityEvents

t = pd.date_range("2024-01-01 06:00", periods=960, freq="1min")
df = pd.concat([
    pd.DataFrame({"systime": t, "uuid": "line1:ok", "value_integer": np.arange(960)}),
    pd.DataFrame({"systime": t, "uuid": "line1:nok", "value_integer": np.arange(960) // 30}),
    ts_shape.make_id_signal("line1:part_id", ["PN-100", "PN-200"], hold=480, freq="1min", start="2024-01-01 06:00"),
], ignore_index=True)

qt = QualityEvents(df)
result = qt.nok_by_shift(ok_counter_uuid="line1:ok", nok_counter_uuid="line1:nok")
print(result[["date", "shift", "ok_parts", "nok_parts", "first_pass_yield_pct"]])
by_part = qt.quality_by_part(ok_counter_uuid="line1:ok", nok_counter_uuid="line1:nok", part_id_uuid="line1:part_id")
```

## Methods

| Method | Returns |
|---|---|
| `nok_by_shift(ok_counter_uuid, nok_counter_uuid)` | per date and shift: `ok_parts`, `nok_parts`, `total_parts`, `nok_rate_pct`, `first_pass_yield_pct`, `quality_pct` |
| `quality_by_part(ok_counter_uuid, nok_counter_uuid, part_id_uuid)` | per part number: same metrics |
| `nok_by_reason(nok_counter_uuid, defect_reason_uuid)` | per reason: `nok_parts`, `share_of_total_pct` |
| `daily_quality_summary(ok_counter_uuid, nok_counter_uuid)` | per date: same metrics as `nok_by_shift` |

## Notes

- Pass cumulative counters; the module takes the increments. Convert per-cycle OK/NOK flags to cumulative sums first.
- With `handle_resets=True` (default) a decreasing counter is a reset and the parts after it still count; a `resets` column is added. Increments are computed once over each whole counter, so the groups add up to the counter total. Per shift and day, the increment between two readings is spread over the time between them and split at the shift boundaries in proportion to time (parts belong to the shift they were produced in; counts can be fractional); per part and reason it is booked to the group of the reading that reports it. `handle_resets=False` uses `max(0, last - first)` per group.
- `quality_pct` equals `first_pass_yield_pct`; it is the column name other modules (OEE, shift handover, period summary) join on.
- Default shifts are 06-14, 14-22, 22-06 (`shift_definitions=` to change).
