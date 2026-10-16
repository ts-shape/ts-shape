# OperatorPerformanceEvents

Output, efficiency against a per-shift target, first-pass yield and a ranking per operator, from an operator-ID string signal and part counters.

`ts_shape.events.production.operator_performance` · [Production guide](../../guides/production.md) · [API reference](../../reference/ts_shape/events/production/operator_performance.md)

## Example

```python
import numpy as np
import pandas as pd
import ts_shape
from ts_shape.events.production.operator_performance import OperatorPerformanceEvents

t = pd.date_range("2024-01-01 06:00", periods=960, freq="1min")
df = pd.concat([
    ts_shape.make_id_signal("line1:operator", ["Anna", "Ben"], hold=480, freq="1min", start="2024-01-01 06:00"),
    pd.DataFrame({"systime": t, "uuid": "line1:ok", "value_integer": np.arange(960) // 2}),
    pd.DataFrame({"systime": t, "uuid": "line1:nok", "value_integer": np.arange(960) // 60}),
], ignore_index=True)

ops = OperatorPerformanceEvents(df)
print(ops.operator_efficiency(operator_uuid="line1:operator", counter_uuid="line1:ok", target_per_shift=250))
quality = ops.quality_by_operator(operator_uuid="line1:operator", ok_uuid="line1:ok", nok_uuid="line1:nok")
```

## Methods

| Method | Returns (one row per operator) |
|---|---|
| `production_by_operator(operator_uuid, counter_uuid)` | `total_produced`, `shifts_worked`, `avg_per_shift` (+ `resets`) |
| `operator_efficiency(operator_uuid, counter_uuid, target_per_shift)` | `total_produced`, `target`, `efficiency_pct` |
| `quality_by_operator(operator_uuid, ok_uuid, nok_uuid)` | `ok_count`, `nok_count`, `first_pass_yield_pct` (+ `resets`) |
| `operator_comparison(operator_uuid, counter_uuid)` | `total_produced`, `rank`, `share_of_best_pct` |

## Notes

- Counter increments are computed once over the whole signal and booked to the operator (and shift) of the reading that reports them, so operators add up to the counter total. With `handle_resets=True` (default) a decreasing counter is a reset and later production still counts; `handle_resets=False` uses `max(0, last - first)` per group and loses a reset.
- Shifts default to 06:00-14:00, 14:00-22:00, 22:00-06:00 (`shift_definitions=` to change). With `production_day=True` (default) the post-midnight part of a night shift belongs to the shift's start date.
- Raw output is only comparable across operators with similar shift lengths and product mix.
