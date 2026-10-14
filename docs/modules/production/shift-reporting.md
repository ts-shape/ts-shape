# ShiftReportEvents

Production per shift, shift-to-shift comparison, actual against per-shift targets, and the best and worst shifts, from a production counter.

`ts_shape.events.production.shift_reporting` · [Production guide](../../guides/production.md) · [API reference](../../reference/ts_shape/events/production/shift_reporting.md)

## Example

```python
import numpy as np
import pandas as pd
from ts_shape.events.production.shift_reporting import ShiftReportEvents

t = pd.date_range("2024-01-01 06:00", periods=7 * 1440, freq="1min")
df = pd.DataFrame({"systime": t, "uuid": "line1:parts", "value_integer": np.arange(len(t)) // 2})

reporter = ShiftReportEvents(df, shift_definitions={
    "morning": ("06:00", "14:00"), "afternoon": ("14:00", "22:00"), "night": ("22:00", "06:00")})
print(reporter.shift_production(counter_uuid="line1:parts").head())
comparison = reporter.shift_comparison(counter_uuid="line1:parts", days=7)
result = reporter.shift_targets(counter_uuid="line1:parts",
                                targets={"morning": 250, "afternoon": 240, "night": 200})
```

## Methods

| Method | Returns |
|---|---|
| `shift_production(counter_uuid, part_id_uuid=None, date=None)` | `date`, `shift`, `quantity` (+ `part_number` with `part_id_uuid`, + `resets`) |
| `shift_comparison(counter_uuid, days=7)` | per shift: `avg_quantity`, `min_quantity`, `max_quantity`, `std_quantity`, `days_count` |
| `shift_targets(counter_uuid, targets, date=None)` | `date`, `shift`, `actual`, `target`, `variance`, `achievement_pct` |
| `best_and_worst_shifts(counter_uuid, days=30)` | the five best and five worst shifts: `rank_type` (`best`/`worst`), `date`, `shift`, `quantity` |

## Notes

- Shift quantities add up to the counter total: reset-aware increments are computed once over the whole signal and each is booked to the shift of the reading that reports it, so parts made between the last reading of one shift and the first of the next count in the next shift. `date` only selects which shifts are reported. `handle_resets=False` uses the per-shift `max(0, last - first)` and loses that production.
- Without `shift_definitions`, shifts are `shift_1` (06:00-14:00), `shift_2` (14:00-22:00) and `shift_3` (22:00-06:00). `targets` keys must match the shift names.
- With `production_day=True` (default) the post-midnight part of a night shift belongs to the shift's start date.
- `best_and_worst_shifts(..., as_frame=False)` returns the deprecated dict.
