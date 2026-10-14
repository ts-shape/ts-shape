# TargetEvents

Actual against target per shift or per day, and the share of days on target, for any counter signal.

`ts_shape.events.production.target_tracking` · [Production guide](../../guides/production.md) · [API reference](../../reference/ts_shape/events/production/target_tracking.md)

## Example

```python
import numpy as np
import pandas as pd
from ts_shape.events.production.target_tracking import TargetEvents

t = pd.date_range("2024-01-01", periods=7 * 1440, freq="1min")
df = pd.DataFrame({"systime": t, "uuid": "line1:parts", "value_integer": np.arange(len(t))})

tracker = TargetEvents(df)  # shifts shift_1/2/3: 06-14, 14-22, 22-06
by_shift = tracker.compare_to_target(metric_uuid="line1:parts",
                                     targets={"shift_1": 500, "shift_2": 480, "shift_3": 400})
daily = tracker.target_achievement_summary(metric_uuid="line1:parts", daily_target=1400)
print(tracker.target_hit_rate(metric_uuid="line1:parts", daily_target=1400))
```

## Methods

| Method | Returns |
|---|---|
| `compare_to_target(metric_uuid, targets)` | per date and shift: `actual`, `target`, `variance`, `achievement_pct`, `status` (+ `resets`) |
| `target_achievement_summary(metric_uuid, daily_target)` | per date: same columns (+ `resets`) |
| `target_hit_rate(metric_uuid, daily_target)` | one row: `total_days`, `days_on_target`, `hit_rate_pct`, `avg_achievement_pct` |

## Notes

- `status` is `on_target` at 100 % achievement or more, `warning` from 90 %, else `below_target`. A day counts as on target at 100 % or more.
- Counters are reset-aware by default (`handle_resets=True`); `handle_resets=False` uses `max(0, last - first)` per group.
- Without `shift_definitions`, shifts are `shift_1`, `shift_2`, `shift_3`; `targets` keys must match. A night shift belongs to its start date (`production_day=True`).
- `target_hit_rate(..., as_frame=False)` returns the deprecated dict.
