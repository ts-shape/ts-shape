# ProductionScheduleAdherenceEvents

Planned against actual production for each scheduled period (shift, work order), the rate needed to recover a period that is behind, alerts for periods outside the deviation bands, and a roll-up across the schedule.

`ts_shape.events.production.schedule_adherence` · [Production guide](../../guides/production.md) · [API reference](../../reference/ts_shape/events/production/schedule_adherence.md)

## Example

```python
import numpy as np
import pandas as pd
from ts_shape.events.production.schedule_adherence import ProductionScheduleAdherenceEvents

# part counter read every minute: 50 parts/h on the early shift, 40 parts/h on the late shift
t = pd.date_range("2024-05-02 06:00", "2024-05-02 22:00", freq="1min")
rate = np.where(t.hour < 14, 50 / 60, 40 / 60)
df = pd.DataFrame({"systime": t, "uuid": "counter:line-1",
                   "value_double": np.floor(np.cumsum(rate))})

schedule = pd.DataFrame({
    "start": pd.to_datetime(["2024-05-02 06:00", "2024-05-02 14:00"]),
    "end": pd.to_datetime(["2024-05-02 14:00", "2024-05-02 22:00"]),
    "planned_qty": [400, 400],
    "order_id": ["WO-1001", "WO-1002"],
})

adherence = ProductionScheduleAdherenceEvents(df, schedule)
periods = adherence.adherence_by_period("counter:line-1")
print(periods[["order_id", "planned_qty", "actual_qty", "schedule_adherence_pct", "deviation_type"]])
alerts = adherence.deviation_events("counter:line-1")
summary = adherence.adherence_summary("counter:line-1")
print(summary[["total_planned_qty", "total_actual_qty", "overall_adherence_pct", "worst_order_id"]])
```

The late shift made 320 of 400 parts (80 %): `behind`, and the worst order of the day.

## Methods

| Method | Returns |
|---|---|
| `adherence_by_period(counter_uuid, handle_resets=True, as_of=None, ahead_pct=105.0, behind_pct=95.0, critical_pct=80.0)` | one row per scheduled period: `order_id`, `planned_qty`, `actual_qty`, `schedule_adherence_pct`, `deviation_units`, `deviation_type`, `recovery_required_rate`, `remaining_seconds`, `resets` |
| `deviation_events(counter_uuid, ..., include_ahead=False)` | point events at the period start for periods behind (or ahead) of plan: `deviation_type`, `severity`, `direction`, `threshold` |
| `adherence_summary(counter_uuid, ...)` | one row: `total_planned_qty`, `total_actual_qty`, `overall_adherence_pct`, `periods_on_track`, `periods_behind`, `periods_critical_behind`, `periods_ahead`, `on_plan_rate_pct`, `worst_order_id`, `worst_adherence_pct` |

## Notes

- `deviation_type` is `ahead` above `ahead_pct`, `on_track` down to `behind_pct`, `behind` down to `critical_pct`, and `critical_behind` below it. The bands must be ordered `critical_pct <= behind_pct <= ahead_pct`.
- `recovery_required_rate` is the parts per hour needed over the rest of the period to finish on plan, evaluated at `as_of` (default: the last timestamp in the data). It is 0 for a period already on plan and NaN for a period that is over.
- The counter is reset-aware by default. A reading on a boundary shared by two periods is counted once, for the period that ends there.
- `overall_adherence_pct` divides the summed quantities, so a large order weighs more than a small one.
- The schedule columns default to `start`, `end`, `planned_qty` and `order_id`; pass `schedule_start_column=`, `schedule_end_column=`, `planned_qty_column=` and `order_id_column=` to the constructor to use an MES export as it is.
