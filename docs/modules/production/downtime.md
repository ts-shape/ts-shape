# DowntimeEvents

Downtime and availability per shift, per reason code and over time, from a machine-state signal and an optional reason-code signal.

`ts_shape.events.production.downtime_tracking` · [Production guide](../../guides/production.md) · [API reference](../../reference/ts_shape/events/production/downtime_tracking.md)

## Example

```python
import numpy as np
import pandas as pd
from ts_shape.events.production.downtime_tracking import DowntimeEvents

rng = np.random.default_rng(42)
t = pd.date_range("2024-01-01 06:00", periods=480, freq="1min")
state = rng.choice(["Running", "Stopped"], size=480, p=[0.8, 0.2])
reason = np.where(state == "Running", "NONE", rng.choice(["MECH", "ELEC", "MATERIAL"], size=480))
df = pd.concat([pd.DataFrame({"systime": t, "uuid": "line1:state", "value_string": state}),
                pd.DataFrame({"systime": t, "uuid": "line1:reason", "value_string": reason})])

dt = DowntimeEvents(df)  # shifts 06-14, 14-22, 22-06
print(dt.downtime_by_shift(state_uuid="line1:state", max_gap="15min"))
pareto = dt.top_downtime_reasons(state_uuid="line1:state", reason_uuid="line1:reason", top_n=5)
trend = dt.availability_trend(state_uuid="line1:state", window="1h", max_gap="15min")
```

## Methods

| Method | Returns |
|---|---|
| `downtime_by_shift(state_uuid, running_value='Running', max_gap=None)` | per date and shift: `total_seconds`, `downtime_seconds`, `uptime_seconds`, `availability_pct`, `excluded_seconds` |
| `downtime_by_reason(state_uuid, reason_uuid, stopped_value='Stopped')` | per reason: `occurrences`, `total_seconds`, `avg_seconds`, `pct_of_total` |
| `top_downtime_reasons(state_uuid, reason_uuid, top_n=5)` | Pareto: `reason`, `total_seconds`, `pct_of_total`, `cumulative_pct` |
| `availability_trend(state_uuid, window='1D', max_gap=None)` | per period: `availability_pct`, `uptime_seconds`, `downtime_seconds`, `excluded_seconds` |

Every `*_seconds` column has a `*_minutes` twin.

## Notes

- The state may be a string signal in `value_string` (`running_value='Running'`, `stopped_value='Stopped'`) or a boolean run signal in `value_bool` (True = running). The boolean signal is used when the state rows carry no `value_string`; `running_value=True` or `stopped_value=False` selects it explicitly. Both give the same downtime.
- `max_gap` treats a gap between two readings longer than the cap as no data instead of booking it to the preceding state; `excluded_seconds` is the time removed (0 without a cap). With `specs=` (see [signals](../../guides/signals.md)) a state signal's `max_gap` is applied by default.
- Default shifts are 06:00-14:00, 14:00-22:00, 22:00-06:00; pass `shift_definitions={"day": ("06:00", "18:00"), "night": ("18:00", "06:00")}` for your schedule.
