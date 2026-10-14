# DowntimeEvents

> Track machine downtimes by shift and reason from state and reason-code signals.

**Module:** `ts_shape.events.production.downtime_tracking`
**Guide:** [Production Monitoring](../../guides/production.md)

---

## When to Use

Use for downtime Pareto analysis. Categorizes downtime by reason code and shift to identify the biggest availability losses. This is the standard approach for continuous improvement — focus resources on the top downtime reasons to get the greatest return on investment.

---

## Quick Example

```python
from ts_shape.events.production.downtime_tracking import DowntimeEvents
import pandas as pd
import numpy as np

# Long format: a machine-state string signal and a reason-code string signal
np.random.seed(42)
times = pd.date_range("2024-01-01 06:00", periods=480, freq="1min")
states = np.random.choice(["Running", "Stopped"], size=480, p=[0.80, 0.20])
reasons = np.where(states == "Running", "NONE", np.random.choice(
    ["MECHANICAL", "ELECTRICAL", "MATERIAL", "CHANGEOVER"], size=480
))
df = pd.concat([
    pd.DataFrame({"systime": times, "uuid": "line1:state", "value_string": states}),
    pd.DataFrame({"systime": times, "uuid": "line1:reason", "value_string": reasons}),
], ignore_index=True)

dt = DowntimeEvents(df)  # default shifts: 06-14, 14-22, 22-06
by_shift = dt.downtime_by_shift(state_uuid="line1:state", running_value="Running")
by_reason = dt.downtime_by_reason(
    state_uuid="line1:state", reason_uuid="line1:reason", stopped_value="Stopped"
)
pareto = dt.top_downtime_reasons(state_uuid="line1:state", reason_uuid="line1:reason", top_n=5)
trend = dt.availability_trend(state_uuid="line1:state", window="1h", max_gap="15min")
print(pareto)
```

---

## Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `downtime_by_shift(state_uuid, running_value='Running', max_gap=None)` | Aggregate downtime per date and shift | DataFrame with `date`, `shift`, `total_minutes`, `total_seconds`, `downtime_minutes`, `downtime_seconds`, `uptime_minutes`, `uptime_seconds`, `availability_pct`, `excluded_seconds` |
| `downtime_by_reason(state_uuid, reason_uuid, stopped_value='Stopped')` | Break down downtime by reason code with duration totals | DataFrame with `reason`, `occurrences`, `total_minutes`, `total_seconds`, `avg_minutes`, `avg_seconds`, `pct_of_total` |
| `top_downtime_reasons(state_uuid, reason_uuid, top_n=5)` | Pareto analysis of the top N downtime reasons by cumulative duration | DataFrame with `reason`, `total_minutes`, `total_seconds`, `pct_of_total`, `cumulative_pct` (sorted by duration) |
| `availability_trend(state_uuid, window='1D', max_gap=None)` | Track availability over time windows for trend analysis | DataFrame with `period`, `availability_pct`, `uptime_minutes`, `uptime_seconds`, `downtime_minutes`, `downtime_seconds`, `excluded_seconds` |

---

Durations are reported in seconds (`*_seconds`); the `*_minutes` columns are kept as a convenience. `excluded_seconds` is the time `max_gap` removed as "no data" (0 without a cap).

## Tips & Hints

!!! tip "String or boolean run state"
    The state signal may be a string signal in `value_string` (`running_value='Running'`, `stopped_value='Stopped'`) or a boolean run signal in `value_bool` (True = running). The boolean signal is used automatically when the state rows carry no `value_string`; passing `running_value=True` (or `stopped_value=False`) selects it explicitly. Both forms give the same downtime.

!!! tip "Define shifts in your configuration"
    Shift boundaries default to 06:00-14:00, 14:00-22:00, 22:00-06:00. Pass `shift_definitions={"day": ("06:00", "18:00"), "night": ("18:00", "06:00")}` to match your plant's actual schedule. Misaligned shift definitions will attribute downtime to the wrong crew.

!!! info "Related modules"
    - [Machine State](machine-state.md) — provides the run/idle signal used to detect downtime intervals
    - [OEE Calculator](oee-calculator.md) — uses availability from downtime analysis
    - [Alarm Management](alarm-management.md) — correlate alarms with downtime events
    - [Changeover](changeover.md) — distinguish planned changeover downtime from unplanned stops

---

## See Also

- [Production Monitoring Guide](../../guides/production.md)
- [API Reference](../../reference/ts_shape/events/production/downtime_tracking.md)
