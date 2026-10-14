# DowntimeTracking

> Track machine downtimes by shift and reason from state and reason-code signals.

**Module:** `ts_shape.events.production.downtime_tracking`
**Guide:** [Production Monitoring](../../guides/production.md)

---

## When to Use

Use for downtime Pareto analysis. Categorizes downtime by reason code and shift to identify the biggest availability losses. This is the standard approach for continuous improvement — focus resources on the top downtime reasons to get the greatest return on investment.

---

## Quick Example

```python
from ts_shape.events.production.downtime_tracking import DowntimeTracking
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

dt = DowntimeTracking(df)  # default shifts: 06-14, 14-22, 22-06
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
| `downtime_by_shift(state_uuid, running_value='Running', max_gap=None)` | Aggregate downtime per date and shift | DataFrame with `date`, `shift`, `total_minutes`, `downtime_minutes`, `uptime_minutes`, `availability_pct` |
| `downtime_by_reason(state_uuid, reason_uuid, stopped_value='Stopped')` | Break down downtime by reason code with duration totals | DataFrame with `reason`, `occurrences`, `total_minutes`, `avg_minutes`, `pct_of_total` |
| `top_downtime_reasons(state_uuid, reason_uuid, top_n=5)` | Pareto analysis of the top N downtime reasons by cumulative duration | DataFrame with `reason`, `total_minutes`, `pct_of_total`, `cumulative_pct` (sorted by duration) |
| `availability_trend(state_uuid, window='1D', max_gap=None)` | Track availability over time windows for trend analysis | DataFrame with `period`, `availability_pct`, `uptime_minutes`, `downtime_minutes` |

---

## Tips & Hints

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
