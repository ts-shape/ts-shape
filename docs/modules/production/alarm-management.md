# AlarmManagementEvents

> Analyze alarm signals following ISA-18.2 alarm management principles.

**Module:** `ts_shape.events.production.alarm_management`
**Guide:** [OEE Analytics](../../guides/oee-analytics.md)

---

## When to Use

Use for ISA-18.2 compliant alarm management. Identifies chattering alarms, standing alarms, and alarm frequency patterns that indicate system health or nuisance alarm issues. Regular alarm rationalization is critical — operators who receive too many alarms become desensitized, leading to missed critical events.

---

## Quick Example

```python
from ts_shape.events.production.alarm_management import AlarmManagementEvents
import pandas as pd
import numpy as np

np.random.seed(42)
df = pd.DataFrame({
    "timestamp": pd.date_range("2024-01-01 06:00", periods=500, freq="30s"),
    "high_temp_alarm": np.random.choice([True, False], size=500, p=[0.15, 0.85]),
    "low_pressure_alarm": np.random.choice([True, False], size=500, p=[0.05, 0.95]),
})

alarms = AlarmManagementEvents(df, timestamp_column="timestamp")
freq = alarms.alarm_frequency(window="1h")
chattering = alarms.chattering_detection(min_transitions=5, window="10min")
standing = alarms.standing_alarms(min_duration="1h")
print(freq.head())
```

---

## Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `alarm_frequency(window='1h')` | Count alarm activations per time window to identify alarm flood periods | DataFrame with window and activation count per alarm |
| `alarm_duration_stats()` | Compute statistics on how long each alarm stays active (ON duration) | DataFrame with min, mean, max, and percentile durations |
| `chattering_detection(min_transitions=5, window='10min')` | Detect chattering alarms that toggle rapidly within a short window | DataFrame of chattering alarm instances with transition count |
| `standing_alarms(min_duration='1h')` | Find alarms that remain active beyond a threshold, indicating stale or ignored alarms | DataFrame of standing alarm intervals |

---

## Tips & Hints

!!! tip "Target fewer than 6 alarms per operator per hour"
    ISA-18.2 recommends a manageable alarm rate. Use `alarm_frequency` to check if your plant exceeds this guideline and prioritize chattering alarms for rationalization.

!!! info "Related modules"
    - [Machine State](machine-state.md) — correlate alarms with machine run/idle state
    - [Downtime Tracking](downtime.md) — link alarm events to downtime reasons
    - [Duty Cycle](duty-cycle.md) — detect excessive cycling that may trigger nuisance alarms

---

## See Also

- [OEE Analytics Guide](../../guides/oee-analytics.md)
- [API Reference](../../reference/ts_shape/events/production/alarm_management.md)
