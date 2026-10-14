# AlarmManagementEvents

Alarm frequency, alarm-on durations, chattering and standing alarms (ISA-18.2 style) from one boolean alarm signal (`value_bool`, True = active).

`ts_shape.events.production.alarm_management` · [Production guide](../../guides/production.md) · [API reference](../../reference/ts_shape/events/production/alarm_management.md)

## Example

```python
import numpy as np
import pandas as pd
from ts_shape.events.production.alarm_management import AlarmManagementEvents

rng = np.random.default_rng(42)
df = pd.DataFrame({
    "systime": pd.date_range("2024-01-01 06:00", periods=500, freq="30s"),
    "uuid": "line1:temp_high_alarm",
    "value_bool": rng.random(500) < 0.15,
})
alarms = AlarmManagementEvents(df, alarm_uuid="line1:temp_high_alarm")
print(alarms.alarm_frequency(window="1h").head())
chattering = alarms.chattering_detection(min_transitions=5, window="10min")
standing = alarms.standing_alarms(min_duration="1h")
```

## Methods

| Method | Returns |
|---|---|
| `alarm_frequency(window='1h')` | activations per window: `start`, `end`, `duration_seconds`, `alarm_count` |
| `alarm_duration_statistics()` | one row: `alarm_count`, `min_`/`avg_`/`max_`/`total_duration_seconds` of the alarm-on intervals |
| `chattering_detection(min_transitions=5, window='10min')` | windows with at least `min_transitions` state changes: `start`, `end`, `transition_count` |
| `standing_alarms(min_duration='1h')` | alarm-on intervals longer than `min_duration`: `start`, `end`, `duration_seconds` |

## Notes

- An activation is a False to True transition. With `include_initial=True` (default) an alarm already active in the first sample counts as an activation; pass `include_initial=False` to ignore it.
- An alarm-on interval ends at the next False sample, or at the last sample if the alarm is still active at the end of the data.
- Missing values count as False.
