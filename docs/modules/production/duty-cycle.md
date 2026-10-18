# DutyCycleEvents

On/off intervals, duty cycle per window, transition counts and excessive cycling from one boolean signal (True = on), e.g. a heater, motor or valve.

`ts_shape.events.production.duty_cycle` · [Production guide](../../guides/production.md) · [API reference](../../reference/ts_shape/events/production/duty_cycle.md)

## Example

```python
import numpy as np
import pandas as pd
from ts_shape.events.production.duty_cycle import DutyCycleEvents

df = pd.DataFrame({
    "systime": pd.date_range("2024-01-01 06:00", periods=3600, freq="1s"),
    "uuid": "oven1:heater_on",
    "value_bool": np.arange(3600) % 90 < 60,  # on 60 s, off 30 s
})
dc = DutyCycleEvents(df, signal_uuid="oven1:heater_on")
result = dc.duty_cycle_by_window(window="15min")
print(result[["start", "on_time_seconds", "off_time_seconds", "duty_cycle_pct"]])
intervals = dc.on_off_intervals()
excessive = dc.excessive_cycling(max_transitions=20, window="15min")
```

## Methods

| Method | Returns |
|---|---|
| `on_off_intervals(max_gap=None)` | every interval: `start`, `end`, `state` (`on`/`off`), `duration_seconds`, `duration` (Timedelta). With `max_gap` (e.g. `"15min"`) an interval ends at its last sample before a longer gap and a new one starts after it |
| `duty_cycle_by_window(window='1h', time_weighted=True)` | per window: `on_time_seconds`, `off_time_seconds`, `duty_cycle_pct` |
| `cycle_count(window='1h')` | per window: `on_count`, `off_count`, `total_transitions` |
| `excessive_cycling(max_transitions=20, window='1h')` | windows above `max_transitions`: `transition_count`, `avg_on_duration_seconds`, `avg_off_duration_seconds` |

## Notes

- A missing value (`None`, `NaN`, `pd.NA`) is unknown, not off: the time from that sample to the next is in neither `on_time_seconds` nor `off_time_seconds`, in no on/off interval and no transition. `duty_cycle_pct` is the on share of the known time (NaN for a window without any).
- With `data_end=` in the constructor, the last interval is held until `data_end`, and `duty_cycle_by_window` (time-weighted) and `cycle_count` report the windows up to it. A naive `data_end` on tz-aware data is local time of the data's timezone.
- Intervals close at the next transition by default (`interval_end="next_change"`), so signals that only log a row per change are measured correctly. `interval_end="last_sample"` ends each interval at its own last sample.
- `time_weighted=True` (default) weights each sample by the time to the next one. `time_weighted=False` averages the samples, which matches the time share only for uniform sampling (`On@09:00, Off@09:59` reads 50 %).
- For absolute run hours, start counts and an hour meter, use [RuntimeAccountingEvents](runtime-accounting.md).
