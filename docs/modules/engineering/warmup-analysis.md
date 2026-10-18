# WarmUpCoolDownEvents

Warm-up and cool-down ramps of a signal (ovens, extruders, molds, hydraulics), their consistency, and the time from each ramp start to a target value.

`ts_shape.events.engineering.warmup_analysis` · [Engineering guide](../../guides/engineering.md) · [API reference](../../reference/ts_shape/events/engineering/warmup_analysis.md)

## Example

```python
import numpy as np
import pandas as pd
from ts_shape.events.engineering.warmup_analysis import WarmUpCoolDownEvents

t = pd.date_range("2024-01-01 05:00", periods=600, freq="10s")
temp = np.interp(np.arange(600), [0, 30, 210, 400, 520, 599], [20, 20, 185, 185, 40, 40])
df = pd.DataFrame({"systime": t, "uuid": "oven_zone1", "value_double": temp})

wu = WarmUpCoolDownEvents(df, "oven_zone1")
result = wu.detect_warmup(50.0, min_duration="1min")
print(result[["start", "start_value", "end_value", "duration_seconds", "avg_rate"]])
cooldowns = wu.detect_cooldown(30.0, min_duration="1min")
time_to_temp = wu.time_to_target(180.0, direction="rising")
```

## Methods

| Method | Returns |
|---|---|
| `detect_warmup(min_rise, min_duration='1min', max_pause=0)` | `start`, `end`, `start_value`, `end_value`, `rise`, `duration_seconds`, `avg_rate` |
| `detect_cooldown(min_fall, min_duration='1min', max_pause=0)` | same with `fall` |
| `warmup_consistency(min_rise, min_duration='1min')` | per warm-up: `warmup_index`, `avg_rate`, `deviation_from_median_duration` |
| `time_to_target(target_value, direction='rising')` | per ramp: `target_reached_at`, `time_to_target_seconds`, `overshoot` |

## Notes

- A ramp is a run of rising (or falling) steps with a total change of at least `min_rise` / `min_fall` lasting `min_duration`. `max_pause` is the number of consecutive flat or reversed steps tolerated inside a ramp; with the default 0 any pause ends it, so noisy signals need a pause allowance or smoothing.
- `overshoot` in `time_to_target` is the peak beyond the target from the time it is reached until the next ramp starts (or the data ends), so a later ramp's peak is not booked to an earlier one.
- A time-to-target that grows over weeks points to degrading heaters, fouling or insulation.
