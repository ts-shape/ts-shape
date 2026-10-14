# MicroStopEvents

Short idle intervals (default under 30 s), their frequency, the time they cost and when in the day they cluster, from a boolean run signal (True = running).

`ts_shape.events.production.micro_stop_detection` · [Production guide](../../guides/production.md) · [API reference](../../reference/ts_shape/events/production/micro_stop_detection.md)

## Example

```python
import numpy as np
import pandas as pd
from ts_shape.events.production.micro_stop_detection import MicroStopEvents

df = pd.DataFrame({
    "systime": pd.date_range("2024-01-01 06:00", periods=7200, freq="1s"),
    "uuid": "line1:running",
    "value_bool": np.arange(7200) % 120 >= 8,  # 8 s stop every 2 min
})
ms = MicroStopEvents(df, run_state_uuid="line1:running")
stops = ms.detect_micro_stops(max_duration="30s")
print(ms.micro_stop_impact(window="15min", max_duration="30s").head())
patterns = ms.micro_stop_patterns(hour_grouping=True)
```

## Methods

| Method | Returns |
|---|---|
| `detect_micro_stops(max_duration='30s', min_duration='0s')` | `start`, `end`, `duration_seconds`, `duration`, `preceding_run_duration` |
| `micro_stop_frequency(window='1h', max_duration='30s')` | per window: `count`, `total_lost_time`, `pct_of_window` |
| `micro_stop_impact(window='1h', max_duration='30s')` | per window: `total_run_time`, `total_micro_stop_time`, `availability_loss_pct` |
| `micro_stop_patterns(hour_grouping=True, max_duration='30s')` | per `hour` of day (or 8-hour `shift` block with `hour_grouping=False`): `avg_count`, `avg_lost_time` |

## Notes

- `max_duration` is passed to each method separately. Set it just below the threshold at which stops are reported as downtime (30 s suits fast discrete lines; slower assembly may need 60-120 s).
- Missing values count as idle.
- Intervals close at the next transition by default (`interval_end="next_change"`), so a run signal logged only on change is measured correctly; `data_end` bounds the last open interval. With `interval_end="last_sample"`, every interval of such a signal has zero duration and no lost time is reported.
