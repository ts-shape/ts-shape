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
result = ms.micro_stop_impact(window="15min", max_duration="30s").head()
print(result[["start", "total_micro_stop_time_seconds", "performance_loss_pct"]])
patterns = ms.micro_stop_patterns(hour_grouping=True)
```

## Methods

| Method | Returns |
|---|---|
| `detect_micro_stops(max_duration='30s', min_duration='0s')` | `start`, `end`, `duration_seconds`, `duration`, `preceding_run_duration` |
| `micro_stop_frequency(window='1h', max_duration='30s')` | per window: `micro_stop_count`, `total_lost_time_seconds`, `share_of_window_pct` (each right after its deprecated twin `count`, `total_lost_time`, `pct_of_window`) |
| `micro_stop_impact(window='1h', max_duration='30s')` | per window: `total_run_time_seconds`, `total_micro_stop_time_seconds`, `performance_loss_pct` (= `availability_loss_pct`) |
| `micro_stop_patterns(hour_grouping=True, max_duration='30s', shift_definitions=None)` | per `hour` of day (or per `shift` with `hour_grouping=False`): `avg_count`, `avg_lost_time_seconds` |

## Notes

- `max_duration` is passed to each method separately. Set it just below the threshold at which stops are reported as downtime (30 s suits fast discrete lines; slower assembly may need 60-120 s).
- Missing values count as idle.
- `micro_stop_impact` classifies a stop by its full duration and splits its time and the run time at the window boundaries; `micro_stop_frequency` counts each stop (and its lost time) in the window it starts in.
- `performance_loss_pct` is the micro-stop time as a share of run + micro-stop time. Minor stoppages are a performance (speed) loss in Nakajima's six big losses, not an availability loss; `availability_loss_pct` carries the same value under its original name for compatibility.
- Intervals close at the next transition by default (`interval_end="next_change"`), so a run signal logged only on change is measured correctly; `data_end` bounds the last open interval. With `interval_end="last_sample"`, every interval of such a signal has zero duration and no lost time is reported.
- `total_lost_time`, `total_run_time`, `total_micro_stop_time` and `avg_lost_time` are kept as deprecated twins (seconds) right before their `_seconds` columns.
- `micro_stop_patterns(hour_grouping=False)` groups by the shifts in `shift_definitions`, by default `shift_1` (06:00-14:00), `shift_2` (14:00-22:00) and `shift_3` (22:00-06:00) as in the rest of ts-shape (formerly the fixed names `morning`, `afternoon`, `night`; pass them as `shift_definitions` to keep them). A night shift that crosses midnight is one shift: its stops after midnight count on the shift's start date. `avg_count` and `avg_lost_time_seconds` average over the days (shifts) that had stops.
