# LineThroughputEvents

Parts per window, takt adherence, a counter-only OEE estimate and throughput trends from a part counter and/or a cycle trigger.

`ts_shape.events.production.line_throughput` · [Production guide](../../guides/production.md) · [API reference](../../reference/ts_shape/events/production/line_throughput.md)

## Example

```python
import numpy as np
import pandas as pd
from ts_shape.events.production.line_throughput import LineThroughputEvents

t = pd.date_range("2024-01-01 06:00", periods=480, freq="30s")
df = pd.concat([
    pd.DataFrame({"systime": t, "uuid": "line1:part_counter", "value_integer": np.arange(480) // 2}),
    pd.DataFrame({"systime": t, "uuid": "line1:cycle_trigger", "value_bool": np.arange(480) % 2 == 0}),
], ignore_index=True)

tp = LineThroughputEvents(df)
print(tp.count_parts(counter_uuid="line1:part_counter", window="15min").head())
takt = tp.takt_adherence(cycle_uuid="line1:cycle_trigger", takt_time="55s")
oee = tp.throughput_oee(counter_uuid="line1:part_counter", window="1h", target_rate=60)
```

## Methods

| Method | Returns |
|---|---|
| `count_parts(counter_uuid, window='1min', handle_resets=True)` | per window: `count` (+ `resets`) |
| `takt_adherence(cycle_uuid, takt_time='60s', min_violation='0s')` | one row per cycle: `systime`, `cycle_time_seconds`, `violation` |
| `throughput_oee(counter_uuid, window='1h', target_rate=None, availability_threshold=0.0)` | per window: `actual_count`, `target_count`, `availability`, `performance`, `oee_score` (0-1), `run_seconds`, `planned_seconds` (+ `resets`) |
| `throughput_trends(counter_uuid, window='1h', trend_window=24)` | per window: `count`, `moving_avg`, `trend_direction`, `degradation_detected` |
| `cycle_quality_check(cycle_uuid, expected_cycle_time=None, tolerance_pct=0.1)` | one row per cycle: `cycle_time_seconds`, `expected_time`, `deviation_pct`, `is_valid`, `quality_flag` |

## Notes

- Counter resets: with `handle_resets=True` (default) a decreasing counter is a reset and the post-reset value counts as new production; `resets` gives the number per window. `handle_resets=False` uses the window-over-window difference of the counter maximum clipped at zero, which loses the production around a reset.
- Cycles: a boolean trigger (`value_bool`) uses rising edges; a numeric column uses every increase (a reset is not a cycle boundary); a string column uses every value change. A cycle is a violation when it is strictly longer than `takt_time + min_violation`.
- `throughput_oee` has no run-state signal: a window counts as available when its count reaches `availability_threshold × target_rate` (default 0: any part). Quality is assumed 1.0, and `target_rate` (parts per window) defaults to the 95th percentile of the counts. Use [OEEEvents](oee-calculator.md) when a run signal and reject counter exist.
- `cycle_quality_check` uses the median cycle time when `expected_cycle_time` is not given.
