# BottleneckDetectionEvents

Finds the station that constrains a line: the one with the highest utilization per window, from one boolean run signal per station.

`ts_shape.events.production.bottleneck_detection` · [Production guide](../../guides/production.md) · [API reference](../../reference/ts_shape/events/production/bottleneck_detection.md)

## Example

```python
import numpy as np
import pandas as pd
from ts_shape.events.production.bottleneck_detection import BottleneckDetectionEvents

rng = np.random.default_rng(42)
times = pd.date_range("2024-01-01 06:00", periods=360, freq="1min")
p_run = {"line1:st_a:run": 0.92, "line1:st_b:run": 0.98, "line1:st_c:run": 0.85}
df = pd.concat([pd.DataFrame({"systime": times, "uuid": u, "value_bool": rng.random(360) < p})
                for u, p in p_run.items()], ignore_index=True)

bd = BottleneckDetectionEvents(df)
print(bd.detect_bottleneck(list(p_run), window="1h"))
summary = bd.throughput_constraint_summary(list(p_run), window="1h")
```

## Methods

| Method | Returns |
|---|---|
| `station_utilization(station_uuids, window='1h')` | per window and station: `start`, `end`, `uuid`, `utilization_pct` |
| `detect_bottleneck(station_uuids, window='1h')` | per window: `start`, `end`, `bottleneck_uuid`, `utilization_pct` |
| `shifting_bottleneck(station_uuids, window='1h')` | one row each time the bottleneck changes: `systime`, `from_uuid`, `to_uuid`, `previous_utilization`, `new_utilization` |
| `throughput_constraint_summary(station_uuids, window='1h')` | one row per station: `uuid`, `bottleneck_count`, `bottleneck_pct`, `avg_utilization_pct`, `most_frequent_bottleneck` (bool); `start`/`end`/`duration_seconds` are the analysed span |

## Notes

- The bottleneck is the station with the highest utilization: it runs most because the others wait for it.
- `time_weighted=True` (default) weights each sample by the time to the next sample, so change-encoded and irregular signals give wall-clock utilization; the last sample of a station carries no weight. `time_weighted=False` averages the samples, which equals the time share only for uniform sampling.
- `throughput_constraint_summary(..., as_frame=False)` returns the deprecated dict form.
