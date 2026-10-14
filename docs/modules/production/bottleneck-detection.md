# BottleneckDetectionEvents

> Identify which station constrains a production line by analyzing utilization.

**Module:** `ts_shape.events.production.bottleneck_detection`
**Guide:** [OEE Analytics](../../guides/oee-analytics.md)

---

## When to Use

Use for multi-station line optimization. The bottleneck determines line output — finding it is the first step in improving throughput. This module compares utilization across stations and detects when the bottleneck shifts between stations over time.

---

## Quick Example

```python
from ts_shape.events.production.bottleneck_detection import BottleneckDetectionEvents
import pandas as pd
import numpy as np

# Long format: one boolean run-state signal per station (True = running)
np.random.seed(42)
times = pd.date_range("2024-01-01 06:00", periods=360, freq="1min")
p_run = {"line1:station_a:run": 0.92, "line1:station_b:run": 0.98, "line1:station_c:run": 0.85}
df = pd.concat([
    pd.DataFrame({"systime": times, "uuid": uuid,
                  "value_bool": np.random.rand(len(times)) < p})
    for uuid, p in p_run.items()
], ignore_index=True)

bd = BottleneckDetectionEvents(df)
station_uuids = list(p_run)
utilization = bd.station_utilization(station_uuids, window="1h")
bottleneck = bd.detect_bottleneck(station_uuids, window="1h")
shifts = bd.shifting_bottleneck(station_uuids, window="1h")
summary = bd.throughput_constraint_summary(station_uuids, window="1h")
print(bottleneck)
```

---

## Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `station_utilization(station_uuids, window='1h', time_weighted=True)` | Compute per-station uptime percentage per window | DataFrame with `start`, `uuid`, `utilization_pct` |
| `detect_bottleneck(station_uuids, window='1h', time_weighted=True)` | Identify the bottleneck station (highest utilization) per window | DataFrame with `start`, `end`, `bottleneck_uuid`, `utilization_pct` |
| `shifting_bottleneck(station_uuids, window='1h', time_weighted=True)` | Detect when the bottleneck shifts from one station to another | DataFrame with `systime`, `from_uuid`, `to_uuid`, `previous_utilization`, `new_utilization` |
| `throughput_constraint_summary(station_uuids, window='1h', time_weighted=True)` | Aggregate summary of how often each station is the bottleneck | `DataFrame`, one row per station: `start`, `end`, `duration_seconds` (the analysed span), `uuid`, `source_uuid`, `bottleneck_count`, `bottleneck_pct`, `avg_utilization_pct`, `most_frequent_bottleneck` (bool). `as_frame=False` returns the deprecated dict |

---

## Tips & Hints

!!! tip "The bottleneck has the highest utilization"
    Counter-intuitively, the station with the most uptime (not the most downtime) is usually the bottleneck — it is running flat-out trying to keep up. Stations with lower utilization are often waiting.

!!! info "Related modules"
    - [Flow Constraints](flow-constraints.md) — detects blocked/starved conditions between station pairs
    - [Machine State](machine-state.md) — provides the per-station run/idle signals
    - [Micro-Stop Detection](micro-stops.md) — uncovers hidden losses at the bottleneck station

---

## See Also

- [OEE Analytics Guide](../../guides/oee-analytics.md)
- [API Reference](../../reference/ts_shape/events/production/bottleneck_detection.md)
