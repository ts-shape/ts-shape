# LineThroughputEvents

> Count parts per window and check takt adherence from counter or trigger signals.

**Module:** `ts_shape.events.production.line_throughput`
**Guide:** [Production Monitoring](../../guides/production.md)

---

## When to Use

Use for real-time throughput monitoring. Counts parts from monotonic counters and checks cycle times against takt time targets. Ideal for lines where a PLC counter increments on each part produced or a trigger signal fires per cycle.

---

## Quick Example

```python
from ts_shape.events.production.line_throughput import LineThroughputEvents
import pandas as pd
import numpy as np

# Long format: a monotonic part counter and a boolean cycle trigger
np.random.seed(0)
times = pd.date_range("2024-01-01 06:00", periods=480, freq="15s")
done = times[0] + pd.to_timedelta(np.random.normal(58, 4, 120).cumsum(), unit="s")
df = pd.concat([
    pd.DataFrame({"systime": times, "uuid": "line1:part_counter",
                  "value_integer": np.arange(1, 481)}),
    pd.DataFrame({"systime": np.concatenate([done, done + pd.Timedelta("1s")]),
                  "uuid": "line1:cycle_trigger",
                  "value_bool": [True] * 120 + [False] * 120}),
], ignore_index=True)

tp = LineThroughputEvents(df)
parts = tp.count_parts(counter_uuid="line1:part_counter", window="5min", handle_resets=True)
takt = tp.takt_adherence(cycle_uuid="line1:cycle_trigger", takt_time="60s")
oee = tp.throughput_oee(counter_uuid="line1:part_counter", window="15min", target_rate=60)
print(parts.head())
print(takt[takt["violation"]].head())
```

---

## Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `count_parts(counter_uuid, window='1min', handle_resets=False)` | Count produced parts per time window from a monotonic counter signal | DataFrame with `start`, `uuid`, `source_uuid`, `is_delta`, `count` (plus `resets` when `handle_resets=True`) |
| `takt_adherence(cycle_uuid, takt_time='60s', min_violation='0s')` | Measure each cycle (rising edges of a boolean trigger, or increases of a numeric counter -- a reset is not a cycle) against the takt time; `violation` = cycle strictly longer than `takt_time + min_violation` | DataFrame with `systime`, `uuid`, `source_uuid`, `is_delta`, `cycle_time_seconds`, `violation` |
| `throughput_oee(counter_uuid, window='1h', target_rate=None, availability_threshold=0.0)` | Counter-only OEE estimate per window (quality assumed 1.0; `target_rate` defaults to the 95th percentile of counts) | DataFrame with `start`, `actual_count`, `target_count`, `availability`, `performance`, `oee_score` |
| `throughput_trends(counter_uuid, window='1h', trend_window=24)` | Analyze throughput trends with moving averages and degradation detection | DataFrame with `start`, `count`, `moving_avg`, `trend_direction`, `degradation_detected` |
| `cycle_quality_check(cycle_uuid, expected_cycle_time=None, tolerance_pct=0.1)` | Validate each cycle time against an expected value (median if not given) | DataFrame with `systime`, `cycle_time_seconds`, `expected_time`, `deviation_pct`, `is_valid`, `quality_flag` |

---

## Tips & Hints

!!! tip "Handle counter resets"
    Monotonic counters may reset at shift boundaries or after a PLC restart. By default `count_parts` clips negative window deltas to zero, which loses the production around a reset. Pass `handle_resets=True` to treat a decreasing counter as a reset, count the post-reset value as new production, and get a `resets` column per window.

!!! info "Related modules"
    - [Cycle Time Tracking](cycle-time.md) — per-part cycle time analysis
    - [Part Production Tracking](part-tracking.md) — production quantities by part number
    - [OEE Calculator](oee-calculator.md) — uses throughput for the performance component

---

## See Also

- [Production Monitoring Guide](../../guides/production.md)
- [API Reference](../../reference/ts_shape/events/production/line_throughput.md)
