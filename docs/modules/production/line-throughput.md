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

df = pd.DataFrame({
    "timestamp": pd.date_range("2024-01-01 06:00", periods=480, freq="15s"),
    "part_counter": np.arange(1, 481),
    "cycle_time_s": np.random.normal(loc=58, scale=4, size=480),
})

tp = LineThroughputEvents(df, timestamp_column="timestamp")
parts = tp.count_parts(counter_uuid="part_counter", window="1min")
violations = tp.takt_adherence(cycle_uuid="cycle_time_s", takt_time="60s")
print(parts.head())
```

---

## Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `count_parts(counter_uuid, window='1min')` | Count produced parts per time window from a monotonic counter signal | DataFrame with window start and part count |
| `takt_adherence(cycle_uuid, takt_time='60s')` | Identify cycles that exceed the target takt time | DataFrame of violations with timestamp and overshoot |
| `throughput_oee(counter_uuid, window='1h')` | Compute OEE-style throughput metrics per window | DataFrame with actual vs ideal counts and performance ratio |
| `throughput_trends(counter_uuid, window='1h')` | Analyze throughput trends over time for shift or daily review | DataFrame with trend statistics per window |
| `cycle_quality_check(cycle_uuid)` | Validate cycle signal integrity: gaps, duplicates, out-of-range values | Dict of quality check results |

---

## Tips & Hints

!!! tip "Handle counter resets"
    Monotonic counters may reset at shift boundaries or after a PLC restart. `count_parts` automatically detects and compensates for resets by looking at negative deltas.

!!! info "Related modules"
    - [Cycle Time Tracking](cycle-time.md) — per-part cycle time analysis
    - [Part Production Tracking](part-tracking.md) — production quantities by part number
    - [OEE Calculator](oee-calculator.md) — uses throughput for the performance component

---

## See Also

- [Production Monitoring Guide](../../guides/production.md)
- [API Reference](../../reference/ts_shape/events/production/line_throughput.md)
