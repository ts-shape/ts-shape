# CycleTimeEvents

> Track cycle times by part number from part-ID and cycle-trigger signals.

**Module:** `ts_shape.events.production.cycle_time_tracking`
**Guide:** [Production Monitoring](../../guides/production.md)

---

## When to Use

Use to track individual cycle durations and identify slow cycles. Essential for performance analysis in OEE and for detecting tool wear or process drift. When cycle times gradually increase over a shift, it often indicates tool degradation or thermal effects that need attention.

---

## Quick Example

```python
from ts_shape.events.production.cycle_time_tracking import CycleTimeEvents
import pandas as pd
import numpy as np

# Long format: a part-number string signal and a boolean cycle-complete pulse
np.random.seed(42)
cycle_times = np.random.normal(loc=55, scale=3, size=200)
cycle_times[50:55] = 90  # simulate slow cycles
done = pd.Timestamp("2024-01-01 06:00") + pd.to_timedelta(cycle_times.cumsum(), unit="s")
pulses = pd.DataFrame({
    "systime": np.concatenate([done, done + pd.Timedelta("1s")]),
    "uuid": "line1:cycle_complete",
    "value_bool": [True] * 200 + [False] * 200,
})
parts = pd.DataFrame({
    "systime": [pd.Timestamp("2024-01-01 06:00"), done[99]],
    "uuid": "line1:part_number",
    "value_string": ["PN-100", "PN-200"],
})
df = pd.concat([pulses, parts], ignore_index=True)

ct = CycleTimeEvents(df)
ids = dict(part_id_uuid="line1:part_number", cycle_trigger_uuid="line1:cycle_complete")
by_part = ct.cycle_time_by_part(**ids, max_cycle_gap="5min")
stats = ct.cycle_time_statistics(**ids)
slow = ct.detect_slow_cycles(**ids, threshold_factor=1.5)
trend = ct.cycle_time_trend(**ids, part_number="PN-100", window_size=20)
print(stats)
```

---

## Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `cycle_time_by_part(part_id_uuid, cycle_trigger_uuid, max_cycle_gap=None)` | Compute cycle time for each part number from the intervals between cycle-complete triggers | DataFrame with `systime`, `part_number`, `cycle_time_seconds` |
| `cycle_time_statistics(part_id_uuid, cycle_trigger_uuid)` | Calculate count, min, avg, max, std, and median cycle time per part number | DataFrame with `part_number`, `count`, `min_seconds`, `avg_seconds`, `max_seconds`, `std_seconds`, `median_seconds` |
| `detect_slow_cycles(part_id_uuid, cycle_trigger_uuid, threshold_factor=1.5)` | Flag cycles exceeding a multiple of the median cycle time for that part | DataFrame with `systime`, `part_number`, `cycle_time_seconds`, `median_seconds`, `deviation_factor`, `is_slow` |
| `cycle_time_trend(part_id_uuid, cycle_trigger_uuid, part_number, window_size=20)` | Analyze cycle time trend for a specific part number over time | DataFrame with `systime`, `cycle_time_seconds`, `moving_avg`, `trend` |
| `hourly_cycle_time_summary(part_id_uuid, cycle_trigger_uuid)` | Summarize cycle times per hour per part for shift review dashboards | DataFrame with `hour`, `part_number`, `cycles_completed`, `avg_cycle_time`, `min_cycle_time`, `max_cycle_time` |

---

## Tips & Hints

!!! tip "Use threshold_factor relative to each part"
    The `detect_slow_cycles` method computes the threshold per part number, so a 1.5x factor means 1.5 times the median for that specific SKU — no need to hard-code absolute limits for each part.

!!! info "Related modules"
    - [Line Throughput](line-throughput.md) — takt adherence from the throughput perspective
    - [Part Production Tracking](part-tracking.md) — production quantities per part
    - [OEE Calculator](oee-calculator.md) — uses cycle time for the performance component
    - [Micro-Stop Detection](micro-stops.md) — brief stops that inflate apparent cycle time

---

## See Also

- [Production Monitoring Guide](../../guides/production.md)
- [API Reference](../../reference/ts_shape/events/production/cycle_time_tracking.md)
