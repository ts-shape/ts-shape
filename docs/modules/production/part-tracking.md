# PartProductionTracking

> Track production quantities by part number from part-ID and counter signals.

**Module:** `ts_shape.events.production.part_tracking`
**Guide:** [Production Monitoring](../../guides/production.md)

---

## When to Use

Use for part-level production reporting. Tracks how many of each part number were produced per hour, day, or custom period. Essential for production planning reconciliation and for computing per-part OEE when a line produces multiple SKUs.

---

## Quick Example

```python
from ts_shape.events.production.part_tracking import PartProductionTracking
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "timestamp": pd.date_range("2024-01-01 06:00", periods=480, freq="1min"),
    "part_id": (["PN-100"]*160 + ["PN-200"]*160 + ["PN-100"]*160),
    "part_counter": np.arange(1, 481),
})

pt = PartProductionTracking(df, timestamp_column="timestamp")
hourly = pt.production_by_part(part_id_uuid="part_id", counter_uuid="part_counter", window="1h")
daily = pt.daily_production_summary(part_id_uuid="part_id", counter_uuid="part_counter")
totals = pt.production_totals(part_id_uuid="part_id", counter_uuid="part_counter")
print(hourly.head(10))
```

---

## Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `production_by_part(part_id_uuid, counter_uuid, window='1h', handle_resets=False)` | Count parts produced per part number per time window | DataFrame with window, part_id, and quantity (plus `resets` when `handle_resets=True`) |
| `daily_production_summary(part_id_uuid, counter_uuid, handle_resets=False)` | Aggregate daily production totals by part number | DataFrame with date, part_id, and daily quantity |
| `production_totals(part_id_uuid, counter_uuid, handle_resets=False)` | Compute total production by part number over the entire date range | DataFrame with part_id and total quantity |
| `detect_resets(part_id_uuid, counter_uuid)` | Find when the counter was reset | DataFrame with timestamp, part_id, `count_before`, `count_after`, `drop` |

---

## Counter Resets

Production counters are assumed to increase as parts are produced. Many counters reset back to zero (or a lower value) — at a shift change, a part change, or a controller restart. By default (`handle_resets=False`), a window where the counter drops reports `quantity = max(0, last_count - first_count)`, which **clamps to 0 and loses all production in that window**.

Pass `handle_resets=True` to count production correctly across resets. The quantity is then the sum of per-reading increments, where a drop is treated as a reset that contributes the new counter value. The result gains a `resets` column counting resets per window.

```python
# Reset-aware hourly production
hourly = pt.production_by_part(
    part_id_uuid="part_id", counter_uuid="part_counter",
    window="1h", handle_resets=True,
)

# Inspect exactly when the counter reset
resets = pt.detect_resets(part_id_uuid="part_id", counter_uuid="part_counter")
print(resets)   # timestamp, part_number, count_before, count_after, drop
```

---

## Tips & Hints

!!! tip "Enable reset handling when counters reset"
    If your counter resets at part changes, shift boundaries, or controller restarts, pass `handle_resets=True` so production is not silently dropped. Use `detect_resets()` first to confirm whether and when your counter resets.

!!! info "Related modules"
    - [Line Throughput](line-throughput.md) — aggregate throughput without part-level breakdown
    - [Cycle Time Tracking](cycle-time.md) — per-part cycle time analysis
    - [Batch Tracking](batch-tracking.md) — batch-level tracking for process industries
    - [Quality Tracking](quality-tracking.md) — per-part quality metrics

---

## See Also

- [Production Monitoring Guide](../../guides/production.md)
- [API Reference](../../reference/ts_shape/events/production/part_tracking.md)
