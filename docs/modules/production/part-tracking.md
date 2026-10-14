# PartProductionEvents

> Track production quantities by part number from part-ID and counter signals.

**Module:** `ts_shape.events.production.part_tracking`
**Guide:** [Production Monitoring](../../guides/production.md)

---

## When to Use

Use for part-level production reporting. Tracks how many of each part number were produced per hour, day, or custom period. Essential for production planning reconciliation and for computing per-part OEE when a line produces multiple SKUs.

---

## Quick Example

The tracker reads long-format timeseries data: one row per signal reading, identified by a `uuid` column. The part number lives in a string signal (`value_string`) and the production count in a counter signal (`value_integer`).

```python
from ts_shape.events.production.part_tracking import PartProductionEvents
import pandas as pd
import numpy as np

t = pd.date_range("2024-01-01 06:00", periods=480, freq="1min")

# Part-number signal (which part is running) and a counter signal
parts = pd.DataFrame({
    "uuid": "part_id",
    "systime": t,
    "value_string": (["PN-100"] * 160 + ["PN-200"] * 160 + ["PN-100"] * 160),
})
counter = pd.DataFrame({
    "uuid": "part_counter",
    "systime": t,
    "value_integer": np.arange(1, 481),
})
df = pd.concat([parts, counter], ignore_index=True)

pt = PartProductionEvents(df, time_column="systime")
hourly = pt.production_by_part(part_id_uuid="part_id", counter_uuid="part_counter", window="1h")
daily = pt.daily_production_summary(part_id_uuid="part_id", counter_uuid="part_counter")
totals = pt.production_totals(part_id_uuid="part_id", counter_uuid="part_counter")
print(hourly.head(10))

# If the counter resets (shift/part change, controller restart), enable
# reset handling so production is not silently dropped:
hourly = pt.production_by_part(
    part_id_uuid="part_id", counter_uuid="part_counter",
    window="1h", handle_resets=True,
)
resets = pt.detect_resets(part_id_uuid="part_id", counter_uuid="part_counter")
```

---

## Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `production_by_part(part_id_uuid, counter_uuid, window='1h', handle_resets=True)` | Count parts produced per part number per time window | DataFrame with `start`, `part_number`, `quantity`, `first_count`, `last_count` (plus `resets` unless `handle_resets=False`) |
| `daily_production_summary(part_id_uuid, counter_uuid, handle_resets=True)` | Aggregate daily production totals by part number | DataFrame with `date`, `part_number`, `total_quantity`, `hours_active` (plus `resets` unless `handle_resets=False`) |
| `production_totals(part_id_uuid, counter_uuid, start_date=None, end_date=None, handle_resets=True)` | Compute total production by part number, optionally limited to a date range | DataFrame with `part_number`, `total_quantity`, `days_produced` (plus `resets` unless `handle_resets=False`) |
| `detect_resets(part_id_uuid, counter_uuid)` | Find when the counter was reset | DataFrame with the time column (e.g. `systime`), `part_number`, `count_before`, `count_after`, `drop` |

---

## Counter Resets

Production counters are assumed to increase as parts are produced. Many counters reset back to zero (or a lower value) — at a shift change, a part change, or a controller restart. By default (`handle_resets=True`) the quantity is the sum of per-reading increments, where a drop is treated as a reset that contributes the new counter value, and the result has a `resets` column counting resets per window.

With `handle_resets=False`, a window where the counter drops reports `quantity = max(0, last_count - first_count)`, which clamps to 0 and loses the production in that window.

```python
# Reset-aware hourly production
hourly = pt.production_by_part(
    part_id_uuid="part_id", counter_uuid="part_counter",
    window="1h",
)

# Inspect exactly when the counter reset
resets = pt.detect_resets(part_id_uuid="part_id", counter_uuid="part_counter")
print(resets)   # systime, part_number, count_before, count_after, drop
```

---

## Tips & Hints

!!! tip "Check when counters reset"
    Reset handling is on by default, so production after a reset at a part change, shift boundary or controller restart is counted. Use `detect_resets()` to see whether and when your counter resets.

!!! info "Related modules"
    - [Line Throughput](line-throughput.md) — aggregate throughput without part-level breakdown
    - [Cycle Time Tracking](cycle-time.md) — per-part cycle time analysis
    - [Batch Tracking](batch-tracking.md) — batch-level tracking for process industries
    - [Quality Tracking](quality-tracking.md) — per-part quality metrics

---

## See Also

- [Production Monitoring Guide](../../guides/production.md)
- [API Reference](../../reference/ts_shape/events/production/part_tracking.md)
