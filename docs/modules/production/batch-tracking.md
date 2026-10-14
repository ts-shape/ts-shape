# BatchTrackingEvents

> Track batch/recipe production from a batch-ID string signal.

**Module:** `ts_shape.events.production.batch_tracking`
**Guide:** [OEE Analytics](../../guides/oee-analytics.md)

---

## When to Use

Use for batch production environments (food, pharma, chemicals). Tracks batch boundaries from a batch-ID signal and computes duration and yield statistics. Essential for regulatory traceability and for optimizing batch scheduling to minimize transition losses.

---

## Quick Example

```python
from ts_shape.events.production.batch_tracking import BatchTrackingEvents
import pandas as pd
import numpy as np

# Long format: a batch-ID string signal plus a monotonic part counter
times = pd.date_range("2024-01-01 06:00", periods=360, freq="1min")
batch = pd.DataFrame({
    "systime": times,
    "uuid": "line1:batch_id",
    "value_string": ["BATCH-001"] * 90 + ["BATCH-002"] * 120 + ["BATCH-003"] * 150,
})
counter = pd.DataFrame({
    "systime": times,
    "uuid": "line1:part_counter",
    "value_integer": np.arange(1, 361),
})
df = pd.concat([batch, counter], ignore_index=True)

bt = BatchTrackingEvents(df, batch_uuid="line1:batch_id")
batches = bt.detect_batches()
stats = bt.batch_duration_statistics()
yields = bt.batch_yield(counter_uuid="line1:part_counter")
matrix = bt.batch_transition_matrix()
print(batches)
```

---

## Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `detect_batches()` | Identify batch start and end times from value changes in the batch-ID signal | DataFrame with `batch_id`, `start`, `end`, `duration_seconds`, `sample_count`, `uuid`, `source_uuid` |
| `batch_duration_statistics()` | Compute duration statistics (count, min, avg, max, total) grouped by batch type | DataFrame with `batch_id`, `count`, `min_`/`avg_`/`max_`/`total_duration_seconds` |
| `batch_yield(counter_uuid, handle_resets=True)` | Calculate production quantity per batch from a monotonic counter signal | DataFrame with `batch_id`, `start`, `end`, `duration_seconds`, `quantity` (plus `resets` unless `handle_resets=False`) |
| `batch_transition_matrix(layout='long')` | Count which batch follows which, useful for scheduling analysis | DataFrame with `from_state`, `to_state`, `transition_count`, one row per observed transition (the empty result keeps these columns). `layout='wide'` returns the from-batch by to-batch pivot table with a `total` column |

---

## Tips & Hints

!!! tip "Watch for empty or null batch IDs"
    Gaps in the batch-ID signal (empty strings or nulls) often represent cleaning or changeover periods. Filter these out or treat them as explicit changeover events.

!!! info "Related modules"
    - [Changeover](changeover.md) — detects product changes, complementary for discrete manufacturing
    - [Part Production Tracking](part-tracking.md) — tracks production by part number rather than batch
    - [Quality Tracking](quality-tracking.md) — quality metrics that can be joined to batch data

---

## See Also

- [OEE Analytics Guide](../../guides/oee-analytics.md)
- [API Reference](../../reference/ts_shape/events/production/batch_tracking.md)
