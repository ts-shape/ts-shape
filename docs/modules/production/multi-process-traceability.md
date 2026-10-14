# MultiProcessTraceabilityEvents

> Trace items across a multi-station topology with parallel paths.

**Module:** `ts_shape.events.production.multi_process_traceability`
**Guide:** [Traceability Guide](../../guides/traceability.md)

---

## When to Use

Use for complex manufacturing topologies with parallel process lines, merges, and splits. Tracks items through non-linear routings where a single item may visit multiple stations simultaneously (e.g., sub-assemblies built in parallel then merged). This is the most powerful traceability module, suited for automotive or electronics assembly lines.

---

## Quick Example

```python
from ts_shape.events.production.multi_process_traceability import MultiProcessTraceabilityEvents
import pandas as pd

# Long format: each station publishes the ID of the item it is working on
t = pd.Timestamp("2024-01-01 06:00")
rows = [
    # (minute, uuid, item_id)
    (0, "cell_a1:order_id", "ORD-1"), (4, "cell_a1:order_id", "ORD-1"),
    (2, "cell_b1:order_id", "ORD-1"), (6, "cell_b1:order_id", "ORD-1"),  # parallel
    (8, "merge:order_id", "ORD-1"), (12, "merge:order_id", "ORD-1"),
    (5, "cell_a1:order_id", "ORD-2"), (9, "cell_a1:order_id", "ORD-2"),
    (13, "merge:order_id", "ORD-2"), (16, "merge:order_id", "ORD-2"),
]
df = pd.DataFrame({
    "systime": [t + pd.Timedelta(minutes=m) for m, _, _ in rows],
    "uuid": [u for _, u, _ in rows],
    "value_string": [v for _, _, v in rows],
})

tracer = MultiProcessTraceabilityEvents(
    df,
    processes=[
        {"id_uuid": "cell_a1:order_id", "station": "Cell A1"},
        {"id_uuid": "cell_b1:order_id", "station": "Cell B1"},
        {"id_uuid": "merge:order_id", "station": "Merge"},
    ],
    # optional: handovers=[{"uuid": "a1:to_merge", "from_station": "Cell A1",
    #                       "to_station": "Merge"}]
)

# Full timeline across all stations
timeline = tracer.build_timeline()

# End-to-end lead time per item
lead = tracer.lead_time()

# Items at multiple stations simultaneously
parallel = tracer.parallel_activity()

# Most common routing paths
paths = tracer.routing_paths()
print(lead[["item_id", "station_path", "lead_time_seconds"]])
```

---

## Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `build_timeline(infer_dwell_from_next=False)` | Full timeline of every item at every station (`item_id`, `station`, `start`, `end`, `duration_seconds`, `step_sequence`, `is_parallel`) | `DataFrame` |
| `lead_time()` | End-to-end lead time per item (`first_seen`, `last_seen`, `lead_time_seconds`, `station_path`, `has_parallel`) | `DataFrame` |
| `parallel_activity()` | Detect items at multiple stations simultaneously (`station_a`, `station_b`, `overlap_start`, `overlap_end`, `overlap_seconds`) | `DataFrame` |
| `handover_log()` | Log of handover events between stations, resolved to item IDs (requires `handovers`) | `DataFrame` |
| `station_statistics(infer_dwell_from_next=False)` | Dwell time statistics per station and cell (`item_count`, `min_`/`avg_`/`max_`/`total_dwell_seconds`) | `DataFrame` |
| `routing_paths()` | Frequency of each unique routing path, with lead-time statistics | `DataFrame` |

---

## Tips & Hints

!!! tip "One process entry per ID signal"
    Each entry in `processes` maps one station's item-ID signal (`id_uuid`) to a station name. Parallel cells of the same station type can share a `station` name; `station_statistics()` still separates them by `id_uuid`. Visit order is derived from timestamps, and overlapping visits are flagged as parallel.

!!! tip "Stations that log once on arrival"
    If a station only logs the item ID when it arrives, pass `infer_dwell_from_next=True` to `build_timeline()` / `station_statistics()` so dwell is measured to the next station arrival instead of collapsing to zero.

!!! info "Related modules"
    - [Order Traceability](order-traceability.md) — simpler linear tracing
    - [Routing Traceability](routing-traceability.md) — routing with state signals
    - [Bottleneck Detection](bottleneck-detection.md) — identify bottlenecks in multi-station flows

---

## See Also

- [Traceability Guide](../../guides/traceability.md)
- [API Reference](../../reference/ts_shape/events/production/multi_process_traceability.md)
