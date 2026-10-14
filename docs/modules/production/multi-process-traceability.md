# MultiProcessTraceabilityEvents

Item timelines, lead times, parallel processing and routing paths across a station topology with parallel cells, merges and splits. Each station publishes the ID of the item it is working on.

`ts_shape.events.production.multi_process_traceability` · [Production guide](../../guides/production.md) · [API reference](../../reference/ts_shape/events/production/multi_process_traceability.md)

## Example

```python
import pandas as pd
from ts_shape.events.production.multi_process_traceability import MultiProcessTraceabilityEvents

t0 = pd.Timestamp("2024-01-01 06:00")
rows = [(0, "cell_a1:order_id", "ORD-1"), (4, "cell_a1:order_id", "ORD-1"),
        (2, "cell_b1:order_id", "ORD-1"), (6, "cell_b1:order_id", "ORD-1"),  # parallel to A1
        (8, "merge:order_id", "ORD-1"), (12, "merge:order_id", "ORD-1"),
        (5, "cell_a1:order_id", "ORD-2"), (9, "cell_a1:order_id", "ORD-2"),
        (13, "merge:order_id", "ORD-2"), (16, "merge:order_id", "ORD-2")]
df = pd.DataFrame([{"systime": t0 + pd.Timedelta(minutes=m), "uuid": u, "value_string": v}
                   for m, u, v in rows])
tracer = MultiProcessTraceabilityEvents(df, processes=[
    {"id_uuid": "cell_a1:order_id", "station": "Cell A1"},
    {"id_uuid": "cell_b1:order_id", "station": "Cell B1"},
    {"id_uuid": "merge:order_id", "station": "Merge"}])
print(tracer.lead_time()[["item_id", "station_path", "lead_time_seconds", "has_parallel"]])
```

## Methods

| Method | Returns |
|---|---|
| `build_timeline(infer_dwell_from_next=False)` | one row per item and station visit: `item_id`, `station`, `id_uuid`, `start`, `end`, `duration_seconds`, `step_sequence`, `is_parallel` |
| `lead_time()` | per item: `first_seen`, `last_seen`, `lead_time_seconds`, `stations_visited`, `station_path`, `has_parallel` |
| `parallel_activity()` | overlapping visits: `item_id`, `station_a`, `station_b`, `overlap_start`, `overlap_end`, `overlap_seconds` |
| `handover_log()` | one row per handover trigger: `timestamp`, `item_id`, `from_station`, `to_station`, `handover_value` |
| `station_statistics(infer_dwell_from_next=False)` | per station and `id_uuid`: `item_count`, `min_`/`avg_`/`max_`/`total_dwell_seconds` |
| `routing_paths()` | per path: `item_count`, `avg_`/`min_`/`max_lead_time_seconds`, `has_parallel_steps` |

## Notes

- `processes` maps each station's item-ID signal (`id_uuid`) to a station name. Parallel cells of one station type can share a name; `station_statistics()` still separates them by `id_uuid`. Visit order comes from timestamps; overlapping visits are flagged as parallel.
- Optional `handovers=[{"uuid", "from_station", "to_station"}]`: a value > 0 (or True) is a transfer, and the item is taken from the from-station's ID signal at that time.
- By default a visit ends at the item's last sample at the station, so a station that logs the ID only on arrival gives a dwell of 0. `infer_dwell_from_next=True` ends each sequential visit at the item's next station arrival; parallel visits keep their own span.
