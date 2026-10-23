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
| `build_timeline(infer_dwell_from_next=False, interval_end="last_sample")` | one row per item and station visit: `item_id`, `station`, `id_uuid`, `start`, `end`, `duration_seconds`, `step_sequence`, `is_parallel` |
| `lead_time()` | per item: `first_seen`, `last_seen`, `lead_time_seconds`, `stations_visited`, `station_path`, `has_parallel` |
| `parallel_activity()` | overlapping visits: `item_id`, `station_a`, `station_b`, `overlap_start`, `overlap_end`, `overlap_seconds` |
| `handover_log()` | one row per handover trigger: `timestamp`, `item_id`, `from_station`, `to_station`, `handover_value` |
| `station_statistics(infer_dwell_from_next=False, interval_end="last_sample")` | per station and `id_uuid`: `item_count`, `min_`/`avg_`/`max_`/`total_dwell_seconds` |
| `routing_paths()` | per path: `item_count`, `avg_`/`min_`/`max_lead_time_seconds`, `has_parallel_steps` |
| `to_intervals(object_type="serial", interval_end="last_sample")` | one row per visit: `oid`, `type`, `start`, `end`, `duration_seconds`, `uuid` and `source_uuid` (the cell's `id_uuid`), `station` |

## Notes

- `processes` maps each station's item-ID signal (`id_uuid`) to a station name. Parallel cells of one station type can share a name; `station_statistics()` still separates them by `id_uuid`. Visit order comes from timestamps; overlapping visits are flagged as parallel.
- Optional `handovers=[{"uuid", "from_station", "to_station"}]`: a value > 0 (or True) is a transfer, and the item is taken from the from-station's ID signal at that time.
- By default a visit ends at the item's last sample at the station, so a station that logs the ID only on arrival gives a dwell of 0. `infer_dwell_from_next=True` ends each sequential visit at the item's next station arrival; parallel visits keep their own span.
- `interval_end="next_change"` ends a visit at the next different reading on the cell's ID signal (the next item or an empty reading) instead of at its last sample, so the item is held until it changes, like [`object_intervals`](../../guides/eventlog/objects.md); a signal's last visit is held until the last timestamp of the process signals. A visit followed by the same item after a time gap keeps its last sample (the gap is a leave). `is_parallel` uses the chosen ends. The default `"last_sample"` keeps the results as before. `infer_dwell_from_next` is applied after it.
- `to_intervals(object_type="serial")` returns the visits as an object interval table: the columns of `object_intervals` (`oid`, `type`, `start`, `end` in UTC, `duration_seconds`, `uuid`, `source_uuid`) plus `station`. It feeds `presence_events`, `object_kpis`, `wip_over_time`, `attach_counts` and `relate` directly; see [Object analytics](../../guides/eventlog/object-analytics.md#traceability-classes).
- In `to_event_log`, the item column binds objects of type `serial` and the station column objects of type `station` (plus `asset` from `source_uuid`, here the `id_uuid`). These links do not change the event ids. Pass `objects={"serial": None}` to leave one out, or `objects={"work_order": "item_id"}` to name the type yourself.
- `build_timeline` converts to interval events (`start` / `end`) in `to_event_log`; `lead_time`, `parallel_activity` and `handover_log` lack the canonical time columns (`start` / `end`, `systime`) and do not convert yet.
