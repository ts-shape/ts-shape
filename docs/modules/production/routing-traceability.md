# RoutingTraceabilityEvents

Item routing from an item-ID signal paired with a routing/state signal whose value encodes the current process step or station; useful when one station performs several steps.

`ts_shape.events.production.routing_traceability` · [Production guide](../../guides/production.md) · [API reference](../../reference/ts_shape/events/production/routing_traceability.md)

## Example

```python
import numpy as np
import pandas as pd
import ts_shape
from ts_shape.events.production.routing_traceability import RoutingTraceabilityEvents

t = pd.date_range("2024-01-01 06:00", periods=60, freq="1min")
df = pd.concat([
    ts_shape.make_id_signal("line1:serial", ["SN-1", "SN-2"], hold=30, freq="1min", start="2024-01-01 06:00"),
    pd.DataFrame({"systime": t, "uuid": "line1:step", "value_integer": np.tile(np.repeat([1, 2, 3], 10), 2)}),
], ignore_index=True)

tracer = RoutingTraceabilityEvents(df, id_uuid="line1:serial", routing_uuid="line1:step",
                                   state_map={1: "Welding", 2: "Painting", 3: "Assembly"})
print(tracer.lead_time()[["item_id", "routing_path", "lead_time_seconds"]])
stats = tracer.station_statistics()
```

## Methods

| Method | Returns |
|---|---|
| `build_routing_timeline(infer_dwell_from_next=False, interval_end="last_sample")` | one row per item and step: `item_id`, `routing_value`, `station_name`, `start`, `end`, `duration_seconds`, `sample_count`, `station_sequence` |
| `lead_time()` | per item: `first_station`, `last_station`, `first_seen`, `last_seen`, `lead_time_seconds`, `stations_visited`, `routing_path` |
| `station_statistics(infer_dwell_from_next=False, interval_end="last_sample")` | per step: `routing_value`, `item_count`, `min_`/`avg_`/`max_`/`total_dwell_seconds` |
| `routing_paths()` | per path: `item_count`, `avg_`/`min_`/`max_lead_time_seconds` |
| `to_intervals(object_type="serial", interval_end="last_sample")` | one row per visit: `oid`, `type`, `start`, `end`, `duration_seconds`, `uuid` (the ID signal), `source_uuid`, `station` (the step name) |

## Notes

- `item_id` is text. A number with an integer value renders without `.0` (`1001.0` is `1001`, as in `to_intervals()` and the other traceability classes), so `1001.0` and `"1001"` are one item.
- Each routing sample is assigned the most recent item ID at that time; contiguous samples with the same (item, step) form one interval. A routing sample with no item (none read yet, or a missing or empty ID reading) or without a routing value is no visit: it ends the interval before it. Rows without a timestamp are left out; readings at the same timestamp keep their input order.
- `exclude=` drops ID readings that are not item IDs, like [`ObjectSpec.exclude`](../../guides/eventlog/objects.md): `"misreads"` (`NOREAD`, `#N/A`, `0`, `65535`, blanks, ...; see `ts_shape.utils.id_codes.is_misread`), a list of values, or a callable returning a mask. Between two readings of the same item an excluded reading is a gap (the item is held across it); anywhere else it is an empty reading (no item). The default `None` keeps every reading; `"misreads"` is planned to become the default in the next major release.
- `state_map` translates routing values (int, float or str keys) to names; without it the raw values are the labels. `station_map` is a deprecated alias.
- Defaults: ID in `value_string`, routing value in `value_integer` (`value_column_id=`, `value_column_routing=` to change).
- By default an interval ends at its last routing sample, so a change-only routing signal gives a dwell of 0; `infer_dwell_from_next=True` ends it at the item's next routing arrival.
- `interval_end="next_change"` ends a visit at the next different reading on the routing signal (the next routing sample with another item or step, or with no item) instead of at its last sample, so the item is held until it changes, like [`object_intervals`](../../guides/eventlog/objects.md); the last segment is held until the last timestamp of the two signals. An item change on the ID signal alone is seen at the next routing sample. The default `"last_sample"` keeps the results as before. `infer_dwell_from_next` is applied after it.
- `to_intervals(object_type="serial")` returns the visits as an object interval table: the columns of `object_intervals` (`oid`, `type`, `start`, `end` in UTC, `duration_seconds`, `uuid`, `source_uuid`) plus `station`. It feeds `presence_events`, `object_kpis`, `wip_over_time`, `attach_counts` and `relate` directly; see [Object analytics](../../guides/eventlog/object-analytics.md#traceability-classes).
- In `to_event_log`, the item column binds objects of type `serial` and the station column objects of type `station` (plus `asset` from `source_uuid`). These links do not change the event ids. Pass `objects={"serial": None}` to leave one out, or `objects={"work_order": "item_id"}` to name the type yourself.
