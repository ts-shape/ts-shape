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
| `build_routing_timeline(infer_dwell_from_next=False)` | one row per item and step: `item_id`, `routing_value`, `station_name`, `start`, `end`, `duration_seconds`, `sample_count`, `station_sequence` |
| `lead_time()` | per item: `first_station`, `last_station`, `first_seen`, `last_seen`, `lead_time_seconds`, `stations_visited`, `routing_path` |
| `station_statistics(infer_dwell_from_next=False)` | per step: `routing_value`, `item_count`, `min_`/`avg_`/`max_`/`total_dwell_seconds` |
| `routing_paths()` | per path: `item_count`, `avg_`/`min_`/`max_lead_time_seconds` |

## Notes

- Each routing sample is assigned the most recent item ID at that time; contiguous samples with the same (item, step) form one interval.
- `state_map` translates routing values (int, float or str keys) to names; without it the raw values are the labels. `station_map` is a deprecated alias.
- Defaults: ID in `value_string`, routing value in `value_integer` (`value_column_id=`, `value_column_routing=` to change).
- By default an interval ends at its last routing sample, so a change-only routing signal gives a dwell of 0; `infer_dwell_from_next=True` ends it at the item's next routing arrival.
