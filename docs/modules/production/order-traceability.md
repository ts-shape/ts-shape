# ValueTraceabilityEvents

Timeline, lead time and current location of a shared identifier (serial number, order ID, batch code) across stations, where each station writes the identifier it is processing to its own string signal.

`ts_shape.events.production.order_traceability` · [Production guide](../../guides/production.md) · [API reference](../../reference/ts_shape/events/production/order_traceability.md)

## Example

```python
import pandas as pd
import ts_shape
from ts_shape.events.production.order_traceability import ValueTraceabilityEvents

stations = {"line1:cutting:serial": "Cutting", "line1:welding:serial": "Welding",
            "line1:painting:serial": "Painting"}
df = pd.concat([
    ts_shape.make_id_signal(uuid, ["SN-1", "SN-2", "SN-3"], hold=10, freq="1min",
                            start=f"2024-01-01 06:{10 * i:02d}")
    for i, uuid in enumerate(stations)
], ignore_index=True)

tracer = ValueTraceabilityEvents(df, station_uuids=stations)
print(tracer.lead_time()[["identifier", "station_path", "lead_time_seconds"]])
status = tracer.current_status()
```

## Methods

| Method | Returns |
|---|---|
| `build_timeline(infer_dwell_from_next=False)` | one row per identifier and station: `identifier`, `station_name`, `start`, `end`, `duration_seconds`, `sample_count`, `station_sequence` |
| `lead_time()` | per identifier: `first_station`, `last_station`, `first_seen`, `last_seen`, `lead_time_seconds`, `stations_visited`, `station_path` |
| `current_status()` | per identifier: `current_station`, `arrived_at`, `time_at_station_seconds` |
| `station_dwell_statistics(infer_dwell_from_next=False)` | per station: `identifier_count`, `min_`/`avg_`/`max_`/`total_dwell_seconds` |

## Notes

- `station_uuids` maps each station signal UUID to a station name. The identifier is read from `value_string` (`value_column=` to change).
- By default a visit ends at the identifier's last sample at the station, so a station that logs the identifier once on arrival reports a dwell of 0. `infer_dwell_from_next=True` ends each visit at the identifier's next station arrival; the last visit stays open.
- Identifiers must match exactly across stations: `Order-42` and `Order-042` are different items.
- `OrderTraceabilityEvents` is a deprecated alias of this class.
