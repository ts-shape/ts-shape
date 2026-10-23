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
| `build_timeline(infer_dwell_from_next=False, interval_end="last_sample")` | one row per identifier and station: `identifier`, `station_name`, `start`, `end`, `duration_seconds`, `sample_count`, `station_sequence` |
| `lead_time()` | per identifier: `first_station`, `last_station`, `first_seen`, `last_seen`, `lead_time_seconds`, `stations_visited`, `station_path` |
| `current_status(infer_dwell_from_next=False, interval_end="last_sample")` | per identifier: `current_station`, `arrived_at`, `time_at_station_seconds` |
| `station_dwell_statistics(infer_dwell_from_next=False, interval_end="last_sample")` | per station: `identifier_count`, `min_`/`avg_`/`max_`/`total_dwell_seconds` |
| `to_intervals(object_type="serial", interval_end="last_sample")` | one row per visit: `oid`, `type`, `start`, `end`, `duration_seconds`, `uuid`, `source_uuid`, `station` |

## Notes

- `station_uuids` maps each station signal UUID to a station name. The identifier is read from `value_string` (`value_column=` to change).
- By default a visit ends at the identifier's last sample at the station, so a station that logs the identifier once on arrival reports a dwell of 0. `infer_dwell_from_next=True` ends each visit at the identifier's next station arrival; the last visit stays open.
- Identifiers must match exactly across stations: `Order-42` and `Order-042` are different items.
- `OrderTraceabilityEvents` is a deprecated alias of this class.
- `lead_time()`: `last_seen` (and `end`) is the latest visit end, also when an identifier is at two stations at once and the earlier visit ends later.
- `exclude=` drops readings that are not identifiers, like [`ObjectSpec.exclude`](../../guides/eventlog/objects.md): `"misreads"` (`NOREAD`, `#N/A`, `0`, `65535`, blanks, ...; see `ts_shape.utils.id_codes.is_misread`), a list of values, or a callable returning a mask. Between two readings of the same identifier an excluded reading is a gap (the identifier is held across it); anywhere else it is an empty reading. With `exclude="misreads"`, `to_intervals(interval_end="next_change")` gives the presences of `object_intervals` with its default `ObjectSpec`. The default `None` keeps every reading; `"misreads"` is planned to become the default in the next major release.
- Readings at the same timestamp keep their input order.
- `interval_end="next_change"` ends a visit at the next different reading on the station signal (the next identifier or an empty reading) instead of at its last sample, so the identifier is held until it changes, like [`object_intervals`](../../guides/eventlog/objects.md); a station's last visit is held until the last timestamp of the station signals. The default `"last_sample"` keeps the results as before. `infer_dwell_from_next` is applied after it.
- `to_intervals(object_type="serial")` returns the visits as an object interval table: the columns of `object_intervals` (`oid`, `type`, `start`, `end` in UTC, `duration_seconds`, `uuid`, `source_uuid`) plus `station`. It feeds `presence_events`, `object_kpis`, `wip_over_time`, `attach_counts` and `relate` directly; see [Object analytics](../../guides/eventlog/object-analytics.md#traceability-classes).
- In `to_event_log`, the identifier column binds objects of type `serial` and the station column objects of type `station` (plus `asset` from `source_uuid`, the station signal). These links do not change the event ids. Pass `objects={"serial": None}` to leave one out, or `objects={"work_order": "identifier"}` to name the type yourself.
