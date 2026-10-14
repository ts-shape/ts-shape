# Object detection

Event detectors answer *what happened*. Object detection answers *to which thing*:
the order, serial, batch, carrier, coil or recipe a signal carries. You declare which
signal holds the ids of which object type; ts-shape turns the readings into objects
with lifetimes, relations between them, and links from events to the objects that
were present. The result is an [`EventLog`](index.md) in OCEL 2.0 form.

```python
import pandas as pd
from ts_shape.eventlog import ObjectSpec, detect_objects, object_intervals

t = pd.date_range("2024-01-01 06:00", periods=6, freq="10min")
df = pd.concat([
    pd.DataFrame({"systime": t[[0, 3]], "uuid": "line:batch", "value_string": ["B1", "B2"]}),
    pd.DataFrame({"systime": t[[0, 1, 2, 3, 4, 5]], "uuid": "line:serial",
                  "value_string": ["S1", "S2", "", "S3", "S4", "S4"]}),
], ignore_index=True)

specs = [ObjectSpec("line:batch", "batch"), ObjectSpec("line:serial", "serial")]
intervals = object_intervals(df, specs)
print(intervals[["oid", "type", "start", "end"]])

log = detect_objects(df, specs, hierarchy={"serial": "batch"})
print(log.o2o)
```

S2 is present 06:10–06:20: the empty reading at 06:20 ends it, and nothing is present
until S3 at 06:30. B2 is sent once at 06:30 and stays present to the end of the data
(06:50). With the hierarchy, each serial is `part_of` the batch it overlaps longest
(S1 and S2 → B1, S3 and S4 → B2).

## Declare the objects

`ObjectSpec(uuid, object_type, value_column="value_string", min_duration=None,
id_template="{value}", attributes={})` says that signal `uuid` carries ids of
`object_type`.

- Unknown object types are registered automatically.
- `id_template` builds the object id, e.g. `"{type}:{value}"` when two types can use
  the same id. Sharing an id between two types gives a `DataQualityWarning`.
- `attributes={"recipe": "line:recipe"}` records another signal's value as an object
  attribute: its value at the start, and every change while the object is present.
- `min_duration="5s"` drops id blips shorter than that; the neighbours close over
  the blip. The last id on a signal is never dropped.

`object_specs_from_metadata(metadata)` builds the specs from a metadata table.

## When is an object present?

- An id is present from its first reading until the next reading on the same signal:
  another id, or an empty reading (`""`, whitespace, NaN, None).
- A gap in the data does not end a presence; the id is held across it.
- An id that appears again later is the same object with another presence.
- A signal whose last reading is still an id keeps that object present until the
  last timestamp in the data, like the last segment of `SegmentExtractor`.
- Times are returned in UTC; naive input is read as UTC.

`object_intervals` returns one row per presence: `oid`, `type`, `start`, `end`,
`duration_seconds`, `uuid`, `source_uuid`, and one column per attribute.

## Relations between objects

`detect_objects(df, specs, hierarchy=None, infer_o2o=True)` returns an `EventLog`
with `objects`, `o2o` and `object_changes` (each object's active/released lifecycle
and attribute changes).

- With a `hierarchy` (`{"serial": "batch", "batch": "order"}`), each child presence
  is `part_of` the declared parent it overlaps longest. Only declared pairs are
  related.
- Without a hierarchy, objects of different types that are present at the same time
  are `co_occurs`. Touching end-to-start does not count.

## Link events to objects

```python
from ts_shape.eventlog import attach_objects, to_event_log
from ts_shape.events.production.machine_state import MachineStateEvents

state = pd.DataFrame({"systime": t, "uuid": "line:run",
                      "value_bool": [True, True, False, True, True, False]})
events = MachineStateEvents(state, "line:run").detect_run_idle()
log = to_event_log(events, detector="MachineStateEvents.detect_run_idle")
log = attach_objects(log, df, specs, hierarchy={"serial": "batch"})
print(log.relations[["ocel:eid", "ocel:oid", "ocel:type"]].head())
```

An event is linked to every object present at its timestamp. At a hand-over instant,
when one object ends and the next starts at the same time on the same signal, only
the incoming object gets the event. Otherwise the last sample of a presence belongs
to the object. `relate(log, intervals)` does the same with an interval table you
built yourself; a missing end means the object is still present.

Objects are not scoped to a station: an event links to objects of every signal in
`df`. Run per line, or give each line its own object type, when that matters.

## Export

`to_event_log_ocel(log)` returns the OCEL 2.0 tables. Use
`to_event_log_ocel(log, pm4py=True)` for pm4py: it adds `ocel:activity` and
`ocel:timestamp` to the relations and widens `object_changes` to one column per
field, so the log exports to OCEL JSON, XML and SQLite.
