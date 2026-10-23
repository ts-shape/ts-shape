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
id_template="{value}", attributes={}, exclude="misreads")` says that signal `uuid`
carries ids of `object_type`.

- Unknown object types are registered automatically.
- `id_template` builds the object id, e.g. `"{type}:{value}"` when two types can use
  the same id. Sharing an id between two types gives a `DataQualityWarning`.
- `attributes={"recipe": "line:recipe"}` records another signal's value as an object
  attribute: its value at the start, and every change while the object is present.
- `min_duration="5s"` drops id blips shorter than that; the neighbours close over
  the blip. The last id on a signal is never dropped.
- `exclude` removes readings that are not ids. The default `"misreads"` uses
  [`is_misread`](../../reference/ts_shape/utils/id_codes.md): scanner no-reads and
  placeholders (`NOREAD`, `NO READ`, `ERR`, `ERROR`, `#N/A`, `N/A`, `NA`, `NULL`,
  `NONE`, `NAN`, any case), text without a letter or digit (`"????????"`,
  `"--------"`), PLC default numbers (`0`, `-1`, `32767`, `-32768`, `65535`, ...,
  also as text and with leading zeros), one repeated character (`"XXXXXXXX"`), and
  truncated reads (a short read that is the start of the id next to it). Pass a list
  of values to exclude exactly those, a function that takes the signal's values in
  time order and returns a boolean mask, or `None` to keep every reading.

`object_specs_from_metadata(metadata)` builds the specs from a metadata table.
Without metadata, [`identify_objects`](../object-identification.md) finds the id
signals and their types from the values alone; `report.to_object_specs()` returns the
specs.

## When is an object present?

- An id is present from its first reading until the next reading on the same signal:
  another id, or an empty reading (`""`, whitespace, NaN, None).
- A gap in the data does not end a presence; the id is held across it.
- An excluded reading (a misread) is never an object. Between two readings of the
  same id it counts as a gap, so `A, NOREAD, A` is one presence of A. Anywhere else
  it counts as an empty reading: in `A, NOREAD, B`, A ends at the NOREAD.
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

- The lifecycle joins presences of one object that overlap, or that touch at the
  same station. A hand-over to another station at the same instant starts a new
  presence: the lifecycle reads `active` again there (one value per instant, and
  the incoming presence owns it). For enter and leave events per station, lead
  times and WIP, see [Object analytics](object-analytics.md).
- With a `hierarchy`, each child presence is `part_of` the declared parent it
  overlaps longest. Only declared pairs are related; without a hierarchy `o2o` is
  empty.
- A hierarchy entry names object types (`{"serial": "batch", "batch": "order"}`) or
  id signals. Signal entries keep two lines apart, because a line-1 serial can only
  be part of a line-1 batch:

  ```python
  hierarchy = {
      "L1:serial": ["L1:batch", "L1:hu"],  # two parents: batch and handling unit
      "L2:serial": "L2:batch",
  }
  ```

  A name that is the uuid of one of the specs' signals is a signal, otherwise an
  object type. A list of parents gives one `part_of` per parent.
- `infer_o2o="co_occurs"` also relates every two objects of different types that
  are present at the same time with `co_occurs` (touching end-to-start does not
  count; pairs already `part_of` are not repeated). On a plant this can be hundreds
  of thousands of rows. `infer_o2o=False` skips `o2o`. `derive_o2o(intervals,
  hierarchy=..., co_occurs=False)` does the same from an interval table.

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

An event is linked to every object whose presence overlaps it:

- An interval event (`ts_shape:start_timestamp` to `ocel:timestamp`) links to every
  object present during part of it, also one that left before the event ended. A
  run that spans two batches links to both. Only touching at an end point is not an
  overlap.
- A point event links to the objects present at its timestamp. At a hand-over
  instant, when one object ends and the next starts at the same time on the same
  signal, only the incoming object gets the event. Otherwise the last sample of a
  presence belongs to the object.

`relate(log, intervals)` does the same with an interval table you built yourself; a
missing end means the object is still present.

### Scope

By default an event links to the objects of every signal in `df`, so a stop on line
1 also gets the serials of line 2. Pass `scope` to keep each line apart:

```python
scope = {"line:batch": "L1", "line:serial": "L1", "line:run": "L1"}
# a second line adds its signals: {"line2:batch": "L2", "line2:run": "L2", ...}
events = to_event_log(
    MachineStateEvents(state, "line:run").detect_run_idle(),
    detector="MachineStateEvents.detect_run_idle",
)
scoped = attach_objects(events, df, specs, scope=scope)
print(scoped.relations[["ocel:eid", "ocel:oid", "ocel:type"]].head())
```

`scope` maps signal uuids to a label. An object gets the label of its id signal
(or of its `source_uuid`). An event gets the label of its source signal: the
`source_uuid` / `<pack>:source_uuid` column `to_event_log` writes, or its `asset`
relation. An event links only to objects with the same label. Events and objects
without a label are linked to nothing by time, and a `DataQualityWarning` names
them.

Without `scope`, a `DataQualityWarning` suggests one when an event links to more
than `fanout_threshold` (default 1) objects of one type present at the same time.
One signal carries one object at a time, so this means several signals of that type,
for example two lines. Raise the threshold, or pass `fanout_threshold=None`, when
several stations of one line really share the events.

Naive timestamps are read as UTC everywhere. When `link_events_to_objects` or
`relate` gets one table tz-naive and the other tz-aware, a `DataQualityWarning` says
so: local wall-clock times would be linked shifted by the UTC offset, so localize
them first.

## Export

`to_event_log_ocel(log)` returns the OCEL 2.0 tables. Use
`to_event_log_ocel(log, pm4py=True)` for pm4py: it adds `ocel:activity` and
`ocel:timestamp` to the relations and widens `object_changes` to one column per
field, so the log exports to OCEL JSON, XML and SQLite.
