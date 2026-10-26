# Object detection

Event detectors answer *what happened*. Object detection answers *to which thing*:
the order, serial, batch, carrier, coil or recipe a signal carries. You declare which
signal holds the ids of which object type; ts-shape turns the readings into objects
with lifetimes, relations between them, and links from events to the objects that
were present. The result is an [`EventLog`](index.md) in OCEL 2.0 form.

```python
import pandas as pd
from ts_shape.eventlog import ObjectSpec, detect_objects, object_intervals, where_used

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
id_template="{value}", attributes={}, exclude="misreads", delta_column="is_delta",
cycle_uuid=None, cycle_value_column="value_bool")` says that signal `uuid` carries
ids of `object_type`.

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

## Change flags (is_delta)

Many historians flag the row where a value was *set*: `is_delta=True` when the
operator scans or sets up a material number, `is_delta=False` on the repeats sent
every few seconds until the next change. A re-setup with the same material number
is a new flagged row without a value change. `ObjectSpec.delta_column` (default
`"is_delta"`) reads that flag:

```python
stamps = pd.date_range("2024-01-01 06:00", periods=7, freq="2s")
material = pd.DataFrame({
    "systime": stamps,
    "uuid": "ext:material",
    "value_string": ["M-100", "M-100", "M-100", "M-100", "M-100", "M-200", "M-200"],
    "is_delta": [True, False, False, True, False, True, False],
})
spec = ObjectSpec("ext:material", "material")
print(object_intervals(material, [spec])[["oid", "start", "end"]])
```

```text
     oid                     start                       end
0  M-100 2024-01-01 06:00:00+00:00 2024-01-01 06:00:06+00:00
1  M-100 2024-01-01 06:00:06+00:00 2024-01-01 06:00:10+00:00
2  M-200 2024-01-01 06:00:10+00:00 2024-01-01 06:00:12+00:00
```

The re-setup at 06:00:06 starts a second presence of M-100 (still one object).

- The flag is used on a signal that has both `True` and `False` rows (hold and
  repeat): there it tells a written change from a snapshot. A value change starts a
  new presence with or without the flag.
- A change-only store flags every row it keeps, and the only unchanged rows it keeps
  are re-sends (keep-alives, reconnects). On a signal whose rows are all `True`, an
  unchanged flagged row is a re-send and does not split the presence. All `False` or
  missing flags also leave the value changes alone.
- A flag right after a misread or an empty reading adds nothing: it marks the change
  back from the misread, so `A, NOREAD, A` is still one presence.
- A missing flag is no flag: it never starts a presence, and it does not count as a
  `False` row (`True` and missing only is a change-only store). Without the column,
  or with `delta_column=None`, presences follow value changes only, as before.

## Cycles

In a discrete process a cycle runs while a bool signal is `True`, for example from
"door closed" to "door open". The serial number is scanned somewhere inside the
cycle, and the part it names is the one made in that cycle. `cycle_uuid` makes the
cycle the serial's presence:

```python
s = pd.Timestamp("2024-01-01 08:00")
sec = pd.to_timedelta
door = pd.DataFrame({
    "systime": s + sec([0, 60, 70, 120], unit="s"),
    "uuid": "cell:door_closed",
    "value_bool": [True, False, True, False],
})
serial = pd.DataFrame({
    "systime": s + sec([20, 40, 60, 80, 85, 105, 125], unit="s"),
    "uuid": "cell:serial",
    "value_string": ["SN-1", "SN-1", "SN-1", "SN-1", "SN-2", "SN-2", "SN-2"],
    "is_delta": [True, False, False, False, True, False, False],
})
cell = pd.concat([door, serial], ignore_index=True)
serial_spec = ObjectSpec("cell:serial", "serial", cycle_uuid="cell:door_closed")
print(object_intervals(cell, [serial_spec])[["oid", "start", "end"]])
```

```text
    oid                     start                       end
0  SN-1 2024-01-01 08:00:00+00:00 2024-01-01 08:01:00+00:00
1  SN-2 2024-01-01 08:01:10+00:00 2024-01-01 08:02:00+00:00
```

SN-1 is scanned at 08:00:20 and is present for its cycle, 08:00–08:01, although the
scanner repeats it until 08:01:20. Without the cycle it would run from the scan to
the next scan.

- A presence belongs to the cycle `[rise, fall)` its first reading (the scan) falls
  in. Repeats of the id in the next cycle do not count; only a new scan does.
- Several scans of one id in one cycle give one presence. Two different ids in one
  cycle each get the whole cycle, with a `DataQualityWarning`.
- A presence that starts outside every cycle keeps its own span (scan to next
  change); one `DataQualityWarning` per call gives the count.
- A cycle still `True` at the cycle signal's last reading ends at the last timestamp
  of the data. Missing readings of the cycle signal are skipped.
- `cycle_value_column` (default `"value_bool"`) names the cycle signal's column;
  numbers count as `True` when non-zero. A `cycle_uuid` that is not in the frame
  raises `ValueError`.

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

## Usage links (made_from / made_with / made_by)

Material, purchased-part and pre-product numbers are set at the start of a run and
often stay the same over many cycles. Every part made while such a number is held
used it: a bill-of-materials link that `hierarchy` (one parent, the longest
overlap) does not express. `uses` declares these links per child signal:

```python
setup = pd.DataFrame({
    "systime": [s, s, s + sec(65, unit="s")],
    "uuid": ["cell:material", "cell:tool", "cell:material"],
    "value_string": ["MAT-A", "T-7", "MAT-B"],
    "is_delta": True,
})
plant = pd.concat([cell, setup], ignore_index=True)
cell_specs = [
    serial_spec,
    ObjectSpec("cell:material", "material"),
    ObjectSpec("cell:tool", "tool"),
]
bom = detect_objects(
    plant,
    cell_specs,
    uses={"cell:serial": {"cell:material": "made_from", "cell:tool": "made_with"}},
)
print(bom.o2o)
```

```text
  ocel:oid ocel:oid_2 ocel:qualifier
0     SN-1      MAT-A      made_from
1     SN-2      MAT-B      made_from
2     SN-1        T-7      made_with
3     SN-2        T-7      made_with
```

The material changes at 08:01:05, between the two cycles; the tool is set once and
held to the end.

- Keys are child signals, values map parent signals to a qualifier: `made_from`
  (material), `made_with` (tool), `made_by` (operator) or any text. Names follow the
  `hierarchy` rule: a uuid of one of the specs' signals, else an object type.
- Each child presence is linked to every parent presence it overlaps (touching
  end-to-start is no overlap). A parent held over the whole recording, even one read
  only once, links every child. Several materials give several links.
- A serial whose cycle spans a material change is linked to both materials: the
  part may hold material of either lot.
- One row per (child, parent, qualifier). `uses` works next to `hierarchy`, and
  `infer_o2o="co_occurs"` leaves out the pairs it already relates.
  `infer_o2o=False` skips it with the rest of `o2o`.
- `derive_o2o`, `relate` and `attach_objects` take `uses=` too.

Genealogy follows the new qualifiers: `where_used(bom, "MAT-A")` lists the serials
made from MAT-A, `where_from(bom, "SN-1")` the material and tool SN-1 was made from
and with.

```python
print(where_used(bom, "T-7"))
```

```text
    oid object_type  qualifier  depth  via
0  SN-1      serial  made_with      1  T-7
1  SN-2      serial  made_with      1  T-7
```

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
