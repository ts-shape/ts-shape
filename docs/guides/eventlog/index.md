# Event log (XES and OCEL)

`ts_shape.eventlog` turns detector output into one canonical event log whose
column names follow the [OCEL 2.0](https://www.ocel-standard.org/) and
[XES](https://xes-standard.org/) standards. Use it to combine the events of
several detectors, attach objects such as assets and batches, and hand the
result to process-mining tools (pm4py, Celonis, Disco, OCEL viewers).
ts-shape writes no files and imports no process-mining library; it returns
DataFrames.

Related pages: [Object detection](objects.md) (batches, serials, recipes as
OCEL objects), [Labeling and taxonomy](taxonomy.md) (activity names and
standard attributes), [Adapters](adapters.md) (how a detector frame is
converted, and custom adapters).

## Quick start

```python
import ts_shape
from ts_shape.eventlog import concat, to_event_log, to_event_log_ocel, to_event_log_xes

running = ts_shape.make_timeseries(["press-01:running"], n_points=600, freq="1min", value_column="value_bool")
torque = ts_shape.make_timeseries(["press-01:torque"], n_points=600, freq="1min", n_outliers=3)
torque["batch_id"] = ["B-1"] * 300 + ["B-2"] * 300

states = ts_shape.MachineStateEvents(running, run_state_uuid="press-01:running").detect_run_idle(min_duration="5min")
outliers = ts_shape.OutlierDetectionEvents(torque, value_column="value_double").detect_outliers_zscore(n_sigma=4.0)

state_log = to_event_log(states, detector="MachineStateEvents.detect_run_idle")
outlier_log = to_event_log(
    outliers,
    detector="OutlierDetectionEvents.detect_outliers_zscore",
    objects={"batch": "batch_id"},            # link each event to the batch in this column
    qualifiers={"batch": "during_batch"},
)
log = concat(state_log, outlier_log)          # sorted by time, objects de-duplicated

xes = to_event_log_xes(log, case_object_type="asset")   # one case per asset
tables = to_event_log_ocel(log)               # .events .objects .relations .o2o .object_changes
```

`detector` names the method that produced the frame as `"Class.method"`; it
selects the label rule (activity name, shape, standard attributes) from
`ts_shape.eventlog.taxonomy.REGISTRY`. Every public detector method has an
entry. Deprecated class and method names are accepted and resolved to the
current ones.

The `OCEL2Tables` returned by `to_event_log_ocel` map one to one onto pm4py's
`OCEL` constructor.

## The canonical schema

An `EventLog` holds the five OCEL 2.0 tables as DataFrames: `events`,
`objects`, `relations` (event to object), `o2o` (object to object) and
`object_changes` (time-varying object attributes). The last two are empty
unless you supply them.

### events

| Column | Type | Content |
|---|---|---|
| `ocel:eid` | string | content-based id, a UUIDv5 of detector, `source_uuid`, timestamps, activity and row values |
| `ocel:activity` | string | dotted activity name, for example `production.machine_state.run` |
| `ocel:timestamp` | datetime, UTC | event time; the interval end for intervals |
| `ts_shape:start_timestamp` | datetime, UTC | interval start; `NaT` for point events |
| `ts_shape:duration_s` | float | interval duration in seconds |
| `ts_shape:detector` | string | `"Class.method"` that produced the event |
| `ts_shape:pack` | string | `quality`, `production`, `engineering`, `maintenance`, `energy`, `correlation`, `supplychain` or `development` |
| `ts_shape:severity` | string | `info`, `warn` or `critical` |
| `ts_shape:value` | float | main numeric value, where there is one |
| `<pack>:<column>` | any | the detector's own columns, prefixed with the pack name |

These nine `ocel:` and `ts_shape:` columns are present in every log. Other
columns (the [standard attributes](taxonomy.md#standard-attribute-extension)
such as `ts_shape:method`, and the `<pack>:<column>` columns) appear only when
they hold a value, so logs of different detectors have different extra
columns. `concat` takes the union and fills gaps with `NaN`;
`align_columns(*logs)` gives each log the same full column set before you
stack them yourself.

### objects and relations

`objects` has `ocel:oid` and `ocel:type`. The standard types are `asset`,
`tool`, `sensor`, `signal`, `station`, `cycle`, `batch`, `lot`, `recipe`,
`work_order`, `shift`, `material`, `part`, `serial`, `article` and
`operator`; `register_object_type("name")` adds one.

`relations` links events to objects: `ocel:eid`, `ocel:oid`, `ocel:type` and
`ocel:qualifier` (the role of the object, for example `during_batch`).

`o2o` (`ocel:oid`, `ocel:oid_2`, `ocel:qualifier`) relates objects, for example
a sensor mounted on an asset. `object_changes` (`ocel:oid`, `ocel:type`,
`ocel:field`, `ocel:value`, `ocel:timestamp`) records attribute values that
change over time.

## Attaching objects

Objects get onto events in two ways:

1. **Automatically.** Most label rules declare `produces_objects=("asset",)`;
   the adapter then creates an `asset` object from the `source_uuid` column.
2. **By the caller**, with `objects=`. A binding value is a column name, a
   callable `row -> oid`, a `pd.Series` aligned with the detector output by
   index label (a length or label mismatch raises `ValueError`), or any other
   value -- including a string that is not a column name -- used as the id for
   every row. `None` links no object.

```python
log = to_event_log(
    outliers,
    detector="OutlierDetectionEvents.detect_outliers_zscore",
    objects={
        "batch": "batch_id",                 # column
        "operator": "op-7",                  # not a column: constant id
        "station": lambda row: "line-1",
    },
    qualifiers={"batch": "during_batch", "operator": "operated_by"},
    o2o=[{"ocel:oid": "press-01:torque", "ocel:oid_2": "line-1", "ocel:qualifier": "part_of"}],
    object_changes=[{
        "ocel:oid": "press-01:torque", "ocel:type": "asset",
        "ocel:field": "firmware", "ocel:value": "v2",
        "ocel:timestamp": "2025-01-01T03:00:00Z",
    }],
)
```

Every `ocel:oid` referenced in `o2o` or `object_changes` must exist in
`objects`, otherwise validation raises. [Object detection](objects.md) derives
objects from id signals instead of columns.

## XES export

`to_event_log_xes(log, case_object_type=...)` flattens the log to one case per
object of the given type, with the XES columns `case:concept:name`,
`concept:name`, `time:timestamp` and `lifecycle:transition`. The same log can
be flattened per asset, per batch or per cycle. A log without objects of that
type raises an error instead of inventing case ids.

- `lifecycle="single"` (default): one row per event with
  `lifecycle:transition="complete"`.
- `lifecycle="two_row"`: intervals become a `start` and a `complete` row,
  paired by `concept:instance`.

`org:resource` is filled only from `operator` objects; the column is omitted
when the log has none.

A runnable script with output is
[`examples/eventlog_demo.py`](https://github.com/ts-shape/ts-shape/blob/main/examples/eventlog_demo.py).
