# Adapters

An adapter converts one detector's output frame into an
[`EventLog`](index.md#the-canonical-schema). ts-shape has one generic adapter
(`ts_shape.eventlog.adapters.adapt`), configured per detector method by a
`LabelRule` in `REGISTRY`. For a new detector method you normally add a
`LabelRule` ([Adding a new detector method](taxonomy.md#adding-a-new-detector-method));
a custom adapter is needed only for output the generic adapter cannot express.

## Shapes

The rule's `shape` decides where the event time comes from:

| Shape | Required columns | `ocel:timestamp` | `ts_shape:start_timestamp`, `ts_shape:duration_s` |
|---|---|---|---|
| `point` | `systime` | `systime` | `NaT`, `NaN` |
| `interval` | `start`, `end` | `end` | `start`, `end - start` in seconds |
| `summary` | `start`, `end` | `end` | `start`, `end - start` in seconds |
| `static` | none | `to_event_log(..., timestamp=...)`, the same for all rows; default the Unix epoch (`1970-01-01T00:00Z`) | `NaT`, `NaN` |

A frame without the required columns raises an error naming the detector.
Naive timestamps are read as UTC. Interval and summary rows whose `end` is
before their `start` are dropped with a `DataQualityWarning`.
The implementation is `adapt` in `src/ts_shape/eventlog/adapters.py`.

## LabelRule fields

| Field | Default | Effect |
|---|---|---|
| `template` | required | the `ocel:activity`; `{column}` placeholders are filled per row |
| `pack` | required | stored as `ts_shape:pack` and used as prefix of the detector's own columns |
| `shape` | `"point"` | see above |
| `produces_objects` | `("asset",)` | object types created automatically from standard columns (`source_uuid` becomes an `asset`); `()` for none |
| `severity_field` | `None` | numeric column bucketed into `ts_shape:severity` (a boolean column maps `True` to `warn`, `False` to `info`); without it a `severity` column with `info` / `warn` / `critical` is passed through |
| `value_field` | `None` | column used as `ts_shape:value`; without it the first of `value`, `value_double`, `value_integer` |
| `drop_fields` | `()` | output columns not copied into the event log |
| `standard_attrs` | `{}` | mapping to the [standard attribute keys](taxonomy.md#standard-attribute-extension) |

## What `to_event_log` does

1. Looks up `(class, method)` from `detector="Class.method"` in `REGISTRY`
   (deprecated names are resolved to the current ones). An unknown name
   raises `KeyError`.
2. Uses a custom adapter if one is registered for that method.
3. Otherwise runs the generic adapter:
    - takes the timestamps according to the shape;
    - renders the activity from the template (a missing value renders as
      `unknown`, with a `DataQualityWarning`);
    - builds a content-based `ocel:eid`: a UUIDv5 of the detector, the row's
      `source_uuid`, its timestamps, the activity and every other column of
      the row. The same input always gives the same ids, independent of row
      order, and the same detector on two machines does not collide;
    - fills `ts_shape:severity`, `ts_shape:value` and the standard attributes;
    - copies every other output column as `<pack>:<column>` and drops such
      columns when they are completely empty;
    - creates the automatic objects and those passed with `objects=`.
4. Validates the result: required columns and dtypes, unique `ocel:eid`, and
   every relation pointing to an existing event and object.

For `MachineStateEvents.detect_run_idle` (template
`production.machine_state.{state}`, shape `interval`) an output row with
`state="run"`, `source_uuid="asset-A"`, `start` and `end` becomes an event
with activity `production.machine_state.run`, timestamp `end`, start
timestamp `start`, an `asset` object `asset-A`, and the attributes
`production:uuid`, `production:is_delta` and `production:duration_seconds`.

## Custom adapters

Register a custom adapter with `@register_adapter("Class", "method")` when
the generic one does not fit: several events per output row, no usable time
columns, objects that depend on the data, or state carried across rows. The
adapter receives the detector frame and returns an `EventLog`; it is validated
like any other.

```python
import pandas as pd
from ts_shape.eventlog import EventLog, register_adapter, to_event_log
from ts_shape.eventlog import schema as S
from ts_shape.eventlog.taxonomy import REGISTRY, LabelRule

# A real detector registers its rule in src/ts_shape/eventlog/taxonomy.py.
REGISTRY[("MyDetector", "alarm_windows")] = LabelRule(
    template="production.custom.{kind}", pack="production", shape="point",
)

@register_adapter("MyDetector", "alarm_windows")
def raise_and_clear(detector_frame, *, rule, detector, objects, qualifiers):
    """Two events per row: one when the alarm is raised, one when it clears."""
    events, relations = [], []
    for i, row in detector_frame.iterrows():
        for kind in ("raised", "cleared"):
            eid = f"e-{detector}-{i}-{kind}"
            events.append({
                S.OCEL_EID: eid,
                S.OCEL_ACTIVITY: f"production.custom.{kind}",
                S.OCEL_TIMESTAMP: pd.Timestamp(row[f"{kind}_at"], tz="UTC"),
                S.TS_DETECTOR: detector,
                S.TS_PACK: rule.pack,
            })
            relations.append({
                S.OCEL_EID: eid, S.OCEL_OID: row["asset_id"],
                S.OCEL_TYPE: "asset", S.OCEL_QUALIFIER: "produced_on",
            })
    return EventLog(
        events=pd.concat([S.empty_events(), pd.DataFrame(events)], ignore_index=True),
        objects=pd.DataFrame({
            S.OCEL_OID: detector_frame["asset_id"].astype("string").unique(),
            S.OCEL_TYPE: "asset",
        }),
        relations=pd.concat([S.empty_relations(), pd.DataFrame(relations)], ignore_index=True),
    )

frame = pd.DataFrame({"asset_id": ["press-01"], "raised_at": ["2025-01-01 08:00"], "cleared_at": ["2025-01-01 08:05"]})
log = to_event_log(frame, detector="MyDetector.alarm_windows")
```

The returned log must have at least `ocel:eid` (unique), `ocel:activity`,
`ocel:timestamp`, `ts_shape:detector` and `ts_shape:pack` in `events`, and
every relation must point to an existing event and object.
`clear_overrides()` removes registered adapters again.
