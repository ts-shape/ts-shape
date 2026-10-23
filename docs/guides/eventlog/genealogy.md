# Genealogy

Traceability asks two questions of the object relations in an
[`EventLog`](index.md): *where was this used?* (a recalled material lot: which
batches, serials and packing units contain it) and *where did this come from?*
(a returned serial: which batch, which material). ts-shape answers both from the
`o2o` table that [object detection](objects.md) builds from your `hierarchy`.

```python
import pandas as pd
from ts_shape.eventlog import ObjectSpec, detect_objects

t = pd.date_range("2024-01-01 06:00", periods=6, freq="10min")
df = pd.concat([
    pd.DataFrame({"systime": t[[0]], "uuid": "line:material", "value_string": ["M1"]}),
    pd.DataFrame({"systime": t[[0, 3]], "uuid": "line:batch", "value_string": ["B1", "B2"]}),
    pd.DataFrame({"systime": t, "uuid": "line:serial",
                  "value_string": ["S1", "S2", "S3", "S4", "S5", "S6"]}),
    pd.DataFrame({"systime": t[[0, 2, 4]], "uuid": "line:hu",
                  "value_string": ["HU1", "HU2", "HU3"]}),
], ignore_index=True)

specs = [
    ObjectSpec("line:material", "material"),
    ObjectSpec("line:batch", "batch"),
    ObjectSpec("line:serial", "serial"),
    ObjectSpec("line:hu", "handling_unit"),
]
log = detect_objects(df, specs, hierarchy={
    "line:serial": ["line:batch", "line:hu"],  # a serial is in a batch and a box
    "line:batch": "line:material",
})
print(log.where_used("B1"))
print(log.where_from("S2"))
```

`part_of` edges point from the child to the parent, so the two directions are:

- `where_used(oid)` walks from a parent to its children and on: B1's serials,
  then whatever those serials are part of downstream (none here: the boxes are
  parents of the serials, not children).
- `where_from(oid)` walks from a child to its parents and on: S2's batch and box,
  then the batch's material.

Both return one row per edge into a newly reached object:

| column | meaning |
|---|---|
| `oid` | the object reached |
| `object_type` | its `ocel:type` (NA when the oid is only in `o2o`) |
| `qualifier` | the edge's `ocel:qualifier` |
| `depth` | steps from the start object (1 = direct) |
| `via` | the object one step closer to the start |

Extra `o2o` columns (a table you built yourself may carry, say, a validity time)
are appended per edge. Each object is reported at its shortest distance; when two
edges reach it at that distance (a serial reworked from one batch into another
has two batches) it has a row per edge. Cycles end: an object is visited once.

## Lineage

`log.lineage(oid)` returns both directions in one frame with a signed `depth`:
what the object came from at negative depth, the object itself at 0 (its
`qualifier` and `via` are NA), and where it was used at positive depth.

```python
print(log.lineage("B2"))
```

## Depth, qualifiers and time windows

- `depth=1` keeps only the direct neighbours; `depth=None` (default) walks to the
  end of the graph.
- `qualifiers` picks the `o2o` qualifiers to follow. The default follows every
  qualifier except the symmetric `co_occurs` (two objects present at the same
  time), which says nothing about genealogy.
- `start` / `end` restrict the walk to objects present in the half-open window
  `[start, end)`. `o2o` has no validity time of its own, so presence comes from
  the `lifecycle` rows of `object_changes` (`active` to `released`); touching the
  window at an end point is not an overlap. Objects without lifecycle rows (for
  example imported from another system) count as present. Naive times are UTC.

```python
print(log.where_used("M1", start="2024-01-01 06:00", end="2024-01-01 06:25"))
```

B2 starts at 06:30, after the window, so neither it nor its serials are reached.

The functions `where_used(log, oid)`, `where_from(log, oid)` and
`lineage(log, oid)` in `ts_shape.eventlog` do the same. An `oid` that is neither
in `objects` nor in `o2o` raises `ObjectNotFoundError` (a `KeyError`); a known
object without relations gives an empty frame. The walk is a breadth-first
search over arrays: 100k edges take well under a second.

## Sub-logs

Two filters return a new `EventLog` whose tables stay consistent: no relation,
`o2o` row or object change refers to a dropped event or object.

- `log.filter_by_object_type(types, keep_events="linked")` keeps the objects of
  `types` and the events linked to at least one of them (`keep_events="all"`
  keeps every event, without its relations to other objects).
- `log.filter_by_time(start, end)` keeps the events whose `ocel:timestamp` is in
  `[start, end)`, the objects present in the window (or without lifecycle rows,
  or linked to a kept event), and the object changes in the window plus each
  field's last value before `start`.

```python
serials = log.filter_by_object_type(["serial", "batch"])
print(serials.o2o)
morning = log.filter_by_time("2024-01-01 06:00", "2024-01-01 06:20")
print(sorted(morning.objects["ocel:oid"]))
```

`filter_by_pack` and `filter_by_object` select by event instead.
