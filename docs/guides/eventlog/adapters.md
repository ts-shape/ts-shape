# Event Log: adapter anatomy & custom adapters

How a detector's raw DataFrame becomes a canonical `EventLog`. This is the
mechanics companion to the [Event Log overview](index.md) (the schema and
exports) and the [Labelling standard & taxonomy](taxonomy.md) (the naming and
attribute rules).

---

## Adapter anatomy

An **adapter** is the function that turns one detector's legacy DataFrame
into a canonical [`EventLog`](index.md#the-canonical-schema). ts-shape ships
*one* generic adapter (`adapters.adapt`) plus a registry of per-method
[`LabelRule`](#the-labelrule-fields) entries that parameterize it. In
practice you almost never write a custom adapter; you add a `LabelRule`
to the registry and the generic adapter does the rest.

### The four shapes

Every detector method is classified into one of four shapes. The shape
tells the generic adapter which legacy columns to look for and how to
populate `ocel:timestamp` / `ts_shape:start_timestamp` /
`ts_shape:duration_s`.

| Shape | When to use it | Time columns probed | Resulting timestamps |
|---|---|---|---|
| `point` | One event per row, single timestamp (e.g. outlier detected at T). | `systime` (canonical). Falls back to the first datetime column if absent. | `ocel:timestamp` = the detected time; `ts_shape:start_timestamp` = `NaT`; `ts_shape:duration_s` = `NaN`. |
| `interval` | Each row spans a window with explicit `start` and `end` (e.g. a run/idle interval). | Start: `start`. End: `end`. | `ocel:timestamp` = end; `ts_shape:start_timestamp` = start; `ts_shape:duration_s` = `(end - start).total_seconds()`. Falls back to `point` shape if start/end columns are absent. |
| `summary` | Each row is an aggregate over a window (KPI per shift, daily mean, etc.). | Start: `start`. End: `end`. Both come from the canonical summary schema declared in `events/_output.py`. | `ocel:timestamp` = `end`; `ts_shape:start_timestamp` = `start`; `ts_shape:duration_s` = `(end - start).total_seconds()`. |
| `static` | No natural time (e.g. a Gauge R&R summary, a routing-paths table). | None — a fixed `now-UTC` is broadcast to every row. | All rows share the same `ocel:timestamp = now`; `start_timestamp`/`duration_s` are null. |

The shape is declared in the `LabelRule` and lives in
`src/ts_shape/eventlog/taxonomy.py`. The branches that implement each
shape live in `src/ts_shape/eventlog/adapters.py` (function `adapt`).

### The `LabelRule` fields

| Field | Type | Default | What it controls |
|---|---|---|---|
| `template` | `str` | required | The `ocel:activity` value. May contain `{column}` placeholders that get substituted from the legacy row at adapter time. Example: `"production.machine_state.{state}"`. |
| `pack` | `str` | required | One of `quality`, `production`, `engineering`, `maintenance`, `supplychain`, `energy`, `correlation`. Stored as `ts_shape:pack` and used as the prefix for detector-specific attributes. |
| `shape` | `str` | `"point"` | One of `point`, `interval`, `summary`, `static` — see above. |
| `produces_objects` | `tuple[str, ...]` | `("asset",)` | Object types the adapter auto-extracts from standard legacy columns (e.g. `source_uuid → asset`). Empty tuple = events only, no auto-extracted objects. Caller-supplied bindings via `objects=` are honored regardless. |
| `severity_field` | `str \| None` | `None` | Name of a numeric column to bucket into `ts_shape:severity`. Falls back to a `severity` column (passed through verbatim) when omitted. |
| `value_field` | `str \| None` | `None` | Name of the numeric column to expose as `ts_shape:value`. Falls back to `value` / `value_double` / `value_integer` when omitted. |
| `drop_fields` | `tuple[str, ...]` | `()` | Legacy columns to *not* promote to attributes (e.g. internal helper columns). |

### What `to_event_log()` does, step by step

1. Parse the `detector="ClassName.method_name"` string and look up
   `(ClassName, method_name)` in
   `ts_shape.eventlog.taxonomy.REGISTRY`. Missing entry → `KeyError`
   (the coverage test prevents this from ever shipping).
2. Check the `_OVERRIDES` table for a function registered with
   `@register_adapter("ClassName", "method_name")`. If one is
   registered, call it with `(legacy_df, *, rule, detector,
   objects, qualifiers)` and use its return value directly.
3. Otherwise call `adapters.adapt(legacy_df, rule=…, detector=…,
   objects=…, qualifiers=…)`:
    - **Resolve timestamps** based on `rule.shape` (see the four-shapes
      table above).
    - **Render `ocel:activity`** per row by substituting `{column}`
      placeholders in `rule.template` with values from the legacy row.
      Missing columns render as `unknown` (never raise).
    - **Generate stable `ocel:eid`** as
      `"e-" + uuid5(namespace, f"{detector}|{ts.isoformat()}|{i}|{activity}")`.
      Same input → same eid; safe to re-run.
    - **Map severity**: if `rule.severity_field` is set and numeric,
      bucket via `< 3.0 → info`, `3.0–4.5 → warn`, `≥ 4.5 → critical`.
      Falls back to a literal `severity` column when present.
    - **Pull value**: if `rule.value_field` is set, coerce to float and
      expose as `ts_shape:value`. Falls back to `value` /
      `value_double` / `value_integer`.
    - **Prefix attributes**: every legacy column not consumed above is
      added as `<pack>:<col>` so it's namespaced and never clashes with
      OCEL/XES columns.
    - **Trim empty extras**: any non-core column (a `<pack>:<col>`
      passthrough or a standard-attr extension) that is entirely empty is
      dropped — only the 9 canonical core columns are kept unconditionally,
      preserving a stable append-friendly schema.
    - **Auto-extract objects**: for each type in `rule.produces_objects`,
      look for the standard binding column (today: `source_uuid → asset`)
      and create relations. Merge in caller-supplied `objects=` (any
      type allowed; `qualifiers=` provides the role string).
4. Run `schema.validate(...)` on the result — checks required columns,
   dtypes, unique `ocel:eid`, and that every relation references an
   existing event and object.

### Concrete walkthrough — `MachineStateEvents.detect_run_idle`

The legacy DataFrame returned by `detect_run_idle()` looks like this:

| `start` | `end` | `uuid` | `source_uuid` | `is_delta` | `state` | `duration_seconds` |
|---|---|---|---|---|---|---|
| `2026-05-07 08:00:00+00:00` | `2026-05-07 08:04:30+00:00` | `prod:run_idle` | `asset-A` | `False` | `run` | `270.0` |

`to_event_log(legacy_df, detector="MachineStateEvents.detect_run_idle")`
applies the registry entry
`LabelRule(template="production.machine_state.{state}",
pack="production", shape="interval", produces_objects=("asset",))`
and produces:

| Legacy column | Lands in… | Why |
|---|---|---|
| `start` | `ts_shape:start_timestamp` | Interval-shape start probe matched. |
| `end` | `ocel:timestamp` | Interval-shape end probe matched. |
| (computed) | `ts_shape:duration_s = 270.0` | `(end - start).total_seconds()`. |
| `state = "run"` | `ocel:activity = "production.machine_state.run"` | Substituted into `{state}` placeholder. |
| `source_uuid` | `ocel:oid` (in `objects` & `relations`), with `ocel:type = "asset"` | Auto-extracted because `produces_objects` includes `"asset"`. |
| `uuid` | `production:uuid` (event attribute) | Non-canonical column → prefixed and attached. |
| `is_delta` | `production:is_delta` (event attribute) | Same. |
| `duration_seconds` | `production:duration_seconds` (event attribute) | Same. The canonical `ts_shape:duration_s` is always recomputed. |
| (computed) | `ocel:eid = "e-<uuid5>"` | Stable hash of `(detector, timestamp, row-key, activity)`. |
| (constant) | `ts_shape:detector = "MachineStateEvents.detect_run_idle"` | From the `detector=` argument. |
| (constant) | `ts_shape:pack = "production"` | From the `LabelRule`. |

If the caller had passed `objects={"batch": "batch_id"}`, an additional
`batch` object would have been bound from the (caller-provided)
`batch_id` column, with relation qualifier from the `qualifiers={"batch":
"during_batch"}` mapping.

### When to override with a custom adapter

Reach for `@register_adapter` only when the generic adapter cannot
express what your detector returns:

- The legacy DataFrame is irregular (no time column, multiple
  sub-frames, nested dict columns).
- You need to emit **multiple events per legacy row** — e.g. one row
  describes a batch with five sub-stage transitions and you want one
  event per transition.
- You want `produces_objects` to depend on **runtime data** rather than
  on a static rule.
- You need cross-row state (running totals, sessions) that the row-by-row
  generic adapter cannot compute.

In every other case — including new methods on existing detectors —
just add a `LabelRule`. See [Adding a new detector method](taxonomy.md#adding-a-new-detector-method).

---

## Custom adapters

For the rare detector whose output doesn't fit any of the four shapes
(see [When to override](#when-to-override-with-a-custom-adapter)),
register an override with `@register_adapter`. The normalizer consults
the override **before** the generic adapter and validates the result
the same way.

### Adapter signature

```python
def my_adapter(
    legacy_df: pd.DataFrame,
    *,
    rule: LabelRule,            # the registry entry for this method
    detector: str,              # "MyDetector.weird_method"
    objects:    Mapping[str, object] | None,
    qualifiers: Mapping[str, str] | None,
) -> EventLog: ...
```

### Invariants the override must satisfy

- `events` has the columns listed in the [Events table](index.md#events): at
  minimum `ocel:eid` (unique), `ocel:activity`, `ocel:timestamp`,
  `ts_shape:detector`, `ts_shape:pack`.
- Every `ocel:eid` referenced from `relations` exists in `events`.
- Every `(ocel:oid, ocel:type)` pair in `relations` exists in `objects`.

`to_event_log()` runs `schema.validate(...)` on the returned
`EventLog`, so violations surface immediately.

### Working example

```python
import pandas as pd
from ts_shape.eventlog import EventLog, register_adapter
from ts_shape.eventlog import schema as S
from ts_shape.eventlog.taxonomy import REGISTRY, LabelRule

# 1. Make the registry aware of the method (real detectors do this in
#    src/ts_shape/eventlog/taxonomy.py — done inline here for brevity).
REGISTRY[("MyDetector", "weird_method")] = LabelRule(
    template="production.custom.{kind}",
    pack="production",
    shape="point",
    produces_objects=("asset",),
)

# 2. Register an override that emits TWO events per legacy row
#    (one "raised", one "cleared") — something the generic adapter
#    cannot express.
@register_adapter("MyDetector", "weird_method")
def expand_pairs(legacy_df, *, rule, detector, objects, qualifiers):
    rows: list[dict] = []
    relations: list[dict] = []
    for i, row in legacy_df.iterrows():
        for kind in ("raised", "cleared"):
            eid = f"e-{detector}-{i}-{kind}"
            rows.append({
                S.OCEL_EID: eid,
                S.OCEL_ACTIVITY: f"production.custom.{kind}",
                S.OCEL_TIMESTAMP: pd.Timestamp(row[f"{kind}_at"], tz="UTC"),
                S.TS_DETECTOR: detector,
                S.TS_PACK: rule.pack,
            })
            relations.append({
                S.OCEL_EID: eid,
                S.OCEL_OID: row["asset_id"],
                S.OCEL_TYPE: "asset",
                S.OCEL_QUALIFIER: "produced_on",
            })
    events = pd.concat([S.empty_events(), pd.DataFrame(rows)], ignore_index=True)
    rels = pd.concat([S.empty_relations(), pd.DataFrame(relations)], ignore_index=True)
    objs = pd.DataFrame({
        S.OCEL_OID: legacy_df["asset_id"].astype("string").unique(),
        S.OCEL_TYPE: "asset",
    })
    return EventLog(events=events, objects=objs, relations=rels)
```

Once registered, `to_event_log(df, detector="MyDetector.weird_method")`
calls the override automatically.
