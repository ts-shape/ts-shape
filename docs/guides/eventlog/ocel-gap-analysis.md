# OCEL 2.x integration: process overview and gap analysis

Reference: *OCEL (Object-Centric Event Log) 2.1 Specification*, v2.1.0pre5
(RWTH Aachen, 2026-10-01). It keeps the OCEL 2.0 metamodel and adds the
single-file CSV format (§9), the bundled CSV/Parquet format (§10) and GZIP (§11).

This page compares that spec with what `ts_shape.eventlog` does today. It is
written for manufacturing: signal-derived machine events joined to physical
and logistic objects (asset, station, batch, lot, serial, coil, tool,
recipe, work order).

---

## 1. Process overview: from shop-floor signals to an OCEL 2.x log

```
 ┌──────────────┐  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
 │1 Source data │→ │2 Event extraction│→ │3 Object identif. │→ │4 Relations       │
 │ historian,   │  │ 333 detector     │  │ ObjectSpec on id │  │ E2O: attach_/link│
 │ PLC/UNS, MES │  │ rules → events   │  │ signals; asset   │  │ O2O: hierarchy,  │
 │ metadata     │  │ (eid, activity)  │  │ from source_uuid │  │ uses, co_occurs  │
 └──────────────┘  └──────────────────┘  └──────────────────┘  └──────────────────┘
          ┌──────────────────────────────────────────────────────────────┘
          ▼
 ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
 │5 Object dynamics │→ │6 Spec conformance│→ │7 Serialization   │→ │8 Consumption     │
 │ object_changes:  │  │ ids, PKs, types, │  │ SQLite/JSON/XML/ │  │ pm4py, OCPM tools│
 │ lifecycle, attrs │  │ qualifiers, time │  │ CSV/.ocel.zip/gz │  │ lakehouse, import│
 └──────────────────┘  └──────────────────┘  └──────────────────┘  └──────────────────┘
```

| Step | What OCEL 2.x requires | ts-shape today |
|---|---|---|
| 1 Source | System-agnostic. Events come from ERP, MES and other IT systems. | Loaders for timeseries and metadata (parquet, Timescale, Azure, S3, Databricks, API) |
| 2 Events | Atomic, typed, timestamped, unique id, typed attributes | `to_event_log` + `taxonomy.REGISTRY` (333 `LabelRule`s). Deterministic `ocel:eid`, activity `pack.family.method` |
| 3 Objects | Typed, **globally unique id**, attributes with base (t=0) and timed values | `detect_objects` / `ObjectSpec` / `object_specs_from_metadata`. 16 standard types + `register_object_type` |
| 4 Relations | Qualified E2O `(e, q, o)` and O2O `(o, q, o')`, each a set (PK = all columns) | `attach_objects`, `link_events_to_objects`, `derive_o2o`, `relate` |
| 5 Dynamics | `oaval(o, oa, t)` with value valid from `t`. Initial values at 1970-01-01 | `object_changes` (long form): `lifecycle` active/released plus attributes at presence start |
| 6 Conformance | Constraints §6.7 (PKs, FKs, type maps), §9/§10 validity rules | `schema.validate` checks columns, unique eid, E2O/O2O/changes FKs |
| 7 Serialization | SQLite (§6), XML (§7), JSON (§8), CSV (§9), bundled CSV/Parquet (§10), gzip (§11) | **None.** `to_event_log_ocel` returns DataFrames; `pm4py=True` reshapes them for pm4py's `OCEL` |
| 8 Consumption | Any conformant tool; round-trip between formats | Hand-off to pm4py only. **No import** |

---

## 2. Gap analysis

Legend: ✅ covered · ⚠️ partial / non-conformant edge cases · ❌ missing

### 2.1 Metamodel (spec §3–4, Definition 2)

| # | Spec element | Status | Finding | Manufacturing impact |
|---|---|---|---|---|
| M1 | Events `E`, `evtype`, `time` | ✅ | `ocel:eid` unique (validated), UTC tz-aware timestamps | — |
| M2 | **Object id uniqueness.** `object.ocel_id` is the PK. §9: an id under two object types is invalid | ❌ | ts-shape keys objects by `(oid, type)`. `validate` accepts `X/batch` and `X/lot` together (checked). Asset oids are raw `source_uuid` values (`"line-1"`, `"temp"`), while detected objects use `{type}:{value}` templates, so ids can collide | A batch `4711` and a lot `4711` merge, or are rejected, in every OCEL DB or tool |
| M3 | E2O qualifier ∈ `U_qual` (part of the PK) | ⚠️ | Adapter-generated asset bindings have `ocel:qualifier = <NA>`. SQLite rejects a NULL PK column; CSV §9 expects `""` | Every detector event's link to its asset has no role |
| M4 | E2O / O2O are **sets** (PK over all columns) | ⚠️ | `(eid, oid, qualifier)` and `(oid, oid_2, qualifier)` uniqueness is not validated. `concat` of overlapping logs can create duplicates | Duplicate rows inflate object-centric DFG frequencies |
| M5 | Event attributes typed per event type (`eatype`, disjoint sets) | ⚠️ | One wide, sparse `events` frame. Pack columns (`quality:uuid`, `quality:is_delta`, …) are on every row of the pack. No per-activity attribute set is declared | Needed for `eventTypes` (JSON/XML), `event_<type>` tables (SQLite) and `ocel-meta.json` |
| M6 | Attribute primitive types: string, time, integer, float, boolean | ❌ | No mapping from pandas dtypes to OCEL types. `ocel:value` in `object_changes` is `object` dtype with mixed types | Typed export of setpoints, temperatures, counts and flags |
| M7 | Object base values (t = 0 / 1970-01-01) vs timed changes | ⚠️ | `objects` holds only `oid` and `type`. **All** attributes are timed rows in `object_changes` at presence start. §10 forbids epoch rows in change tables and puts initial values in the object table | Master data (article no., customer PN, material grade) should be static. Recipe or tool wear stay timed |
| M8 | Object dynamics `oaval`, one value per `(o, oa, t)` | ✅/⚠️ | Enforced by `drop_duplicates` in `_object_changes`, but not by `validate` for user-supplied tables | — |
| M9 | O2O qualified relations | ✅ | `part_of`, `made_from`, `made_with`, `made_by`, `co_occurs` (opt-in) | Genealogy (`where_used` / `where_from`) already builds on it |
| M10 | **Events are atomic** | ⚠️ | Interval events (`ts_shape:start_timestamp`, `duration_s`) use the end as `ocel:timestamp` and keep the start as an attribute. XES export has `two_row`, but OCEL export has no equivalent | Downtime, setup and steady-state *spans* are central. Tools see only the end point |
| M11 | Presence as an object attribute (`lifecycle` active/released) | ✅ (design choice) | This is valid OCEL. `presence_events` can also emit them as events | Keep both options and document which one is used for OCPM |
| M12 | E2O relation carries `ocel:type` | ✅ (extra) | Redundant with `objects`. Must be dropped (SQLite/bundled) or ignored on export | — |
| M13 | Event/object type → table-name map (`event_map_type`, `object_map_type`, injective) | ❌ | Needed for SQLite. Activities such as `production.flow.{x}_summary` and types such as `work_order` need a safe mapping | — |
| M14 | Semantic: object = real entity | ⚠️ | `OutlierDetectionEvents` links to oid `temp` (a **signal** uuid) typed `asset`. `signal` and `sensor` exist as standard types but are not used for this | Machine-centric OCPM would treat each sensor as its own machine |

### 2.2 Exchange formats (spec §6–11)

| # | Format | Write | Read | Notes |
|---|---|---|---|---|
| F1 | In-memory relational tables | ✅ | ✅ (`to_event_log(objects=…)`) | `OCEL2Tables`, pm4py layout |
| F2 | SQLite (§6): `event`, `object`, `event_<t>`, `object_<t>`, `event_object`, `object_object`, map tables | ❌ | ❌ | stdlib `sqlite3` is enough. `object_<t>` needs `ocel_changed_field` plus base rows at epoch |
| F3 | JSON (§8) `.json` / `.jsonocel` | ❌ | ❌ | Only via pm4py, which is not a dependency. Needs `eventTypes` / `objectTypes` with attribute types (M5, M6) |
| F4 | XML (§7) `.xml` / `.xmlocel` | ❌ | ❌ | Same as F3, with lower priority |
| F5 | Single-file CSV (§9) `.ocel.csv` | ❌ | ❌ | New in 2.1. Human-readable, good for process engineers in Excel. Needs reference escaping (`\/ \# \{`), JSON attributes, deterministic row order and canonical number typing |
| F6 | **Bundled CSV/Parquet (§10) `.ocel.zip`** + `ocel-meta.json` | ❌ | ❌ | New in 2.1. **Best fit for ts-shape**: dataframe-native, `pyarrow` is already an optional dependency, maps straight onto Databricks/lakehouse. Needs percent-encoded file names, per-type tables, `TIMESTAMP(UTC, MICROS)`, non-null fixed columns, and no epoch rows in change tables |
| F7 | GZIP (§11) `.json.gz` / `.xml.gz` | ❌ | ❌ | Trivial once F3/F4 exist. Add decompression-bomb limits on read |
| F8 | Conformance fixture (spec running example, P2P) | ❌ | — | Needed as a round-trip golden test for every format |

### 2.3 Validation (spec §6.7, §9, §10 validity rules)

| Rule | `schema.validate` |
|---|---|
| Unique event id | ✅ |
| Unique object id (global) | ❌ (M2) |
| E2O FK → event, object | ✅ |
| O2O FK → object | ✅ |
| E2O / O2O PK uniqueness | ❌ (M4) |
| Non-null fixed columns (`ocel_qualifier`, …) | ❌ (M3) |
| One type per `(object type, attribute)` | ❌ (M6) |
| One value per `(oid, attr, t)` | ⚠️ (M8) |
| No epoch rows in change tables (bundled) | ❌ (M7) |
| Attribute set per event type disjoint | n/a. Holds by construction once per-type columns are derived |

---

## 3. Recommended roadmap

### P0: spec-conformance layer (no new dependencies)

1. **`validate(log, level="ocel")`**: a strict mode that adds the ❌ rules in §2.3.
   The current `validate` stays as the "canonical ts-shape" level.
2. **`to_ocel_spec(log, *, interval="attribute"|"start_complete", oid_policy=...)`**
   returns a normalized `EventLog` that:
   - makes oids globally unique (default `{type}:{value}` everywhere, including
     assets; error or warn when two types share an id) (M2, M14);
   - sets a default qualifier for adapter bindings (e.g. `emitted_by` for asset,
     `observed_on` for signal) and turns any remaining NULL into `""` (M3);
   - removes duplicate E2O and O2O rows (M4);
   - optionally splits interval events into `<activity>.start` /
     `<activity>.complete` with a shared attribute (M10);
   - moves time-invariant attributes to object base values (M7).
3. **Type catalog**: `ocel_types(log) → (event_types, object_types)` with
   attribute name → OCEL primitive type, from dtypes and `STANDARD_ATTR_TYPES`.
   The `LabelRule` registry can declare event-type attributes statically, so the
   catalog does not depend on the data (M5, M6, M13).

### P1: writers

4. `write_ocel(log, path)`, dispatching on extension:
   - `.ocel.zip` / directory → bundled **Parquet** (default) or CSV (F6)
   - `.sqlite` → relational (F2)
   - `.json` / `.json.gz` → JSON (F3, F7)
   Use stdlib plus optional `pyarrow`. pm4py stays an *interop test* extra only.

### P2: readers + conformance

5. `read_ocel(path) → EventLog` for the same formats (and `.ocel.csv`, F5).
   Use it to merge **external MES/ERP OCEL extracts** (work orders, material
   lots, customer orders) with signal-derived machine events on a shared oid.
   This join between IT and OT is the main manufacturing use case.
6. Golden round-trip tests on the spec running example and a manufacturing
   fixture (line → station → batch → serial, coil `made_from`, tool `made_with`).
   Add a pm4py import smoke test in an optional CI job.
7. Single-file CSV writer (F5) and XML (F4).

### P3: manufacturing semantics (spec §12 asks for domain-standard types)

8. Publish the ts-shape object types and qualifiers as a controlled vocabulary,
   aligned to ISA-95 (enterprise/site/area/line/cell, material lot/sublot,
   work order, personnel, equipment). Derive the equipment-hierarchy O2O
   (`station part_of line`) from metadata.
9. Keep the signal ≠ asset distinction (M14): `signal observed_on asset` as O2O,
   with events linked to the asset.
10. Incremental / partitioned export (per shift or day) for streaming lines.
    This goes beyond the spec, but follows the bundled layout per partition.

---

## 4. Evidence (code locations)

- Canonical tables and validation: `src/ts_shape/eventlog/schema.py` (`validate`, `empty_*`)
- `EventLog` model: `src/ts_shape/eventlog/model.py`
- Exporter (DataFrames only, pm4py layout): `src/ts_shape/eventlog/ocel.py`
- Object detection, `object_changes` (lifecycle + attributes at presence start): `src/ts_shape/eventlog/objects.py` (`_object_changes`)
- Adapter E2O bindings (qualifier per object type, else NULL): `src/ts_shape/eventlog/adapters.py` (`_to_relations`)
- XES interval handling (`two_row`): `src/ts_shape/eventlog/flat.py`
- Checked behaviour: `validate` accepts the same oid under two types; the
  outlier detector links events to `oid="temp"` typed `asset` with a `<NA>` qualifier.
