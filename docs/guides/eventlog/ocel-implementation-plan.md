# OCEL 2.x implementation plan

Follow-up to the [gap analysis](ocel-gap-analysis.md). Gap ids (M1–M14,
F1–F8) refer to that page.

## Guiding decisions

1. **The canonical `EventLog` stays as it is.** ts-shape keys objects by
   `(oid, type)` (see `concat`, `_restrict_to_events`, `object_changes`), and
   E2O rows carry `ocel:type`. Changing that would break the detectors, the
   genealogy code and the analytics. Instead, a **spec export layer** turns a
   canonical log into a spec-conformant log, and readers turn a spec log back
   into a canonical one.
2. **No new required dependencies.** SQLite, JSON, XML, CSV and ZIP use the
   stdlib. Parquet uses the existing optional `pyarrow` (`parquet` extra).
   pm4py is used only in an opt-in interop test.
3. **One normalized intermediate representation (IR)** feeds every writer and
   comes out of every reader, so the spec rules are implemented once:
   `OCELSpecLog` = per-type event tables, per-type object tables (base values),
   per-type change tables, `e2o`, `o2o`, plus a typed attribute catalog.
4. **Deterministic output.** The same log gives byte-identical files (sorted
   rows, sorted types, fixed float/time formatting). This matches the
   deterministic `ocel:eid` philosophy and the §9 "exporter order".

## Target module layout

```
src/ts_shape/eventlog/ocelspec/
  __init__.py        # public: to_ocel_spec, from_ocel_spec, write_ocel, read_ocel,
                     #         validate_ocel, ocel_types, OCELSpecLog, AttributeType
  types.py           # AttributeType enum, dtype <-> OCEL type mapping, canonical lexical forms
  catalog.py         # ocel_types(): event/object type catalog (data + LabelRule registry)
  model.py           # OCELSpecLog dataclass (the IR)
  normalize.py       # to_ocel_spec / from_ocel_spec (canonical <-> IR)
  validate.py        # validate_ocel(): spec §6.7 / §9 / §10 rules, error report
  io_bundle.py       # .ocel.zip / directory, csv + parquet (§10)
  io_sqlite.py       # .sqlite (§6)
  io_json.py         # .json / .jsonocel (+ .gz) (§8, §11)
  io_xml.py          # .xml / .xmlocel (+ .gz) (§7, §11)
  io_csv.py          # .ocel.csv single file (§9)
  dispatch.py        # write_ocel / read_ocel by extension
tests/eventlog/ocelspec/
  fixtures/p2p.py    # spec running example (§5) built as an OCELSpecLog
  fixtures/line.py   # manufacturing fixture (line → station → batch → serial, coil, tool)
  test_*.py
```

`to_event_log_ocel` (DataFrames, pm4py layout) stays unchanged.

---

## Phase 0: spec conformance core (gaps M2–M8, M10, M12, M13)

### PR 1: attribute types and type catalog

- `types.py`
  - `AttributeType = Literal["string", "time", "integer", "float", "boolean"]`.
  - `infer_type(series) -> AttributeType`: map pandas dtypes
    (`string`/`object`→string, `Int*`/`int*`→integer, `float*`→float,
    `bool`/`boolean`→boolean, `datetime64[*, tz]`→time). For object columns,
    apply the §9 canonical-lexical rule (int → float → bool → time → string)
    and never change a value: `"007"` stays a string.
  - `format_value(v, type)` / `parse_value(text, type)` for the CSV and XML
    lexical forms. Times are ISO 8601 UTC with `+00:00`; floats use the
    shortest round-trip repr.
- `catalog.py`: `ocel_types(log) -> TypeCatalog`
  - Event types: per `ocel:activity`, the columns that are non-null for at
    least one event of that activity, each with a single type. Excluded:
    `ocel:eid`, `ocel:activity`, `ocel:timestamp`. `ts_shape:*` columns are
    included by default (`include_internal=True`).
  - Object types: per `ocel:type`, the `ocel:field` values in
    `object_changes`. The type is unified over all values of that
    `(type, field)`. Mixed types fall back to string with a
    `DataQualityWarning`.
  - Injective, file- and SQL-safe name map (`map_type_name`): identity when
    already safe, otherwise sanitized with a collision suffix (M13).
- Tests: dtype matrix; canonical-lexical edge cases from §9 (`007`, `+5`,
  `1.50`, out-of-range int64); catalog of the P2P fixture equals spec Tables 1/2.

### PR 2: `to_ocel_spec` / `from_ocel_spec` (canonical ↔ IR)

`to_ocel_spec(log, *, oid_policy="error", default_qualifiers=None,
interval="attribute", static_fields="auto", include_internal=True) -> OCELSpecLog`

| Option | Behaviour | Gap |
|---|---|---|
| `oid_policy="error"` | Raise `OCELConformanceError` listing the oids shared by two types | M2 |
| `oid_policy="prefix"` | Rewrite every oid as `{type}:{oid}` unless it already has that prefix, consistently across E2O, O2O and changes. The reverse map is stored in the IR (`oid_map`) | M2, M14 |
| `oid_policy="prefix_conflicts"` | Prefix only the colliding oids | M2 |
| `default_qualifiers` | `{object_type: qualifier}` for NULL E2O qualifiers. Built-in default: `{"asset": "emitted_by", "station": "at_station", "signal": "observed_on"}`, anything else `""`. Never NULL | M3 |
| E2O / O2O dedupe | Always: E2O on `(eid, oid, qualifier)`, O2O on `(src, tgt, qualifier)`. Drops `ocel:type` from E2O | M4, M12 |
| `interval="attribute"` | Atomic event at `ocel:timestamp` (end). `ts_shape:start_timestamp` is kept as a `time` attribute | M10 |
| `interval="start_complete"` | Each interval event becomes `<activity>.start` at the start and `<activity>.complete` at the end. Ids are `<eid>:start` / `<eid>:complete`, linked by the attribute `ts_shape:instance` = original eid, with the same E2O on both | M10 |
| `static_fields="auto"` | A field with exactly one value per object (or listed in `static_fields=[...]`) becomes a base value in the object table. Everything else stays a timed change. Change rows at the epoch move to base values. `lifecycle` is always timed | M7 |
| One value per `(oid, field, t)` | Conflicts raise (strict) or keep the last row with a warning | M8 |

`from_ocel_spec(spec, *, restore_oids=True) -> EventLog` is the inverse. It
puts object types back on E2O from the object table, turns base values into
epoch-timestamped change rows, and undoes the `oid_map` prefixing. If
`ts_shape:detector` / `ts_shape:pack` are missing (foreign logs), it fills
them with `"external"` and with the activity's first dotted segment or
`"external"`.

Tests:
- canonical → IR → canonical round-trip on the object-detection demo log
  (equality modulo the documented normalizations);
- oid collision under each policy;
- NULL qualifier defaulting;
- `start_complete` split: two event types, shared E2O;
- static/dynamic split: no epoch rows in the change tables.

### PR 3: `validate_ocel(spec_or_log, *, raise_=True) -> list[Violation]`

Implements §6.7 and the §9/§10 validity rules on the IR: unique event ids;
globally unique object ids; E2O/O2O foreign keys; PK uniqueness; non-null
fixed columns; one type per attribute scope; attributes only of their own
type; one value per `(oid, attr, t)`; no epoch rows in change tables; `o2o`
not used as an activity name (CSV). It returns structured `Violation(rule,
table, row, message)`, so CI and users get every violation at once. The
canonical `schema.validate` stays as it is, and `EventLog.validate_ocel()`
becomes a convenience method.

---

## Phase 1: writers (F6, F2, F3, F7)

### PR 4: bundled CSV/Parquet `.ocel.zip` (§10). Priority writer

- `write_ocel_bundle(spec, path, *, storage="parquet"|"csv", compress=True)`.
  The path is a `.ocel.zip` file, or a directory for the uncompressed layout.
- File names use UTF-8 percent-encoding (keep `A-Za-z0-9._-`, uppercase `%HH`).
  `ocel-meta.json` has `ocelVersion: "2.0"`, `bundleFormatVersion: "1.0"`,
  the type → file maps, attribute types and relation files.
- Parquet: fixed columns are required string / `TIMESTAMP(UTC, MICROS)`,
  attribute columns are nullable, and timestamps are never written as strings.
- Every object type gets a change table, even an empty one (header only or
  zero rows).
- `pyarrow` is imported lazily. Without it, `storage="parquet"` raises a clear
  error that names the `ts-shape[parquet]` extra.

### PR 5: SQLite (§6)

- Tables `event`, `object`, `event_map_type`, `object_map_type`,
  `event_<map>`, `object_<map>` (base rows at `1970-01-01 00:00:00 UTC`, then
  change rows with `ocel_changed_field`), `event_object`, `object_object`.
  PKs and FKs as in §6.7, `PRAGMA foreign_keys=ON`, and inserts in one
  transaction with `executemany`.
- Attribute column names are double-quoted (`"ts_shape:severity"` is valid).

### PR 6: JSON (+ gzip) (§8, §11)

- `eventTypes` / `objectTypes` with typed attributes, `objects` (timed
  attribute values plus O2O `relationships`) and `events` (attributes plus E2O
  `relationships`). Streams the output with `json.JSONEncoder.iterencode` to
  keep memory flat. `.json.gz` / `.jsonocel.gz` go through `gzip.open`.

### PR 7: `write_ocel(log_or_spec, path, **opts)` dispatch

- Picks the writer by extension: `.ocel.zip`, `.sqlite` / `.db`, `.json`,
  `.jsonocel`, `.json.gz`, `.xml`, `.xmlocel`, `.xml.gz` and `.ocel.csv`.
- Passing an `EventLog` runs `to_ocel_spec(**opts)` and `validate_ocel` first.
- Also exposed as `EventLog.to_ocel(path, **opts)` and on `Pipeline`
  (`pipeline.to_ocel(path)`). Docs and changelog entries are part of this PR.

---

## Phase 2: readers + conformance (F1–F8 read side)

### PR 8: readers for bundle, SQLite and JSON (+ gzip)

- `read_ocel(path) -> EventLog` (canonical, via `from_ocel_spec`) and
  `read_ocel_spec(path) -> OCELSpecLog`.
- Bundle: trust `ocel-meta.json` only, reject absolute or `..` paths, ignore
  undeclared entries, and keep whitespace exactly as stored.
- Gzip and ZIP limits: a `max_uncompressed_bytes` cap (default 2 GiB) with
  streaming decompression (§11 robustness).
- **IT/OT merge use case:** `concat(read_ocel("mes_export.json"), machine_log)`
  joins on a shared oid (e.g. `work_order:WO-123`). It is documented with
  `oid_policy="prefix"` as the convention for ts-shape-produced logs.

### PR 9: golden conformance suite

- `fixtures/p2p.py` builds the spec running example (§5, Tables 4–21).
- For each format: write → read → `OCELSpecLog` equality. Compare the bundled
  CSV against the spec's Tables 24–28, and SQLite against Tables 4–21.
- Manufacturing fixture round-trip through every format, including
  `start_complete` and the genealogy queries (`where_used` gives the same
  result before and after the round-trip).
- Optional pm4py interop job (`pytest -m pm4py`, separate CI matrix entry):
  `pm4py.read_ocel2_json/sqlite` on our output must load with matching counts.

### PR 10: single-file CSV `.ocel.csv` (§9) and XML (§7)

- CSV: the four row types, reference grammar `id#qualifier{json}` separated
  by `/`, backslash escaping of `/ # { \`, the exporter order of §9, and type
  inference as in PR 1.
- XML: written with `xml.etree.ElementTree.iterparse` / `xmlfile`-style
  incremental output. Reading uses `defusedxml` if it is installed and
  otherwise disables external entities.

---

## Phase 3: manufacturing semantics (M14, spec §12)

### PR 11: signal vs asset, equipment hierarchy

- Adapter default: a detector's `source_uuid` binding becomes object type
  `signal`, plus O2O `signal observed_on asset` when metadata provides the
  asset. This is behind a flag first (`signal_objects=True`). The default flips
  in the next minor release with a deprecation warning, because it changes
  `produces_objects` for many rules.
- `equipment_hierarchy_from_metadata(meta)` → O2O
  `station part_of line part_of area part_of site` (ISA-95 levels).

### PR 12: controlled vocabulary

- `ts_shape.eventlog.vocabulary`: object types (ISA-95 aligned: equipment,
  material lot/sublot, work order, personnel, tool, recipe) and the E2O/O2O
  qualifiers already in use (`identified_by`, `during_batch`, `made_of`,
  `part_of`, `made_from`, `made_with`, `made_by`, `co_occurs`, plus the new
  defaults), each with a definition. It is exported into the type catalog
  and the docs.
- A registry conformance test makes every `LabelRule.produces_objects` and
  qualifier part of the vocabulary.

### PR 13: partitioned / incremental export

- `write_ocel(..., partition_by="day"|"shift")` writes one bundle per
  partition. Objects that span partitions are repeated, with base values taken
  from the state at the partition start (reusing `filter_by_time`).
  A `stream.py` hook appends partitions as windows close.

---

## Sequencing, effort, acceptance

| PR | Depends on | Effort | Done when |
|---|---|---|---|
| 1 types/catalog | — | S | P2P catalog equals spec Tables 1/2 |
| 2 normalize | 1 | M | demo log round-trips; all policy tests green |
| 3 validate_ocel | 2 | S | every §6.7/§9/§10 rule has a failing and a passing test |
| 4 bundle | 2, 3 | M | P2P bundle CSV equals spec Tables 24–28; Parquet types per §10 |
| 5 sqlite | 2, 3 | M | schema equals Fig. 3/4; FK check passes |
| 6 json(+gz) | 2, 3 | S | P2P JSON equals the §8 example semantically |
| 7 dispatch/API | 4–6 | S | `log.to_ocel("x.ocel.zip")` documented and in changelog |
| 8 readers | 4–7 | M | write → read identity for 3 formats |
| 9 conformance | 8 | S | golden suite + optional pm4py job green |
| 10 csv/xml | 8 | M | P2P CSV equals spec Listing 1 |
| 11 signal/asset | 2 | M | flag on: outlier events link signal + asset, no regressions |
| 12 vocabulary | 11 | S | registry conformance test green |
| 13 partitions | 7 | M | per-day bundles concat back to the full log |

Every PR must pass: `ruff check src tests examples scripts`, `mypy` and
`pytest`. New public names go into `__all__` and pass
`tests/test_named_public_functions.py`. Each PR includes a docs page section
and a `CHANGES.txt` entry.

## Open decisions (defaults proposed)

1. **Default `oid_policy`**: `"error"` (proposed) or `"prefix"`. `"prefix"`
   is friendlier, but it changes ids that users may join on.
2. **Default interval export**: `"attribute"` (proposed, lossless and simple)
   or `"start_complete"` (better for OCPM performance analysis).
3. **Default asset qualifier**: `emitted_by` (proposed).
4. **`ts_shape:*` internal attributes in exports**: included (proposed), with
   `include_internal=False` to strip them.
5. **Signal-as-object default flip (PR 11)**: whether it waits for the next
   minor release (proposed) or stays opt-in permanently.
