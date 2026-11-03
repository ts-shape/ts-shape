# OCEL 2.0 implementation plan: canonical model

Follow-up to the [gap analysis](ocel-gap-analysis.md). Gap ids (M1–M14)
refer to that page.

## Scope

- **The canonical `EventLog` itself is made OCEL 2.0-conformant.** It is the
  metamodel of spec §3 and Definition 2, enforced by `validate`. There is no
  conversion layer and no second representation.
- **No file formats.** SQLite, JSON, XML, CSV and `.ocel.zip` are out of
  scope. `to_event_log_ocel` stays the DataFrame hand-off.
- **Functional correctness comes first.** Every row in `events` must be a real,
  atomic occurrence on the shop floor. Every object must be a real entity with
  one identity. Every relation must say what role the object plays.

## Target invariants (what `validate` enforces when done)

| # | Invariant | Spec | Today |
|---|---|---|---|
| I1 | `ocel:eid` unique | Def. 2, §6.7 | ✅ |
| I2 | `ocel:oid` **globally unique**; `ocel:type` is a function of the oid | §6.7 PK, §9 | ❌ keyed by `(oid, type)` |
| I3 | `relations.ocel:type` and `object_changes.ocel:type` equal the type in `objects` (kept as denormalized columns for filtering) | Def. 2 `objtype` | not checked |
| I4 | E2O is a set: `(eid, oid, qualifier)` unique; qualifier non-null | Def. 2 E2O ⊆ E×U_qual×O | ❌ NULL qualifiers, no PK check |
| I5 | O2O is a set: `(oid, oid_2, qualifier)` unique; qualifier non-null | Def. 2 O2O | ❌ no PK check |
| I6 | Events are atomic: one timestamp, no duration semantics | §2 "Events are atomic" | ❌ 76 interval rules |
| I7 | Events are occurrences: no fabricated timestamps, no window aggregates | §2, Def. 2 `time` | ❌ 62 static rules at 1970-01-01, 145 summary rules |
| I8 | `oaval` is a partial function: one value per `(oid, field, t)`; static values at t = 0 (epoch) | Def. 2, §4 | ⚠️ only in `detect_objects`; static master data timed at presence start |
| I9 | One primitive type (string, time, integer, float, boolean) per attribute scope: `(activity, attr)` and `(object type, field)` | §7/§8 attribute types | ❌ |
| I10 | Every E2O, O2O and change row references an existing event/object | §6.7 FKs | ✅ |

---

## Work packages

The work packages are ordered by dependency. Each one is a PR that tightens
`validate` by the invariants it delivers, so the whole test suite (≈270
eventlog tests plus the registry conformance guard) proves the invariant
across all 333 detector rules.

### WP1: global object identity (I2, I3; gap M2)

**Functional:** an object id names one thing on the shop floor. Batch `4711`
and lot `4711` are two things, so they need two ids.

**Technical:**
1. **Type-namespaced ids by default:** `"{type}:{value}"`.
   - `adapters._to_relations`: every binding, including the default
     `source_uuid` binding, renders through the template unless the value
     already starts with `"{type}:"`.
   - `ObjectSpec.id_template` default changes from `"{value}"` to
     `"{type}:{value}"`. `object_specs_from_metadata` and `signals.identify`
     follow.
   - `to_event_log(objects=…)` with user-provided literals gets the same
     treatment. Callers who already have unique ids can opt out with
     `id_template="{value}"`.
2. **One key:** `oid` replaces `(oid, type)` as the object key in
   `model._objects_in`, `model._restrict_to_events`, `query._key_index` /
   `_pairs_in`, `analysis._log_presences`, `concat` (dedupe objects by oid;
   if one oid arrives with two types, raise `ValueError` naming both) and
   `schema._unknown_pairs`, which becomes an oid-only FK check.
3. `validate`: duplicate oid → error (I2); relation or change type ≠ object
   type → error (I3).
4. Docstrings: remove "an object id used by two types is two objects"
   (`concat`) and the `(oid, type)` wording in `model`, `query` and `objects`.

**Migration:** this is a breaking change to oid values. Add a `CHANGES.txt`
entry and a `docs/migration.md` section with a one-line opt-out per API.

**Tests:** collision raises in `validate` and in `concat`; every registry
rule's sample output satisfies I2/I3 (extend
`tests/test_detector_registry_conformance.py`); genealogy queries return the
same lineage on the demo fixtures with the new ids.

### WP2: qualified relations as sets (I4, I5; gaps M3, M4)

**Functional:** a qualifier states the role an object plays in an event.
Without one, OCPM cannot tell the machine that *caused* a stop from the
station where a part *waited*.

**Technical:**
1. A new module `eventlog/vocabulary.py` holds the controlled qualifier
   vocabulary, with a definition for each entry. It starts from the
   qualifiers already in use (`identified_by`, `during_batch`, `made_of`,
   `part_of`, `made_from`, `made_with`, `made_by`, `co_occurs`) and adds a
   default E2O role per standard object type:

   | object type | default E2O qualifier |
   |---|---|
   | asset | `on_asset` |
   | station | `at_station` |
   | signal | `observed_signal` |
   | sensor | `observed_by` |
   | shift | `in_shift` |
   | operator | `by_operator` |
   | work_order | `for_order` |
   | batch / lot | `during_batch` / `during_lot` |
   | serial / part | `identified_by` |
   | material | `made_of` |
   | tool | `with_tool` |
   | recipe | `with_recipe` |
   | cycle | `in_cycle` |

   `register_object_type(name, default_qualifier=...)` extends it.
2. `LabelRule` gains `qualifiers: Mapping[str, str]` for rule-specific roles
   (e.g. a bottleneck summary links `station` as `constraint`). The
   `qualifiers=` argument of `to_event_log`, `attach_objects` and
   `link_events_to_objects` overrides it. Resolution order: argument → rule →
   vocabulary default. The result is never NULL.
3. Every producer removes duplicate rows on the full PK
   (`_to_relations`, `_link`, `_infer_o2o`, `concat` already does this).
4. `validate`: NULL qualifier → error; duplicate PK → error.

**Tests:** every registry rule emits only non-null qualifiers from the
vocabulary (conformance guard); overriding works at each level; `concat` of a
log with itself is idempotent on all tables.

### WP3: object attributes over time (I8, I9; gaps M6, M7, M8)

**Functional:** master data (article number, customer part number, material
grade) is static. Recipe, tool wear, lifecycle and location change over
time. Def. 2 models both with `oaval(o, oa, t)`, using t = 0 for static
values.

**Technical:**
1. `object_changes` stays the single source of truth (long form). Static
   values get `ocel:timestamp = 1970-01-01T00:00:00Z`, following Def. 2 and
   §6.4.
2. `ObjectSpec.static_attributes: Collection[str]` lists the attributes that
   are recorded at t = 0, once per object. A static attribute that changes
   value is an error (an object has one article number). The metadata loader
   and `object_specs_from_metadata` mark master-data columns as static.
   `lifecycle` and the remaining attributes stay timed.
3. A new module `eventlog/types.py`:
   - `AttributeType` holds the five OCEL primitive types;
   - `attribute_type(series)` maps a pandas dtype to an OCEL type, with the
     canonical-lexical rule for object columns (`"007"` stays a string);
   - `object_changes.ocel:value` is typed per `(type, field)` on write. A
     scope with mixed types is coerced to string with a
     `DataQualityWarning`.
4. `describe(log)` (in `introspect.py`) reports the metamodel: event types
   with their attribute names and types (`eatype`), and object types with
   their fields, types and whether each is static or timed (`oatype`).
5. `validate`: duplicate `(oid, field, t)` → error (I8); more than one type
   per scope → error (I9).

**Tests:** static attributes appear once at the epoch; a conflicting static
value raises; the type matrix (int/float/bool/time/string, nullable dtypes,
the `"007"`/`"+5"`/`"1.50"` edge cases); `describe` on the spec's running
example equals spec Tables 1/2.

### WP4: what is an event (I6, I7; gap M10, plus the rule audit)

This is the largest functional change. Of the 333 registry rules:

| shape | rules | today | OCEL-correct modelling |
|---|---|---|---|
| point | 50 | one event at `systime` | ✅ unchanged |
| interval | 76 | one event at the end; start and duration as attributes | **two atomic events**: `<activity>.start` at the start and `<activity>.end` at the end, sharing `ts_shape:instance`, with the same E2O set on both. Duration becomes derived, not stored. A machine stopping and a machine restarting are two occurrences, and OCPM measures time between events |
| summary | 145 | one event at the window end carrying a KPI (Cpk, hit rate, constraint stats) | **object attribute changes**: field = KPI name, value = KPI, t = window end, written to `object_changes` of the measured object (asset, signal, station). A KPI describes the object's state; it does not happen. A summary that is a verdict (e.g. "bottleneck at station X") stays an event (rule flag `emits_event=True`) |
| static | 62 | one event at the fabricated 1970-01-01 timestamp | **no events.** Per-object results become static object attributes (t = 0). Pairwise results (correlation, root cause, lag between two signals) become **O2O** relations (`correlates_with`, `root_cause_of`, `leads`) with rank and score as attributes of the source object. Results with no object are not OCEL data: `to_event_log` returns them unchanged with a `DataQualityWarning`, and they stay available as the detector DataFrame |

**Technical:**
1. `taxonomy.LabelRule.shape` keeps its four values. `adapters` gains one
   mapping function per target (event pair, attribute change, O2O) instead
   of one adapter per shape. `LabelRule.emits_event` and
   `LabelRule.o2o_qualifier` drive the choice.
2. Interval consumers move to the event pair:
   - `link_events_to_objects` builds temporal overlap from the
     `.start`/`.end` pair (joined on `ts_shape:instance`);
   - `flat.to_event_log_xes` maps the pair directly to the
     `lifecycle:transition` values `start` / `complete` (`two_row` becomes
     the only mode);
   - `analysis` and `lambda_rules.backtest` use the pair for durations.
3. `ts_shape:start_timestamp` / `ts_shape:duration_s` are removed from
   `EVENT_OPTIONAL_COLUMNS`. `STATIC_TIMESTAMP` and the `timestamp=`
   argument of `to_event_log` are deprecated, then removed.
4. Each of the 207 summary and static rules gets an explicit decision in the
   registry (`emits_event`, KPI field names, `o2o_qualifier`). The audit is
   reviewed pack by pack (production, quality, maintenance, energy,
   engineering, correlation, supply chain, development), one commit per pack.
5. `validate`: an event at the epoch → error (I7); `ts_shape:start_timestamp`
   present → error (I6).

**Tests:** for every interval rule, the start and end events exist, share
E2O and are ordered; summary KPIs show up as `object_changes` on the right
object; static correlation rules produce O2O; no event is at the epoch;
XES export of an interval detector still gives start/complete traces.

### WP5: signal vs asset (gap M14; equipment hierarchy)

**Functional:** a detector reads a **signal** (`source_uuid`, e.g. `temp`).
The process happens on an **asset** (machine, station, line). Today 245
rules bind `source_uuid` as `asset`, so every sensor becomes a machine.

**Technical:**
1. The default binding `source_uuid` → **`signal`** (`signal:<uuid>`).
   `produces_objects=("asset", …)` in rules now means "link the asset of the
   signal".
2. The signal → asset mapping comes from metadata (`loader.metadata`,
   `context`), or from `to_event_log(asset_of={signal_uuid: asset_id} |
   callable)`. It produces O2O `signal measured_on asset` and E2O to both
   (`observed_signal`, `on_asset`). Without a mapping, only the signal is
   linked, with a one-time `DataQualityWarning`. The signal is never
   relabelled as an asset.
3. Equipment hierarchy from metadata, aligned to ISA-95: O2O
   `asset part_of station part_of line part_of area part_of site`. A new
   function `equipment_hierarchy(meta) -> EventLog` gives objects plus O2O;
   `concat` merges it with the event logs.

**Tests:** an outlier detector with a mapping links `signal:temp` and
`asset:M1` plus O2O; without a mapping, only the signal and a warning;
`filter_by_object("asset:M1")` returns events of all signals measured on M1;
the genealogy traversal follows the hierarchy.

### WP6: conformance suite and docs

1. **Spec running example as an in-memory `EventLog`** (§5, Tables 1–21).
   `validate` passes, `describe` matches Tables 1/2, and the
   `oaval` lookups at the spec's times match (PO1 quantity at 2022-01-11 and
   2022-01-13, R3 blocked and released).
2. **Manufacturing reference log:** a line → station → batch → serial
   hierarchy with coil `made_from`, tool `made_with`, operator, shift, work
   order, and a downtime (interval) and Cpk (summary) on the same asset.
   Every invariant from I1 to I10 is asserted explicitly.
3. **Registry-wide guard:** every rule's synthetic output goes through
   `to_event_log`, then `validate`, with I1–I10.
4. Docs: update `guides/eventlog/index.md`, `objects.md`, `taxonomy.md` and
   `adapters.md` for the new semantics. Write `docs/migration.md` with
   before/after for oids, qualifiers, interval pairs and summary/static
   rules.

---

## Sequencing

| WP | Depends on | Size | Breaking for users |
|---|---|---|---|
| WP1 identity | — | M | oid values |
| WP2 qualifiers | WP1 | S | qualifier values (NULL → role) |
| WP3 attributes | WP1 | M | static attributes move to t = 0 |
| WP5 signal/asset | WP1, WP2 | M | asset links of 245 rules |
| WP4 event semantics | WP2, WP3, WP5 | L | interval / summary / static output |
| WP6 conformance | each WP adds its part | S | — |

WP4 comes last: its summary → KPI and static → O2O mappings need the typed
attributes (WP3) and the correct target object (WP5).

Every PR passes `ruff check src tests examples scripts`, `mypy` and
`pytest`, and adds a `CHANGES.txt` entry and a migration note.

## Decisions to confirm before WP4

1. **Interval events:** two atomic events `.start` / `.end` (proposed), or
   one event at the start with the end as an attribute.
2. **Summary rules:** KPI → object attribute changes (proposed), with
   `emits_event=True` only for verdicts.
3. **Static rules:** object attributes / O2O, never events (proposed).
4. **Breaking changes:** released together in one minor version with a
   migration guide (proposed), or behind a compatibility flag for one release.
