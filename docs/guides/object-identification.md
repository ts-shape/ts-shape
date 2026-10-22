# Object identification

[Object detection](eventlog/objects.md) needs to know which signals carry ids and of
what: `ObjectSpec("line:a", "serial")`. `identify_objects` finds those signals for
you. It screens every signal for values that name a physical object or a
manufacturing entity, then classifies the type of each one: serial, batch, work
order, carrier, tool, operator, and so on.

The evidence is the **data only**: the values and how they behave over time. The
uuid, labels, descriptions and other metadata are never read, so a signal called
`batch_id` that holds temperatures is still a measurement. The one exception is
`hints={uuid: object_type}`, which you set yourself to pin a type.

The result is a separate report. It does not change `SignalSpec.kind` or what
`infer_specs` returns.

## Workflow

`identify_objects` → review the table → `to_object_specs` → `detect_objects` →
`to_event_log`.

```python
import numpy as np
import pandas as pd
from ts_shape import identify_objects

t = pd.date_range("2025-03-03 06:00", periods=720, freq="1min")  # 12 h
part = np.arange(720) // 3  # a part every 3 minutes
rng = np.random.default_rng(0)
running = (np.arange(720) // 45) % 4 != 3
df = pd.concat(
    [
        pd.DataFrame({"systime": t, "uuid": "line:a",
                      "value_string": [f"SN{250000 + p}" for p in part]}),
        pd.DataFrame({"systime": t, "uuid": "line:b",
                      "value_string": [f"B250303-{1 + i // 120:03d}" for i in range(720)]}),
        pd.DataFrame({"systime": t, "uuid": "line:c", "value_integer": part}),
        pd.DataFrame({"systime": t, "uuid": "line:d",
                      "value_double": 60 + rng.normal(0, 0.5, 720)}),
        pd.DataFrame({"systime": t, "uuid": "line:e",
                      "value_string": np.where(running, "RUN", "IDLE")}),
    ],
    ignore_index=True,
)

report = identify_objects(df)
print(report.table[["uuid", "role", "object_type", "type_confidence", "parent_uuid"]])
```

```text
     uuid         role object_type  type_confidence parent_uuid
0  line:a   identifier      serial         0.775510      line:b
1  line:b   identifier       batch         0.622222        None
2  line:c      counter        None              NaN        None
3  line:d  measurement        None              NaN        None
4  line:e        state        None              NaN        None
```

`line:a` is a serial nested in the batch `line:b`; the counter, the temperature and
the run state are screened out. Review the reasons before you rely on a type:

```python
print(report.explain("line:a"))
```

```text
line:a: identifier (value_string)
  identifier score 0.80; type serial (confidence 0.78), runner-up batch (0.15)
  720 samples, 240 distinct, 240 runs, median hold 180 s, revisit 0.00, ...
  - 240 distinct values (> 16) +0.30
  - shape A{2}9{6} (100% regular) +0.25
  - held 3 samples per run (median) +0.15
  - single tokens of 2..40 characters +0.10
  - serial 1.90: values never come back (revisit <= 0.02) +0.25; many distinct
    values (>= 50) +0.25; held about one cycle (median <= 1800 s) +0.25; next value
    is 1..5 above (>= 80%) +0.15; changes once per counted part (aligned with a
    counter) +0.50; each presence lies inside one presence of another id +0.10;
    changes whenever a coarser non-recurring id changes +0.15; prefix family 'SN'
    +0.25
  - nested in line:b (depth 1)
  type scores: serial 1.90, batch 0.55, handling_unit 0.40
```

Then turn the confident identifiers into object specs, detect the objects and build
the event log:

```python
from ts_shape.eventlog import attach_objects, detect_objects, to_event_log
from ts_shape.events.production.machine_state import MachineStateEvents

specs = report.to_object_specs(min_confidence=0.6)
print([(s.uuid, s.object_type) for s in specs])

objects = detect_objects(df, specs, hierarchy={"serial": "batch"})
print(objects.objects.groupby("ocel:type").size())

state = pd.DataFrame({"systime": t, "uuid": "line:e", "value_bool": running})
events = MachineStateEvents(state, "line:e").detect_run_idle()
log = to_event_log(events, detector="MachineStateEvents.detect_run_idle")
log = attach_objects(log, df, specs, hierarchy={"serial": "batch"})
print(log.relations["ocel:type"].value_counts().sort_index())
```

`report.summary()` gives a one-screen overview, `report.identifiers()` the
identifier rows, and `report.type_scores[uuid]` every type's raw score. The report is
also a read-only mapping: `report["line:a"]` is the type of a signal, `None` when it
is not an identifier, and `dict(report)` gives all of them.

## The report table

One row per signal, sorted by uuid:

| Column | Meaning |
|---|---|
| `role` | `identifier`, `state`, `free_text`, `measurement`, `counter`, `timestamp`, `constant` or `insufficient_data` |
| `identifier_score` | Screening score, 0-1 (identifier at 0.5 or more) |
| `object_type`, `type_confidence` | The type (or group) and its confidence, 0-1 |
| `runner_up_type`, `runner_up_confidence` | The best type outside it |
| `sample_count`, `distinct_count`, `run_count` | Rows, distinct non-empty values, value runs |
| `mean_hold_seconds`, `median_hold_seconds` | How long an id is held (to the next value or empty reading) |
| `revisit_ratio` | Share of value runs that bring back an earlier value |
| `changes_per_hour`, `empty_ratio` | Change rate; share of empty readings |
| `pattern`, `pattern_share` | Dominant shape signature (`A` upper, `a` lower, `9` digit: `A{2}-9{5}`) and its share |
| `sequential_ratio` | Share of changes to a number 1-5 higher |
| `recurrence_cv` | Regularity of re-appearance (low: a pool cycling at a fixed pace) |
| `parent_uuid` | The id signal this one is nested in |
| `evidence` | Short tuple of reasons, each with its weight |

The id semantics are those of the id-consuming detectors and of object detection: a
value is held until the next value, a change starts a new item, and an empty
reading (`""`, whitespace, NaN) means "no item".

## How screening works

Each signal is reduced to its run-length encoding and its distinct values, so
screening is O(n) and holds one signal in memory at a time. 1 million rows over 20
signals take under a second. `value_string`, `value_integer` and integral
`value_double` can carry ids.

**Clean-up first.** Readers misread, so before anything is measured:

- spellings that differ only in case, surrounding blanks or leading zeros
  (`vr-415` and `VR-415`, `0000140630` and `140630`) are one value;
- a *misread* keeps the previous reading: a value that never lasts more than two
  samples, is rare (under a quarter of the runs or samples of the typical value)
  and is foreign to the typical values -- no letter or digit (`????????`), a PLC
  default number (0, -1, 32767, 65535, ...), a shape no typical value has
  (`NOREAD` among `FX-13`), or a truncation of one (`FX-`). In an integer column a
  PLC default in a shape of its own is a misread however long it lasts;
- a single empty sample inside a hold (`A A _ A`) is a dropped reading, not "no
  item"; one-sample ids cleared by an empty (`A _ B _`) keep their empties.

The evidence counts the misreads. Shares of a format or a shape are weighed by each
value's runs, so a misread that survives cannot outvote a pool of three tools.

The negative roles are checked first, in this order:

1. `insufficient_data`: fewer than `min_samples` rows (default 3: a recipe logged on
   change has only a handful of rows over days), or no non-empty reading.
2. `constant`: one value.
3. `state`: booleans.
4. `measurement`: non-integral floats, or decimal numbers stored as text (`60.9`).
5. `timestamp`: ISO and local date/time formats; compact dates (`20240315`,
   `20240315143000`, and `240315` on the day it appears); numbers or digit strings
   that track the sample time (epoch s/ms/us/ns); any short text of digits and date
   punctuation that a date parser reads as the moment the value appears (within 2
   days; a date alone within 31 days before).
6. `counter`: monotonic with small steps and resets, or a count that steps up by
   1-5 at nearly every change and restarts (a reset polled late). A fixed-width
   (5+ digits) number that only rises and never resets is a serial instead.
7. `measurement`: a quantized analog value (small steps up and down), or a number
   that changes at nearly every sample with no format, no holding and no
   circulating pool.
8. `state`: at most 16 values that are word-like (RUN/IDLE, Running), a small dense
   integer enumeration (starting at -1, 0 or 1 and filling half its range: 1/2/3),
   or a phase sequence (values in a fixed order, each held under 15 s: step names).
   Integer *codes* stay: sparse (17, 23, 5), large (4711) or starting elsewhere
   (9..13), and a tiny pool that takes turns every cycle (parallel machines).
9. `free_text`: sentences (two or more plain words, even when they embed ids:
   `Batch B24W07-008 released`), or spaces with a variable or long length and
   irregular shapes.

What is left gets an identifier score, the sum of these features (clipped to 0-1):

| Feature | Condition | Weight |
|---|---|---|
| `high_cardinality` | more than 16 distinct values | +0.30 |
| `fixed_pool` | 3-16 values that recur regularly (recurrence CV <= 0.5) | +0.30 |
| `low_cardinality_codes` | 2-16 values that are codes, not words | +0.20 |
| `regular_format` | one shape, coarse shape, shape family (`M-7`, `M-58`) or segment structure (`RZ_PP_GF30_v5`) for 80 %+ of the values | +0.25 |
| `known_format` | a format detector matches 90 %+ of the values | +0.35 |
| `held` | median of 2+ samples per run | +0.15 |
| `event_stamped` | one-sample ids cleared by an empty reading (80 %+) | +0.15 |
| `token` | no spaces, 2-40 characters (1-40 for integers) | +0.10 |
| `changes_every_sample` | a number of more than 16 values, new at nearly every sample, no format, not a running number, no circulating pool | -0.30 |

**Fault and alarm codes.** After the cross-signal stage, an identifier from a
recurring pool that is present less than 20 % of the time, in no fixed order, with
no known format and changing with no other id becomes a `state`: that is what a
fault code (`E1043`, set at a stop and cleared) does. A partner changes with its
order; a carrier on a loop keeps a fixed order or circulates.

## How types are identified

Facts about an identifier add the weights of the type rules (`TYPE_RULES` in
`ts_shape.signals.identify`) to the types they point at. The best type wins at a
score of 0.5 or more; its confidence is
`best / (best + runner_up) * min(1, best / 0.8)`. Below 0.5 the type is
`identifier`, and the evidence says which facts were missing.

**Format detectors** (strong evidence, validated):

| Format | Type | Weight |
|---|---|---|
| GS1 element string, AI (00) / (21) / (10) / (01) | handling_unit / serial / batch (lot 1.0) / article | 1.2 |
| GTIN-8/12/13/14 with a valid mod-10 check digit | article | 1.0 |
| SSCC-18 with a valid check digit (also an integer without its leading zeros) | handling_unit | 1.2 |
| SAP-style 18-digit zero-padded material number | material | 1.2 |
| Date code YYMMDD / YYWW / YYDDD + sequence, a valid date at most 31 days before the value first appears | batch (0.7), lot (0.6) | |
| EPC hex, 24 or 32 hex characters | rfid_tag | 1.2 |
| Location code: aisle-rack-level (`A-01-03`) or lettered levels (`R08-S05-F4`) | location | 1.2 |
| UUID / GUID | type-neutral: screening only | |
| Prefix family (`SN`, `WO`, `FA`, `SO`, `PO`, `WT`, `WZ`, ... on the value) | the family's type | 0.25 |

A prefix family is never decisive alone: 0.25 is below the type threshold.

**Behaviour and structure** (the distinguishing evidence per type):

| Type | Distinguishing evidence |
|---|---|
| `serial` | 50+ values, never revisited, held about one cycle (<= 30 min), sequential; changes once per counted part; nested in a batch; changes whenever its batch changes |
| `batch` | long holds, never revisited, a running number; contains serials; changes whenever its coarser work order changes; date codes; held under 30 min counts against it |
| `lot` | long holds, occasionally revisited, unrelated to the other ids; date codes, `LOT` prefix |
| `work_order` | hours-long holds, occasionally revisited (a resumed order); coarser than its batches; recipe and material change with it; contains batches |
| `order` (`sales_order`, `purchase_order`) | event-stamped or held, never revisited, arrivals spread like independent events (hold CV 0.75-1.6), a recurring partner changes with it, no production structure under it |
| `recipe` | a tiny pool (<= 4), often revisited, long holds; changes only when a work order of the production hierarchy changes |
| `material` / `article` | like recipe with more values; contains batches; SAP material numbers, GTIN |
| `tool` / `fixture` | a tiny pool, long or very long holds, changed on its own (no other id changes with it) |
| `carrier` | a circulating pool (a tour of as many runs as values shows nearly all of them), recurring regularly at cycle rate |
| `rfid_tag` | EPC hex (on a circulating carrier) |
| `asset` | a tiny pool that takes turns every cycle (parallel machines), in any order; a two-level code (`L1-M03`) |
| `station` | numbers in steps of 10 (`OP10`, `OP20`), fixed successor |
| `location` | location codes; a fixed count of parts per value with no running number |
| `operator` | changes at shift boundaries (30 %+ and 3+ within 15 min), never held longer than a shift, the badges come back |
| `business_partner` (`customer`, `supplier`) | often revisited, changes only when an event-like id outside production (an order) changes |
| `handling_unit` | SSCC; a fixed count of parts per value (a package of N) with a running number |

**Cross-signal structure** runs among the screened identifiers only (at most 64),
O(signals² × runs) with searchsorted on the run boundaries. Change times match
within 5 s, or within 5 s plus the polling interval of the signals (up to 15 min)
when chance alone would not give the match (the other signal changes at most 10 %
of the time within that window):

- **Nesting**: each presence of A lies inside one presence of B, A has at least 1.5×
  as many presences, and B's changes coincide with a change of A. The closest
  container is recorded as `parent_uuid`, and the depth is evidence (serial ⊂ batch ⊂
  material). Resources, people and serials (tools, carriers, stations, operators,
  partners) are never containers: a long-running tool contains every batch without
  owning it.
- **Attributes**: A changes only when B changes (70 %+), B is not revisited, and A
  recurs (revisit > 0.2): A is an attribute of B. When B is part of the production
  hierarchy (it nests batches or serials, or is coarser or finer than another
  unique id) A is a production attribute (recipe, material); otherwise B is an
  order-like host, and an event-like B makes A its business partner. A unique A
  that changes only with a unique, finer B is the coarser of the two (a work order
  over its batches).
- **Part clocks**: the changes of a serial-like id, or the increments of a counter,
  counted between an id's changes. A fixed count (CV <= 0.3) is a package of N
  parts (handling unit) or its storage location; one change per counted part is a
  serial.
- **Unrelated**: an id that changes with no other id at all (a tool, a lot).

The operator evidence uses the shift boundaries of `shift_definitions` (default
`ts_shape.utils.shifts.DEFAULT_SHIFTS`: 06:00, 14:00, 22:00), read on the local wall
clock for tz-aware data.

## Limits

- **Groups.** When the data cannot tell group members apart, the report gives the
  group instead of guessing:
    - `order`: a sales order or a purchase order, without a format cue.
    - `business_partner`: a customer or a supplier, without a format cue.
    - `setup`: a recipe, tool or fixture. This is reported when a tiny pool of codes
      held for hours changes with no other id, so no work order shows it is a
      recipe, and no format shows it is a tool or a fixture.

  A group is reported when two of its members score within 0.15 of each other. A
  prefix family (`SO`, `PO`, `CUST`, `SUP`, `WZ`, `FX`, ...), a registered pattern or a
  hint separates them.
- **Types that need a format cue:** `rfid_tag` (EPC), `location` and
  `handling_unit` without a part clock (no serial or counter signal), and `fixture`
  (it scores just below `tool` on the same behaviour). Register a pattern for your
  site's format.
- **Weak pairs** (separated by small margins of behaviour):
    - recipe / tool / fixture: a tiny pool of codes held for hours that changes with
      no other id is a tool, a fixture, or a recipe whose work order is not in the
      data. Only the relation to the work order separates them; without it the
      report gives the `setup` group.
    - recipe / material / article: separated by the pool size (a recipe has a tiny
      pool), nesting and formats.
    - customer / material: both change only with a unique id; the host's
      production structure separates them, so an order next to a production order
      with no batch or serial signal reads alike.
    - batch / lot / work order: structure (nesting, the coarser-finer relation),
      revisits and a running number.
    - station / carrier and asset / station: the pool size and a fixed order.
- **Encoding.** An id written once per change, without an empty reading after it, is
  read as held until the next value. Only ids cleared by an empty reading count as
  event-stamped.
- **Counters.** An integer that counts up with a fixed width of 5+ digits and never
  resets is read as a serial, not a counter.

Pin what the data cannot tell with `hints`:

```python
pinned = identify_objects(df, hints={"line:b": "lot"})
print(pinned.table.set_index("uuid").loc["line:b", ["object_type", "type_confidence"]])
```

## Register types and patterns

New types are added with `register_identifier_type(name, group=None)`; they are also
registered as OCEL object types, so `detect_objects` and the OCEL export accept them.
`register_identifier_pattern(name, regex_or_callable, object_type, weight=...)` adds a
site-specific format as evidence: when 90 %+ of a signal's values match (the whole
value, for a regex), the signal counts as having a known format and `object_type`
gains `weight` (1.0 or more is decisive).

```python
from ts_shape import register_identifier_pattern, register_identifier_type

register_identifier_type("coil")
register_identifier_pattern("coil number", r"CL\d{2}-\d{5}", "coil", weight=1.2)

coils = pd.DataFrame({
    "systime": pd.date_range("2025-03-03", periods=200, freq="2min"),
    "uuid": "mill:coil",
    "value_string": [f"CL25-{40000 + i // 10}" for i in range(200)],
})
print(identify_objects(coils).table[["uuid", "object_type", "type_confidence"]])
```

A pattern can also target a group, and a type registered with `group=` shares the
group's rules. The vocabulary and the groups live in one dict,
`ts_shape.signals.IDENTIFIER_TYPES` (type → group or None).
