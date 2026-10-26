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

`identify_objects` → review the table → `report.to_event_log` (or step by step:
`to_object_specs` → `detect_objects` with `report.hierarchy()` → `attach_objects`).

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

Then build the event log. `report.to_event_log` runs the whole chain in one call:
the confident identifiers become object specs, the nesting becomes the `part_of`
hierarchy, and the events are linked to the objects:

```python
from ts_shape.events.production.machine_state import MachineStateEvents

state = pd.DataFrame({"systime": t, "uuid": "line:e", "value_bool": running})
events = MachineStateEvents(state, "line:e").detect_run_idle()

log = report.to_event_log(df, events, detector="MachineStateEvents.detect_run_idle")
print(log)
print(log.o2o.head(3))
```

```text
EventLog(events=8, objects=247, relations=260, o2o=240, object_changes=492)
          ocel:oid         ocel:oid_2 ocel:qualifier
0  serial:SN250000  batch:B250303-001        part_of
1  serial:SN250001  batch:B250303-001        part_of
2  serial:SN250002  batch:B250303-001        part_of
```

`events` is a detector frame (with `detector=`), an `EventLog`, a
`{detector: frame}` mapping or a list of these. Without `events` you get the
objects alone (`objects`, `o2o`, `object_changes`). The defaults are
`min_confidence=0.6`, `namespace=True` (object ids are `"{type}:{value}"`, so an
order and a batch that share a number stay two objects) and `attributes=True`
(see [Attributes](#attributes)). Pass `scope={signal_uuid: line}` when the data
holds several lines or stations, so an event links only to the objects of its own
line. Other keyword arguments (`qualifiers`, `fanout_threshold`, `infer_o2o`, ...)
go to `attach_objects` / `detect_objects`.

The same chain, step by step:

```python
from ts_shape.eventlog import attach_objects, detect_objects, to_event_log

specs = report.to_object_specs(min_confidence=0.6, namespace=True, attributes=True)
print([(s.uuid, s.object_type) for s in specs])
print(report.hierarchy())

objects = detect_objects(df, specs, hierarchy=report.hierarchy())
print(objects.objects.groupby("ocel:type").size())

log = to_event_log(events, detector="MachineStateEvents.detect_run_idle")
log = attach_objects(log, df, specs, hierarchy=report.hierarchy())
print(log.relations["ocel:type"].value_counts().sort_index())
```

`report.hierarchy()` gives the signal-level edges
`{child_uuid: [parent_uuid, packing_parent_uuid]}` from the nesting columns, here
`{'line:a': ['line:b']}`: each serial is `part_of` the batch it lies in, and two
lines never mix. It keeps only the signals `to_object_specs` keeps with the same
`min_confidence`.

`report.summary()` gives a one-screen overview, `report.identifiers()` the
identifier rows, and `report.type_scores[uuid]` every type's raw score. The report is
also a read-only mapping: `report["line:a"]` is the type of a signal, `None` when it
is not an identifier, and `dict(report)` gives all of them.

### Attributes

A signal whose value describes an identifier's objects (the customer of an order,
the material of a work order, a description text) is reported in `attribute_of`:
the uuid of the identifier it describes. `to_object_specs(attributes=True)` adds it
to that spec's `attributes`, so `detect_objects` captures it per object:

```python
t2 = pd.date_range("2025-03-03 06:00", periods=1440, freq="1min")  # 24 h
k = np.arange(1440) // 40  # an order every 40 minutes
pool = ["CUST-0107", "CUST-0212", "CUST-0345", "CUST-0418", "CUST-0526", "CUST-0631"]
who = np.random.default_rng(1).choice(pool, k.max() + 1)
sales = pd.concat(
    [
        pd.DataFrame({"systime": t2, "uuid": "erp:order",
                      "value_string": [f"SO-{2500100 + i}" for i in k]}),
        pd.DataFrame({"systime": t2, "uuid": "erp:customer",
                      "value_string": [who[i] for i in k]}),
    ],
    ignore_index=True,
)
sold = identify_objects(sales)
print(sold.table[["uuid", "object_type", "type_confidence", "attribute_of"]])
specs = {s.uuid: s for s in sold.to_object_specs(0.55, attributes=True)}
print(specs["erp:order"].attributes)
```

```text
           uuid  object_type  type_confidence attribute_of
0  erp:customer     customer         0.604651    erp:order
1     erp:order  sales_order         0.575758         None
{'customer': 'erp:customer'}
```

The customer pool changes only with the order and comes back across orders, and
the order has no production structure under it: that makes it the order's
business partner, and the `SO` prefix of the order makes it a customer (see
[Business partners](#how-types-are-identified)).

The test is on the data only: the signal's value changes only where a presence of
the identifier starts or ends (90 %+ of its changes, far more often than chance), it
keeps one value during 90 %+ of the identifier's presences, and a value of the
identifier that comes back brings back the same value of the signal. The
identifier's production and packing parents are not its attributes (they are in
`hierarchy()`), nor is a non-recurring id holding several of its presences. Of
several identifiers the signal describes, the coarsest one (fewest presences) is
reported. An attribute is named by its type when it is an identifier typed with at
least `min_confidence`, else by its uuid; such an identifier is still its own
object spec when it is kept.

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
| `parent_uuid` | The production parent: the id signal this one is nested in (a serial's batch, a batch's material); `None` when there is none |
| `packing_parent_uuid` | The packing parent: the handling unit (box, package) that holds this serial, next to its production parent |
| `evidence` | Short tuple of reasons, each with its weight |
| `attribute_of` | The identifier whose objects this signal describes (the customer of an order); `None` when there is none. See [Attributes](#attributes) |

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

A prefix family is never decisive alone: 0.25 is below the type threshold. The
prefix is the leading letters of the value, in upper case, shared by most of the
values (`PREFIX_FAMILIES` in `ts_shape.signals.identify` has the full list). The
business-partner prefixes:

| Prefix | Type | Example |
|---|---|---|
| `CUST`, `KD`, `KND`, `KUNDE` (Kunde), `DEB` (Debitor) | customer | `CUST-0107`, `KD10023` |
| `CU-` | customer | `CU-0815` |
| `SUP`, `SUPP`, `SUPPL`, `VEND`, `VDR`, `LF`, `LIEF` (Lieferant), `KRED` (Kreditor) | supplier | `SUP-0045`, `LIEF4711` |
| `V-` | supplier | `V-20017` |

`CU-` and `V-` count only with a separator (`-`, `_`, `/` or `.`) right after the
letters: a lone `V1` is a valve and `CU2` copper as often as a vendor or a
customer.

**Behaviour and structure** (the distinguishing evidence per type):

| Type | Distinguishing evidence |
|---|---|
| `serial` | 50+ values, never revisited, held about one cycle (<= 30 min), sequential; changes once per counted part; nested in a batch; changes whenever its batch changes |
| `batch` | long holds, never revisited, a running number; contains serials; changes whenever its coarser work order changes; date codes; held under 30 min counts against it |
| `lot` | long holds, occasionally revisited, unrelated to the other ids; date codes, `LOT` prefix |
| `work_order` | hours-long holds, occasionally revisited (a resumed order); coarser than its batches; recipe and material change with it; contains batches |
| `order` (`sales_order`, `purchase_order`) | event-stamped or held, never revisited, arrivals spread like independent events (hold CV 0.75-1.6), a recurring partner changes with it, no production structure under it |
| `recipe` | a tiny pool (<= 4), often revisited, long holds; changes only when a work order of the production hierarchy changes |
| `material` / `article` | like recipe with more values; contains batches (the reported parent of a batch or lot); tops a chain of finer unique ids (over its work orders and batches); SAP material numbers, GTIN, plain numbers of 6+ digits |
| `tool` / `fixture` | a tiny pool, long or very long holds, changed on its own (no other id changes with it) |
| `carrier` | a circulating pool (a tour of as many runs as values shows nearly all of them), recurring regularly at cycle rate |
| `rfid_tag` | EPC hex (on a circulating carrier) |
| `asset` | a tiny pool that takes turns every cycle (parallel machines), in any order; a two-level code (`L1-M03`) |
| `station` | numbers in steps of 10 (`OP10`, `OP20`), fixed successor |
| `location` | location codes; a fixed count of parts per value with no running number |
| `operator` | changes at shift boundaries (30 %+ and 3+ within 15 min), never held longer than a shift, the badges come back |
| `business_partner` (`customer`, `supplier`) | often revisited, changes only when an order outside production changes (event-like, or typed as an order); the customer of a sales order, the supplier of a purchase order |
| `handling_unit` | SSCC; a fixed count of parts per value (a package of N) with a running number |

**Cross-signal structure** runs among the screened identifiers only (at most 64),
O(signals² × runs) with searchsorted on the run boundaries. Change times match
within 5 s, or within 5 s plus the polling interval of the signals (up to 15 min)
when chance alone would not give the match (the other signal changes at most 10 %
of the time within that window):

- **Containment**: each presence of A lies inside one presence of B, A has at
  least 1.5× as many presences, and B's changes coincide with a change of A. This
  is type evidence (a serial inside something, a batch-like id that contains
  serials); the reported nesting is decided separately, see
  [Nesting](#nesting). Resources, people and serials (tools, carriers, stations,
  operators, partners) are never containers: a long-running tool contains every
  batch without owning it.
- **Attributes**: A changes only when B changes (70 %+), B is not revisited, and A
  recurs (revisit > 0.2): A is an attribute of B. When B is part of the production
  hierarchy (it nests batches or serials, or is coarser or finer than another
  unique id) A is a production attribute (recipe, material); otherwise B is an
  order-like host, and an event-like B makes A its business partner. A unique A
  that changes only with a unique, finer B is the coarser of the two (a work order
  over its batches).
- **Business partners**: when B has no production structure and its values and
  relations make it an order (a sales or purchase order, or the `order` group,
  0.15 ahead of every other type), the recurring A is its partner: +0.40 for
  `business_partner`, -0.30 for recipe, material and article. A sales order
  0.15 ahead of a purchase order (a prefix such as `SO`) adds +0.20 for
  `customer`, a purchase order for `supplier`; otherwise the group is reported.
  This holds for orders held for minutes or hours as much as for one-sample
  stamps, and needs no prefix on the partner. It does not apply when A has
  production structure of its own (it contains batches or serials, or changes
  with a production id), changes at cycle rate (a machine taking turns per part),
  or changes at fewer than a quarter of B's changes (a tool held over many
  orders).
- **Part clocks**: the changes of a serial-like id, or the increments of a counter,
  counted between an id's changes. A fixed count (CV <= 0.3) is a package of N
  parts (handling unit) or its storage location; one change per counted part is a
  serial.
- **Unrelated**: an id that changes with no other id at all (a tool, a lot).

The operator evidence uses the shift boundaries of `shift_definitions` (default
`ts_shape.utils.shifts.DEFAULT_SHIFTS`: 06:00, 14:00, 22:00), read on the local wall
clock for tz-aware data.

## Nesting

Once the types are known, every serial, batch and lot gets a **production parent**
(`parent_uuid`) and a serial also a **packing parent** (`packing_parent_uuid`):

```python
t = pd.date_range("2025-03-03 06:00", periods=1170, freq="1min")  # a part a minute
lengths = [20, 90, 35, 120, 50, 25, 100, 40, 75, 30] * 2  # batch sizes
batch = [i for i, n in enumerate(lengths) for _ in range(n)]
parts = pd.concat(
    [
        pd.DataFrame({"systime": t, "uuid": "line:serial",
                      "value_string": [f"SN{100000 + i}" for i in range(1170)]}),
        pd.DataFrame({"systime": t, "uuid": "line:batch",
                      "value_string": [f"B250303-{b + 1:03d}" for b in batch]}),
        pd.DataFrame({"systime": t, "uuid": "line:box",  # 24 parts a box
                      "value_string": [f"PK{7000 + i // 24}" for i in range(1170)]}),
    ],
    ignore_index=True,
)
packed = identify_objects(parts)
print(packed.table[["uuid", "object_type", "parent_uuid", "packing_parent_uuid"]])
```

```text
          uuid    object_type parent_uuid packing_parent_uuid
0   line:batch          batch        None                None
1     line:box  handling_unit        None                None
2  line:serial         serial  line:batch            line:box
```

The serial lies inside both the batch and the box; the batch is where it belongs in
production, the box is how it is packed.

**Which parent.** The candidates are ranked by type, and a farther level is taken
only when the data has no id of the nearest one:

| Child | Production parent, in order of preference |
|---|---|
| `serial` | `batch` / `lot` → `work_order` / order → `material` / `article` → `recipe` / `setup` |
| `batch`, `lot` | `material` / `article` / `recipe` (a batch is a quantity of one material made by one recipe) → `work_order` / order |

Among equals the finer id wins (more distinct values: a material inside its
recipe). For a serial, an equal that itself nests both ways in a material or
recipe comes first: a package or a refilled lot on a clock of its own may lie
around the parts too, but in no material. A handling unit, or any id that
holds a fixed count of parts, that nests a serial is its packing parent.

**Mistyped materials.** In a recording of a few days a material is often typed
otherwise on its own values, so a batch's parent is not limited to the ids typed
material, article or recipe:

- a tool or fixture that nests the batch both ways is a candidate of the nearest
  level: a material held for half a day from a pool of three reads like a tool;
- an order, package, batch or lot that nests the batch both ways and holds another
  such id is the material: one unique id inside another over the batch is the
  work order inside the material it makes (a material never revisited reads like
  an order);
- of a material and a recipe that change together one to one, the one whose
  values read more like a material wins: its format and prefix evidence, a number
  of six or more digits (`10423317`) against a number of at most three digits or a
  name of three or more `_`-separated segments (`17`, `RZ_PA66_GF30_v3`).

These apply only to the finest quantity: an id with another order or batch inside
it, both ways, is itself an order or material taken for a batch and gets none of
them. A "serial" that holds other serials (four times as many presences or more,
both ways) and itself nests in a material, recipe or order with a lift of 0.7 or
more is read as their batch: batches shorter than half an hour read as serials on
their values. Tools, fixtures, carriers,
operators, stations, locations and partners are never part of a nesting, and
neither are orders as children (an order is not nested in its material) or
messages that name an id (`"Batch 25061207 released"`).

**Aligned both ways.** A parent must nest the child in both directions:

- the parent keeps its value during the child's presences (90 %+ of them; the
  polling lag of either signal may stick out), and the child has at least 1.25×
  as many presences;
- the parent's changes coincide with a change of the child: a presence starts or
  ends, nothing is present, or the child pauses (a presence 4× its median hold, a
  stopped line), within 5 s or 10 % of the child's median hold, widened by the
  polling period of a polled signal. At least half of them must, and far more
  often than chance would give: the lift `(matched - chance) / (1 - chance)`
  over the share of time such windows cover is 0.4 or more, and the binomial
  probability of the matches by chance is at most 1 %.

The chance correction is what keeps a batch of another line, or a lot refilled on
its own clock, from nesting a serial: any moment lies close to a change of a fast
part clock, so plain coincidence is no evidence.

**Sampling windows.** Each change is also tested on the window its own sampling
leaves open: a value seen at `t` changed after the sample before it, so in
`(t - u, t]`, with `u` the time since that sample, at most the polling lag. A
parent change meets a child change when their windows overlap (within 5 s); the
chance of each parent change is the share of the time where a window as wide as
its own would meet one, and the matches are tested against those chances (at
least 3 matches, a lift of 0.6 or more, and at most 1 % by chance). Two cases
gain:

- a value written when it changes and re-sent every few minutes or hours
  (keep-alive): the re-sends lie on a grid, a read off that grid is a change seen
  when it happened (`u = 0`), however long the re-send interval;
- a material polled every 15 to 60 minutes: its change is seen up to that long
  after the batch change it goes with. The containment test reads the same
  windows: a parent change counts as inside a child presence only when its whole
  window is.

A parent that holds the child in the strong sense needs only 5 % by chance: no
child presence is seen to straddle one of its changes, and the child has four
times as many distinct values (a material pool over its batches).

**Polled signals.** A signal sampled on a regular grid is seen up to one period
late. When that hides when the child changed (a serial read every half cycle, a
batch polled every minute while a part takes 30 s), the alignment cannot be shown.
A preferred parent the child lies inside still wins over a verified, farther one
when it nests in that farther parent itself: both belong to the line the farther
parent puts the child on. Without a verified parent of the nearest level, any
other id that nests the child both ways serves the same way: a farther level, a
tool or fixture, a package, or an order or material typed otherwise. The verified
parent is tried first. Without any such id the child stays without a parent rather
than getting a guess.

What sampling cannot show stays undecided. A material polled every 30 minutes over
batches of 20 to 40 minutes is never placed: any half hour holds a batch change,
so its windows meet them by chance alone.

### Rarely changing materials

In a continuous process the material often changes once a day: a recording of a
few days shows one to three changes. That is too few for the co-change test above
(3 matches at least), yet one exact coincidence can already be very unlikely by
chance. So a parent with at most 3 changes in the child's span is tested on all of
them together:

- **Every change meets the child.** Each change of the parent must meet a change
  of the child's value (a new batch, not the same batch again after an empty
  reading), on the windows their sampling leaves open, within 5 s.
- **The chance model.** A change's chance is the share of the child's span where
  a window as wide as its own would meet a child change: about the number of
  child changes times (10 s plus the sampling windows) over the span. With
  batches every 30 minutes, pushed when they change, that is 0.6 % for one change;
  with batches every 2 hours 0.14 %. The changes are independent draws, so the
  chance of all of them is the product (two changes on 30-minute batches:
  0.003 %). It is computed on the times the changes were seen (pushed when they
  happened) and on the windows the parent's own sampling leaves open; the smaller
  product is doubled when the two tries differ.
- **Multiple testing.** Every batch, lot or serial (and every order or package it
  may be found through) is tried against every id that changes only a few times,
  so the product times the number of pairs tried must stay at most 1 %
  (`NEST_FEW_CHANCE_P`): a Bonferroni correction over the pairs that could reach
  that bound at all (Tarone's correction: a pair whose changes would be likely by
  chance even if every one of them met the child cannot give a false edge, so it
  is not counted). Under the chance model, the chance that this test gives a frame
  one false edge is at most 1 %.
- **Strong containment.** No child presence straddles one of the parent's
  changes, the child has twice as many distinct values as the parent at least and
  6 changes of its own (two ids that both change rarely make no container).
- **Not by the clock.** A parent whose every change falls on a shift boundary or
  midnight (within 5 s on the local wall clock) is no parent, whatever it holds: a
  day order, a date code or a shift label changes there, and a batch made per
  shift or per day starts there too. This applies to every nesting test.

A material on its own values (a tiny pool held for a day) often reads as no type,
as a tool, as a lot or as a batch; with that few changes it is still taken for a
batch's or lot's material. When the batch is polled too coarsely, the material is
found through the work order the batch lies in (the order's changes meet the
material's). Of a recipe and a material that both change with the batch, the
material is taken when it changes at every change of the recipe, as unlikely by
chance (a recipe pushed when it changes, a material polled every minute). An id
that another order or batch nests in, or a tiny pool that changes only a few
times, is no batch to nest itself.

The types follow: a unique id that tops a chain of two finer unique ids held
longer than parts (a material over its work orders over its batches, never
revisited in a short recording) gets +0.45 for material, article and recipe and
-0.30 to -0.40 for work order, batch, lot, order and package; the reported parent
of a batch or lot gets +0.35 for material, article and recipe and -0.30 to -0.40
for tool, fixture, work order, batch, lot, order and package (it is related, not
`unrelated`); and a plain number of 6 or more digits counts -0.25 for a recipe,
tool or fixture (a material number, not a program). A material without a format
cue (`3KA5130-1AB`) can still read as the `setup` group.

What it supports, by the opt-in options of the generator
(`ts_shape._object_signals.generate`): `material_changes_per_day=1.0` runs every
material in campaigns that change about once a day, at the start of a work order;
the batch then nests in it at 0.93 / 0.93 precision / recall on the held-out
split (0.83 / 0.68 before), 0.96 / 0.96 on never-seen seeds (0.93 / 0.87). With
`continuous_process="lot"` (no batches, no work orders: production lots of a few
hours inside the material, parts inside the lots) the lot nests in its material
at 0.94 / 0.61 (0.87 / 0.46); the misses are lots that read as an order or as no
type, which are no children to nest. With `continuous_process="serial"` (parts
directly in the material) there is no edge: see the limits.

`explain()` and `summary()` show both parents (`nested in line:batch (depth 1)`,
`packed in line:box`). On the held-out benchmark split the nesting scores about 0.98
precision and 0.95 recall per edge (0.99 / 0.94 on never-seen seeds); with every
material polled every 30 minutes (`material_poll=1800` of the generator) about
0.84 / 0.82 (0.54 / 0.50 before the sampling windows). The misses left are mostly
batches whose material is typed as something that is never a parent (an operator,
a batch), materials polled too coarsely to show when they changed, and serials
whose batch is polled too coarsely to align and that lie in no other verified id.

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
      production structure separates them. A work order with no batch or serial
      signal can read as an order; its recipe stays a recipe unless the order
      type of the host is clear (0.15 ahead of a work order or batch).
    - batch / lot / work order: structure (nesting, the coarser-finer relation),
      revisits and a running number.
    - station / carrier and asset / station: the pool size and a fixed order.
- **Encoding.** An id written once per change, without an empty reading after it, is
  read as held until the next value. Only ids cleared by an empty reading count as
  event-stamped.
- **Counters.** An integer that counts up with a fixed width of 5+ digits and never
  resets is read as a serial, not a counter.
- **Rarely changing materials** (see [Rarely changing materials](#rarely-changing-materials)):
    - A material that never changes in the recording is a constant: there is no
      evidence the batches lie in it, and no edge.
    - One change is found only when the child's changes are timed closely: one
      change meets a child change by chance with about the share of time the
      child's change windows cover. Batches polled every 5 minutes, changing
      every hour or so, cover about a tenth of the time, so one change is no
      evidence; two or three can be. A material polled coarsely, with one change, is found only
      through a work order that is timed closely.
    - Parts directly in the material (no batch, lot or order between them): a
      part changes every cycle, so any moment meets a part change and a material
      change says nothing. No edge.
    - An id that changes only a few times, each time exactly when a batch starts,
      and never inside a batch, is that batch's parent by the data, whatever it
      is. Only the clock (a shift boundary or midnight) and an independent clock
      of its own are told apart.

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
