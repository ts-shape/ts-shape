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
0  line:a   identifier      serial         0.897436      line:b
1  line:b   identifier       batch         0.648148        None
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
  identifier score 0.80; type serial (confidence 0.90), runner-up sales_order (0.03)
  720 samples, 240 distinct, 240 runs, median hold 180 s, revisit 0.00, ...
  - 240 distinct values (> 16) +0.30
  - shape A{2}9{6} (100% regular) +0.25
  - held 3 samples per run (median) +0.15
  - single tokens of 2..40 characters +0.10
  - serial 1.75: values never come back (revisit <= 0.02) +0.25; many distinct
    values (>= 50) +0.25; held about one cycle (median <= 1800 s) +0.25; next value
    is 1..5 above (>= 80%) +0.15; changes once per counted part (aligned with a
    counter) +0.50; each presence lies inside one presence of another id +0.10;
    prefix family 'SN' +0.25
  - nested in line:b (depth 1)
  type scores: serial 1.75, sales_order 0.20, purchase_order 0.20, batch 0.15, ...
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
identifier rows, and `report.type_scores[uuid]` every type's raw score.

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
screening is O(n) and holds one signal in memory at a time. 10 million rows over 20
signals take about 6-8 seconds. `value_string`, `value_integer` and integral
`value_double` can carry ids.

The negative roles are checked first, in this order:

1. `insufficient_data`: fewer than `min_samples` rows (default 20), or no non-empty
   reading.
2. `constant`: one value.
3. `state`: booleans.
4. `measurement`: non-integral floats.
5. `timestamp`: values parse as dates or times, or numbers that track the sample
   time (epoch).
6. `counter`: monotonic with small steps and resets. A fixed-width (5+ digits)
   number that steps by +1 without a reset is a serial instead.
7. `measurement`: a quantized analog value (small steps up and down), or a number
   that changes at nearly every sample with no format and no holding.
8. `state`: at most 16 word-like values (RUN/IDLE) or small integer codes (1/2/3).
9. `free_text`: spaces, variable or long length, irregular shapes.

What is left gets an identifier score, the sum of these features (clipped to 0-1):

| Feature | Condition | Weight |
|---|---|---|
| `high_cardinality` | more than 16 distinct values | +0.30 |
| `fixed_pool` | 3-16 values that recur regularly (recurrence CV <= 0.5) | +0.30 |
| `low_cardinality_codes` | 2-16 values that are codes, not words | +0.20 |
| `regular_format` | one shape signature for 80 %+ of the values | +0.25 |
| `known_format` | a format detector matches 90 %+ of the values | +0.35 |
| `held` | median of 2+ samples per run | +0.15 |
| `event_stamped` | one-sample ids cleared by an empty reading (80 %+) | +0.15 |
| `token` | no spaces, 2-40 characters | +0.10 |
| `changes_every_sample` | a number, new at nearly every sample, no format | -0.30 |

## How types are identified

Facts about an identifier add the weights of the type rules (`TYPE_RULES` in
`ts_shape.signals.identify`) to the types they point at. The best type wins at a
score of 0.5 or more; its confidence is
`best / (best + runner_up) * min(1, best / 0.8)`. Below 0.5 the type is
`identifier`, and the evidence says which facts were missing.

**Format detectors** (strong evidence, validated):

| Format | Type | Weight |
|---|---|---|
| GS1 element string, AI (00) / (21) / (10) / (01) | handling_unit / serial / batch / article | 1.2 |
| GTIN-8/12/13/14 with a valid mod-10 check digit | article | 1.0 |
| SSCC-18 with a valid check digit | handling_unit | 1.2 |
| SAP-style 18-digit zero-padded material number | material | 1.2 |
| Date code YYMMDD / YYWW / YYDDD + sequence, a valid date at most 31 days before the value first appears | batch (0.7), lot (0.6) | |
| EPC hex, 24 or 32 hex characters | rfid_tag | 1.2 |
| Aisle-rack-level code (`A-01-03`) | location | 1.2 |
| UUID / GUID | type-neutral: screening only | |
| Prefix family (`SN`, `WO`, `FA`, `SO`, `PO`, `WT`, `WZ`, ... on the value) | the family's type | 0.25 |

A prefix family is never decisive alone: 0.25 is below the type threshold.

**Behaviour and structure** (the distinguishing evidence per type):

| Type | Distinguishing evidence |
|---|---|
| `serial` | 50+ values, never revisited, held about one cycle (<= 30 min), sequential; changes once per counted part when a counter is present; nested in a batch or order |
| `batch` | long holds (>= 30 min), never revisited, increasing; contains serials; date codes |
| `lot` | long holds, occasionally revisited (interleaved consumption); date codes, `LOT` prefix |
| `work_order` | hours-long holds, occasionally revisited (a resumed order); contains serials or batches |
| `recipe` | 2-100 values, often revisited, long holds (not days) |
| `material` | like recipe; contains batches; SAP material numbers |
| `article` | like recipe; contains serials; GTIN |
| `tool` / `fixture` | at most 8 values, very long holds (>= 8 h) |
| `carrier` | a fixed pool recurring regularly, each value with a fixed successor |
| `rfid_tag` | EPC hex |
| `asset` | few values, very long holds, a two-level code (`L1-M03`) |
| `station` | few values, numbers in steps of 10 (`OP10`, `OP20`), fixed successor |
| `location` | aisle-rack-level codes |
| `operator` | changes at shift boundaries (80 %+ within 15 min) |
| `order` (`sales_order`, `purchase_order`) | event-stamped, never revisited, changes together with a recurring id |
| `business_partner` (`customer`, `supplier`) | event-stamped, often revisited, changes together with an id that never repeats |
| `handling_unit` | SSCC |

**Cross-signal structure** runs among the screened identifiers only (at most 64),
O(signals² × runs) with searchsorted on the run boundaries:

- **Nesting**: each presence of A lies inside one presence of B, A has at least 1.5×
  as many presences, and B's changes coincide with a change of A. The closest
  container is recorded as `parent_uuid`, and the depth is evidence (serial ⊂ batch ⊂
  material). Resources and people (tools, carriers, stations, operators, partners)
  are never nested: a long-running tool contains every batch without owning it.
- **Co-change**: two event-stamped ids that change together; the recurring one is a
  partner, the one that never repeats an order.
- **Counter alignment**: an id that changes once per counter increment is a serial.

The operator evidence uses the shift boundaries of `shift_definitions` (default
`ts_shape.utils.shifts.DEFAULT_SHIFTS`: 06:00, 14:00, 22:00), read on the local wall
clock for tz-aware data.

## Limits

- **Groups.** Without a format cue, the data cannot tell a sales order from a purchase
  order, or a customer from a supplier. The report gives the group, `order` or
  `business_partner`; a prefix family (`SO`, `PO`, `CUST`, `SUP`, ...) or a registered
  pattern separates them.
- **Types that need a format cue:** `rfid_tag` (EPC), `location` (aisle-rack-level),
  `handling_unit` (SSCC), `asset` (a two-level code), and `fixture` (it scores just
  below `tool` on the same behaviour). Register a pattern for your site's format.
- **Weak pairs** (separated by small margins of behaviour): recipe / material /
  article (nesting and formats separate them), batch / lot / work order (revisits and
  hold length), tool / fixture, station / carrier, order / handling unit, and a batch
  that happens to change at shift boundaries reads as an operator.
- **Encoding.** An id written once per change, without an empty reading after it, is
  read as held until the next value. Only ids cleared by an empty reading count as
  event-stamped.
- **Counters.** An integer that counts up by one with a fixed width of 5+ digits and
  never resets is read as a serial, not a counter.

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
