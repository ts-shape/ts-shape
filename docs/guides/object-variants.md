# Object signal variants (test matrix)

Machine builders do not agree on how objects appear in the time series. A
material number, a serial, a cycle or a roll cut can be written in many ways.
This page lists the variants `identify_objects` and `detect_objects` are
tested against. Each variant is one small, labelled scenario; every scenario
states what the library must find and links to the test that checks it.

Every signal has one uuid and one data type (`value_bool`, `value_integer`,
`value_double` or `value_string`). Most historians add `is_delta`, which is
`True` on the row where the value was written by the machine and `False` on
rows that repeat the held value.

## 1. Process types

| Code | Process | Produced unit | Typical boundary | Typical inputs |
|---|---|---|---|---|
| P-D | Discrete, cycle machine (press, welding, assembly, CNC, test bench) | one part per cycle (serial) | a cycle signal | material / purchased-part / pre-product numbers, tool, program, operator |
| P-DN | Discrete, n parts per cycle (multi-cavity mould, nest) | n serials per cycle | cycle signal, n scans | as P-D |
| P-C | Continuous (extrusion, coating, wire drawing, film) | roll, coil, reel, length section | cut / reel change | material (often constant for hours or days), additive lots, order |
| P-B | Batch (mixing, curing, heat treatment) | batch / charge | batch start and end | recipe, raw-material lots |
| P-L | Line with several stations | the same serial at station after station | per-station cycle | per-station resources |

## 2. Variant dimensions

### ID: how an identifier value is written

| Code | Variant | What the data looks like |
|---|---|---|
| ID1 | hold + repeat | written once (`is_delta=True`), repeated every 1–5 s (`is_delta=False`) until the next value |
| ID2 | change only | one row per new value, nothing in between |
| ID3 | pulse | the value is present for one or a few samples, then the signal is empty / `""` / `0` until the next part |
| ID4 | cleared between units | the value is held during the unit and cleared (`""`, `0`, `NULL`) when the unit leaves |
| ID5 | data type | text, integer, float with `.0`, zero-padded text (`000123`), GS1 / prefixed formats |
| ID6 | same-value re-setup | the same number is set again; only `is_delta=True` shows the new setup |
| ID7 | no `is_delta` column | changes are known only from value changes and sampling |
| ID8 | misreads | `NOREAD`, `????????`, `0`, `65535`, truncated reads in between |
| ID9 | coarse polling | the value is read every 15–60 min |

### CY: how a cycle (discrete unit boundary) is written

| Code | Variant | What the data looks like |
|---|---|---|
| CY1 | level bool | `True` for the whole cycle (e.g. door closed, clamp closed, in-process) |
| CY2 | inverted level | `False` during the cycle (e.g. door open = `False` while running) |
| CY3 | start and end pulses | two bools: a short `True` at start, another at end |
| CY4 | part counter | an integer counter that increments at the end of each cycle (resets now and then) |
| CY5 | step / state number | an integer or text step (`0` idle, `10..90` steps) or state (`IDLE`, `RUN`) |
| CY6 | cycle-time value | a `value_double` with the last cycle's duration, written at cycle end |
| CY7 | no cycle signal | only the serial scan marks the unit |

### SC: where the serial is scanned relative to the cycle

| Code | Variant |
|---|---|
| SC1 | at cycle start |
| SC2 | mid-cycle |
| SC3 | at cycle end (label printed / marked after processing) |
| SC4 | before the cycle starts (scanned while loading, door still open) |
| SC5 | after the cycle ended (scanned when unloading) |
| SC6 | no serial: the unit exists only as a counter increment |

### CO: how a continuous unit boundary is written

| Code | Variant | What the data looks like |
|---|---|---|
| CO1 | unit id | a roll / coil / reel number that changes at the cut |
| CO2 | length counter reset | a metre or length counter that drops to 0 at the cut, no unit id |
| CO3 | cut pulse | a bool pulse at the cut |
| CO4 | no unit | only material and order; the order is the unit |

### MAT: how inputs (material, purchased part, pre-product) relate to units

| Code | Variant |
|---|---|
| MAT1 | constant over the whole recording |
| MAT2 | changes rarely (about once a day), at an order change |
| MAT3 | changes at batch / roll boundaries |
| MAT4 | several input slots (bill of materials: component 1..n), each its own signal |
| MAT5 | changes while a unit is in progress (splice in a continuous line): the unit uses both |
| MAT6 | an input lot (incoming coil, raw-material lot) changes on its own clock, independent of output units |
| MAT7 | input set before the first unit and never again written (only one `is_delta=True` row, then repeats) |

### RES: resources

| Code | Variant |
|---|---|
| RES1 | tool / mould number, constant for long periods |
| RES2 | program / recipe number (integer) |
| RES3 | operator badge, changes at shift change |
| RES4 | tool change shown only by a tool-life counter reset |

### TOP: topology

| Code | Variant |
|---|---|
| TOP1 | one machine, one station |
| TOP2 | several stations in a line, the same serial passes each station |
| TOP3 | parallel cells sharing a station name |
| TOP4 | two independent lines in one frame (links must not cross lines) |

## 3. What must be found

For every scenario the expected result is stated in these terms:

- **Screening and type**: each object signal is an identifier of the right type
  (serial, material, roll, batch, order, tool, operator); every non-object
  signal (cycle bool, counter, step, cycle time, states) is not.
- **Presences**: one presence per unit. A serial's presence is its cycle when
  a cycle signal exists (CY1–CY6), from scan to the next unit otherwise.
- **Cycle signal**: the cycle signal of each serial (`cycle_uuid`).
- **Usage links**: every serial / roll / batch links to each input and
  resource held during its presence (`made_from`, `made_with`, `made_by`),
  including constant inputs.
- **Production nesting**: unit → batch / order / material where the data
  shows it (`part_of`).
- **No false links**: nothing across lines, nothing from states, shift
  labels or clock-driven ids.

## 4. Scenario set

The full cross product is far too large. The scenarios below cover every
variant of every dimension at least once and every pair of the dimensions
that interact (ID × CY, CY × SC, MAT × process type) at least once. Each row
is one generated scenario with ground-truth labels.

| # | Process | ID | CY / CO | SC | MAT | RES | TOP |
|---|---|---|---|---|---|---|---|
| D01 | P-D | ID1 | CY1 | SC2 | MAT1 ×2 | RES1 | TOP1 |
| D02 | P-D | ID2 | CY2 | SC1 | MAT2 | RES2 | TOP1 |
| D03 | P-D | ID3 | CY3 | SC3 | MAT4 ×3 | RES3 | TOP1 |
| D04 | P-D | ID4 | CY4 | SC4 | MAT1 | RES1 | TOP1 |
| D05 | P-D | ID5 (int) | CY5 | SC5 | MAT6 | RES4 | TOP1 |
| D06 | P-D | ID1 + ID8 | CY6 | SC2 | MAT7 | RES2 | TOP1 |
| D07 | P-D | ID7 | CY1 | SC2 | MAT2 | RES3 | TOP1 |
| D08 | P-D | ID1 | CY7 | – | MAT1 | RES1 | TOP1 |
| D09 | P-D | – | CY4 | SC6 | MAT1 | RES1 | TOP1 |
| D10 | P-DN | ID3 ×4 | CY1 | SC2 | MAT1 | RES1 | TOP1 |
| D11 | P-L | ID1 | CY1 per station | SC1 | MAT4 per station | RES1 | TOP2 |
| D12 | P-L | ID1 | CY2 | SC2 | MAT1 | RES3 | TOP3 |
| D13 | P-D | ID1 | CY1 | SC2 | MAT1 | RES1 | TOP4 |
| D14 | P-D | ID9 | CY4 | SC2 | MAT2 | RES2 | TOP1 |
| C01 | P-C | ID1 | CO1 | – | MAT1 | RES2 | TOP1 |
| C02 | P-C | ID6 | CO2 | – | MAT2 (re-setup same value) | RES3 | TOP1 |
| C03 | P-C | ID2 | CO3 | – | MAT5 (splice) | RES1 | TOP1 |
| C04 | P-C | ID1 | CO4 | – | MAT2 + MAT6 | RES3 | TOP1 |
| C05 | P-C | ID7 | CO1 | – | MAT1 | – | TOP4 |
| C06 | P-C | ID9 | CO1 | – | MAT3 | RES2 | TOP1 |
| B01 | P-B | ID1 | batch start/end bool | – | MAT4 (raw lots) | RES2 | TOP1 |
| B02 | P-B | ID4 | step number | – | MAT3 | RES3 | TOP1 |

Each scenario's status (supported, partly supported with numbers, not
supported with the reason) is reported by the variant test suite and listed
in the table in the next section once it runs.

## 5. Status

To be filled from `tests/test_object_variants.py`.
