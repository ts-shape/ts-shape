# Signal Normalization (Production)

> Turn any machine's part counters, status signals, cycle-time sources and identifier
> signals into canonical production events — so one report works across a whole fleet.

**Module:** `ts_shape.events.production.signals`
**Guide:** [Production Monitoring](../../guides/production.md)

---

## When to Use

Use this layer whenever you are reporting on **more than one machine** and they do not
share a signal convention — which, past a handful of assets, is always.

The rest of the production pack assumes a specific encoding: a boolean run flag, a
monotonic counter, a string part number. That assumption holds on one machine and
breaks across a plant. Across a fleet of a few hundred you will meet, all reporting
into the same dashboard:

| Quantity | Encodings actually in use |
|----------|---------------------------|
| Part count | 16-bit totalizer that wraps at 65535; totalizer reset every shift; `+1` per event; a part-done pulse meaning 8 parts; an absolute batch quantity written once |
| Quality | separate OK and NOK counters; total + NOK only; a boolean verdict per part; an integer defect code where `0` is good |
| Machine status | one `running` boolean; a bespoke PLC state code; `"AUTO"` / `"STOERUNG"`; four boolean lamps that overlap; a stacklight colour; a PackML state number |
| Cycle time | published in ms, s or min; derived from counter gaps; from one trigger's edges; from a start/end pair; from a part-present flag; only from run time ÷ parts |
| Context | order, batch, part, serial, customer part, variant, tool, operator — sparse, padded, placeholder-laden, and sometimes written *after* the part |

Normalize once, and every metric above — production, quality, availability, cycle time,
OEE — is computed against canonical quantities instead of against tags.

---

## Quick Example

```python
from ts_shape.events.production.signals import (
    CounterKind, CounterMode, CounterSpec, ProductionCounterNormalizer, Rollover,
)

counts = ProductionCounterNormalizer(
    df,
    counters=[
        # a 16-bit register that wraps, and a reject pulse
        CounterSpec(uuid="m1.total", kind=CounterKind.TOTAL,
                    mode=CounterMode.CUMULATIVE, rollover_at=Rollover.UINT16),
        CounterSpec(uuid="m1.reject", kind=CounterKind.NOK, mode=CounterMode.PULSE),
    ],
    machine_id="M1",
)

counts.counts_by_window("1h")   # qty_ok / qty_nok / qty_total + rates
counts.anomalies()              # every reset, wrap-around and glitch it interpreted
counts.diagnostics()            # per-counter health, one row per signal
```

Fleet scale — the same report over hundreds of machines, configured as data:

```python
from ts_shape.events.production.signals import FleetProductionReport, FleetSpec

# One row per signal, maintained by the people who know the machines.
fleet = FleetSpec.from_records(tag_table.to_dict("records"))
report = FleetProductionReport(fleet, df, until="2025-03-03 14:00")

report.production_by_window("1h")                # machine x hour
report.production_by_context(["order_number"])   # machine x order
report.oee("1D")                                 # availability x performance x quality
report.fleet_summary(group_by=["plant", "line"]) # roll-up
report.signal_coverage()                         # which tags never reported
report.anomalies()                               # resets, unmapped state codes
```

See [`examples/global_production_reporting_demo.py`](https://github.com/ts-shape/ts-shape/blob/main/examples/global_production_reporting_demo.py)
for a runnable four-machine fleet where no two machines are wired alike.

---

## Classes

| Class | Purpose |
|-------|---------|
| `ProductionCounterNormalizer` | Any part-counter encoding → canonical OK/NOK/total quantities, with reset, wrap-around and glitch handling |
| `MachineStatusNormalizer` | Any status encoding → canonical `MachineState` intervals, with debounce, staleness and unmapped-code reporting |
| `CycleTimeNormalizer` | Any cycle-time source → per-part durations, with unit inference, cavity division and artefact rejection |
| `ProductionContextResolver` | Identifier signals → context columns on any event frame, with validity limits and late binding |
| `FleetProductionReport` | Composes all four over a `FleetSpec` to produce one comparable frame per machine |

### Configuration types

`CounterSpec`, `StatusSpec`, `CycleSpec`, `ContextSpec`, `MachineSpec`, `FleetSpec` —
frozen dataclasses that validate themselves and round-trip through plain dicts, so a
fleet configuration is JSON/YAML, a config table, or a row per machine in a database.
`FleetSpec.from_records()` builds one straight from a tidy `(machine, role, uuid)` table.

### Canonical vocabularies

| Enum | Values |
|------|--------|
| `MachineState` | `producing`, `idle`, `setup`, `down`, `blocked`, `starved`, `planned_stop`, `off`, `unknown` |
| `CounterKind` | `ok`, `nok`, `total`, `rework`, `scrap` |
| `CounterMode` | `cumulative`, `incremental`, `pulse`, `quality_flag`, `defect_code`, `batch_total` |
| `StatusMode` | `boolean`, `code_map`, `one_hot`, `stacklight`, `packml` |
| `CycleMode` | `explicit`, `counter_delta`, `trigger_edge`, `trigger_pair`, `part_present`, `state_interval` |
| `ContextBinding` | `last_known`, `next_known`, `nearest` |

---

## Key Methods

### `ProductionCounterNormalizer`

| Method | Purpose | Returns |
|--------|---------|---------|
| `count_events()` | One point event per counted quantity; `quantity` is always a delta whatever the source encoding | Point frame |
| `counts_by_window(freq)` | `qty_ok` / `qty_nok` / `qty_total` / rates per window | Summary frame |
| `counts_by_context(dimensions)` | The same quantities per order, part or batch | Summary frame |
| `totals()` | One summary row for the whole range | Summary frame |
| `anomalies()` | Every reset, wrap-around and implausible jump, with severity | Point frame |
| `diagnostics()` | Per-counter health: readings, monotonicity, resets, contribution | Static frame |

### `MachineStatusNormalizer`

| Method | Purpose | Returns |
|--------|---------|---------|
| `status_intervals()` | Contiguous canonical state intervals, with the raw value kept for drill-down | Interval frame |
| `state_summary(freq)` | Seconds per state, `availability_pct`, `utilization_pct` | Summary frame |
| `downtime_intervals(min_duration)` | Only the intervals consuming planned busy time | Interval frame |
| `transitions()` | Point events at each state change, with the previous duration | Point frame |
| `state_at(timestamps)` | As-of state lookup, used to gate other metrics | Series |
| `unmapped_values()` | Raw state codes no mapping covers | Static frame |

### `CycleTimeNormalizer`

| Method | Purpose | Returns |
|--------|---------|---------|
| `cycles()` | One record per accepted cycle: per-part time, raw time, parts per cycle | Interval frame |
| `rejected()` | Candidates filtered out, with the reason (`too_long`, `gated_state`, …) | Interval frame |
| `cycle_statistics(group_by)` | Count, mean, median, p05/p95, target deviation | Summary frame |
| `by_window(freq)` | The same statistics per window | Summary frame |
| `deviations(target)` | Cycles off nameplate, slow and fast | Point frame |

### `ProductionContextResolver`

| Method | Purpose | Returns |
|--------|---------|---------|
| `attach(events)` | Add one column per dimension to any event frame, preserving row order | Same frame + columns |
| `segments(dimensions)` | When each order / batch / part combination ran | Interval frame |
| `changes()` | Changeover events per dimension | Point frame |
| `coverage()` | Fill rate and distinct values per dimension | Static frame |

### `FleetProductionReport`

| Method | Purpose | Returns |
|--------|---------|---------|
| `production_by_window(freq)` | Canonical quantities per machine and window, one stable schema | Summary frame |
| `production_by_context(dimensions)` | The same per order, part or customer part, across the fleet | Summary frame |
| `status_by_window(freq)` | Per-state seconds and availability per machine | Summary frame |
| `cycle_by_window(freq)` / `cycle_by_context(dimensions)` | Cycle statistics per machine | Summary frame |
| `oee(freq)` | Availability × performance × quality from canonical inputs | Summary frame |
| `machine_summary()` | One row per machine: performance *and* instrumentation facts | Static frame |
| `fleet_summary(group_by)` | Roll-up by plant, line or any asset attribute | Static frame |
| `signal_coverage()` | Every configured signal, and whether data arrived | Static frame |
| `anomalies()` | Fleet-wide data-quality findings, worst first | Static frame |
| `normalizers(machine_id)` | The normalizer bundle the report itself used, for drill-down | `MachineNormalizers` |

---

## Tips & Hints

!!! warning "Unknown is not zero"
    A machine that counts a total but measures no quality reports `qty_ok` as `NaN`,
    not as the total — and its `oee_pct` is `NaN`, not an optimistic number. This is
    deliberate: an OEE assembled from guessed inputs gets compared against machines
    whose inputs were real, and the badly instrumented machines end up looking like
    the best performers. Pass `assume_all_ok=True` per machine, deliberately, when it
    is actually true.

!!! tip "Set `stale_after` on every status signal"
    Without it, a dead gateway leaves the last known state running to the end of the
    report, and a machine that "produced" through a two-day outage is the single most
    common cause of a suspiciously good availability figure.

!!! tip "Set `max_seconds` on any derived cycle time"
    Edge- and counter-derived cycles measure the gap between parts, so the gap across
    a lunch break arrives as one 45-minute "cycle". A handful of them dominates every
    mean. Check `rejected()` to see what was removed and why.

!!! tip "Bound context validity with `valid_for`"
    An order number written once at order start and forward-filled without limit
    attaches last month's order to today's parts. And use
    `ContextBinding.NEXT_KNOWN` for anything the machine publishes *after* the part it
    identifies — a serial number, a measured weight, a test verdict — otherwise every
    value lands on the following part, off by one, systematically.

!!! info "Commission with the diagnostics, not the numbers"
    At fleet scale a wrong tag is far more common than a wrong metric, and it is
    invisible in a production report — the machine simply shows no output. Run
    `signal_coverage()`, `anomalies()` and `unmapped_values()` before trusting any
    figure.

!!! info "Related modules"
    - [OEE Calculator](oee-calculator.md) — single-machine OEE from explicit inputs
    - [Machine State](machine-state.md) — the simpler boolean-only state detector
    - [Part Tracking](part-tracking.md) / [Cycle Time](cycle-time.md) — single-encoding equivalents
    - [Quality Tracking](quality-tracking.md) — shift-level NOK and first-pass yield

---

## See Also

- [Production Monitoring Guide](../../guides/production.md)
- [API Reference](../../reference/ts_shape/events/production/signals/index.md)
