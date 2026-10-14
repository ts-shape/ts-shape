# FlowMetricsEvents

> Work in process, throughput and lead time between an entry and an exit signal, tied together by Little's Law.

**Module:** `ts_shape.events.production.flow_metrics`  
**Guide:** [Production Monitoring](../../guides/production.md)

---

## When to Use

Use when you can see units entering and leaving a process (a line, a cell, a buffer, an oven) as two boolean signals, such as light barriers or part-present bits, and want the flow metrics an industrial engineer works with:

- how much WIP sits in the process, hour by hour;
- how many units come out per hour;
- how long each unit takes from entry to exit;
- whether these agree with Little's Law (`WIP = throughput × lead time`), and how much of the lead time is value-add (Process Cycle Efficiency).

---

## Quick Example

```python
from ts_shape.events.production.flow_metrics import FlowMetricsEvents

flow = FlowMetricsEvents(
    df,
    entry_uuid="line3:infeed_part_present",
    exit_uuid="line3:outfeed_part_present",
)

flow.wip_over_time(window="1h")
flow.throughput(window="1h")
flow.lead_time()

summary = flow.flow_summary(value_add_seconds=120, window="1h")
print(summary[["start", "wip_mean", "throughput_per_hour", "consistency_ratio"]])
```

---

## Key Methods

| Method | Shape | Returns |
|---|---|---|
| `wip_over_time(window)` | summary | `start`, `end`, `duration_seconds`, `wip_mean` (time-weighted), `wip_max`, `wip_min` |
| `throughput(window)` | summary | `start`, `end`, `duration_seconds`, `units_out`, `throughput_per_hour` |
| `lead_time()` | point | `systime` (exit time), `uuid`, `source_uuid`, `lead_time_seconds`, `unit_index` |
| `flow_summary(value_add_seconds, window)` | summary | `wip_mean`, `throughput_per_hour`, `lead_time_mean_seconds`, `littles_law_lead_time_seconds`, `consistency_ratio`, plus `process_cycle_efficiency_pct` when `value_add_seconds` is given |

## Notes

- Each rising edge of the entry or exit signal is one unit. WIP is the running count of entries minus exits, starting at 0 at the first edge, so units already inside the process when the data begins are not counted.
- Lead time is FIFO-matched: the nth entry is paired with the nth exit. This is only correct for a first-in-first-out process without scrap removed between entry and exit.
- `consistency_ratio` = measured mean lead time / Little's Law lead time (WIP / throughput). It is close to 1 for a stable FIFO process; large deviations point to missed edges, unbalanced counts or a process that is filling or draining.
- `process_cycle_efficiency_pct` = value-add time / mean lead time × 100. `value_add_seconds` is keyword-only and takes seconds or a pandas offset string such as `"2min"`.
- Throughput and lead time are bucketed by exit time.
