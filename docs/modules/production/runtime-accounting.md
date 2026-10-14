# RuntimeAccountingEvents

> Operating hours from one run signal: total run and idle time, utilization, start count, run time per window and a cumulative hour meter.

**Module:** `ts_shape.events.production.runtime_accounting`  
**Guide:** [Production Monitoring](../../guides/production.md)

---

## When to Use

Use when you need absolute run time rather than a percentage: hours since the last service for a compressor, daily operating hours per machine for cost allocation, starts per day for a motor whose wear depends on the number of starts, or a software hour meter for equipment without a physical one. For duty-cycle percentages and on/off cycle counts see [Duty Cycle](duty-cycle.md).

---

## Quick Example

```python
from ts_shape.events.production.runtime_accounting import RuntimeAccountingEvents

rt = RuntimeAccountingEvents(df, "comp2:motor_running")

rt.runtime_summary(max_gap="15min")
rt.runtime_per_window("1D", max_gap="15min")

meter = rt.operating_hours_meter("1h", max_gap="15min")
print(meter[["start", "cumulative_run_hours"]].tail())
```

---

## Key Methods

| Method | Shape | Returns |
|---|---|---|
| `runtime_summary(max_gap)` | summary | one row: `start`, `end`, `duration_seconds`, `run_seconds`, `run_hours`, `idle_seconds`, `start_count`, `longest_run_seconds`, `mean_run_seconds`, `utilization_pct` |
| `runtime_per_window(window, max_gap)` | summary | per window: `start`, `end`, `duration_seconds`, `run_seconds`, `run_hours`, `start_count`, `utilization_pct` |
| `operating_hours_meter(window, max_gap)` | summary | per window: `start`, `end`, `duration_seconds`, `run_seconds`, `cumulative_run_hours` |

## Notes

- Each sample's state holds until the next sample; the last sample contributes no time. A sample's whole duration is booked to the window it starts in, even when it reaches into the next one.
- Pass `max_gap` (keyword-only) for signals with logging outages: a gap to the next sample longer than `max_gap` counts as neither run nor idle time. Without it, an outage that began while the machine was running is booked as run time.
- `utilization_pct` = run time / (run + idle) time with data, not / calendar time.
- `start_count` counts transitions to running. A run already in progress at the first sample counts as a start.
- The meter starts at 0 at the beginning of the data and only covers windows that contain samples. Add the physical meter reading at that time as an offset if you want to match it.
