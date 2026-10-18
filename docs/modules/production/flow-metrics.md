# FlowMetricsEvents

Work in process, throughput and lead time between an entry and an exit signal (two boolean signals such as light barriers or part-present bits), checked against Little's Law (`WIP = throughput × lead time`).

`ts_shape.events.production.flow_metrics` · [Production guide](../../guides/production.md) · [API reference](../../reference/ts_shape/events/production/flow_metrics.md)

## Example

```python
import numpy as np
import pandas as pd
from ts_shape.events.production.flow_metrics import FlowMetricsEvents

t = pd.date_range("2024-01-01 06:00", periods=480, freq="30s")
pulse = np.tile([True, False], 240)  # one part per minute
df = pd.concat([
    pd.DataFrame({"systime": t, "uuid": "line3:infeed", "value_bool": pulse}),
    pd.DataFrame({"systime": t + pd.Timedelta("5min"), "uuid": "line3:outfeed", "value_bool": pulse}),
], ignore_index=True)

flow = FlowMetricsEvents(df, entry_uuid="line3:infeed", exit_uuid="line3:outfeed")
summary = flow.flow_summary(value_add_seconds=120, window="1h")
print(summary[["start", "wip_mean", "throughput_per_hour", "consistency_factor"]])
```

## Methods

| Method | Returns |
|---|---|
| `wip_over_time(window='1h')` | per window: `wip_mean` (time-weighted), `wip_max`, `wip_min` |
| `throughput(window='1h')` | per window: `units_out`, `throughput_per_hour` |
| `lead_time()` | one row per unit: `systime` (exit time), `lead_time_seconds`, `unit_index` |
| `flow_summary(value_add_seconds=None, window='1h')` | per window: `wip_mean`, `throughput_per_hour`, `lead_time_mean_seconds`, `littles_law_lead_time_seconds`, `consistency_factor` (+ `process_cycle_efficiency_pct`) |

## Notes

- Each rising edge of the entry or exit signal is one unit. WIP is the running count of entries minus exits, starting at 0 at the first edge, so units already inside when the data begins are not counted.
- Lead time is FIFO-matched: the nth entry is paired with the nth exit. This is only correct for a first-in-first-out process without parts removed between entry and exit.
- `consistency_factor` = measured mean lead time / Little's Law lead time (WIP / throughput). It is close to 1 for a stable FIFO process; large deviations point to missed edges, unbalanced counts or a process that is filling or draining.
- `process_cycle_efficiency_pct` = value-add time / mean lead time × 100. `value_add_seconds` takes seconds or an offset string such as `"2min"`.
- Throughput and lead time are bucketed by exit time. All method arguments are keyword-only.
