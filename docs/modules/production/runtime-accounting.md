# RuntimeAccountingEvents

Operating hours from one boolean run signal: run and idle time, utilization, start count, run time per window and a cumulative hour meter. Use it for hours since service, operating hours per machine or starts per day; for duty-cycle percentages and on/off cycle counts see [DutyCycleEvents](duty-cycle.md).

`ts_shape.events.production.runtime_accounting` · [Production guide](../../guides/production.md) · [API reference](../../reference/ts_shape/events/production/runtime_accounting.md)

## Example

```python
import numpy as np
import pandas as pd
from ts_shape.events.production.runtime_accounting import RuntimeAccountingEvents

df = pd.DataFrame({
    "systime": pd.date_range("2024-01-01", periods=2880, freq="1min"),
    "uuid": "comp2:motor_running",
    "value_bool": np.arange(2880) % 60 < 45,  # 45 min on, 15 min off
})
rt = RuntimeAccountingEvents(df, "comp2:motor_running")
print(rt.runtime_summary(max_gap="15min"))
daily = rt.runtime_by_window(window="1D", max_gap="15min")
meter = rt.operating_hours_meter(window="1h", max_gap="15min")
```

## Methods

All arguments are keyword-only.

| Method | Returns |
|---|---|
| `runtime_summary(max_gap=None)` | one row: `run_seconds`, `run_hours`, `idle_seconds`, `start_count`, `longest_run_seconds`, `mean_run_seconds`, `utilization_pct` |
| `runtime_by_window(window='1D', max_gap=None)` | per window: `run_seconds`, `run_hours`, `start_count`, `utilization_pct` |
| `operating_hours_meter(window='1h', max_gap=None)` | per window: `run_seconds`, `cumulative_run_hours` |

## Notes

- Each sample's state holds until the next sample; the last sample contributes no time. Per window, a sample's interval is split at the window boundaries (ISO 22400-2 time model), so a run across midnight counts in both days.
- `max_gap`: a gap to the next sample longer than this counts as neither run nor idle time. Without it, a logging outage that began while running is booked as run time.
- `utilization_pct` = run time / (run + idle) time with data, not / calendar time.
- `start_count` counts transitions to running; a run already in progress at the first sample counts as a start.
- The meter starts at 0 at the beginning of the data and covers every window that holds a sample or observed time. Add the physical meter reading at that time as an offset to match it.
