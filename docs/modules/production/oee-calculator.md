# OEEEvents

Daily OEE = Availability × Performance × Quality from a boolean run signal, a part counter with an ideal cycle time, and optional total/reject counters.

`ts_shape.events.production.oee_calculator` · [Production guide](../../guides/production.md) · [API reference](../../reference/ts_shape/events/production/oee_calculator.md)

## Example

```python
import numpy as np
import pandas as pd
from ts_shape.events.production.oee_calculator import OEEEvents

t = pd.date_range("2024-01-01 06:00", periods=480, freq="1min")
df = pd.concat([
    pd.DataFrame({"systime": t, "uuid": "line1:running", "value_bool": np.arange(480) % 60 < 52}),
    pd.DataFrame({"systime": t, "uuid": "line1:parts", "value_integer": np.arange(480)}),
    pd.DataFrame({"systime": t, "uuid": "line1:rejects", "value_integer": np.arange(480) // 40}),
], ignore_index=True)

daily = OEEEvents(df).oee(run_state_uuid="line1:running", counter_uuid="line1:parts",
                          ideal_cycle_time=55.0, total_uuid="line1:parts",
                          reject_uuid="line1:rejects", max_gap="15min")
print(daily[["start", "availability_pct", "performance_pct", "quality_pct", "oee_pct"]])
```

## Methods

| Method | Returns (one row per day) |
|---|---|
| `availability(run_state_uuid, planned_time_hours=None, max_gap=None)` | `run_seconds`, `planned_seconds`, `availability_pct`, `excluded_seconds` |
| `performance(counter_uuid, ideal_cycle_time, run_state_uuid=None)` | `actual_parts`, `ideal_parts`, `run_seconds`, `performance_pct` (+ `resets`) |
| `quality(total_uuid, reject_uuid)` | `total_parts`, `reject_parts`, `good_parts`, `quality_pct` (+ `resets`) |
| `oee(run_state_uuid, counter_uuid, ideal_cycle_time, total_uuid=None, reject_uuid=None, planned_time_hours=None)` | `availability_pct`, `performance_pct`, `quality_pct`, `oee_pct`, then the basis: `run_seconds`, `planned_seconds`, `excluded_seconds`, `actual_parts`, `ideal_parts`, `total_parts`, `reject_parts`, `good_parts`, `resets` |

## Notes

- Rows are calendar days: `start` is midnight, `duration_seconds` the real day length (23 h or 25 h on DST days with tz-aware data).
- Time is split at midnight (ISO 22400-2 time model): each reading's state holds until the next reading, and a run across midnight counts in both days.
- Availability = run time / planned time. Without `planned_time_hours` the planned time is the time the state signal covers that day. A day the state signal does not cover has NaN availability, performance and OEE (its run time is unknown).
- Performance = actual parts × `ideal_cycle_time` (seconds per part) / run time. Without `run_state_uuid`, `performance` uses the time the counter readings cover that day as run time.
- A percentage above 100 % (availability over a fixed planned time, performance) is reported as computed, not clipped, with a `DataQualityWarning` naming the day: the inputs disagree.
- `max_gap` (e.g. `"15min"`) treats a longer gap between readings as no data and removes it from run and planned time; without it, a logging gap while the last state was running counts as run time. `excluded_seconds` is the removed time.
- Counters are reset-aware by default (`handle_resets=True`): a decreasing counter is a reset and the production after it is counted; the increment between two readings is spread over the time between them and split at midnight in proportion to time, so parts belong to the day they were produced in (part counts can be fractional).
- Without `total_uuid`/`reject_uuid`, quality is 100 % and the quality basis columns are NaN.
