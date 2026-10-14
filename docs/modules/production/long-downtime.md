# LongDowntimeEvents

> Find downtimes longer than a threshold (default 3 h) in a run/stop signal, and count production between consecutive long stops.

**Module:** `ts_shape.events.production.long_downtime_events`  
**Guide:** [Production Monitoring](../../guides/production.md)

---

## When to Use

Use when you care about the long stops only: weekends, planned shutdowns, multi-hour breakdowns, a machine that sat idle overnight. The long downtimes split the history into production campaigns, and `count_events_between_gaps` tells you how much was produced in each one (parts, cycles, state changes). For short stops see [Micro-Stop Detection](micro-stops.md); for downtime by reason see [Downtime](downtime.md).

---

## Quick Example

```python
import pandas as pd

from ts_shape.events.production.long_downtime_events import LongDowntimeEvents

ld = LongDowntimeEvents(df, "press7:running")

stops = ld.detect_long_downtime(
    min_gap="3h",
    interval_end="next_change",                 # the state is logged on change only
    data_end=pd.Timestamp("2026-10-14 06:00"),
)

campaigns = ld.count_events_between_gaps(
    df,
    "press7:part_counter_increment",
    min_gap="3h",
    value_column="value_integer",
    aggregation="sum",
)
```

---

## Key Methods

| Method | Shape | Returns |
|---|---|---|
| `detect_long_downtime(min_gap, interval_end, data_end)` | interval | `start`, `end`, `duration_seconds`, `downtime_index`, `uuid`, `source_uuid`, `is_delta` |
| `count_events_between_gaps(production_df, production_uuid, min_gap, value_column, aggregation)` | interval | one row per window between consecutive long downtimes: `start`, `end`, `window_duration_seconds`, `event_count`, `downtime_index` (of the following downtime), `uuid`, `source_uuid`, `is_delta` |

## Notes

- **State.** By default `value_column` (`value_bool`) is cast to bool: True is running, False is down. For a numeric state, pass `value_range=(low, high)`: values inside the inclusive range count as running.
- **`interval_end`.** `"last_sample"` (default) ends a downtime at its last sample. A signal logged only on change has one row per stop, so every stop would have zero length and be missed. Use `"next_change"` for such signals; it closes each downtime at the next transition. A downtime still open at the end of the data is bounded by `data_end` (for example "now"), otherwise it is dropped.
- **Counting.** `count_events_between_gaps` calls `detect_long_downtime` with the default `interval_end="last_sample"` and needs at least two long downtimes. Only rows strictly between the end of one downtime and the start of the next are counted. `aggregation` is `"count"` (rows), `"sum"` (of `value_column`) or `"transitions"` (value changes, the first row included); anything else raises `ValueError`.
- The constructor raises `ValueError` if `state_uuid` is not in the DataFrame.
