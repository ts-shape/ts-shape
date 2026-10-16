# LongDowntimeEvents

Downtimes longer than a threshold (default 3 h) in a run/stop signal, and the production counted between consecutive long stops. Use it for weekends, planned shutdowns and multi-hour breakdowns; for short stops see [MicroStopEvents](micro-stops.md), for downtime by reason [DowntimeEvents](downtime.md).

`ts_shape.events.production.long_downtime_events` · [Production guide](../../guides/production.md) · [API reference](../../reference/ts_shape/events/production/long_downtime_events.md)

## Example

```python
import pandas as pd
from ts_shape.events.production.long_downtime_events import LongDowntimeEvents

state = pd.DataFrame({  # logged on change only
    "systime": pd.to_datetime(["2024-01-01 06:00", "2024-01-01 14:00", "2024-01-02 06:00",
                               "2024-01-02 14:00", "2024-01-03 06:00"]),
    "uuid": "press7:running", "value_bool": [True, False, True, False, True]})
parts = pd.DataFrame({"systime": pd.date_range("2024-01-02 06:00", periods=480, freq="1min"),
                      "uuid": "press7:part_done", "value_integer": 1})

df = pd.concat([state, parts], ignore_index=True)
ld = LongDowntimeEvents(df, "press7:running")
result = ld.detect_long_downtime(min_gap="3h", data_end=pd.Timestamp("2024-01-03 08:00"))
print(result[["start", "end", "duration_seconds"]])
campaigns = ld.count_events_between_gaps(df, "press7:part_done", min_gap="3h", aggregation="sum")
```

## Methods

| Method | Returns |
|---|---|
| `detect_long_downtime(min_gap='3h', interval_end='next_change', data_end=None)` | `start`, `end`, `duration_seconds`, `downtime_index` |
| `count_events_between_gaps(production_df, production_uuid, min_gap='3h', value_column='value_integer', aggregation='count')` | one row per window between consecutive long downtimes: `start`, `end`, `window_duration_seconds`, `event_count`, `downtime_index` (of the following downtime) |

## Notes

- State: `value_column` (`value_bool`) is cast to bool, True = running. For a numeric state pass `value_range=(low, high)`: values inside the inclusive range count as running.
- `interval_end="next_change"` (default) closes each downtime at the next transition, which works for dense and change-only signals. A downtime still open at the end of the data is bounded by `data_end` (for example "now"), otherwise it is dropped. `"last_sample"` ends a downtime at its last sample, so on a change-only signal every stop is a zero-length row and is missed.
- `count_events_between_gaps` needs at least two long downtimes and counts only rows strictly between the end of one and the start of the next. `aggregation` is `"count"` (rows), `"sum"` (of `value_column`) or `"transitions"` (value changes, first row included); anything else raises `ValueError`.
- The constructor raises `ValueError` if `run_state_uuid` is not in the DataFrame.
