# SetupTimeEvents

Setup (changeover) durations, statistics per product transition and trends, from a machine-state string signal in which setup rows carry a given value (default `"Setup"`).

`ts_shape.events.production.setup_time_tracking` · [Production guide](../../guides/production.md) · [API reference](../../reference/ts_shape/events/production/setup_time_tracking.md)

## Example

```python
import pandas as pd
import ts_shape
from ts_shape.events.production.setup_time_tracking import SetupTimeEvents

df = pd.concat([
    ts_shape.make_id_signal("line1:state", ["Running", "Setup", "Running", "Setup", "Running"],
                            hold=60, freq="1min", start="2024-01-01 06:00"),
    ts_shape.make_id_signal("line1:part_id", ["PN-100", "PN-200", "PN-300"],
                            hold=100, freq="1min", start="2024-01-01 06:00"),
], ignore_index=True)

st = SetupTimeEvents(df)
result = st.setup_durations(run_state_uuid="line1:state", setup_value="Setup")
print(result[["start", "end", "duration_minutes", "shift"]])
by_product = st.setup_by_product(run_state_uuid="line1:state", part_id_uuid="line1:part_id")
stats = st.setup_statistics(run_state_uuid="line1:state")
```

## Methods

| Method | Returns |
|---|---|
| `setup_durations(run_state_uuid, setup_value='Setup')` | one row per setup: `start`, `end`, `duration_seconds`, `date`, `shift` |
| `setup_by_product(run_state_uuid, part_id_uuid, setup_value='Setup')` | per `from_product` → `to_product`: `avg_seconds`, `min_seconds`, `max_seconds`, `count` |
| `setup_statistics(run_state_uuid, setup_value='Setup')` | one row: `total_setups`, `total_seconds`, `avg_seconds`, `median_seconds`, `std_seconds`, `of_available_time_pct` |
| `setup_trend(run_state_uuid, setup_value='Setup', window='1W')` | per `period`: `avg_setup_seconds`, `setup_count`, `total_setup_seconds` |

Every `*_seconds` column has a `*_minutes` twin.

## Notes

- A setup is a contiguous run of samples equal to `setup_value`. It ends at the next non-setup sample, or at its last sample at the end of the data; zero-length setups are dropped.
- If the state signal encodes both internal (machine stopped) and external setup, filter to the internal setup value to measure lost machine time.
- `setup_trend` with `window='1W'` bins Monday to Monday; `period` is the pandas bin label (the week's Sunday).
- Default shifts are 06-14, 14-22, 22-06; a night shift belongs to its start date (`production_day=True`).
