# LineBalancingEvents

Station cycle times, line balance efficiency, balance delay, smoothness index, the bottleneck station and a Yamazumi table, from one cycle-completion trigger (boolean) per station.

`ts_shape.events.production.line_balancing` · [Production guide](../../guides/production.md) · [API reference](../../reference/ts_shape/events/production/line_balancing.md)

## Example

```python
import numpy as np
import pandas as pd
from ts_shape.events.production.line_balancing import LineBalancingEvents

t = pd.date_range("2024-01-01 06:00", periods=3600, freq="1s")
stations = {"asm1:st10_done": "St 10 Press-fit", "asm1:st20_done": "St 20 Screwing",
            "asm1:st30_done": "St 30 Leak test"}
df = pd.concat([pd.DataFrame({"systime": t, "uuid": u, "value_bool": np.arange(3600) % k == 0})
                for u, k in zip(stations, [50, 55, 45])], ignore_index=True)

lb = LineBalancingEvents(df, station_uuids=stations)
print(lb.balance_metrics(takt_time="60s", window="1h", require_all_stations=True))
yamazumi = lb.yamazumi(demand=480, available_time="8h")  # takt 60 s
```

## Methods

| Method | Returns |
|---|---|
| `station_cycle_times(window='1h')` | per window and station: `station_name`, `cycle_time_mean`, `cycle_time_median`, `cycle_time_std`, `cycle_count` |
| `balance_metrics(takt_time=None, demand=None, available_time=None, window='1h', require_all_stations=False)` | per window: `n_stations`, `bottleneck_uuid`, `bottleneck_cycle_time`, `takt_seconds`, `balance_efficiency_pct`, `balance_delay_pct`, `smoothness_index`, `theoretical_min_stations` (+ `stations_missing`) |
| `yamazumi(takt_time=None, demand=None, available_time=None)` | per station over all data: `cycle_time_mean`, `takt_seconds`, `loading_pct`, `idle_to_takt_seconds`, `is_bottleneck` |

## Notes

- Cycle time is the time between consecutive rising edges of a station's trigger, assigned to the window of the later completion. It includes waiting and blocked time, so it is the station's effective cycle, not its work content.
- Balance efficiency = Σ station cycle times / (`n_stations` × bottleneck cycle time); balance delay = 100 % − efficiency.
- Smoothness index = √Σ (bottleneck − station time)²; 0 is a perfectly balanced line.
- Theoretical minimum stations = ⌈Σ station times / takt⌉; `NaN` without a takt.
- Takt comes from `takt_time` (seconds or an offset string such as `"55s"`), else from `available_time / demand`. All method arguments are keyword-only.
- By default `n_stations` counts only stations that completed a cycle in the window, so a stopped station drops out and the line looks balanced. `require_all_stations=True` divides by all configured stations and adds `stations_missing`.
- `station_uuids` maps trigger UUID to station name; give it in line order, as `yamazumi` keeps that order.
