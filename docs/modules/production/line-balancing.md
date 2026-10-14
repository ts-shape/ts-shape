# LineBalancingEvents

> Line balance efficiency, balance delay, smoothness index, the bottleneck station and a Yamazumi table, from one cycle-completion signal per station.

**Module:** `ts_shape.events.production.line_balancing`  
**Guide:** [Production Monitoring](../../guides/production.md)

---

## When to Use

Use on a paced assembly or production line where each station signals the end of its cycle (a boolean trigger, e.g. "part released"). It answers the classic industrial-engineering questions:

- What is each station's actual cycle time, hour by hour?
- How well balanced is the line, and which station is the bottleneck?
- Against a takt time (given directly or from demand and available time): how loaded is each station, and how many stations would the work content need at minimum?

---

## Quick Example

```python
from ts_shape.events.production.line_balancing import LineBalancingEvents

lb = LineBalancingEvents(
    df,
    station_uuids={
        "asm1:st10_cycle_done": "St 10 Press-fit",
        "asm1:st20_cycle_done": "St 20 Screwing",
        "asm1:st30_cycle_done": "St 30 Leak test",
    },
)

lb.station_cycle_times(window="1h")
lb.balance_metrics(takt_time="55s", window="1h", require_all_stations=True)

# Takt from demand: 480 units in an 8 h shift = 60 s
lb.yamazumi(demand=480, available_time="8h")
```

---

## Key Methods

| Method | Shape | Returns |
|---|---|---|
| `station_cycle_times(window)` | summary | per window and station: `start`, `end`, `duration_seconds`, `uuid`, `station_name`, `cycle_time_mean`, `cycle_time_median`, `cycle_time_std`, `cycle_count` |
| `balance_metrics(takt_time, demand, available_time, window, require_all_stations)` | summary | per window: `n_stations`, `bottleneck_uuid`, `bottleneck_cycle_time`, `takt_seconds`, `balance_efficiency_pct`, `balance_delay_pct`, `smoothness_index`, `theoretical_min_stations` (+ `stations_missing`) |
| `yamazumi(takt_time, demand, available_time)` | static | per station over all data: `uuid`, `station_name`, `cycle_time_mean`, `takt_seconds`, `loading_pct`, `idle_to_takt_seconds`, `is_bottleneck` |

## Definitions

- **Cycle time** is the time between consecutive rising edges of a station's trigger, assigned to the window of the later completion. Waiting and blocked time are included, so this is the station's effective cycle, not its pure work content.
- **Balance efficiency** = Σ station cycle times / (`n_stations` × bottleneck cycle time); **balance delay** = 100 % − efficiency.
- **Smoothness index** = √Σ (bottleneck − station time)²; 0 is a perfectly balanced line.
- **Theoretical minimum stations** = ⌈Σ station times / takt⌉; `NaN` without a takt.
- **Takt** comes from `takt_time` (seconds or an offset string such as `"55s"`), or else from `available_time / demand`. The takt arguments are keyword-only.
- By default `n_stations` counts only stations that completed a cycle in the window, so a stopped station drops out and the line looks balanced. Pass `require_all_stations=True` to divide by all configured stations and get a `stations_missing` column.
- `station_uuids` maps trigger UUID to station name; give it in line order, as `yamazumi` keeps that order.
