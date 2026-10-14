# EnergyEfficiencyEvents

> Track energy efficiency metrics against production and machine state.

**Module:** `ts_shape.events.energy.efficiency_tracking`
**Guide:** [Production Guide](../../guides/production.md)

---

## When to Use

Use to track energy efficiency over time and identify waste. Detects energy consumed during idle periods and compares efficiency across shifts or production runs. Essential for continuous improvement programs targeting energy reduction.

---

## Quick Example

```python
from ts_shape.events.energy.efficiency_tracking import EnergyEfficiencyEvents

# df: systime | uuid | value_double (kWh deltas) | value_integer | value_bool
tracker = EnergyEfficiencyEvents(df)

# Rolling energy efficiency (units per kWh)
trend = tracker.efficiency_trend("meter:main", "counter:parts", window="1D")

# Detect energy waste while the machine is idle (state: True = running)
waste = tracker.idle_energy_waste("meter:main", "machine:running", window="1h")
print(f"Total idle waste: {waste['waste_energy'].sum():.1f} kWh")

# Specific energy consumption (kWh per unit) per day
sec = tracker.specific_energy_consumption("meter:main", "counter:parts", window="1D")

# Compare efficiency across shifts (default: three 8-hour shifts from 06:00)
comparison = tracker.efficiency_comparison("meter:main", "counter:parts")
```

---

## Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `efficiency_trend(meter_uuid, counter_uuid)` | Rolling energy efficiency over time (`trend_window` windows) | `start`, `uuid`, `source_uuid`, `is_delta`, `energy`, `units`, `efficiency`, `rolling_avg_efficiency`, `trend_direction` |
| `idle_energy_waste(meter_uuid, state_uuid)` | Energy consumed while the machine ran less than 10 % of the window (run state held until the next change; `time_weighted=False` averages samples) | `start`, `uuid`, `source_uuid`, `is_delta`, `energy_consumed`, `machine_running_pct` (0-100), `machine_running_ratio` (0-1), `is_idle_waste`, `waste_energy` |
| `specific_energy_consumption(meter_uuid, counter_uuid)` | SEC (kWh per unit) per period (default `window="1D"`) | `start`, `uuid`, `source_uuid`, `is_delta`, `total_energy`, `total_output`, `sec`, `sec_trend` |
| `efficiency_comparison(meter_uuid, counter_uuid)` | Compare efficiency across shifts (`shift_definitions`) | `shift`, `avg_energy`, `avg_output`, `avg_efficiency`, `total_energy`, `total_output` |

---

## Tips & Hints

!!! tip "Track idle waste to find quick wins"
    `idle_energy_waste()` often reveals the lowest-hanging fruit for energy savings. Machines left running during breaks or changeovers can account for 10-20% of total consumption.

!!! info "Related modules"
    - [`EnergyConsumptionEvents`](consumption-analysis.md) - Raw consumption analysis and peak demand detection
    - [`SignalCorrelationEvents`](../correlation/signal-correlation.md) - Correlate efficiency changes with process variables

---

## See Also

- [Production Guide](../../guides/production.md)
- [API Reference](../../reference/ts_shape/events/energy/efficiency_tracking.md)
