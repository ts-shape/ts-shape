# EnergyConsumptionEvents

> Analyze energy consumption patterns from meter/sensor signals.

**Module:** `ts_shape.events.energy.consumption_analysis`
**Guide:** [Production Guide](../../guides/production.md)

---

## When to Use

Use for energy management and ISO 50001 compliance. Tracks consumption patterns, identifies peak demand periods, and calculates specific energy consumption per unit produced. Works with utility meter data, sub-meter readings, or any cumulative/instantaneous power signal.

---

## Quick Example

```python
from ts_shape.events.energy.consumption_analysis import EnergyConsumptionEvents

# df: systime | uuid | value_double (kWh deltas) | value_integer (counter)
energy = EnergyConsumptionEvents(df)

# Aggregate consumption per shift
consumption = energy.consumption_by_window("meter:main", window="8h", agg="sum")

# Flag peak demand windows (above the 95th percentile by default)
peaks = energy.peak_demand_detection("meter:main", window="15min")
print(peaks[peaks["is_peak"]].nlargest(1, "demand"))

# Compare actual vs a rolling baseline (flag >15% deviation)
deviations = energy.consumption_baseline_deviation(
    "meter:main", baseline_periods=24, deviation_threshold=0.15
)

# Energy per production unit (pairs the meter with a part counter)
epu = energy.energy_per_unit("meter:main", "counter:parts", window="1D")
```

!!! warning "Values must be kWh deltas"
    Methods sum `value_double` per window, so readings must be energy *deltas*
    (kWh since the previous sample). For an instantaneous kW signal, integrate it
    first with [`PowerIntegrator`](power-integration.md).

---

## Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `consumption_by_window(meter_uuid)` | Aggregate consumption per time window (`agg`: sum/mean/max) | `start`, `uuid`, `source_uuid`, `is_delta`, `consumption` |
| `peak_demand_detection(meter_uuid)` | Flag windows above `threshold` (or the `percentile`, default 0.95) | `start`, `uuid`, `source_uuid`, `is_delta`, `demand`, `threshold`, `is_peak` |
| `consumption_baseline_deviation(meter_uuid)` | Actual vs rolling baseline (`deviation_threshold` is a fraction, default 0.2) | `start`, `uuid`, `source_uuid`, `is_delta`, `consumption`, `baseline`, `deviation_pct`, `is_anomaly` |
| `energy_per_unit(meter_uuid, counter_uuid)` | Energy per production unit | `start`, `uuid`, `source_uuid`, `is_delta`, `energy`, `units_produced`, `energy_per_unit` |

---

## Tips & Hints

!!! tip "Align windows with shift boundaries"
    Use window sizes that match your operational shifts (e.g., `"8h"` for 8-hour shifts). Misaligned windows split consumption across shifts and obscure real patterns.

!!! info "Related modules"
    - [`EnergyEfficiencyEvents`](efficiency-tracking.md) - Track efficiency trends and idle energy waste
    - [`InventoryMonitoringEvents`](../supplychain/inventory-monitoring.md) - Correlate energy use with production output
    - [`DemandPatternEvents`](../supplychain/demand-patterns.md) - Demand patterns that drive energy consumption

---

## See Also

- [Production Guide](../../guides/production.md)
- [API Reference](../../reference/ts_shape/events/energy/consumption_analysis.md)
