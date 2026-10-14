# InventoryMonitoringEvents

> Monitor inventory levels from a timeseries signal and detect supply chain events.

**Module:** `ts_shape.events.supplychain.inventory_monitoring`
**Guide:** [Production Guide](../../guides/production.md)

---

## When to Use

Use for real-time inventory monitoring in manufacturing. Tracks stock levels, calculates consumption rates, and predicts when a stockout will occur based on current consumption trends. Works with any signal representing a quantity that depletes over time and is periodically replenished.

---

## Quick Example

```python
from ts_shape.events.supplychain.inventory_monitoring import InventoryMonitoringEvents

monitor = InventoryMonitoringEvents(
    dataframe=inventory_df,
    level_uuid="stock_level",
)

# Flag intervals where stock stays below 100 units for at least 2 hours
low_stock = monitor.detect_low_stock(min_level=100, hold="2h")
print(f"Low stock events: {len(low_stock)}")

# Daily consumption rate (units per hour); refills do not cancel consumption
rate = monitor.consumption_rate(window="1D")

# Detect when stock falls below the reorder point or safety stock
breaches = monitor.reorder_point_breach(reorder_level=250, safety_stock=80)

# Hours until stockout, based on the last 24 hours of consumption
prediction = monitor.stockout_prediction(consumption_rate_window="24h")
print(prediction[["systime", "current_level", "estimated_stockout_time_hours"]].tail())
```

---

## Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `detect_low_stock()` | Flag intervals below `min_level` for at least `hold` | DataFrame with `start`, `end`, `min_value`, `avg_value`, `duration_seconds` |
| `consumption_rate()` | Consumption rate per window (units per hour) | DataFrame with `start`, `consumption_rate`, `level_start`, `level_end` |
| `reorder_point_breach()` | Detect when stock falls below reorder point or safety stock | DataFrame with `systime`, `current_level`, `breach_type`, `deficit` |
| `stockout_prediction()` | Time estimate until stockout per sample | DataFrame with `systime`, `current_level`, `consumption_rate`, `estimated_stockout_time_hours` |

---

## Tips & Hints

!!! tip "Account for replenishment events"
    By default `consumption_rate()` sums only level decreases and treats a refill like a counter reset. Pass `handle_refills=False` to use the first-minus-last level of each window instead; a refill inside the window then cancels out consumption.

!!! info "Related modules"
    - [`DemandPatternEvents`](demand-patterns.md) - Understand demand patterns driving inventory depletion
    - [`LeadTimeAnalysisEvents`](lead-time-analysis.md) - Monitor supplier lead times that affect reorder timing
    - [`EnergyConsumptionEvents`](../energy/consumption-analysis.md) - Similar depletion pattern analysis for energy resources

---

## See Also

- [Production Guide](../../guides/production.md)
- [API Reference](../../reference/ts_shape/events/supplychain/inventory_monitoring.md)
