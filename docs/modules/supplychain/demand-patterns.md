# DemandPatternEvents

> Analyze demand patterns from a timeseries demand signal.

**Module:** `ts_shape.events.supplychain.demand_pattern`
**Guide:** [Production Guide](../../guides/production.md)

---

## When to Use

Use to understand demand patterns for production planning. Identifies spikes, seasonal patterns, and trends that affect scheduling and inventory management. Works with order counts, request volumes, or any signal representing demand intensity over time.

---

## Quick Example

```python
from ts_shape.events.supplychain.demand_pattern import DemandPatternEvents

analyzer = DemandPatternEvents(
    dataframe=orders_df,
    demand_uuid="order_quantity",
)

# Aggregate demand per day (total, average, and peak)
daily_demand = analyzer.demand_by_period(period="1D")

# Flag days whose demand exceeds mean + 2.5 * std
spikes = analyzer.detect_demand_spikes(threshold_factor=2.5, window="1D")
print(f"Detected {len(spikes)} demand spikes")

# Day-of-week pattern (use a sub-daily period such as "1h" for hour-of-day)
by_weekday = analyzer.seasonality_summary(period="1D")
busiest = by_weekday.loc[by_weekday["avg_demand"].idxmax(), "period_label"]
print("Busiest day:", busiest)

by_hour = analyzer.seasonality_summary(period="1h")
print("Peak hour:", by_hour.loc[by_hour["avg_demand"].idxmax(), "period_label"])
```

---

## Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `demand_by_period()` | Aggregate demand per time period | DataFrame with `start`, `total_demand`, `avg_demand`, `peak_demand` |
| `detect_demand_spikes()` | Flag periods above mean + `threshold_factor` * std | DataFrame with `start`, `demand`, `baseline_mean`, `spike_magnitude` |
| `seasonality_summary()` | Day-of-week (`period="1D"`) or hour-of-day (sub-daily period) patterns | DataFrame with `period_label`, `avg_demand`, `std_demand`, `min_demand`, `max_demand` |

---

## Tips & Hints

!!! tip "Remove known promotions before seasonality analysis"
    Promotional events create artificial spikes that distort `seasonality_summary()`. Filter out known promotion dates or use `detect_demand_spikes()` first to identify and exclude outliers before analyzing seasonal patterns.

!!! info "Related modules"
    - [`InventoryMonitoringEvents`](inventory-monitoring.md) - Monitor stock levels driven by the demand patterns you discover
    - [`LeadTimeAnalysisEvents`](lead-time-analysis.md) - Lead times that constrain how quickly you can respond to demand spikes
    - [`EnergyConsumptionEvents`](../energy/consumption-analysis.md) - Energy demand follows similar seasonal patterns

---

## See Also

- [Production Guide](../../guides/production.md)
- [API Reference](../../reference/ts_shape/events/supplychain/demand_pattern.md)
