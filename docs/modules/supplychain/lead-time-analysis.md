# LeadTimeAnalysisEvents

> Analyze lead times between order placement and delivery.

**Module:** `ts_shape.events.supplychain.lead_time_analysis`
**Guide:** [Production Guide](../../guides/production.md)

---

## When to Use

Use to monitor supplier and internal process lead times. Pairs order and delivery events to compute lead times and flag anomalies that may disrupt production planning. Essential for supply chain reliability assessment and vendor performance tracking.

---

## Quick Example

```python
from ts_shape.events.supplychain.lead_time_analysis import LeadTimeAnalysisEvents

# orders_df holds two signals: "po_placed" and "po_received",
# each row carrying the PO number in value_string
analyzer = LeadTimeAnalysisEvents(dataframe=orders_df)

# Match order placement to delivery events by PO number
lead_times = analyzer.lead_times(
    order_uuid="po_placed",
    delivery_uuid="po_received",
)

# Summary statistics across all orders (single-row DataFrame)
stats = analyzer.lead_time_statistics(order_uuid="po_placed", delivery_uuid="po_received")
print(f"Mean lead time: {stats['mean_hours'].iloc[0]:.1f} h")
print(f"95th percentile: {stats['p95_hours'].iloc[0]:.1f} h")

# Flag orders with abnormally long lead times (mean + 2 * std)
anomalies = analyzer.detect_lead_time_anomalies(
    order_uuid="po_placed", delivery_uuid="po_received", threshold_factor=2.0
)
print(f"Anomalous deliveries: {len(anomalies)}")
```

---

## Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `lead_times()` | Match orders to deliveries and compute durations | DataFrame with `order_time`, `delivery_time`, `lead_time_seconds`, `lead_time_hours`, `order_id` |
| `lead_time_statistics()` | Summary statistics of lead times | Single-row DataFrame with `mean_hours`, `std_hours`, `min_hours`, `max_hours`, `p95_hours`, `count` |
| `detect_lead_time_anomalies()` | Flag lead times above mean + `threshold_factor` * std | DataFrame with `order_time`, `delivery_time`, `lead_time_hours`, `z_score` |

---

## Tips & Hints

!!! tip "Segment lead times by supplier or product"
    Global lead time statistics can hide per-supplier variation. Use separate order/delivery signals per supplier (or filter the input DataFrame per supplier) before calling `lead_time_statistics()` to identify which vendors are unreliable and which are consistently fast.

!!! info "Related modules"
    - [`InventoryMonitoringEvents`](inventory-monitoring.md) - Lead time variability directly affects reorder point calculations
    - [`DemandPatternEvents`](demand-patterns.md) - Demand spikes combined with long lead times create stockout risk
    - [`AnomalyCorrelationEvents`](../correlation/anomaly-correlation.md) - Correlate lead time anomalies with other supply chain disruptions

---

## See Also

- [Production Guide](../../guides/production.md)
- [API Reference](../../reference/ts_shape/events/supplychain/lead_time_analysis.md)
