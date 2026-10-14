# LeadTimeAnalysisEvents

Lead times between order placement and delivery events, their statistics, and abnormally long deliveries.

`ts_shape.events.supplychain.lead_time_analysis` · [Production guide](../../guides/production.md) · [API reference](../../reference/ts_shape/events/supplychain/lead_time_analysis.md)

## Example

```python
import pandas as pd
from ts_shape.events.supplychain.lead_time_analysis import LeadTimeAnalysisEvents

orders = pd.DataFrame({"systime": pd.date_range("2024-01-01", periods=8, freq="1D"), "uuid": "po_placed",
                       "value_string": [f"PO-{i}" for i in range(8)]})
deliveries = orders.assign(uuid="po_received",
                           systime=orders["systime"] + pd.to_timedelta([3, 4, 3, 5, 4, 12, 3, 4], unit="D"))
df = pd.concat([orders, deliveries], ignore_index=True)

lt = LeadTimeAnalysisEvents(df)
lead_times = lt.lead_times("po_placed", "po_received")
print(lt.lead_time_statistics("po_placed", "po_received"))
late = lt.detect_lead_time_anomalies("po_placed", "po_received", threshold_factor=2.0)
```

## Methods

| Method | Returns |
|---|---|
| `lead_times(order_uuid, delivery_uuid)` | per order: `order_time`, `delivery_time`, `lead_time_seconds`, `lead_time_hours`, `order_id` |
| `lead_time_statistics(order_uuid, delivery_uuid)` | one row: `mean_hours`, `std_hours`, `min_hours`, `max_hours`, `p95_hours`, `count` |
| `detect_lead_time_anomalies(order_uuid, delivery_uuid, threshold_factor=2.0)` | lead times above mean + `threshold_factor` × std: `order_time`, `delivery_time`, `lead_time_hours`, `z_score` (`systime` = delivery time) |

## Notes

- The order identifier is read from `value_string` (`value_column=` to change). When the deliveries carry identifiers found among the orders, each order is matched to the first delivery with the same identifier at or after it, and unmatched orders are left out; otherwise orders and deliveries are paired in sequence.
- For per-supplier statistics, use one signal pair per supplier or filter the frame first.
- `calculate_lead_times` is the deprecated name of `lead_times`.
