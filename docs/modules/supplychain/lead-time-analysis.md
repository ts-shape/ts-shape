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
                           systime=orders["systime"] + pd.to_timedelta([3, 4, 4, 5, 5, 12, 4, 4], unit="D"))
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
| `lead_time_statistics(order_uuid, delivery_uuid)` | one row: `mean_hours`, `std_hours`, `min_hours`, `max_hours`, `p95_hours`, `order_count` |
| `detect_lead_time_anomalies(order_uuid, delivery_uuid, threshold_factor=2.0)` | lead times above mean + `threshold_factor` × std: `order_time`, `delivery_time`, `lead_time_hours`, `z_score` (`systime` = delivery time) |

## Notes

- The order identifier is read from `value_string` (`value_column=` to change). Each order, in time order, is paired with the first not yet used delivery at or after it. When the deliveries carry identifiers found among the orders, this one-to-one pairing runs within each identifier: two orders `PO-1` need two `PO-1` deliveries, and one delivery never closes two orders. Otherwise it runs over all orders and deliveries. Orders and deliveries left without a partner (on the identifier path also rows without an identifier, or with one the other side lacks) are dropped with a `DataQualityWarning`; a lead time is never negative.
- A multi-line purchase order with partial deliveries needs line-level identifiers (e.g. `PO-1/10`, `PO-1/20`), one order row and one delivery row per line; with only the PO number, its lines are paired in time order.
- For per-supplier statistics, use one signal pair per supplier or filter the frame first.
- `calculate_lead_times` is the deprecated name of `lead_times`.
