# InventoryMonitoringEvents

Low-stock intervals, consumption rate, reorder-point and safety-stock breaches, and time to stockout, from a stock-level signal.

`ts_shape.events.supplychain.inventory_monitoring` · [Production guide](../../guides/production.md) · [API reference](../../reference/ts_shape/events/supplychain/inventory_monitoring.md)

## Example

```python
import numpy as np
import pandas as pd
from ts_shape.events.supplychain.inventory_monitoring import InventoryMonitoringEvents

t = pd.date_range("2024-01-01", periods=7 * 24, freq="1h")
level = 600 - (np.arange(len(t)) % 72) * 8.0  # consumed 8/h, refilled every 3 days
df = pd.DataFrame({"systime": t, "uuid": "stock_level", "value_double": level})

inv = InventoryMonitoringEvents(df, level_uuid="stock_level")
low = inv.detect_low_stock(100, hold="2h")
rate = inv.consumption_rate(window="1D")
breaches = inv.reorder_point_breach(250, safety_stock=80)
print(inv.stockout_prediction(consumption_rate_window="24h").tail(3))
```

## Methods

| Method | Returns |
|---|---|
| `detect_low_stock(min_level, hold='0s')` | intervals below `min_level` for at least `hold`: `min_value`, `avg_value`, `duration_seconds` |
| `consumption_rate(window='1h', handle_refills=True)` | per window: `consumption_rate` (units per hour), `level_start`, `level_end` |
| `reorder_point_breach(reorder_level, safety_stock=0.0)` | one point per crossing below a level: `current_level`, `breach_type` (`reorder`/`safety_stock`), `deficit` |
| `stockout_prediction(consumption_rate_window='4h')` | per sample: `current_level`, `consumption_rate`, `estimated_stockout_time_hours` |

## Notes

- `handle_refills=True` (default) sums only level decreases, so a refill is treated like a counter reset. With `False` consumption is first − last level per window, so a refill cancels consumption (100 → 40 → 100 → 40 reads as 0).
- `stockout_prediction` projects the time to zero from the consumption rate over the preceding `consumption_rate_window`. Where the level is not falling, `estimated_stockout_time_hours` is NaN (no stockout projected).
- `safety_stock` must be at most `reorder_level`.
- Low-stock intervals end at the next sample after the level recovers (`interval_end="next_change"`, default) or at the last low sample (`"last_sample"`).
