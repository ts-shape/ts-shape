# DemandPatternEvents

Demand per period (total, average, peak), demand spikes, and day-of-week or hour-of-day patterns, from a numeric demand signal (one value per order or reading).

`ts_shape.events.supplychain.demand_pattern` · [Production guide](../../guides/production.md) · [API reference](../../reference/ts_shape/events/supplychain/demand_pattern.md)

## Example

```python
import ts_shape
from ts_shape.events.supplychain.demand_pattern import DemandPatternEvents

df = ts_shape.make_timeseries(["order_quantity"], n_points=2000, freq="1h", baseline=40.0, noise=10.0, n_outliers=3)

dp = DemandPatternEvents(df, demand_uuid="order_quantity")
daily = dp.demand_by_period(window="1D")
spikes = dp.detect_demand_spikes(threshold_factor=2.5, window="1D")
by_weekday = dp.seasonality_summary(window="1D")
print(dp.seasonality_summary(window="1h").nlargest(3, "avg_demand"))
```

## Methods

All arguments are keyword-only.

| Method | Returns |
|---|---|
| `demand_by_period(window='1D')` | per period: `total_demand`, `avg_demand`, `peak_demand` |
| `detect_demand_spikes(threshold_factor=2.0, window='1D')` | periods above mean + `threshold_factor` × std of the period totals: `demand`, `baseline_mean`, `spike_magnitude` |
| `seasonality_summary(window='1D')` | per `period_label`: `avg_demand`, `std_demand`, `min_demand`, `max_demand` |

## Notes

- `seasonality_summary` groups by day of week (Monday = 0 … Sunday = 6) with `window='1D'` and by hour of day (0-23) with any sub-daily window.
- Known promotions or one-off orders distort the seasonality; drop them (e.g. the periods from `detect_demand_spikes`) first.
