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
| `demand_by_period(window='1D', empty_periods='zero')` | per period: `total_demand`, `avg_demand`, `peak_demand`, `order_count` |
| `detect_demand_spikes(threshold_factor=2.0, window='1D', empty_periods='zero')` | periods above mean + `threshold_factor` × std of the period totals: `demand`, `baseline_mean`, `spike_magnitude` |
| `seasonality_summary(window='1D', empty_periods='zero')` | per `period_label`: `avg_demand`, `std_demand`, `min_demand`, `max_demand` |

## Notes

- Each row of the demand signal is one order (event-encoded demand), so a period without rows is a period without demand. `demand_by_period` returns every period from the first to the last one with data; an empty period has `total_demand = 0`, `peak_demand = 0`, `order_count = 0` and no `avg_demand` (the mean order size). `empty_periods='drop'` leaves the empty periods out instead (the behaviour before this option).
- `detect_demand_spikes` and `seasonality_summary` work on these period totals, so by default the zero periods count: the spike baseline is the mean and std over the whole span, and a weekday without orders averages low. Pass `empty_periods='drop'` to use only periods with orders.
- A cumulative counter (a running order total) is not event-encoded: difference it into per-order quantities first, or its readings are summed as if each were a new order.
- `seasonality_summary` groups by day of week (Monday = 0 … Sunday = 6) with `window='1D'` and by hour of day (0-23) with any sub-daily window.
- Known promotions or one-off orders distort the seasonality; drop them (e.g. the periods from `detect_demand_spikes`) first.
