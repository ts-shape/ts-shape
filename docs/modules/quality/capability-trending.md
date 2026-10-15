# CapabilityTrendingEvents

Cp, Cpk, Pp and Ppk per time window, alerts when Cpk drops, a linear Cpk forecast, and estimated yield/DPMO/sigma level, for a numeric signal with specification limits.

`ts_shape.events.quality.capability_trending` · [Quality guide](../../guides/quality.md) · [API reference](../../reference/ts_shape/events/quality/capability_trending.md)

## Example

```python
import ts_shape
from ts_shape.events.quality.capability_trending import CapabilityTrendingEvents

df = ts_shape.make_timeseries(["bore_diameter"], n_points=5000, freq="1min", noise=0.8, drift=2.0)

cap = CapabilityTrendingEvents(df, "bore_diameter", upper_spec=105.0, lower_spec=95.0)
trend = cap.capability_over_time(window="8h")
drops = cap.detect_capability_drop(window="8h", min_cpk=1.33)
forecast = cap.capability_forecast(window="8h", horizon=5, threshold=1.33)
print(forecast[["start", "cpk", "forecast_cpk", "windows_to_threshold"]].tail(1))
```

## Methods

All arguments are keyword-only.

| Method | Returns (per window) |
|---|---|
| `capability_over_time(window='8h')` | `cp`, `cpk`, `pp`, `ppk`, `mean`, `std`, `std_within`, `n_samples` |
| `detect_capability_drop(window='8h', min_cpk=1.33, lookback=5)` | `cpk`, `prev_avg_cpk`, `drop_pct`, `alert` |
| `capability_forecast(window='8h', horizon=5, threshold=1.33)` | `cpk`, `trend_slope`, `forecast_cpk`, `windows_to_threshold` |
| `yield_estimate(window='8h')` | `estimated_yield_pct`, `dpmo`, `sigma_level` (assumes a normal distribution per window) |

## Notes

- Limits are fixed (`upper_spec`, `lower_spec`) or spec signals (`upper_spec_uuid`, `lower_spec_uuid`); with signals each window uses the spec in force for that window.
- Each window uses only its own samples (AIAG SPC manual; Montgomery §8.3): Cp/Cpk use the within σ̂ = MR̄ / 1.128 of consecutive samples (`std_within`), Pp/Ppk the window's sample standard deviation (`std`, ddof = 1). `sigma_method="std"` (constructor) computes Cp/Cpk from `std` too, so Cp = Pp. Before October 2026 Cp/Cpk used the window's `std` and Pp/Ppk the standard deviation of the whole signal (later windows included). Windows with fewer than 2 samples are skipped.
- `alert` is True when Cpk < `min_cpk` or Cpk dropped more than 20 % below the mean of the previous `lookback` windows.
- The forecast fits one straight line through all window Cpk values. `windows_to_threshold` is NaN when the trend is not falling.
