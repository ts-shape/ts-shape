# CapabilityTrendingEvents

> Track process capability indices (Cp, Cpk, Pp, Ppk) over rolling time windows to detect capability degradation before quality escapes occur.

**Module:** `ts_shape.events.quality.capability_trending`  
**Guide:** [Quality Control & SPC](../../guides/quality.md)

---

## When to Use

Use for trending process capability over time. Detects capability degradation before it results in quality escapes. Set up as a daily dashboard to track Cpk trends and predict when intervention is needed. Requires process data with specification limits (fixed `upper_spec`/`lower_spec`, or spec signals via `upper_spec_uuid`/`lower_spec_uuid`) and sufficient sample sizes per window.

---

## Quick Example

```python
from ts_shape.events.quality.capability_trending import CapabilityTrendingEvents

cap = CapabilityTrendingEvents(
    df,
    signal_uuid="bore_diameter",
    upper_spec=105.0,
    lower_spec=95.0,
)

# Compute capability indices per 8-hour window
trend = cap.capability_over_time(window="8h")

# Alert when Cpk drops below 1.33
drops = cap.detect_capability_drop(window="8h", min_cpk=1.33)

# Forecast when Cpk will breach the threshold
forecast = cap.capability_forecast(window="8h", horizon=5, threshold=1.33)
print(forecast[["start", "cpk", "forecast_cpk", "windows_to_threshold"]].tail(1))

# Estimate yield, DPMO, and sigma level
yield_stats = cap.yield_estimate(window="8h")
```

---

## Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `capability_over_time(window='8h')` | Compute Cp, Cpk, Pp, Ppk per window | DataFrame with `start`, `cp`, `cpk`, `pp`, `ppk`, `mean`, `std`, `n_samples` |
| `detect_capability_drop(window='8h', min_cpk=1.33, lookback=5)` | Flag windows where Cpk falls below threshold or degrades vs. the previous windows | DataFrame with `start`, `cpk`, `prev_avg_cpk`, `drop_pct`, `alert` |
| `capability_forecast(window='8h', horizon=5, threshold=1.33)` | Predict future Cpk breach based on trend | DataFrame with `start`, `cpk`, `trend_slope`, `forecast_cpk`, `windows_to_threshold` |
| `yield_estimate(window='8h')` | Estimate yield percentage, DPMO, and sigma level | DataFrame with `start`, `estimated_yield_pct`, `dpmo`, `sigma_level` |

---

## Tips & Hints

!!! tip "Use min_cpk=1.33 as the default threshold"
    A Cpk of 1.33 corresponds to a 4-sigma process (~63 DPMO). For safety-critical processes, raise this to 1.67 (5-sigma). Set `detect_capability_drop` to alert at your threshold so you can intervene before parts go out of spec.

!!! info "Related modules"
    - [Tolerance Deviation](tolerance-deviation.md) — point-in-time capability indices and severity classification
    - [SPC Rules](spc.md) — Western Electric rule violations that often precede capability drops
    - [Sensor Drift](sensor-drift.md) — sensor drift as a root cause of apparent capability degradation

---

## See Also

- [Quality Control & SPC Guide](../../guides/quality.md) — narrative overview
- [API Reference](../../reference/ts_shape/events/quality/capability_trending.md) — full parameter docs
