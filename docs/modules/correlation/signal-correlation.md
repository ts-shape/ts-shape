# SignalCorrelationEvents

> Analyze time-windowed correlations between pairs of numeric signals.

**Module:** `ts_shape.events.correlation.signal_correlation`
**Guide:** [Statistics Guide](../../guides/statistics.md)

---

## When to Use

Use to monitor relationships between process variables. Detects when normally correlated signals (e.g., pressure and temperature) start to diverge, indicating process changes or sensor issues. Also useful for validating sensor redundancy and identifying leading indicators.

---

## Quick Example

```python
from ts_shape.events.correlation.signal_correlation import SignalCorrelationEvents

# df: systime | uuid | value_double
correlator = SignalCorrelationEvents(df)

# Rolling Pearson correlation on a 1-minute grid over 4 hours (240 periods)
rolling = correlator.rolling_correlation(
    "reactor:pressure", "reactor:temperature", window="1min", window_size=240
)

# Find periods where the correlation breaks down
breakdowns = correlator.correlation_breakdown(
    "reactor:pressure", "reactor:temperature", window_size=240, threshold=0.5
)
print(f"Found {len(breakdowns)} correlation breakdown periods")

# Cross-correlation with time lag (up to 30 periods of 1 minute each way)
lag_corr = correlator.lag_correlation(
    "reactor:pressure", "reactor:temperature", window="1min", max_lag=30
)
best_lag = lag_corr[lag_corr["is_best_lag"]].iloc[0]
print(f"Best lag: {best_lag['lag_periods']} periods, r={best_lag['correlation']:.3f}")
```

---

## Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `rolling_correlation(uuid_a, uuid_b)` | Rolling Pearson correlation over `window_size` periods of the `window` grid (defaults `window="1min"`, `window_size=60`) | `systime`, `uuid`, `source_uuid_a`, `source_uuid_b`, `is_delta`, `correlation` |
| `correlation_breakdown(uuid_a, uuid_b)` | Periods where correlation drops below `threshold` (default 0.5) | `start`, `end`, `uuid`, `source_uuid_a`, `source_uuid_b`, `is_delta`, `min_correlation`, `duration_seconds` |
| `lag_correlation(uuid_a, uuid_b)` | Cross-correlation for lags from `-max_lag` to `+max_lag` periods (default 30) | `lag_periods`, `correlation`, `is_best_lag` |

---

## Tips & Hints

!!! tip "Use lag correlation to find leading indicators"
    `lag_correlation()` can reveal that one signal leads another by a fixed delay. Lags are counted in `window` periods, so `lag_periods=5` with `window="1min"` means 5 minutes; a positive lag pairs `uuid_a` at time t with `uuid_b` at t minus the lag. This is valuable for building early warning systems where an upstream variable predicts a downstream change.

!!! info "Related modules"
    - [`AnomalyCorrelationEvents`](anomaly-correlation.md) - Correlate detected anomalies across signals instead of raw values
    - [`DegradationDetectionEvents`](../maintenance/degradation-detection.md) - Detect degradation when correlated signals start diverging
    - [`EnergyEfficiencyEvents`](../energy/efficiency-tracking.md) - Correlate process variables with energy efficiency

---

## See Also

- [Statistics Guide](../../guides/statistics.md)
- [API Reference](../../reference/ts_shape/events/correlation/signal_correlation.md)
