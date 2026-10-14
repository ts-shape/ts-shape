# DegradationDetectionEvents

> Detect degradation patterns in time series signals: trend degradation, variance increases, level shifts, and composite health scores.

**Module:** `ts_shape.events.maintenance.degradation_detection`
**Guide:** [Quality Guide](../../guides/quality.md)

---

## When to Use

Use for predictive maintenance monitoring. Tracks gradual degradation in bearing temperature, motor current, or any signal that slowly drifts before failure. Ideal when you need early warning of equipment deterioration before alarm thresholds are reached.

---

## Quick Example

```python
from ts_shape.events.maintenance.degradation_detection import DegradationDetectionEvents

detector = DegradationDetectionEvents(
    dataframe=sensor_df,
    signal_uuid="bearing_temp",
)

# Detect slow upward drift in bearing temperature (slope in units per second)
trend_events = detector.detect_trend_degradation(
    window="6h", min_slope=0.0001, direction="increasing"
)

# Rolling variance vs. the first window as baseline
variance_events = detector.detect_variance_increase(window="1h", threshold_factor=2.0)

# Permanent mean shifts of at least 3 degrees that persist for 10 minutes
shifts = detector.detect_level_shift(min_shift=3.0, hold="10min")

# Composite 0-100 health score over time, baselined on the first 24 hours
health = detector.health_score(window="1h", baseline_window="24h")
print(health[["systime", "health_score"]].tail())
```

---

## Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `detect_trend_degradation()` | Rolling linear regression slope detection | Intervals with `start`, `end`, `avg_slope`, `total_change`, `duration_seconds` |
| `detect_variance_increase()` | Rolling variance vs. baseline (first window) | Intervals with `baseline_variance`, `current_variance`, `ratio` |
| `detect_level_shift()` | CUSUM-like mean shift detection | Events with `systime`, `shift_magnitude`, `prev_mean`, `new_mean` |
| `health_score()` | Composite 0-100 health score per sample | DataFrame with `systime`, `health_score`, `mean_drift_pct`, `variance_ratio`, `trend_slope` |

---

## Tips & Hints

!!! tip "Set a meaningful baseline window"
    `detect_variance_increase()` uses the first `window` of data as its baseline, and `health_score()` uses the first `baseline_window`. Start the input at a stable, known-good operating period; a poor baseline produces false positives.

!!! info "Related modules"
    - [`FailurePredictionEvents`](failure-prediction.md) - Estimate remaining useful life once degradation is confirmed
    - [`VibrationAnalysisEvents`](vibration-analysis.md) - Specialized degradation detection for vibration signals

---

## See Also

- [Quality Guide](../../guides/quality.md)
- [API Reference](../../reference/ts_shape/events/maintenance/degradation_detection.md)
