# FailurePredictionEvents

> Predict remaining useful life and detect escalating failure patterns.

**Module:** `ts_shape.events.maintenance.failure_prediction`
**Guide:** [Quality Guide](../../guides/quality.md)

---

## When to Use

Use when a degradation trend has been identified and you need to estimate how much time remains before a failure threshold is reached. Pairs well with `DegradationDetectionEvents` for a complete predictive maintenance pipeline.

---

## Quick Example

```python
from ts_shape.events.maintenance.failure_prediction import FailurePredictionEvents

predictor = FailurePredictionEvents(
    dataframe=sensor_df,
    signal_uuid="bearing_temp",
)

# Estimate remaining useful life via linear extrapolation to 95 degrees
# (degradation_rate is a fallback slope in units per second; positive = rising)
rul = predictor.remaining_useful_life(degradation_rate=0.0001, failure_threshold=95.0)
print(rul[["systime", "current_value", "rul_hours", "confidence"]].tail())

# Detect escalating warning/critical breaches per rolling window
exceedances = predictor.detect_exceedance_pattern(
    warning_threshold=80.0, critical_threshold=90.0, window="1D"
)

# Time estimate to reach the critical threshold
tte = predictor.time_to_threshold(threshold=95.0, direction="increasing")
print(tte[["systime", "estimated_time_seconds"]].tail())
```

---

## Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `remaining_useful_life()` | RUL via linear extrapolation to a failure threshold | DataFrame with `systime`, `current_value`, `rul_seconds`, `rul_hours`, `confidence` (`rul_*` is None when no failure is projected) |
| `detect_exceedance_pattern()` | Escalating warning/critical breaches over time | DataFrame with `start`, `warning_count`, `critical_count`, `escalation_detected` |
| `time_to_threshold()` | Time estimate to reach a given threshold | DataFrame with `systime`, `current_value`, `rate_of_change`, `estimated_time_seconds` |

---

## Tips & Hints

!!! tip "Combine with degradation detection"
    Run `DegradationDetectionEvents.detect_trend_degradation()` first to confirm a real trend exists. Calling `remaining_useful_life()` on a stationary signal will return unreliable estimates.

!!! info "Related modules"
    - [`DegradationDetectionEvents`](degradation-detection.md) - Detect degradation trends before estimating RUL
    - [`VibrationAnalysisEvents`](vibration-analysis.md) - Vibration-specific failure indicators for rotating equipment

---

## See Also

- [Quality Guide](../../guides/quality.md)
- [API Reference](../../reference/ts_shape/events/maintenance/failure_prediction.md)
