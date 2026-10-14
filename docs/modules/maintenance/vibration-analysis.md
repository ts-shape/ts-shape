# VibrationAnalysisEvents

> Analyze vibration signals from industrial equipment: RMS exceedance, amplitude growth, and bearing health indicators.

**Module:** `ts_shape.events.maintenance.vibration_analysis`
**Guide:** [Quality Guide](../../guides/quality.md)

---

## When to Use

Use for rotating equipment monitoring (motors, pumps, compressors). Computes standard vibration indicators (RMS, peak, crest factor, kurtosis) for bearing health assessment. Works with accelerometer or velocity sensor data sampled at regular intervals.

---

## Quick Example

```python
from ts_shape.events.maintenance.vibration_analysis import VibrationAnalysisEvents

analyzer = VibrationAnalysisEvents(
    dataframe=vibration_df,
    signal_uuid="accel_x",
)

# Detect RMS exceedance against a known-good baseline
rms_events = analyzer.detect_rms_exceedance(
    baseline_rms=2.5, threshold_factor=1.5, window="1h"
)

# Flag daily windows whose peak-to-peak amplitude grew 10%+ over the first window
growth = analyzer.detect_amplitude_growth(window="1D", growth_threshold=0.1)

# Full bearing health indicators per window
health = analyzer.bearing_health_indicators(window="4h")
print(health[["start", "rms", "peak", "crest_factor", "kurtosis"]].tail())
```

---

## Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `detect_rms_exceedance()` | Rolling RMS vs. `baseline_rms * threshold_factor` | Intervals with `start`, `end`, `rms_value`, `baseline_rms`, `ratio`, `duration_seconds` |
| `detect_amplitude_growth()` | Peak-to-peak amplitude growth vs. the first window | Windows with `start`, `amplitude`, `baseline_amplitude`, `growth_pct` |
| `bearing_health_indicators()` | RMS, peak, crest factor, kurtosis per window | Windows with `start`, `rms`, `peak`, `crest_factor`, `kurtosis` |

---

## Tips & Hints

!!! tip "Use ISO 10816 thresholds for RMS baseline"
    For standard industrial motors, ISO 10816 provides vibration severity thresholds by machine class. Use these as your `baseline_rms` and `threshold_factor` values for meaningful alerts.

!!! info "Related modules"
    - [`DegradationDetectionEvents`](degradation-detection.md) - General-purpose degradation detection for any signal type
    - [`FailurePredictionEvents`](failure-prediction.md) - Estimate remaining useful life from vibration trends
    - [`SignalCorrelationEvents`](../correlation/signal-correlation.md) - Correlate vibration with temperature or current signals

---

## See Also

- [Quality Guide](../../guides/quality.md)
- [API Reference](../../reference/ts_shape/events/maintenance/vibration_analysis.md)
