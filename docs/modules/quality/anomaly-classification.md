# AnomalyClassificationEvents

> Classify anomalies in a numeric signal by type: spike, drift, oscillation, flatline, or level shift.

**Module:** `ts_shape.events.quality.anomaly_classification`  
**Guide:** [Quality Control & SPC](../../guides/quality.md)

---

## When to Use

Use after initial anomaly detection to understand the nature of anomalies. Goes beyond "is this an outlier?" to tell you what kind of anomaly it is, enabling targeted corrective action. Works best when combined with an upstream outlier detection step to narrow the signal regions of interest.

---

## Quick Example

```python
from ts_shape.events.quality.anomaly_classification import AnomalyClassificationEvents

classifier = AnomalyClassificationEvents(df, signal_uuid="temperature_sensor")

# Full multi-type classification in one pass
classified = classifier.classify_anomalies(window="10min", n_sigma=3.0)

# Or detect specific anomaly types individually
flatlines = classifier.detect_flatline(min_duration="1min", tolerance=1e-6)
oscillations = classifier.detect_oscillation(window="1min", min_crossings=6)
drifts = classifier.detect_drift(window="1h", min_slope=0.01)
```

---

## Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `detect_flatline(min_duration='1min', tolerance=1e-6)` | Detect stuck/frozen signal segments | DataFrame with `start`, `end`, `duration`, `stuck_value` |
| `detect_oscillation(window='1min', min_crossings=6)` | Detect excessive oscillation (e.g., control valve hunting) | DataFrame with `start`, `end`, `crossing_count`, `amplitude` |
| `detect_drift(window='1h', min_slope=0.01, slope_per='sample')` | Detect gradual drift in signal mean; `slope_per='second'` makes `slope`/`min_slope` units per second instead of per sample (independent of the sampling rate) | DataFrame with `start`, `end`, `slope`, `direction` |
| `classify_anomalies(window='10min', n_sigma=3.0)` | Multi-type classification: spike, drift, oscillation, flatline, level shift | DataFrame with `start`, `end`, `anomaly_type`, `severity`, `details` |

---

## Tips & Hints

!!! tip "Run outlier detection first, then classify"
    Use [OutlierDetectionEvents](outlier-detection.md) to flag anomalous regions, then restrict the input DataFrame to those regions before running `classify_anomalies()` for root-cause identification. This two-step approach reduces false positives and makes classification more meaningful.

!!! info "Related modules"
    - [Outlier Detection](outlier-detection.md) — detect anomalies before classifying them
    - [Sensor Drift](sensor-drift.md) — specialized drift detection for calibration monitoring
    - [Signal Quality](signal-quality.md) — distinguish data quality issues from true process anomalies

---

## See Also

- [Quality Control & SPC Guide](../../guides/quality.md) — narrative overview
- [API Reference](../../reference/ts_shape/events/quality/anomaly_classification.md) — full parameter docs
