# SensorDriftEvents

> Detect gradual calibration drift in inline sensors by tracking measurement behavior against reference values or historical baselines.

**Module:** `ts_shape.events.quality.sensor_drift`  
**Guide:** [Quality Control & SPC](../../guides/quality.md)

---

## When to Use

Use for continuous monitoring of inline sensor calibration. Detects gradual drift that may not trigger outlier alarms but degrades measurement accuracy over time. Ideal for pH probes, temperature sensors, pressure transmitters, and other instruments that require periodic calibration.

---

## Quick Example

```python
from ts_shape.events.quality.sensor_drift import SensorDriftEvents

drift = SensorDriftEvents(df, value_column="value_double")

# Track zero-point offset over 8-hour windows
zero = drift.detect_zero_drift(window="8h", threshold=None)

# Monitor sensitivity changes (span drift)
span = drift.detect_span_drift(window="8h")

# Get a composite calibration health score
health = drift.calibration_health(window="8h", tolerance=None)
print(health[["start", "health_score"]].head())
```

---

## Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `detect_zero_drift(window='8h', threshold=None)` | Track mean offset from baseline (zero-point drift) | DataFrame with offset per window |
| `detect_span_drift(window='8h')` | Detect sensitivity/gain changes over time | DataFrame with span metrics per window |
| `drift_trend(window='1D', metric='mean')` | Rolling trend of a chosen metric for visualization | DataFrame with trend values |
| `calibration_health(window='8h', tolerance=None)` | Composite health score combining zero and span drift | DataFrame with health scores |

---

## Tips & Hints

!!! tip "Align windows with shift schedules"
    Use `window='8h'` to match typical manufacturing shifts. This makes it easy to correlate drift events with shift handovers and track which shifts see the most calibration degradation.

!!! info "Related modules"
    - [Multi-Sensor Validation](multi-sensor-validation.md) — cross-validate redundant sensors to confirm drift
    - [Anomaly Classification](anomaly-classification.md) — classify drift as a specific anomaly type
    - [Capability Trending](capability-trending.md) — track how sensor drift affects process capability

---

## See Also

- [Quality Control & SPC Guide](../../guides/quality.md) — narrative overview
- [API Reference](../../reference/ts_shape/events/quality/sensor_drift.md) — full parameter docs
