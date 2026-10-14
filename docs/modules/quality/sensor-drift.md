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

# Compare a pH probe against a 7.0 buffer reference
# (or pass reference_uuid= to use a reference signal instead)
drift = SensorDriftEvents(df, signal_uuid="ph_probe_01", reference_value=7.0)

# Track zero-point offset over 8-hour windows (threshold auto-set if None)
zero = drift.detect_zero_drift(window="8h", threshold=0.05)

# Monitor sensitivity changes (span drift; requires a reference)
span = drift.detect_span_drift(window="8h")

# Get a composite calibration health score, normalized to a +/-0.1 pH tolerance
health = drift.calibration_health(window="8h", tolerance=0.1)
print(health[["start", "health_score"]].head())
```

---

## Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `detect_zero_drift(window='8h', threshold=None)` | Track mean offset from reference or first-window baseline (zero-point drift) | DataFrame with `start`, `end`, `mean_offset`, `drift_rate`, `severity` |
| `detect_span_drift(window='8h')` | Detect sensitivity/gain changes over time (requires `reference_uuid` or `reference_value`) | DataFrame with `start`, `sensitivity`, `sensitivity_change_pct` |
| `drift_trend(window='1D', metric='mean')` | Rolling trend of the window `mean` or `std` | DataFrame with `start`, `value`, `slope`, `r_squared`, `direction` |
| `calibration_health(window='8h', tolerance=None)` | Composite 0-100 health score from bias vs. reference and precision | DataFrame with `start`, `bias`, `precision`, `health_score` |

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
