# MultiSensorValidationEvents

> Cross-validate redundant inline sensors measuring the same process variable. Detects disagreement, identifies drifting sensors, and assesses measurement consensus.

**Module:** `ts_shape.events.quality.multi_sensor_validation`  
**Guide:** [Quality Control & SPC](../../guides/quality.md)

---

## When to Use

Use when you have 2+ redundant sensors measuring the same variable. Validates measurement consistency and identifies which sensor is drifting or failing. Common in critical process loops where redundant temperature, pressure, or flow sensors are installed for safety and accuracy.

---

## Quick Example

```python
from ts_shape.events.quality.multi_sensor_validation import MultiSensorValidationEvents

validator = MultiSensorValidationEvents(
    df,
    sensor_columns=["temp_sensor_1", "temp_sensor_2", "temp_sensor_3"],
)

# Flag windows where sensor spread exceeds 0.5 degrees
disagreements = validator.detect_disagreement(threshold=0.5, window="5m")

# Compute pairwise bias between all sensor pairs
bias = validator.pairwise_bias(window="1h")

# Identify which sensor deviates most from the group
outlier = validator.identify_outlier_sensor(window="1h", method="median")
print(outlier[["start", "outlier_sensor", "deviation"]].head())
```

---

## Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `detect_disagreement(threshold, window='5m')` | Flag windows where sensor spread exceeds threshold | DataFrame with disagreement events |
| `pairwise_bias(window='1h')` | Mean difference between all sensor pairs per window | DataFrame with pairwise bias values |
| `consensus_score(window='1h')` | Measurement consensus metric across all sensors | DataFrame with consensus scores |
| `identify_outlier_sensor(window='1h', method='median')` | Identify the sensor furthest from group consensus | DataFrame with outlier sensor labels |

---

## Tips & Hints

!!! tip "Set threshold from historical data"
    Determine the `detect_disagreement` threshold from a known-good calibration period. Compute the maximum spread during that period and add a small margin — this avoids false alarms from normal sensor-to-sensor variation.

!!! info "Related modules"
    - [Sensor Drift](sensor-drift.md) — deep-dive into drift for the sensor identified as outlier
    - [Signal Quality](signal-quality.md) — check individual sensor data quality before cross-validation
    - [Gauge R&R](gauge-rr.md) — formal measurement system analysis for sensor qualification

---

## See Also

- [Quality Control & SPC Guide](../../guides/quality.md) — narrative overview
- [API Reference](../../reference/ts_shape/events/quality/multi_sensor_validation.md) — full parameter docs
