# SignalQualityEvents

> Detect data quality issues in a numeric signal: missing data, irregular sampling, out-of-range values, and data completeness.

**Module:** `ts_shape.events.quality.signal_quality`  
**Guide:** [Quality Control & SPC](../../guides/quality.md)

---

## When to Use

Use to validate data quality before analysis. Check for gaps in historian data, irregular sampling from PLCs, or sensor failures producing out-of-range values. Run this as a first step in any analytics pipeline to avoid drawing conclusions from incomplete or corrupted data.

---

## Quick Example

```python
from ts_shape.events.quality.signal_quality import SignalQualityEvents

sq = SignalQualityEvents(df, signal_uuid="tank_level")

# Find gaps where expected samples are missing
gaps = sq.detect_missing_data(expected_freq="1s", tolerance_factor=2.0)

# Check sampling regularity per hour
regularity = sq.sampling_regularity(window="1h")

# Flag physically impossible readings
out_of_range = sq.detect_out_of_range(min_value=0.0, max_value=100.0)

# Compute hourly completeness percentage
completeness = sq.data_completeness(window="1h", expected_freq="1s")
```

---

## Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `detect_missing_data(expected_freq='1s', tolerance_factor=2.0)` | Find gaps exceeding expected sampling interval | DataFrame with `start`, `end`, `gap_duration`, `expected_samples_missing` |
| `sampling_regularity(window='1h')` | Compute inter-sample interval statistics per window | DataFrame with `start`, `mean_interval`, `std_interval`, `min_interval`, `max_interval`, `regularity_score` |
| `detect_out_of_range(min_value, max_value)` | Flag intervals outside physical or expected range | DataFrame with `start`, `end`, `duration`, `min_observed`, `max_observed`, `direction` |
| `data_completeness(window='1h', expected_freq='1s')` | Calculate percentage of expected samples received | DataFrame with `start`, `expected_count`, `actual_count`, `completeness_pct` |

---

## Tips & Hints

!!! tip "Set tolerance_factor based on your data source"
    For PLC data with a strict 1-second cycle, a `tolerance_factor=2.0` works well. For historian data that may batch-compress, increase to 3.0 or higher to avoid false gap detections.

!!! info "Related modules"
    - [Outlier Detection](outlier-detection.md) — detect statistical outliers after confirming data quality
    - [Anomaly Classification](anomaly-classification.md) — classify flatlines and other anomaly types
    - [Multi-Sensor Validation](multi-sensor-validation.md) — cross-check redundant sensors for consistency

---

## See Also

- [Quality Control & SPC Guide](../../guides/quality.md) — narrative overview
- [API Reference](../../reference/ts_shape/events/quality/signal_quality.md) — full parameter docs
