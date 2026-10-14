# OutlierDetectionEvents

> Processes time series data to detect outliers based on specified statistical methods.

**Module:** `ts_shape.events.quality.outlier_detection`  
**Guide:** [Quality Control & SPC](../../guides/quality.md)

---

## When to Use

Use when you need to find anomalous data points in sensor signals before further analysis. Works with any numeric timeseries column. Choose Z-score for normal distributions, IQR for skewed data, MAD for noisy/heavy-tailed signals, or IsolationForest for complex multivariate patterns.

---

## Quick Example

```python
from ts_shape.events.quality.outlier_detection import OutlierDetectionEvents

detector = OutlierDetectionEvents(df, value_column="value_double")

# Z-score method for normally distributed signals
outliers_z = detector.detect_outliers_zscore(threshold=3.0)

# IQR method for skewed distributions
outliers_iqr = detector.detect_outliers_iqr(threshold=(1.5, 1.5))

# ML-based detection for complex patterns
outliers_if = detector.detect_outliers_isolation_forest(contamination=0.1)
```

---

## Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `detect_outliers_zscore(threshold=3.0)` | Z-score outlier detection for normally distributed data | DataFrame with outlier flags |
| `detect_outliers_iqr(threshold=(1.5, 1.5))` | IQR-based detection for skewed distributions | DataFrame with outlier flags |
| `detect_outliers_mad(threshold=3.5)` | Median Absolute Deviation detection for heavy-tailed signals | DataFrame with outlier flags |
| `detect_outliers_isolation_forest(contamination=0.1)` | ML-based detection for complex multivariate patterns | DataFrame with outlier flags |

---

## Tips & Hints

!!! tip "Choose the right method for your distribution"
    Z-score assumes normality and works best for Gaussian signals. For process data with occasional spikes or skew, prefer IQR or MAD. Use Isolation Forest when you suspect multivariate interactions between features.

!!! note "Statistics per signal"
    Each `uuid` in the frame is scored against its own statistics (`group_by_uuid=True`, the default), so a temperature around 80 and a pressure around 2 in one frame do not share one mean and standard deviation. Pass `group_by_uuid=False` to score all rows as one population.

!!! info "Related modules"
    - [Anomaly Classification](anomaly-classification.md) — classify detected outliers by type (spike, drift, oscillation)
    - [Signal Quality](signal-quality.md) — check data completeness before running outlier detection
    - [SPC Rules](spc.md) — rule-based detection using Western Electric Rules

---

## See Also

- [Quality Control & SPC Guide](../../guides/quality.md) — narrative overview
- [API Reference](../../reference/ts_shape/events/quality/outlier_detection.md) — full parameter docs
