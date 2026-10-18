# OutlierDetectionEvents

Outlier samples by Z-score, IQR, MAD or Isolation Forest, scored per signal, with outliers close in time grouped together.

`ts_shape.events.quality.outlier_detection` · [Quality guide](../../guides/quality.md) · [API reference](../../reference/ts_shape/events/quality/outlier_detection.md)

## Example

```python
import ts_shape
from ts_shape.events.quality.outlier_detection import OutlierDetectionEvents

df = ts_shape.make_timeseries(["press:force", "press:temp"], n_points=500, n_outliers=4)

detector = OutlierDetectionEvents(df, value_column="value_double", time_threshold="5min")
outliers = detector.detect_outliers_zscore(n_sigma=3.0)
print(outliers[["systime", "source_uuid", "value_double", "severity"]])
robust = detector.detect_outliers_mad(n_sigma=3.5)
```

## Methods

| Method | Use for |
|---|---|
| `detect_outliers_zscore(n_sigma=3.0)` | roughly normal signals |
| `detect_outliers_iqr(threshold=(1.5, 1.5))` | skewed signals; lower and upper IQR multipliers |
| `detect_outliers_mad(n_sigma=3.5)` | heavy-tailed or noisy signals (modified z-score) |
| `detect_outliers_isolation_forest(contamination=0.1, random_state=42)` | needs scikit-learn |

Each returns the outlier rows as point events: the original columns, `source_uuid` (the signal), `uuid` (the event UUID) and `severity` (the score).

## Notes

- Each `uuid` is scored against its own statistics (`group_by_uuid=True`, default); `group_by_uuid=False` pools all rows into one population. `signal_uuids=` limits the analysis to some signals.
- Outliers less than `time_threshold` (default `5min`) apart form one group, and only the first and last row of a group are returned. `include_singles=False` drops isolated outliers.
- NaN values are ignored. When more than half of the values are identical (MAD = 0), the MAD method falls back to the mean absolute deviation; if that is 0 too, nothing is flagged.
