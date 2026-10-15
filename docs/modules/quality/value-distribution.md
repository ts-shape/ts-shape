# ValueDistributionEvents

Distribution changes of a numeric signal: the dominant mode per window and when it moves, a bimodality check, normality per window, and percentile bands over time.

`ts_shape.events.quality.value_distribution` · [Quality guide](../../guides/quality.md) · [API reference](../../reference/ts_shape/events/quality/value_distribution.md)

## Example

```python
import ts_shape
from ts_shape.events.quality.value_distribution import ValueDistributionEvents

df = ts_shape.make_timeseries(["extruder1:melt_pressure"], n_points=1440, freq="1min", baseline=180.0, noise=2.0)
df.loc[720:, "value_double"] += 12.0  # new material lot

vd = ValueDistributionEvents(df, "extruder1:melt_pressure")
modes = vd.detect_mode_changes(window="1h", n_modes=2, min_separation=1.0)
result = modes[modes["mode_changed"]]
print(result[["start", "dominant_mode", "n_modes_detected", "mode_changed"]])
bimodal = vd.detect_bimodal(min_samples=30)
bands = vd.percentile_tracking(percentiles=(5, 50, 95), window="1h")
```

## Methods

All arguments are keyword-only.

| Method | Returns |
|---|---|
| `detect_mode_changes(window='1h', n_modes=2, min_separation=1.0)` | per window: `dominant_mode`, `n_modes_detected`, `mode_values` (list), `mode_changed` |
| `detect_bimodal(min_samples=30)` | one row over all data: `is_bimodal`, `dip_score`, `n_samples`, `mode_1`, `mode_2`, `valley` |
| `normality_windows(window='1h', alpha=0.05, min_samples=20)` | per window: `n_samples`, `is_normal`, `p_value`, `skewness`, `kurtosis` |
| `percentile_tracking(percentiles=(5, 25, 50, 75, 95), window='1h')` | per window: `n_samples` and one column per percentile (`p5`, `p50`, …) |

## Notes

- Modes are peaks of a Gaussian kernel density estimate. `min_separation` is in units of the signal's overall standard deviation; a window is `mode_changed` when its dominant mode moves by more than that from the previous window. Windows with fewer than 5 samples are skipped.
- `detect_bimodal` is a KDE-based approximation, not Hartigan's dip test: `dip_score` = 1 − valley density / height of the smaller of the two tallest peaks, and `is_bimodal` is `dip_score > 0.5`. It returns an empty frame below `min_samples`.
- `normality_windows` uses Shapiro-Wilk (on a random 5,000-sample subset for larger windows). `kurtosis` is excess kurtosis (0 for a normal distribution).
