# ValueDistributionEvents

> Is the process behaving normally? Mode changes, bimodality, per-window normality tests and percentile tracking for one numeric signal.

**Module:** `ts_shape.events.quality.value_distribution`  
**Guide:** [Quality Control & SPC](../../guides/quality.md)

---

## When to Use

Use before you trust capability indices or SPC limits, which assume a stable, roughly normal distribution. Typical questions:

- Does the signal switch between two operating modes (two products, two recipes, a valve that is open or shut)?
- Is the overall distribution bimodal, so a single mean and sigma are meaningless?
- Which hours are not normally distributed?
- Is the spread widening even though the mean holds steady?

---

## Quick Example

```python
from ts_shape.events.quality.value_distribution import ValueDistributionEvents

vd = ValueDistributionEvents(df, "extruder1:melt_pressure")

modes = vd.detect_mode_changes(window="1h", n_modes=2, min_separation=1.0)
print(modes[modes["mode_changed"]])

vd.detect_bimodal(min_samples=30)
vd.normality_windows(window="1h", alpha=0.05)
vd.percentile_tracking(percentiles=(5, 50, 95), window="1h")
```

---

## Key Methods

| Method | Shape | Returns |
|---|---|---|
| `detect_mode_changes(window, n_modes, min_separation)` | summary | per window: `start`, `dominant_mode`, `n_modes_detected`, `mode_values` (list), `mode_changed` |
| `detect_bimodal(min_samples)` | static | one row: `is_bimodal`, `dip_score`, `n_samples`, `mode_1`, `mode_2`, `valley` |
| `normality_windows(window, alpha, min_samples)` | summary | per window: `start`, `n_samples`, `is_normal`, `p_value`, `skewness`, `kurtosis` |
| `percentile_tracking(percentiles, window)` | summary | per window: `start`, `n_samples`, one column per percentile (`p5`, `p50`, `p95`, ...) |

## Notes

- Modes are peaks of a Gaussian kernel density estimate. `min_separation` is in units of the signal's overall standard deviation; a window is flagged `mode_changed` when its dominant mode moves by more than that from the previous window. Windows with fewer than 5 samples are skipped.
- `detect_bimodal` is a KDE-based approximation, not the full Hartigan dip test: `dip_score` = 1 − valley density / height of the smaller of the two tallest peaks, and `is_bimodal` is `dip_score > 0.5`. It returns an empty frame below `min_samples`.
- `normality_windows` uses Shapiro-Wilk (on a 5,000-sample random subset for larger windows). `kurtosis` is excess kurtosis (0 for a normal distribution).
