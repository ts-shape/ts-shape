# ChangePointEvents

Finds every sample where a signal's level (or level and spread) shifts, such as tool changes, new material lots or recalibrations, and the stable segments between them. Needs no window length.

`ts_shape.events.quality.change_point` · [Quality guide](../../guides/quality.md) · [API reference](../../reference/ts_shape/events/quality/change_point.md)

## Example

```python
import ts_shape
from ts_shape.events.quality.change_point import ChangePointEvents

df = ts_shape.make_timeseries(["line1:bore_diameter"], n_points=600, noise=0.5)
df.loc[300:, "value_double"] += 2.0  # level shift after a tool change

cp = ChangePointEvents(df, "line1:bore_diameter")
print(cp.detect_change_points()[["systime", "mean_before", "mean_after", "shift_sigma"]])
segments = cp.segment_statistics()
spread_changes = cp.detect_change_points(cost="normal_meanvar")
```

## Methods

| Method | Returns |
|---|---|
| `detect_change_points(cost='normal_mean', penalty=None, min_size=5, method='auto')` | one point per change at the first sample of the new segment: `mean_before`, `mean_after`, `std_before`, `std_after`, `shift`, `shift_sigma` |
| `segment_statistics(...)` (same arguments) | one interval per segment: `start`, `end` (start of the next segment), `segment`, `mean`, `std`, `sample_count` |

## Notes

- `method="pelt"` is exact: it minimizes a Gaussian cost plus a penalty per change, but becomes quadratic on long series with few changes. `method="binseg"` (binary segmentation) is O(n log n) and approximate. `method="auto"` (the default) runs PELT on up to 20,000 samples (`PELT_WARN_SAMPLES`) and binary segmentation above; pass `method="pelt"` to get the exact result on a long series (it warns above 20,000 samples), or resample first.
- `penalty=None` uses the BIC (`2 log n` for `normal_mean`, `3 log n` for `normal_meanvar`) and finds no changes on stationary noise. Raise it to keep only larger shifts.
- `normal_mean` estimates one noise level for the whole signal, robustly from first differences, so level shifts do not inflate it. On a signal whose spread changes it over-segments the noisy stretch; use `normal_meanvar` there.
- `min_size` is the minimum number of samples per segment.
- `shift_sigma` is the shift in pooled segment standard deviations; NaN (never `inf`) when both segments are constant.
- The functions `pelt(x, ...)` and `binseg(x, ...)` in the same module work on any 1-D array and return the indices where new segments start.
