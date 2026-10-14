# ChangePointEvents

> Find every point where a signal's level (or spread) shifted: tool changes, new material lots, recalibrations, corrected setpoints.

**Module:** `ts_shape.events.quality.change_point`  
**Guide:** [Quality Control & SPC](../../guides/quality.md)

---

## When to Use

Use for retrospective analysis of a single signal: "when did this dimension move, and by how much?" A control chart tells you that a point is out of limits. Change-point detection segments the history into stable stretches and places each shift at the exact sample. It needs no window length.

- **PELT** (`method="pelt"`, default) is exact: it returns the segmentation that minimizes a Gaussian cost plus a penalty per change.
- **Binary segmentation** (`method="binseg"`) is fast (O(n log n)) and approximate. Use it, or resample first, on long series with few changes, where PELT becomes quadratic. It warns above 20,000 samples.

---

## Quick Example

```python
from ts_shape.events.quality.change_point import ChangePointEvents

cp = ChangePointEvents(df, "line1:bore_diameter")

# One point event per shift, with the level before and after
shifts = cp.detect_change_points()
print(shifts[["systime", "mean_before", "mean_after", "shift_sigma"]])

# The stable stretches between the shifts
segments = cp.segment_statistics()

# Changes in spread too (e.g. a worn spindle that got noisier)
cp.detect_change_points(cost="normal_meanvar")
```

---

## Key Methods

| Method | Shape | Returns |
|---|---|---|
| `detect_change_points(cost, penalty, min_size, method)` | point | `systime` of the first sample of each new segment, `mean_before/after`, `std_before/after`, `shift`, `shift_sigma` |
| `segment_statistics(cost, penalty, min_size, method)` | interval | `start`, `end` (start of the next segment), `segment`, `mean`, `std`, `sample_count` |

## Tuning

- `penalty=None` uses the BIC (`2 log n` for `normal_mean`, `3 log n` for `normal_meanvar`). On stationary noise it finds no changes. Raise it to keep only larger shifts; lower it to see smaller ones.
- `normal_mean` estimates one noise level for the whole signal, robustly from the first differences, so level shifts do not inflate it. On a signal whose spread changes it over-segments the noisy stretch; use `normal_meanvar` there.
- `min_size` is the minimum number of samples per segment (default 5).

The low-level functions `pelt(x, ...)` and `binseg(x, ...)` work on any 1-D array and return the indices of new segments.
