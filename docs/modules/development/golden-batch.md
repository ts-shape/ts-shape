# GoldenBatchDeviationEvents

> Score how far a new batch's trajectory deviates from a golden (reference) batch: pointwise, as the area between the curves, or as a dynamic-time-warping shape distance.

**Module:** `ts_shape.events.development.golden_batch`

---

## When to Use

Use in batch processes (pharma, food, specialty chemicals) where one run that delivered the target product serves as the reference trajectory. Compare every new batch of the same signal (reactor temperature, pH, torque) against it:

- `pointwise` answers "what was the largest deviation, and when in the batch?";
- `area` measures cumulative deviation over the whole batch;
- `dtw` compares the trajectory's *shape* even when the new batch ran faster or slower in places.

---

## Quick Example

```python
from ts_shape.events.development.golden_batch import GoldenBatchDeviationEvents

gb = GoldenBatchDeviationEvents(golden_df, "reactor2:jacket_temp", n_resample=256)

gb.compare(batch_df, mode="pointwise")
gb.compare(batch_df, mode="area")
gb.compare(batch_df, mode="dtw", dtw_band_frac=0.1)

gb.phase_breakdown(batch_df, batch_df, phase_uuid="reactor2:phase_name")
```

---

## Key Methods

| Method | Shape | Returns |
|---|---|---|
| `compare(batch_df, mode, dtw_band_frac)` | interval | one row for the batch: `start`, `end`, `duration_seconds`, `uuid`, `mode`, `deviation_score`, `max_abs_residual`, `batch_time_at_max` |
| `phase_breakdown(batch_df, phase_df, phase_uuid)` | interval | one row per phase occurrence: `start`, `end`, `duration_seconds`, `phase`, `deviation_score`, `max_abs_residual` |

## Notes

- **Normalized batch time.** The constructor's DataFrame holds the golden batch. Both batches are mapped to batch time 0 to 1 and linearly interpolated onto `n_resample` evenly spaced points (at least 4; default 256), so batches of different wall-clock length are comparable. `batch_time_at_max` is on that 0-to-1 scale (0.5 = halfway through the batch).
- **Scores.** `pointwise`: the largest absolute residual (equal to `max_abs_residual`). `area`: trapezoid integral of the absolute residual over batch time 0 to 1, i.e. the mean absolute deviation. `dtw`: DTW distance with a Sakoe-Chiba band of `dtw_band_frac` × `n_resample` points, divided by the warping-path length. All are in signal units.
- **Phases.** `phase_breakdown` reads phase names from `value_string` of `phase_uuid` in `phase_df` (often the same frame as `batch_df`) and scores each contiguous phase with the area method. Each phase segment is resampled onto the full grid and compared with the *whole* golden trajectory, not with the matching phase of the golden batch, so the per-phase scores are only meaningful when the golden reference you pass covers that phase alone. `phase_uuid` is keyword-only.
- The constructor raises `ValueError` when the golden batch has no samples for `signal_uuid`.
