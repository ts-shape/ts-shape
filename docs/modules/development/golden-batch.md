# GoldenBatchDeviationEvents

Scores how far a batch's trajectory of one signal (reactor temperature, pH, torque) deviates from a golden reference batch: the largest pointwise deviation, the area between the curves, or a dynamic-time-warping shape distance; overall or per recipe phase.

`ts_shape.events.development.golden_batch` · [Engineering guide](../../guides/engineering.md) · [API reference](../../reference/ts_shape/events/development/golden_batch.md)

## Example

```python
import pandas as pd
import ts_shape
from ts_shape.events.development.golden_batch import GoldenBatchDeviationEvents

def batch(n, drift, seed):  # one batch: temperature plus its phase signal
    return pd.concat([
        ts_shape.make_timeseries(["reactor2:jacket_temp"], n_points=n, freq="1min", drift=drift, seed=seed),
        ts_shape.make_id_signal("reactor2:phase", ["HEAT_UP", "HOLD", "COOL_DOWN"], hold=n // 3, freq="1min"),
    ], ignore_index=True)

golden, new = batch(240, 40.0, 1), batch(300, 46.0, 2)

gb = GoldenBatchDeviationEvents(golden, "reactor2:jacket_temp", n_resample=256)
result = gb.compare(new, mode="dtw", dtw_band_frac=0.1)
print(result[["mode", "deviation_score", "max_abs_residual", "batch_time_at_max"]])
phases = gb.phase_breakdown(new, new, phase_uuid="reactor2:phase")
```

## Methods

| Method | Returns |
|---|---|
| `compare(batch_df, mode='pointwise', dtw_band_frac=0.1)` | one row: `start`, `end`, `mode`, `deviation_score`, `max_abs_residual`, `batch_time_at_max` |
| `phase_breakdown(batch_df, phase_df, phase_uuid)` | one row per phase occurrence: `phase`, `start`, `end`, `deviation_score`, `max_abs_residual` |

## Notes

- The constructor's DataFrame holds the golden batch; it raises `ValueError` without samples of `signal_uuid`. Both batches are mapped to batch time 0 to 1 and interpolated onto `n_resample` points (at least 4), so batches of different length are comparable. `batch_time_at_max` is on that scale (0.5 = halfway).
- Scores, all in signal units: `pointwise` is the largest absolute residual; `area` the integral of the absolute residual over batch time (the mean absolute deviation); `dtw` the DTW distance within a Sakoe-Chiba band of `dtw_band_frac` × `n_resample` points, divided by the warping-path length.
- `phase_breakdown` reads phase names from `value_string` of `phase_uuid` in `phase_df` (often the same frame as `batch_df`). A phase spanning the fraction [a, b] of the batch is compared with the fraction [a, b] of the golden batch, scored with the area method. This assumes the phase sequence and relative phase lengths are the same in both batches.
