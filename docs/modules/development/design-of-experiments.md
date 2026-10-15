# DesignOfExperimentsEvents

Recovers the runs of a DOE campaign from historian data, where factor signals (setpoints) were stepped through a few levels, and estimates the main effect of each factor level on a response signal. No regression model is fitted, so it works for whatever design was run: one-factor-at-a-time, full or fractional factorial, Latin square.

`ts_shape.events.development.design_of_experiments` · [Engineering guide](../../guides/engineering.md) · [API reference](../../reference/ts_shape/events/development/design_of_experiments.md)

## Example

```python
import numpy as np
import pandas as pd
from ts_shape.events.development.design_of_experiments import DesignOfExperimentsEvents

t = pd.date_range("2024-01-01", periods=480, freq="1min")
temp = np.repeat([200.0, 220.0, 200.0, 220.0], 120)
speed = np.tile(np.repeat([80.0, 120.0], 60), 4)
visc = 500 - 2 * (temp - 200) - 0.5 * (speed - 80) + np.random.default_rng(0).normal(0, 2, 480)
df = pd.concat([pd.DataFrame({"systime": t, "uuid": u, "value_double": v}) for u, v in
                [("pilot1:barrel_temp_sp", temp), ("pilot1:screw_speed_sp", speed), ("pilot1:melt_viscosity", visc)]])

doe = DesignOfExperimentsEvents(df, ["pilot1:barrel_temp_sp", "pilot1:screw_speed_sp"])
runs = doe.detect_runs(min_duration="10min", stability_tol=0.02)
result = doe.effects("pilot1:melt_viscosity", statistic="settled", min_duration="10min")
print(result[["factor", "level", "n_runs", "response_mean", "main_effect"]])
```

## Methods

| Method | Returns |
|---|---|
| `detect_runs(min_duration='5min', stability_tol=0.02, n_levels=None)` | one interval per run: `start`, `end`, `run_id`, one `factor__<uuid>_level` column per factor |
| `effects(response_uuid, statistic='mean', min_duration='5min', stability_tol=0.02, n_levels=None)` | per factor and level: `factor`, `level`, `n_runs`, `response_mean`, `response_std`, `main_effect` |

## Notes

- Runs: factor signals are pivoted to wide form and forward-filled. A new run starts whenever any factor changes by more than `stability_tol` (relative to its value, default 2 %) from the previous sample. Runs shorter than `min_duration` or with fewer than two samples are dropped as transients. A run's level is the mean factor value in it.
- Levels: without `n_levels`, each distinct (rounded) run mean is its own level, which is right for factors driven to explicit setpoints. For noisy factors pass `n_levels={uuid: k}` to quantile-bin the run means into `k` levels (labels 0 to k−1).
- Effects: the response is aggregated per run with `statistic`: `mean`, `median`, `max`, `min` or `settled` (mean of the final third, for responses that lag the factor change). `main_effect` = level mean − grand mean over all runs. Interactions are not estimated.
- The constructor raises `ValueError` for an empty `factor_uuids` list. `compute_effects` is the deprecated name of `effects`.
