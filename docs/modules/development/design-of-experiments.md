# DesignOfExperimentsEvents

> Recover the run structure of a DOE campaign from continuous process data and estimate the main effect of each factor level on a response.

**Module:** `ts_shape.events.development.design_of_experiments`

---

## When to Use

Use when an experiment was run on a real line or pilot plant and all you have is the historian: factor signals (temperature setpoint, screw speed, feed rate) that were stepped through a few levels, and a response signal (yield, viscosity, wall thickness). The detector finds the intervals where every factor sat still at one level, tags each run with its factor-level combination, and averages the response per level. It fits no regression model, so it works for whatever design was actually run: one-factor-at-a-time, full or fractional factorial, Latin square.

---

## Quick Example

```python
from ts_shape.events.development.design_of_experiments import DesignOfExperimentsEvents

doe = DesignOfExperimentsEvents(
    df,
    ["pilot1:barrel_temp_sp", "pilot1:screw_speed_sp"],
)

runs = doe.detect_runs(min_duration="10min", stability_tol=0.02)

effects = doe.effects(
    "pilot1:melt_viscosity",
    statistic="settled",    # mean of the last third of each run
    min_duration="10min",
)
print(effects.sort_values("main_effect"))
```

---

## Key Methods

| Method | Shape | Returns |
|---|---|---|
| `detect_runs(min_duration, stability_tol, n_levels)` | interval | `start`, `end`, `duration_seconds`, `uuid`, `run_id`, one `factor__<uuid>_level` column per factor |
| `effects(response_uuid, statistic, min_duration, stability_tol, n_levels)` | summary | one row per factor and level: `factor`, `level`, `n_runs`, `response_mean`, `response_std`, `main_effect` |

## Notes

- **Runs.** Factor signals are pivoted to wide form and forward-filled. A new run starts whenever any factor changes by more than `stability_tol` (relative to its value, default 2 %) from the previous sample. Runs shorter than `min_duration` are dropped as transients. The level is the mean factor value within the run.
- **Levels.** Without `n_levels`, each distinct (rounded) run mean is its own level, which is right when the factors were driven to explicit setpoints. For noisy factors pass `n_levels={uuid: k}` to quantile-bin the run means into `k` levels (labels 0 to k−1).
- **Effects.** The response is aggregated per run with `statistic`: `"mean"`, `"median"`, `"max"`, `"min"` or `"settled"` (mean of the final third, for responses that lag the factor change). `main_effect` = level mean − grand mean over all runs. Interactions are not estimated.
- All method arguments except `response_uuid` are keyword-only. The constructor raises `ValueError` for an empty `factor_uuids` list.
