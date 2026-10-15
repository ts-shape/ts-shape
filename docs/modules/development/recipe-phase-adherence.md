# RecipePhaseAdherenceEvents

Checks every phase of a batch (heat-up, hold, cool-down, …) against a recipe spec for duration, hold value, peak, trough and ramp rate, with pass/fail and the failed criteria per phase occurrence. Input is a phase-name signal and one process value.

`ts_shape.events.development.recipe_phase_adherence` · [Engineering guide](../../guides/engineering.md) · [API reference](../../reference/ts_shape/events/development/recipe_phase_adherence.md)

## Example

```python
import numpy as np
import pandas as pd
import ts_shape
from ts_shape.events.development.recipe_phase_adherence import RecipePhaseAdherenceEvents

phase = ts_shape.make_id_signal("reactor2:phase", ["HEAT_UP", "HOLD", "COOL_DOWN"], hold=60, freq="30s")
temp = np.concatenate([np.linspace(20, 80, 60), np.full(60, 80.5), np.linspace(80, 25, 60)])
df = pd.concat([phase, pd.DataFrame({"systime": phase["systime"], "uuid": "reactor2:temp", "value_double": temp})])

spec = {"HEAT_UP": {"duration_seconds": (1200, 1800), "ramp_rate_max": 0.05},
        "HOLD": {"duration_seconds": (1800, 2000), "hold_value": (79.0, 81.0)},
        "COOL_DOWN": {"trough_value": (20.0, 30.0)}}
phases = RecipePhaseAdherenceEvents(df, "reactor2:phase", "reactor2:temp", spec).evaluate()
print(phases[["phase", "duration_seconds", "observed_ramp_per_s", "criteria_failed", "pass"]])
```

## Methods

| Method | Returns |
|---|---|
| `evaluate(interval_end='next_change', data_end=None)` | one row per phase occurrence: `start`, `end`, `phase`, `mean_value`, `min_value`, `max_value`, `observed_ramp_per_s`, `criteria_failed` (list), `pass` |

Spec keys per phase, all optional; ranges are inclusive `(min, max)` tuples: `duration_seconds` (`duration_s` is the deprecated spelling: it still works, with a `DeprecationWarning`, until the next major release), `hold_value` (phase mean), `peak_value` (phase max), `trough_value` (phase min), `ramp_rate_max` (absolute, value units per second).

## Notes

- Phase names are read from `value_string` of `phase_uuid`; a new occurrence starts whenever the name changes.
- With `interval_end="next_change"` (default) a phase ends at the next phase change and its values are taken from `[start, end)`; the last phase ends at `data_end` if given, else at its own last sample. `"last_sample"` spans the first to the last sample carrying the phase, so a change-only phase signal gives zero-length phases.
- `observed_ramp_per_s` = (last value − first value) / phase duration, not the steepest slope in the phase.
- A phase whose name is not in the spec passes with an empty `criteria_failed`. Giving both `duration_seconds` and `duration_s` for one phase raises `ValueError`. Phases without samples of `value_uuid` are omitted.
- The spec is a plain dict, so it can be loaded from YAML or JSON.
