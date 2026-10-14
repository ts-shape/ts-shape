# RecipePhaseAdherenceEvents

> Check every phase of a batch (heat-up, hold, cool-down, ...) against a recipe spec for duration, hold value, ramp rate, peak and trough, with pass/fail and the failed criteria per phase.

**Module:** `ts_shape.events.development.recipe_phase_adherence`

---

## When to Use

Use when a batch recipe runs as named phases and development has defined what each phase must achieve. With a phase-name signal and one process value you get one event per phase occurrence, telling you whether it met its spec and which criterion failed. The spec is a plain dict, so a process engineer can write it by hand or load it from YAML or JSON.

---

## Quick Example

```python
from ts_shape.events.development.recipe_phase_adherence import RecipePhaseAdherenceEvents

spec = {
    "HEAT_UP": {"duration_s": (1200, 1800), "ramp_rate_max": 0.05},
    "HOLD": {"duration_s": (3600, 3900), "hold_value": (79.0, 81.0), "peak_value": (79.0, 82.0)},
    "COOL_DOWN": {"trough_value": (20.0, 30.0)},
}

rpa = RecipePhaseAdherenceEvents(
    df,
    "reactor2:phase_name",
    "reactor2:product_temp",
    spec,
)

phases = rpa.evaluate()
print(phases.loc[~phases["pass"], ["phase", "start", "criteria_failed"]])
```

---

## Key Methods

| Method | Shape | Returns |
|---|---|---|
| `evaluate()` | interval | one row per phase occurrence: `start`, `end`, `duration_seconds`, `uuid`, `phase`, `mean_value`, `min_value`, `max_value`, `observed_ramp_per_s`, `criteria_failed` (list), `pass` |

## Spec Keys

All keys are optional per phase; ranges are inclusive `(min, max)` tuples.

| Key | Checks |
|---|---|
| `duration_s` | phase duration in seconds |
| `hold_value` | mean value during the phase |
| `peak_value` | maximum value during the phase |
| `trough_value` | minimum value during the phase |
| `ramp_rate_max` | absolute end-to-end ramp rate (value units per second) |

## Notes

- Phase names are read from `value_string` of `phase_uuid`; a new phase occurrence starts whenever the name changes. A phase runs from its first to its last phase-signal sample, so with a change-only phase signal (one row per phase) the duration is 0. Log the phase periodically, or resample it first.
- `observed_ramp_per_s` = (last value − first value) / phase duration. It is not the steepest slope within the phase.
- A phase whose name is not in the spec passes with an empty `criteria_failed` list. Malformed range entries are skipped.
- Phases without samples of `value_uuid` are omitted.
