# EnvironmentalCompensationEvents

> Quantify how much an environmental condition (temperature, humidity, pressure) moves an inline measurement, compensate for it, and flag when conditions leave the valid range.

**Module:** `ts_shape.events.quality.environmental_compensation`  
**Guide:** [Quality Control & SPC](../../guides/quality.md)

---

## When to Use

Use when an inline gauge reads differently as the hall warms up or the humidity changes: a laser micrometer with thermal drift, a scale affected by air pressure, a dimension measured on a part that has not cooled down. You get:

- the correlation between measurement and environment per window, with a significance flag;
- the sensitivity coefficient (measurement units per unit of environment, e.g. mm per °C);
- a compensated signal referred to a nominal condition;
- the windows where the environment was outside the sensor's specified operating range.

---

## Quick Example

```python
from ts_shape.events.quality.environmental_compensation import EnvironmentalCompensationEvents

env = EnvironmentalCompensationEvents(
    df,
    "line1:laser_gauge_width",
    env_uuid="line1:hall_temperature",
    env_value=20.0,   # nominal (reference) temperature for compensation
)

env.detect_correlation(window="4h")
sens = env.sensitivity(window="8h")

corrected = env.compensated_signal()   # coefficient estimated from all data
env.detect_env_exceedance(valid_range=(15.0, 30.0), window="1h")
```

---

## Key Methods

| Method | Shape | Returns |
|---|---|---|
| `detect_correlation(window)` | summary | per window: `window_start`, `correlation`, `p_value`, `significant` |
| `sensitivity(window)` | summary | per window: `window_start`, `coefficient`, `intercept`, `r_squared`, `unit_change_per_degree` |
| `compensated_signal(coefficient)` | point | per sample: `systime`, `uuid`, `value_double` (measured), `compensated_value`, `correction_applied` |
| `detect_env_exceedance(valid_range, window)` | interval | windows outside the range: `window_start`, `window_end`, `env_mean`, `env_min`, `env_max`, `exceedance_type` (`high`/`low`/`both`), `severity` |

## Notes

- The constructor needs `env_uuid` or `env_value` and raises `ValueError` otherwise. Every method reads the environment signal, so in practice pass `env_uuid`; with `env_value` alone all methods return empty frames.
- Both signals are read from the same `value_column`. Each measurement sample is paired with the nearest environment sample within 5 minutes; samples without one are dropped.
- Compensation: `compensated_value = measured − coefficient × (env − env_nominal)`. `env_nominal` is `env_value` if given, else the mean of the environment signal. Without `coefficient`, one linear regression over all paired data estimates it.
- `significant` means Pearson `p < 0.05`. Windows with fewer than 5 paired samples are skipped. `unit_change_per_degree` equals `coefficient`.
- `severity` scales with how far the worst value lies outside `valid_range`, relative to the range width: `low` up to 5 %, `medium` up to 20 %, `high` up to 50 %, `critical` beyond.
