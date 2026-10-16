# EnvironmentalCompensationEvents

How much an environmental condition (temperature, humidity, pressure) moves an inline measurement: correlation and sensitivity per window, a compensated signal referred to a nominal condition, and the windows where the environment left the valid range.

`ts_shape.events.quality.environmental_compensation` · [Quality guide](../../guides/quality.md) · [API reference](../../reference/ts_shape/events/quality/environmental_compensation.md)

## Example

```python
import numpy as np
import pandas as pd
from ts_shape.events.quality.environmental_compensation import EnvironmentalCompensationEvents

t = pd.date_range("2024-01-01", periods=2880, freq="1min")
hall = 22 + 4 * np.sin(np.arange(2880) / 2880 * 4 * np.pi)
width = 50 + 0.01 * (hall - 20) + np.random.default_rng(0).normal(0, 0.002, 2880)
df = pd.concat([pd.DataFrame({"systime": t, "uuid": "line1:hall_temp", "value_double": hall}),
                pd.DataFrame({"systime": t, "uuid": "line1:width", "value_double": width})])

env = EnvironmentalCompensationEvents(df, "line1:width", env_uuid="line1:hall_temp", env_value=20.0)
print(env.sensitivity(window="8h")[["start", "coefficient", "r_squared"]])
corrected = env.compensated_signal()
hot = env.detect_env_exceedance(valid_range=(15.0, 25.0), window="1h")
```

## Methods

| Method | Returns |
|---|---|
| `detect_correlation(window='4h')` | per window: `start`, `end`, `duration_seconds`, `correlation`, `p_value`, `significant` |
| `sensitivity(window='8h')` | per window: `start`, `end`, `duration_seconds`, `coefficient`, `intercept`, `r_squared`, `unit_change_per_degree` |
| `compensated_signal(coefficient=None)` | per sample: `systime`, `value_double` (measured), `compensated_value`, `correction_applied` |
| `detect_env_exceedance(valid_range, window='1h')` | windows outside the range: `start`, `end`, `duration_seconds`, `env_mean`, `env_min`, `env_max`, `exceedance_type` (`high`/`low`/`both`), `severity` |

## Notes

- `env_uuid` is required; the constructor raises `ValueError` without it. `env_value` is the nominal environment for `compensated_signal`.
- Both signals are read from the same `value_column`. Each measurement sample is paired with the nearest environment sample within 5 minutes; unpaired samples are dropped.
- `compensated_value = measured − coefficient × (env − env_nominal)`, with `env_nominal` = `env_value` or else the environment mean. Without `coefficient`, one linear regression over all paired data estimates it.
- `significant` means Pearson `p < 0.05`. Windows with fewer than 5 paired samples are skipped. `unit_change_per_degree` equals `coefficient` (measurement units per environment unit).
- `severity` scales with how far the worst value lies outside `valid_range`, relative to the range width: `low` up to 5 %, `medium` up to 20 %, `high` up to 50 %, `critical` beyond.
