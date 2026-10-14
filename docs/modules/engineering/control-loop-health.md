# ControlLoopHealthEvents

Health of a control loop from its setpoint and actual value (and optionally the controller output), independent of setpoint changes: error integrals per window, oscillation of the control error, output saturation, and a per-shift grade.

`ts_shape.events.engineering.control_loop_health` · [Engineering guide](../../guides/engineering.md) · [API reference](../../reference/ts_shape/events/engineering/control_loop_health.md)

## Example

```python
import numpy as np
import pandas as pd
from ts_shape.events.engineering.control_loop_health import ControlLoopHealthEvents

t = pd.date_range("2024-01-01", periods=8 * 3600, freq="1s")
pv = 50 + 0.8 * np.sin(np.arange(len(t)) / 30)  # hunting loop
df = pd.concat([pd.DataFrame({"systime": t, "uuid": u, "value_double": v}) for u, v in [
    ("flow_sp", np.full(len(t), 50.0)), ("flow_pv", pv), ("valve_out", np.clip(60 + 50 * (50 - pv), 0, 100))]])

loop = ControlLoopHealthEvents(df, setpoint_uuid="flow_sp", actual_uuid="flow_pv", output_uuid="valve_out")
integrals = loop.error_integrals(window="1h")
saturation = loop.output_saturation(upper_limit=100.0, lower_limit=0.0, window="1h")
print(loop.loop_health_summary(window="8h"))
```

## Methods

All arguments are keyword-only.

| Method | Returns (per window) |
|---|---|
| `error_integrals(window='1h', max_gap=None)` | `iae`, `ise`, `itae`, `bias`, `sample_count` |
| `detect_oscillation(window='5min', min_crossings=4)` | oscillating windows: `crossing_count`, `estimated_period_seconds`, `amplitude`, `damping_direction` |
| `output_saturation(upper_limit=100.0, lower_limit=0.0, window='1h')` | `pct_time_at_high`, `pct_time_at_low`, `pct_time_saturated`, `longest_saturation_seconds` (needs `output_uuid`) |
| `loop_health_summary(window='8h', oscillation_window='5min', min_crossings=4)` | `iae`, `bias`, `oscillation_count`, `pct_saturated`, `health_grade` |

## Notes

- Error = actual − setpoint (all signals from `value_double`); a positive `bias` means the process runs above setpoint.
- `error_integrals` with `max_gap=None` weights each error by the time since the previous sample. With `max_gap` each error is held until the next sample and longer intervals contribute nothing, so a communication gap is not integrated as one large error.
- Output samples within 1 % of the `upper_limit − lower_limit` range of a limit count as saturated. Without `output_uuid`, `output_saturation` returns an empty frame.
- `loop_health_summary` detects oscillation in `oscillation_window` slices (a whole shift would almost always have enough zero-crossings) and counts the episodes starting in each report window. `health_grade` counts issues (IAE above 1.5 × the median IAE of all windows, any oscillation, more than 5 % saturated): `good` with none, `fair` with one, `poor` with more.
