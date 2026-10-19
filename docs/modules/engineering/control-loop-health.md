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
result = loop.loop_health_summary(window="8h")
print(result[["iae", "bias", "oscillation_count", "saturated_pct", "health_grade"]])
```

## Methods

All arguments are keyword-only.

| Method | Returns (per window) |
|---|---|
| `error_integrals(window='1h', max_gap=None)` | `iae`, `ise`, `itae`, `bias`, `sample_count` |
| `detect_oscillation(window='5min', min_crossings=4)` | oscillating windows: `crossing_count`, `estimated_period_seconds`, `amplitude`, `damping_direction` |
| `output_saturation(upper_limit=100.0, lower_limit=0.0, window='1h')` | `time_at_high_pct`, `time_at_low_pct`, `time_saturated_pct`, `longest_saturation_seconds` (needs `output_uuid`) |
| `loop_health_summary(window='8h', oscillation_window='5min', min_crossings=4)` | `iae`, `bias`, `oscillation_count`, `saturated_pct`, `health_grade` |

## Notes

- Error = actual − setpoint (all signals from `value_double`); a positive `bias` means the process runs above setpoint.
- `error_integrals` with `max_gap=None` weights each error by the time since the previous sample. With `max_gap` each error is held until the next sample and longer intervals contribute nothing, so a communication gap is not integrated as one large error. A step that crosses a window boundary is split there, so no time is lost at window edges and a window without samples of its own still gets the time a held error covers in it; no time is integrated before the first or after the last sample. `itae` measures time from the window start (`∫ (t − window_start)·|e| dt`). `bias` is the mean error of the samples in the window.
- Output samples within 1 % of the `upper_limit − lower_limit` range of a limit count as saturated. `output_saturation` measures time, not samples: each output sample is held until the next one, the percentages are shares of the observed time in the window (the time after the last sample is not observed), and `longest_saturation_seconds` is the longest stretch of saturated time inside the window. Without `output_uuid`, `output_saturation` returns an empty frame.
- `detect_oscillation` counts sign changes of the error; a zero (or NaN) error carries the last non-zero sign, so `1, 0, -1` is one crossing and touching zero (`1, 0, 1`) is none. A crossing is timed at the sample where the carried sign changes; `estimated_period_seconds` is twice the mean time between crossings.
- `loop_health_summary` detects oscillation in `oscillation_window` slices (a whole shift would almost always have enough zero-crossings) and counts the episodes starting in each report window. `health_grade` counts issues (IAE above 1.5 × the median IAE of all windows, any oscillation, more than 5 % saturated): `good` with none, `fair` with one, `poor` with more.
