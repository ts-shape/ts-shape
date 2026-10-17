# SetpointChangeEvents

Setpoint steps and ramps, and the response of the actual value after each change: settling time, rise time, overshoot and undershoot, decay rate and oscillation frequency.

`ts_shape.events.engineering.setpoint_events` · [Engineering guide](../../guides/engineering.md) · [API reference](../../reference/ts_shape/events/engineering/setpoint_events.md)

## Example

```python
import numpy as np
import pandas as pd
from ts_shape.events.engineering.setpoint_events import SetpointChangeEvents

t = pd.date_range("2024-01-01", periods=3600, freq="1s")
sp = np.where(np.arange(3600) < 1200, 150.0, 170.0)
pv = sp - 20.0 * (np.arange(3600) >= 1200) * np.exp(-np.clip(np.arange(3600) - 1200, 0, None) / 60)
df = pd.concat([pd.DataFrame({"systime": t, "uuid": "temp_sp", "value_double": sp}),
                pd.DataFrame({"systime": t, "uuid": "temp_pv", "value_double": pv})])

sc = SetpointChangeEvents(df, setpoint_uuid="temp_sp")
steps = sc.detect_setpoint_steps(min_delta=5.0, min_hold="30s")
settle = sc.time_to_settle("temp_pv", settle_pct=0.02, hold="30s", lookahead="15min")
print(sc.control_quality_metrics("temp_pv", tol=0.5, lookahead="15min").T)
```

## Methods

| Method | Returns (one row per setpoint change) |
|---|---|
| `detect_setpoint_steps(min_delta, min_hold='0s')` | `magnitude`, `prev_level`, `new_level` |
| `detect_setpoint_ramps(min_rate, min_duration='0s')` | ramp intervals: `avg_rate`, `delta` |
| `detect_setpoint_changes(min_delta=0.0, min_rate=None)` | steps and ramps in one table with `change_type` |
| `time_to_settle(actual_uuid, tol=0.0, settle_pct=None, hold='0s', lookahead='10min')` | `t_settle_seconds`, `settled`, `settled_within_window` |
| `time_to_settle_derivative(actual_uuid, rate_threshold=0.01)` | `t_settle_seconds`, `settled`, `final_rate` |
| `rise_time(actual_uuid, start_pct=0.1, end_pct=0.9)` | `rise_time_seconds`, `reached_end` |
| `overshoot_metrics(actual_uuid, window='10min', deadband=0.0)` | `overshoot_abs`, `overshoot_pct`, `overshoot_ratio`, `t_peak_seconds`, `undershoot_abs`, `undershoot_pct`, `undershoot_ratio`, `oscillation_count`, `oscillation_amplitude` |
| `decay_rate(actual_uuid, lookahead='10min')` | `decay_rate_lambda`, `fit_quality_r2` |
| `oscillation_frequency(actual_uuid, window='10min', deadband=0.0)` | `oscillation_freq_hz`, `period_seconds` |
| `control_quality_metrics(actual_uuid, tol=0.0, settle_pct=None, deadband=None)` | all of the above per change |

## Notes

- A step is a change of at least `min_delta` whose new level holds for `min_hold` (no further change within it). `filter_noise=True` drops changes below `noise_threshold`.
- A ramp is a run where |dS/dt| ≥ `min_rate` (setpoint units per second) lasting at least `min_duration`.
- The settling time (Ogata) is the time after which the actual value enters the ±`tol` band around the new setpoint and stays inside it until the end of the evaluation window, not the first entry: an underdamped response crosses the band on every swing. The window ends after `lookahead` or at the next setpoint change, whichever is first. `settled_within_window` is False (and `t_settle_seconds` NaN) when the value is outside the band at the window's end; `settled` additionally requires the final in-band stretch to last `hold`. `settle_pct` (fraction of the step size, e.g. 0.02) takes precedence over `tol`.
- `overshoot_pct` / `undershoot_pct` are percent of the step size (Ogata's maximum percent overshoot); `overshoot_ratio` / `undershoot_ratio` are the same as fractions.
- `oscillation_count` counts crossings of a band (`deadband`) around the new setpoint: a crossing counts once the error leaves the band on the other side, so noise around the setpoint is not counted. `control_quality_metrics` uses the settle band (`tol`, or `settle_pct` of each step) by default; `overshoot_metrics` and `oscillation_frequency` default to `deadband=0`, every sign change of the error. `oscillation_freq_hz` uses the same crossings.
- `decay_rate` fits error(t) = A · exp(−λ t).
- The actual signal must be in the same frame as the setpoint signal.
