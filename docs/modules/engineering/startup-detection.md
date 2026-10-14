# StartupDetectionEvents

Equipment startups in a numeric signal (speed, current, temperature): by threshold crossing with optional hysteresis, by sustained positive slope, by an adaptive baseline, or across several signals; plus startup quality, phase tracking and failed startups.

`ts_shape.events.engineering.startup_events` · [Engineering guide](../../guides/engineering.md) · [API reference](../../reference/ts_shape/events/engineering/startup_events.md)

## Example

```python
import numpy as np
import pandas as pd
from ts_shape.events.engineering.startup_events import StartupDetectionEvents

t = pd.date_range("2024-01-01", periods=1800, freq="1s")
current = np.interp(np.arange(1800), [0, 300, 360, 1200, 1230, 1799], [0.5, 0.5, 12, 12, 0.5, 0.5])
df = pd.DataFrame({"systime": t, "uuid": "motor_current", "value_double": current})

sd = StartupDetectionEvents(df, target_uuid="motor_current")
startups = sd.detect_startup_by_threshold(threshold=5.0, hysteresis=(5.0, 4.5), min_above="10s")
print(sd.assess_startup_quality(startups))
failed = sd.detect_failed_startups(threshold=5.0, completion_threshold=10.0, max_completion_time="5min")
```

## Methods

| Method | Returns |
|---|---|
| `detect_startup_by_threshold(threshold, hysteresis=None, min_above='0s')` | `start`, `end` (= start + `min_above`), `method`, `threshold` |
| `detect_startup_by_slope(min_slope, slope_window='0s', min_duration='0s')` | runs with slope ≥ `min_slope`: `min_slope`, `avg_slope` |
| `detect_startup_adaptive(baseline_window=None, sensitivity=2.0, min_above='10s', lookback_periods=5)` | `adaptive_threshold`, `baseline_mean`, `baseline_std` |
| `detect_startup_multi_signal(signals, logic='all', time_tolerance='30s')` | `signals_triggered`, `signal_details` |
| `assess_startup_quality(startup_events)` | per startup: `duration`, `smoothness_score`, `anomaly_flags`, `value_change`, `avg_rate`, `max_value`, `stability_score` |
| `track_startup_phases(phases, min_phase_duration='5s')` | per phase: `phase_name`, `phase_number`, `start`, `end`, `duration`, `next_phase`, `completed` |
| `detect_failed_startups(threshold, min_rise_duration='5s', max_completion_time='5min', completion_threshold=None, required_stability='10s')` | `failure_reason`, `max_value_reached`, `time_to_failure` |

## Notes

- Threshold: a startup begins where the signal rises to or above `threshold` (or the `enter` level of `hysteresis=(enter, exit)`) and counts only if it stays at or above the exit level for `min_above`.
- Slope is in signal units per second, between consecutive samples by default; a positive `slope_window` takes it over that trailing window, which smooths noise.
- Adaptive: threshold = baseline mean + `sensitivity` × baseline std, from the samples before each point (the last `lookback_periods` samples, or the span `baseline_window`), so a rising value does not raise its own threshold.
- `detect_startup_multi_signal` takes `{uuid: {"method": "threshold" | "slope", ...}}`; `logic="all"` requires all signals within `time_tolerance`, `"any"` one of them.
- A failed startup crosses `threshold` for at least `min_rise_duration` but does not reach `completion_threshold` (default 2 × `threshold`) within `max_completion_time`, or drops back before holding it for `required_stability`.
- `track_startup_phases` phases use `condition` `threshold` (`min_value`), `range` (`min_value`, `max_value`) or `slope` (`min_slope`).
