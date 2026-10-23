# MachineStateEvents

Run and idle intervals, transitions, rapid toggling and state-data quality from a boolean run signal (True = running).

`ts_shape.events.production.machine_state` · [Production guide](../../guides/production.md) · [API reference](../../reference/ts_shape/events/production/machine_state.md)

## Example

```python
import pandas as pd
from ts_shape.events.production.machine_state import MachineStateEvents

df = pd.DataFrame({
    "systime": pd.date_range("2024-01-01 06:00", periods=120, freq="30s"),
    "uuid": "line1:running",
    "value_bool": [True] * 20 + [False] * 5 + [True] * 40 + [False] * 10 + [True] * 45,
})
events = MachineStateEvents(df, run_state_uuid="line1:running")
result = events.detect_run_idle(min_duration="1min")
print(result[["start", "end", "state", "duration_seconds"]])
transitions = events.transition_events()
quality = events.state_quality_metrics()
```

## Methods

| Method | Returns |
|---|---|
| `detect_run_idle(min_duration='0s', interval_end=None, data_end=None)` | `start`, `end`, `state` (`run`/`idle`), `duration_seconds` |
| `transition_events()` | `systime`, `transition` (`idle_to_run`/`run_to_idle`), `time_since_last_transition_seconds` |
| `detect_rapid_transitions(threshold='5s', min_count=3)` | bursts of at least `min_count` transitions within `threshold`: `start`, `end`, `transition_count`, `duration_seconds` |
| `state_quality_metrics(interval_end=None)` | one row: `total_transitions`, `avg_run_duration_seconds`, `avg_idle_duration_seconds`, `run_idle_factor` (run / idle time; NaN without idle time), `data_gaps_detected`, `rapid_transitions_detected` |

## Notes

- Intervals end at the next transition by default (`interval_end=None` resolves to `"next_change"`), so a PLC that logs a row only on each change is measured correctly. With `specs=` given, the signal's spec decides, and gaps longer than its `max_gap` are no data. `interval_end="last_sample"` ends each interval at its own last sample (zero duration on a change-only signal).
- `data_end` bounds the last, still-open interval. A naive `data_end` (or text) on tz-aware data is local time of the data's timezone.
- A missing state (`None`, `NaN`, `pd.NA`) is unknown, not idle: the time from it to the next known sample is in no run or idle interval, and no transition. `state_quality_metrics` averages and the run/idle ratio use the observed time; an average over no intervals is NaN.
- `min_duration` drops short intervals, e.g. a few seconds of sensor glitch.
- For a numeric state, pass `value_range=(low, high)` to the constructor.
- For a text state (`"RUN"`, `"IDLE"`, …), pass `run_states={"RUN", ...}` with the values that mean running. Without it, every non-empty text counts as running, and the constructor raises a `DataQualityWarning` saying so.
- `state_quality_metrics(..., as_frame=False)` returns the deprecated dict.
