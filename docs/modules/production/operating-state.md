# OperatingStateEvents

Operating states inferred from one continuous signal (spindle current, power, vibration) when the machine logs no explicit run/idle flag: state intervals, typical durations per state, unexpected state transitions and states held too long. With a boolean run signal, use [MachineStateEvents](machine-state.md) instead.

`ts_shape.events.production.operating_state` · [Production guide](../../guides/production.md) · [API reference](../../reference/ts_shape/events/production/operating_state.md)

## Example

```python
import numpy as np
import pandas as pd
from ts_shape.events.production.operating_state import OperatingStateEvents

# spindle current every 30 s: idle ~2 A and cutting ~18 A, alternating every 15 min
t = pd.date_range("2024-01-01 06:00", periods=480, freq="30s")
current = np.tile([2.0] * 30 + [18.0] * 30, 8)
current[240:270] = 18.0  # one cut runs on for 30 min ...
current[330:360] = 2.0   # ... and the line waits 30 min later on
rng = np.random.default_rng(0)
df = pd.DataFrame({"systime": t, "uuid": "spindle:current",
                   "value_double": current + rng.normal(0, 0.3, len(t))})

ops = OperatingStateEvents(df, "spindle:current")
states = ops.infer_states(n_states=2, min_duration="1min")
print(states[["start", "end", "state_label", "mean_value"]].head(4))
stats = ops.state_duration_statistics(n_states=2)
too_long = ops.dwell_time_violation(1, "25min")   # state 1 = "high"
print(too_long[["start", "state_label", "duration_seconds", "excess_seconds"]])
sequence = ops.unusual_state_sequence(expected_order=[0, 1, 0])
```

The cut that ran on is the one `high` interval longer than 25 minutes; it is reported with its excess over the limit.

## Methods

| Method | Returns |
|---|---|
| `infer_states(n_states=2, min_duration='0s')` | intervals: `start`, `end`, `state` (0 = lowest), `state_label`, `mean_value`, `duration_seconds` |
| `state_duration_statistics(n_states=2)` | one row per state: `count`, `mean_duration`, `std_duration`, `min_duration`, `max_duration`, `total_duration` (seconds) |
| `unusual_state_sequence(n_states=2, expected_order=None)` | one row per transition: `transition_time`, `from_state`, `to_state`, `from_label`, `to_label`, `is_unusual` |
| `dwell_time_violation(state, max_duration, n_states=2)` | intervals of `state` longer than `max_duration`: `duration_seconds`, `excess_seconds` |

## Notes

- States are value bins at the quantiles of the signal: with `n_states=2` the split is the median. A machine that spends most of its time in one state gets that state split in two, so choose `n_states` (and the time span) so the states are of similar weight, or threshold the signal yourself and use [MachineStateEvents](machine-state.md).
- Labels are `low`/`high` for two states, `low`/`mid`/`high` for three, `state_0` … otherwise. `state` numbers count from the lowest values.
- An interval ends at its last sample, so a phase of 15 minutes sampled every 30 s lasts 870 s.
- Without `expected_order`, a transition is normal when it accounts for at least 10 % of all transitions. `expected_order=[0, 1, 0]` allows exactly the transitions 0→1 and 1→0.
- Samples without a value are ignored; with fewer valued samples than `n_states` the result is empty.
