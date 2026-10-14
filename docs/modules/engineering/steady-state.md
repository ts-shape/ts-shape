# SteadyStateDetectionEvents

Steady-state intervals (rolling standard deviation below a threshold) and transient intervals of a signal, the share of time steady, and the operating band of each steady interval.

`ts_shape.events.engineering.steady_state_detection` · [Engineering guide](../../guides/engineering.md) · [API reference](../../reference/ts_shape/events/engineering/steady_state_detection.md)

## Example

```python
import numpy as np
import pandas as pd
from ts_shape.events.engineering.steady_state_detection import SteadyStateDetectionEvents

t = pd.date_range("2024-01-01", periods=7200, freq="1s")
level = np.interp(np.arange(7200), [0, 1800, 2400, 5400, 6000, 7199], [150, 150, 180, 180, 160, 160])
df = pd.DataFrame({"systime": t, "uuid": "reactor_temp",
                   "value_double": level + np.random.default_rng(0).normal(0, 0.3, 7200)})

ss = SteadyStateDetectionEvents(df, "reactor_temp")
print(ss.detect_steady_state(window="5min", std_threshold=1.0, min_duration="10min"))
summary = ss.steady_state_statistics(window="5min", std_threshold=1.0)
bands = ss.steady_state_value_bands(window="5min", std_threshold=1.0)
```

## Methods

All arguments are keyword-only.

| Method | Returns |
|---|---|
| `detect_steady_state(window='5min', std_threshold=1.0, min_duration='10min')` | `start`, `end`, `mean_value`, `std_value`, `duration_seconds` |
| `detect_transient_periods(window='5min', std_threshold=1.0)` | `start`, `end`, `max_std`, `duration_seconds` |
| `steady_state_statistics(window='5min', std_threshold=1.0, min_duration='10min')` | one row: `total_steady_seconds`, `total_transient_seconds`, `steady_pct`, `num_steady_periods`, `avg_steady_duration_seconds` |
| `steady_state_value_bands(window='5min', std_threshold=1.0, min_duration='10min')` | per steady interval: `mean_value`, `lower_band`, `upper_band` (mean ± std), `duration_seconds` |

## Notes

- `std_threshold` is in signal units; set it above the sensor noise.
- Intervals end at the next sample after the condition stops holding by default (`interval_end="next_change"`), so change-only signals keep their duration; `"last_sample"` ends them at the last flagged sample.
- `steady_state_statistics(..., as_frame=False)` returns the deprecated dict.
