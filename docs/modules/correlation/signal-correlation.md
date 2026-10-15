# SignalCorrelationEvents

Rolling Pearson correlation between two signals, periods where a normally strong correlation breaks down, and the time lag at which they correlate best.

`ts_shape.events.correlation.signal_correlation` · [Engineering guide](../../guides/engineering.md) · [API reference](../../reference/ts_shape/events/correlation/signal_correlation.md)

## Example

```python
import numpy as np
import pandas as pd
from ts_shape.events.correlation.signal_correlation import SignalCorrelationEvents

rng = np.random.default_rng(0)
t = pd.date_range("2024-01-01", periods=1440, freq="1min")
temp = np.cumsum(rng.normal(0, 0.2, 1440)) + 150
pressure = 0.5 * np.roll(temp, 5) + rng.normal(0, 0.05, 1440)  # follows temperature by 5 min
df = pd.concat([pd.DataFrame({"systime": t, "uuid": "reactor:temperature", "value_double": temp}),
                pd.DataFrame({"systime": t, "uuid": "reactor:pressure", "value_double": pressure})])

sc = SignalCorrelationEvents(df)
rolling = sc.rolling_correlation("reactor:pressure", "reactor:temperature", window="1min", window_size=60)
lags = sc.lag_correlation("reactor:pressure", "reactor:temperature", window="1min", max_lag=30)
print(lags[lags["is_best_lag"]])
```

## Methods

| Method | Returns |
|---|---|
| `rolling_correlation(uuid_a, uuid_b, window='1min', window_size=60)` | per grid point: `systime`, `correlation` |
| `correlation_breakdown(uuid_a, uuid_b, window='1min', window_size=60, threshold=0.5)` | intervals with correlation below `threshold`: `start`, `end`, `min_correlation`, `duration_seconds` |
| `lag_correlation(uuid_a, uuid_b, window='1min', max_lag=30)` | one row per lag from `-max_lag` to `+max_lag`: `lag_periods`, `correlation`, `is_best_lag` |

## Notes

- Both signals are resampled to the `window` grid; `window_size` and `max_lag` count grid periods, so `window_size=240` with `window="1min"` is 4 hours.
- Sign convention: `lag_periods = k` pairs `uuid_a` at time t with `uuid_b` at t − k. A positive best lag means `uuid_b` leads (`uuid_a` follows it by k periods); a negative best lag means `uuid_b` lags `uuid_a`. For b(t) = a(t − 5) the best lag is −5. In the example above pressure (`uuid_a`) follows temperature (`uuid_b`) by 5 min, so the best lag is +5.
- Breakdown intervals end at the next grid point after the condition stops holding (`interval_end="next_change"`, default) or at the last flagged point (`"last_sample"`).
