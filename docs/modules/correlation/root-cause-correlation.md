# RootCauseCorrelationEvents

Ranks candidate signals as possible causes of a set of detected events (stops, alarms, quality failures): for each candidate, the lead time at which it correlates most strongly with the event onsets, with a significance test, a single verdict, and the correlation-by-lag curve behind it.

`ts_shape.events.correlation.root_cause_correlation` · [Engineering guide](../../guides/engineering.md) · [API reference](../../reference/ts_shape/events/correlation/root_cause_correlation.md)

## Example

```python
import numpy as np
import pandas as pd
from ts_shape.events.correlation.root_cause_correlation import RootCauseCorrelationEvents

# ten hours at 1 min; the hydraulic pressure dips five minutes before every stop
t = pd.date_range("2024-06-01", periods=600, freq="1min")
onsets = [90, 210, 330, 450, 570]
rng = np.random.default_rng(3)
hydraulic = rng.normal(180.0, 0.5, len(t))
hydraulic[[i - 5 for i in onsets]] -= 15.0
df = pd.concat([
    pd.DataFrame({"systime": t, "uuid": "press:hydraulic", "value_double": hydraulic}),
    pd.DataFrame({"systime": t, "uuid": "flow:coolant", "value_double": rng.normal(20.0, 1.0, len(t))}),
    pd.DataFrame({"systime": t, "uuid": "temp:spindle", "value_double": rng.normal(45.0, 0.3, len(t))}),
], ignore_index=True)

# any detector output can be the trigger: the onset is its `systime` or `start`
stops = pd.DataFrame({"start": t[onsets], "end": t[onsets] + pd.Timedelta("10min"),
                      "uuid": "prod:long_downtime", "source_uuid": "state:line-1"})

rc = RootCauseCorrelationEvents(df)
candidates = ["press:hydraulic", "flow:coolant", "temp:spindle"]
ranked = rc.rank_causes(stops, candidates, window="1min", max_lag=15)
print(ranked[["candidate_uuid", "correlation_coefficient", "lag_seconds", "significant"]])
verdict = rc.top_cause(stops, candidates, window="1min", max_lag=15)
print(verdict[["top_cause_uuid", "top_cause_lag_seconds", "confidence"]])
profile = rc.lag_profile(stops, "press:hydraulic", window="1min", max_lag=15)
```

The hydraulic pressure leads the stops by 300 s with a strong negative correlation (it dips); coolant flow and spindle temperature are not significant.

## Methods

| Method | Returns |
|---|---|
| `rank_causes(trigger_events, candidate_uuids, window='1min', max_lag=30, p_threshold=0.05, trigger_uuid=None, trigger_event_type=None, only_significant=False)` | one row per candidate, strongest first: `candidate_uuid`, `correlation_coefficient`, `lag_seconds`, `p_value`, `p_value_adjusted`, `significant`, `sample_count`, `trigger_count`, `rank` |
| `top_cause(trigger_events, candidate_uuids, window='1min', max_lag=30, p_threshold=0.05, ...)` | one row: `top_cause_uuid` (None when no candidate is significant), `top_cause_lag_seconds`, `top_cause_correlation`, `confidence`, `candidates_tested`, `candidates_significant` |
| `lag_profile(trigger_events, candidate_uuid, window='1min', max_lag=30)` | one row per tested lag: `lag_seconds`, `correlation`, `p_value`, `sample_count`, `is_best_lag` |

## Notes

- Each candidate is averaged onto a grid of `window` bins; the event onsets become a 0/1 series on the same grid. Lags from 0 to `max_lag` bins are tested, and the lag with the largest absolute Pearson correlation wins. A positive `lag_seconds` means the candidate moves before the events.
- `p_value_adjusted` is Bonferroni-corrected over the `max_lag + 1` lags tested; `significant` compares it with `p_threshold`. `lag_profile` reports the raw per-lag p-values.
- The method finds candidates whose level changes around the onsets (a dip, a spike). A slow drift that only crosses a limit before each event correlates weakly with a sparse onset series; detect the crossings first and correlate those events instead.
- `confidence` in `top_cause` is the absolute correlation of the winning candidate, 0 when none qualifies.
- Correlation is not causation: a candidate that leads the events may share a cause with them.
