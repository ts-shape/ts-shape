# MultivariateSPCEvents

Hotelling T² control chart for correlated variables. It catches faults that break the relationship between variables while each stays inside its own limits, e.g. temperature +1σ and pressure −1σ at correlation 0.95 give T² ≈ 40 against a limit of about 12.

`ts_shape.events.quality.multivariate_spc` · [Quality guide](../../guides/quality.md) · [API reference](../../reference/ts_shape/events/quality/multivariate_spc.md)

## Example

```python
import numpy as np
import pandas as pd
from ts_shape.events.quality.multivariate_spc import MultivariateSPCEvents

rng = np.random.default_rng(0)
t = pd.date_range("2024-01-01", periods=2000, freq="5min")
temp = rng.normal(0, 1, 2000)
pressure = 0.95 * temp + rng.normal(0, 0.3, 2000)
pressure[1800:1820] -= 1.5  # breaks the relationship, each value still within 3 sigma
df = pd.concat([pd.DataFrame({"systime": t, "uuid": u, "value_double": v})
                for u, v in [("reactor:temp", temp), ("reactor:pressure", pressure)]])

mspc = MultivariateSPCEvents(df, ["reactor:temp", "reactor:pressure"], max_gap="10min")
runs = mspc.detect_out_of_control(reference_period=("2024-01-01", "2024-01-05"), min_samples=3)
print(runs[["start", "end", "max_t2", "top_contributor"]])
```

## Methods

All arguments are keyword-only.

| Method | Returns |
|---|---|
| `hotelling_t2(reference_period=None, alpha=0.0027)` | per aligned observation: `t2`, `ucl`, `phase` (`I` = reference, `II` = monitored), `out_of_control` |
| `contributions(reference_period=None, alpha=0.0027, only_out_of_control=True)` | per observation and variable: `contribution` (sums to T²), `contribution_pct`, `z_score` |
| `detect_out_of_control(reference_period=None, alpha=0.0027, min_samples=1)` | runs of alarms: `start`, `end`, `max_t2`, `sample_count`, `top_contributor` |

## Notes

- Alignment: the variables are aligned on the timestamps of the first signal. Every other signal contributes the value in force at that time (no look-ahead), held at most `max_gap`. Rows where a variable has no value are dropped.
- Limits (Montgomery, Introduction to SQC, ch. 11), with `m` reference observations and `p` variables: Phase II (new observations) `p(m+1)(m−1) / (m(m−p)) · F(1−α; p, m−p)`; Phase I (the reference observations) `(m−1)²/m · Beta(1−α; p/2, (m−p−1)/2)`. `alpha=0.0027` matches 3σ.
- `reference_period=None` uses all data (a retrospective Phase I study). For monitoring, pass a period known to be in control; it needs more than `p + 1` observations. With fewer, every method returns an empty result and emits a `DataQualityWarning`.
- A singular covariance (a constant or perfectly collinear variable) falls back to the pseudo-inverse, with a logged warning.
