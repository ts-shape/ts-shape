# ReliabilityEvents

MTBF, MTTR, inherent availability and failure rate (overall or per calendar window), and a Weibull life model, from a signal that says whether an asset is failed or under repair (a fault bit, a breakdown status code, a maintenance mode).

`ts_shape.events.maintenance.reliability` · [Engineering guide](../../guides/engineering.md) · [API reference](../../reference/ts_shape/events/maintenance/reliability.md)

## Example

```python
import numpy as np
import pandas as pd
from ts_shape.events.maintenance.reliability import ReliabilityEvents

rng = np.random.default_rng(3)
up_h, down_h = rng.weibull(2.0, 12) * 150, rng.uniform(1, 6, 12)  # 12 run/repair cycles
starts = np.cumsum(np.column_stack([up_h, down_h]).ravel())
df = pd.DataFrame({"systime": pd.Timestamp("2024-01-01") + pd.to_timedelta(np.r_[0, starts[:-1]], unit="h"),
                   "uuid": "press3:status", "value_string": ["RUN", "BREAKDOWN"] * 12})

rel = ReliabilityEvents(df, "press3:status", failure_states=("BREAKDOWN", "REPAIR"),
                        value_column="value_string", max_gap="30D")
print(rel.mtbf_mttr()[["failures", "mtbf_hours", "mttr_hours", "availability_pct"]])
fit = rel.weibull_fit()
survival = rel.reliability_at([100], age_hours=200)  # 100 more hours after 200 h since repair
```

## Methods

| Method | Returns |
|---|---|
| `failure_events()` | one interval per failure: `repair_hours`, `operating_hours_before`, `first_observed` |
| `mtbf_mttr(window=None, tz=None)` | one row, or one per calendar window (e.g. `window="MS"`): `operating_hours`, `repair_hours`, `no_data_hours`, `failures`, `mtbf_hours`, `mtbf_iso22400_hours`, `mttr_hours`, `availability_pct`, `failure_rate_per_hour` |
| `weibull_fit(include_initial=False)` | one row: `beta`, `eta_hours`, `mttf_hours`, `b10_hours`, `n_failures`, `n_censored`, `failure_pattern` |
| `reliability_at(hours, age_hours=0.0)` | per horizon: `reliability`, `failure_probability`, `hazard_per_hour` |

## Notes

- Each state holds until the next change. A logging gap longer than `max_gap` counts as neither operating nor repair time but as `no_data_hours` (which also covers window time outside the data). Failed intervals separated only by a gap are one failure. `data_end` bounds the last state.
- `failure_states` lists the values that mean failed (default `(True,)` on `value_bool`).
- `mtbf_hours` = operating time / failures; `mtbf_iso22400_hours` = operating time / (failures + 1), averaging over the FE + 1 operating periods of ISO 22400-2 including the one still running. MTTR = repair time / failures. Availability = operating / (operating + repair): inherent availability, not OEE availability.
- Weibull: maximum-likelihood fit with location 0. The operating period still running at the end is right-censored (it has not failed yet). The operating time before the first failure is excluded unless `include_initial=True`, because the asset's age at the start of the data is unknown. When the times between failures (almost) all have the same length the likelihood has no finite maximum: the parameters are NaN, `failure_pattern` is `unknown`, a `DataQualityWarning` is emitted, and `reliability_at` returns NaN values (never `inf`).
- `failure_pattern`: `beta < 0.9` infant mortality (early failures; preventive replacement makes it worse), 0.9-1.1 random (use condition monitoring rather than time-based maintenance), `beta > 1.1` wear-out (time-based replacement before `b10_hours` pays off).
