# ReliabilityEvents

> MTBF, MTTR, availability and a Weibull life model, computed from a failure-state signal.

**Module:** `ts_shape.events.maintenance.reliability`

---

## When to Use

Use when you have a signal that says whether an asset is failed or down for repair: a fault bit, a breakdown status code, a maintenance mode. You get:

- the reliability KPIs maintenance planning runs on (MTBF, MTTR, availability, failure rate), overall or per calendar window;
- a Weibull life model: is this asset failing early (infant mortality), at random, or wearing out? How likely is it to survive the next 100 operating hours?

---

## Quick Example

```python
from ts_shape.events.maintenance.reliability import ReliabilityEvents

rel = ReliabilityEvents(
    df,
    "press3:status",
    failure_states=("BREAKDOWN", "REPAIR"),
    value_column="value_string",
    max_gap="30min",   # a longer logging gap is no data, not uptime
)

rel.failure_events()            # one interval per failure
rel.mtbf_mttr()                 # one row over the whole history
rel.mtbf_mttr("MS", tz="Europe/Berlin")   # per calendar month

fit = rel.weibull_fit()
print(fit[["beta", "eta_hours", "b10_hours", "failure_pattern"]])

# Given it has run 200 h since the last repair: odds of another 100 h?
rel.reliability_at([100], age_hours=200)
```

---

## Key Methods

| Method | Shape | Returns |
|---|---|---|
| `failure_events()` | interval | `repair_hours`, `operating_hours_before`, `first_observed` |
| `mtbf_mttr(window, tz)` | summary | `operating_hours`, `repair_hours`, `no_data_hours`, `failures`, `mtbf_hours`, `mtbf_iso22400_hours`, `mttr_hours`, `availability_pct`, `failure_rate_per_hour` |
| `weibull_fit(include_first)` | static | `beta`, `eta_hours`, `mttf_hours`, `b10_hours`, `n_failures`, `n_censored`, `failure_pattern` |
| `reliability_at(hours, age_hours)` | static | `reliability`, `failure_probability`, `hazard_per_hour` |

## Definitions

- **Time accounting.**
    - Each state holds until the next change.
    - A logging gap longer than `max_gap` counts as neither operating nor repair time; it shows up in `no_data_hours`, which also covers window time outside the data.
    - Failed intervals separated only by a gap are one failure.
- **MTBF.**
    - `mtbf_hours` = operating time / failures.
    - `mtbf_iso22400_hours` = operating time / (failures + 1). ISO 22400-2 averages over the FE + 1 operating periods, including the one still running.
- **MTTR** = repair time / failures.
- **Availability** = operating / (operating + repair). This is inherent availability, not OEE availability.
- **Weibull fit.**
    - It is a maximum-likelihood fit with the location fixed at 0.
    - The operating period still running at the end of the data is right-censored: it has not failed *yet*. Dropping it would bias the life estimate low.
    - The operating time before the first failure is excluded by default, because the asset's age at the start of the data is unknown.
- **Failure pattern.**
    - `beta < 0.9` means infant mortality: early failures, often from installation or quality problems. Preventive replacement makes it worse.
    - `beta` from 0.9 to 1.1 means random failures. Time-based maintenance does not help; use condition monitoring.
    - `beta > 1.1` means wear-out. Time-based replacement before `b10_hours` pays off.
