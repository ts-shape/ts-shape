# MultivariateSPCEvents

> Hotelling T² control chart for correlated process variables: detects faults that break the relationship between variables while each stays inside its own limits.

**Module:** `ts_shape.events.quality.multivariate_spc`  
**Guide:** [Quality Control & SPC](../../guides/quality.md)

---

## When to Use

Use when several variables move together: temperature and pressure in a reactor, spindle load and feed, flow and valve position. One univariate chart per variable has two problems:

- it misses a fault that pushes the variables apart, as long as each stays within 3σ;
- it raises the false-alarm rate with every chart you add.

T² measures each observation's distance from the in-control mean in units of the in-control covariance. With temperature +1σ and pressure −1σ at correlation 0.95, T² is about 40 against a limit of about 12. Each univariate chart sees a 1σ deviation.

---

## Quick Example

```python
from ts_shape.events.quality.multivariate_spc import MultivariateSPCEvents

mspc = MultivariateSPCEvents(
    df, ["reactor:temp", "reactor:pressure", "reactor:flow"], max_gap="5min"
)
reference = ("2024-01-01", "2024-01-08")  # a known in-control week

t2 = mspc.hotelling_t2(reference_period=reference)  # every aligned observation
runs = mspc.detect_out_of_control(reference_period=reference, min_samples=3)
print(runs[["start", "end", "max_t2", "top_contributor"]])

# Which variable drives each alarm?
contrib = mspc.contributions(reference_period=reference)
```

---

## Key Methods

| Method | Shape | Returns |
|---|---|---|
| `hotelling_t2(reference_period, alpha)` | point | `t2`, `ucl`, `phase` (`I` = reference, `II` = monitored), `out_of_control` |
| `contributions(reference_period, alpha, only_out_of_control)` | point | one row per observation and variable: `contribution` (sums to T²), `contribution_pct`, `z_score` |
| `detect_out_of_control(reference_period, alpha, min_samples)` | interval | runs of alarms with `max_t2`, `sample_count`, `top_contributor` |

## Details

- **Alignment:** the variables are aligned on the timestamps of the first signal. Every other signal contributes the value in force at that time, held backward with no look-ahead and never longer than `max_gap`. Rows where a variable has no value are dropped.
- **Limits** (Montgomery, *Introduction to SQC*, ch. 11), with `m` reference observations and `p` variables:
    - **Phase II** (new observations): `p(m+1)(m−1) / (m(m−p)) · F(1−α; p, m−p)`;
    - **Phase I** (the reference observations themselves): `(m−1)²/m · Beta(1−α; p/2, (m−p−1)/2)`.

    `alpha=0.0027` matches 3σ.
- **Reference period:** `None` uses all data, which is a retrospective Phase I study. For monitoring, pass a period known to be in control; it needs more than `p + 1` observations.
- A singular covariance (a constant or perfectly collinear variable) falls back to the pseudo-inverse, with a logged warning.
