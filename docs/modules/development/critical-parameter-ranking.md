# CriticalParameterRankingEvents

> Which input parameters drive the outcome? Rank candidate process parameters by their statistical association with a per-run quality result.

**Module:** `ts_shape.events.development.critical_parameter_ranking`

---

## When to Use

Use in process development to shortlist the critical process parameters (CPPs) worth tight control: given one row per completed run (batch, DOE point, shift) with candidate parameter values and an outcome such as yield or a quality KPI, rank the candidates by how strongly they relate to the outcome.

- `spearman` (default) handles any monotonic relationship and is robust to outliers.
- `pearson` is for linear relationships between continuous variables.
- `anova_f` suits parameters that were set at a few discrete levels, as in a DOE.

---

## Quick Example

```python
from ts_shape.events.development.critical_parameter_ranking import (
    CriticalParameterRankingEvents,
)

candidates = ["inlet_temp_mean", "spray_rate_mean", "atomization_bar", "batch_size_kg"]

cpr = CriticalParameterRankingEvents(df)

ranking = cpr.rank(per_run_df, candidates, "yield_pct", method="spearman")
drivers = cpr.top_drivers(per_run_df, candidates, "yield_pct", method="anova_f", k=3, alpha=0.05)
```

---

## Key Methods

| Method | Shape | Returns |
|---|---|---|
| `rank(per_run_df, candidate_columns, outcome_column, method, anova_bins)` | static | one row per candidate, strongest first: `parameter`, `method`, `statistic`, `p_value`, `abs_effect_size`, `rank` |
| `top_drivers(per_run_df, candidate_columns, outcome_column, method, k, alpha, anova_bins)` | static | the same columns, filtered to `p_value <= alpha` and the first `k` rows |

## Notes

- The methods take a **wide per-run table**, not long-format signal data: one row per run, one column per candidate, plus the outcome column. Build it yourself (for example with the [Feature-Table Pipeline](../../pipelines/feature-pipeline.md)), because the right aggregate per parameter (hold-phase mean, settled value, peak) is an engineering decision. The constructor's DataFrame is not used by `rank`.
- `abs_effect_size` is |r| for `pearson` and `spearman`, and eta-squared (between-group / total sum of squares) for `anova_f`. For `anova_f` each candidate is quantile-binned into `anova_bins` groups; groups with fewer than 2 runs are dropped.
- Candidates with fewer than 3 complete rows, or unknown column names, are skipped. A missing outcome column raises `ValueError`.
- Association is not causation: correlated candidates share credit, and a confounder can rank high. Confirm the shortlist with a designed experiment.
