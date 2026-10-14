# CriticalParameterRankingEvents

Ranks candidate process parameters by their statistical association with a per-run outcome (yield, a quality KPI), to shortlist the critical process parameters worth tight control.

`ts_shape.events.development.critical_parameter_ranking` · [Engineering guide](../../guides/engineering.md) · [API reference](../../reference/ts_shape/events/development/critical_parameter_ranking.md)

## Example

```python
import numpy as np
import pandas as pd
from ts_shape.events.development.critical_parameter_ranking import CriticalParameterRankingEvents

rng = np.random.default_rng(0)
runs = pd.DataFrame({"inlet_temp_mean": rng.normal(60, 3, 40),  # one row per batch
                     "spray_rate_mean": rng.normal(120, 10, 40),
                     "batch_size_kg": rng.choice([200, 250, 300], 40)})
runs["yield_pct"] = 95 - 0.8 * (runs["inlet_temp_mean"] - 57) + rng.normal(0, 1, 40)

cpr = CriticalParameterRankingEvents()  # no long-format data needed
candidates = ["inlet_temp_mean", "spray_rate_mean", "batch_size_kg"]
print(cpr.rank(runs, candidates, "yield_pct", method="spearman"))
drivers = cpr.top_drivers(runs, candidates, "yield_pct", method="anova_f", k=3, alpha=0.05)
```

## Methods

| Method | Returns |
|---|---|
| `rank(per_run_df, candidate_columns, outcome_column, method='spearman', anova_bins=3)` | one row per candidate, strongest first: `parameter`, `method`, `statistic`, `p_value`, `abs_effect_size`, `rank` |
| `top_drivers(..., k=5, alpha=0.05)` | the same columns, limited to `p_value <= alpha` and the first `k` rows |

## Notes

- The methods take a wide per-run table: one row per run (batch, DOE point, shift), one column per candidate, plus the outcome column. Build it yourself (see the [pipeline builder guide](../../guides/pipeline-builder.md)), because the right aggregate per parameter (hold-phase mean, settled value, peak) is an engineering decision. The constructor's long-format DataFrame is optional and not used by `rank`.
- `method`: `spearman` (default) for any monotonic relationship, robust to outliers; `pearson` for linear relationships; `anova_f` for parameters set at a few discrete levels, as in a DOE.
- `abs_effect_size` is |r| for `pearson` and `spearman`, and eta-squared (between-group / total sum of squares) for `anova_f`. For `anova_f` each candidate is quantile-binned into `anova_bins` groups; groups with fewer than 2 runs are dropped.
- Candidates with fewer than 3 complete rows are skipped with a `DataQualityWarning`. An unknown candidate or a missing outcome column raises `ValueError`.
- Association is not causation: correlated candidates share credit and a confounder can rank high. Confirm the shortlist with a designed experiment.
