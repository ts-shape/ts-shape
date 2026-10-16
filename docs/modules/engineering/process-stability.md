# ProcessStabilityIndexEvents

One 0-100 stability score per window (e.g. per shift) for a signal, built from four 0-25 sub-scores for spread, bias, excursions and jitter; plus its trend, the worst windows and the gap to the best window.

`ts_shape.events.engineering.process_stability_index` · [Engineering guide](../../guides/engineering.md) · [API reference](../../reference/ts_shape/events/engineering/process_stability_index.md)

## Example

```python
import ts_shape
from ts_shape.events.engineering.process_stability_index import ProcessStabilityIndexEvents

df = ts_shape.make_timeseries(["reactor_temp"], n_points=4320, freq="1min", baseline=180.0, noise=1.0)
df.loc[3000:3300, "value_double"] += 4.0  # a rough hour or five

psi = ProcessStabilityIndexEvents(df, "reactor_temp", target=180.0, lower_spec=175.0, upper_spec=185.0)
print(psi.stability_score(window="8h")[["start", "stability_score", "grade"]])
trend = psi.score_trend(window="8h", n_windows=7)
worst = psi.worst_periods(window="1h", n=5)
```

## Methods

All arguments are keyword-only.

| Method | Returns |
|---|---|
| `stability_score(window='8h')` | per window: `stability_score`, `variance_score`, `bias_score`, `excursion_score`, `smoothness_score`, `grade` |
| `score_trend(window='8h', n_windows=7)` | per window: `stability_score`, `rolling_avg` (over `n_windows`), `trend_direction`, `score_change` |
| `worst_periods(window='1h', n=5)` | the `n` lowest-scoring windows with sub-scores and `primary_issue` |
| `stability_comparison(window='8h')` | per window: `stability_score`, `best_score`, `gap_to_best`, `of_best_pct` |

## Notes

- With `half_range` = (upper_spec − lower_spec) / 2, the sub-scores are: variance 25 × (1 − std / half_range), bias 25 × (1 − |mean − target| / half_range), excursion 25 × share of samples within spec, smoothness 25 × (1 − mean absolute step / half_range); each ratio is clipped to [0, 1].
- Without `target` the signal mean is used; without spec limits, mean ± 3 standard deviations.
- `grade`: A ≥ 90, B ≥ 75, C ≥ 60, D ≥ 40, else F. `primary_issue` names the lowest sub-score: `high_variance`, `off_target`, `excursions` or `rough_signal`.
- Windows with fewer than two samples are skipped.
