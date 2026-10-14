# ProcessStabilityIndex

> Generate a single 0-100 stability score per shift/day for a signal.

**Module:** `ts_shape.events.engineering.process_stability_index`  
**Guide:** [Process Engineering](../../guides/engineering.md)

---

## When to Use

Use as a single-number KPI for process health dashboards. Answers "is my process running well today?" at a glance. Combine with other modules for detailed root cause analysis when the score drops. Ideal for management dashboards, shift handover summaries, and cross-plant benchmarking.

---

## Quick Example

```python
from ts_shape.events.engineering.process_stability_index import ProcessStabilityIndex

scorer = ProcessStabilityIndex(
    dataframe=process_data,
    signal_uuid="reactor_temperature_01",
    target=180.0,        # optional; defaults to the signal mean
    lower_spec=175.0,    # optional; defaults to mean - 3 std
    upper_spec=185.0,    # optional; defaults to mean + 3 std
)

# Get a 0-100 stability score per 8-hour shift
scores = scorer.stability_score(window="8h")

# Track the score trend with a 7-window rolling average
trend = scorer.score_trend(window="8h", n_windows=7)

# Find the 5 worst 1-hour periods for investigation
worst = scorer.worst_periods(window="1h", n=5)

# Compare each shift to the best-observed shift
comparison = scorer.stability_comparison(window="8h")
```

---

## Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `stability_score(window='8h')` | Composite 0-100 stability score | DataFrame: `start`, `stability_score`, four 0-25 sub-scores, `grade` |
| `score_trend(window='8h', n_windows=7)` | Score trend with an N-window rolling average | DataFrame: `start`, `stability_score`, `rolling_avg`, `trend_direction`, `score_change` |
| `worst_periods(window='1h', n=5)` | N worst-scoring windows | DataFrame of worst windows with sub-scores and `primary_issue` |
| `stability_comparison(window='8h')` | Compare each window to the best-observed window | DataFrame: `start`, `stability_score`, `best_score`, `gap_to_best`, `pct_of_best` |

---

## Tips & Hints

!!! tip "Use worst_periods to drill down from a low score"
    When the shift-level score drops, call `worst_periods(window='1h', n=5)` to pinpoint the exact hours that dragged the score down. Then use module-specific tools (disturbance recovery, rate of change, etc.) on those windows for root cause analysis.

!!! info "Related modules"
    - [Process Windows](process-windows.md) — detailed windowed statistics behind the score
    - [Disturbance Recovery](disturbance-recovery.md) — investigate disturbances that lowered the score
    - [Control Loop Health](control-loop-health.md) — loop-level diagnostics when control quality degrades
    - [Steady State Detection](steady-state.md) — stability score is most meaningful during steady state

---

## See Also

- [Process Engineering Guide](../../guides/engineering.md)
- [API Reference](../../reference/ts_shape/events/engineering/process_stability_index.md)
