# GlobalScorecard & KpiTarget

> Every site, every KPI, against target, with RAG status and a weighted score.

**Module:** `ts_shape.reporting.scorecard`
**Guide:** [Global Reporting Guide](../../guides/global-reporting.md)

---

## When to Use

Use to produce the table a management review runs off. Each `KpiTarget` carries its own direction, tolerance, weight and aggregation, so a scrap target and an OEE target are evaluated correctly in the same call and a site with no data reports `no_data` rather than passing by omission.

---

## Quick Example

```python
from ts_shape.reporting import GlobalScorecard, KpiTarget, MetricSpec

targets = [
    KpiTarget("oee_pct", 75.0,
              spec=MetricSpec.weighted("oee_pct", weight="planned_minutes")),
    KpiTarget("scrap_rate_pct", 2.5, higher_is_better=False),
    KpiTarget("parts_produced", 12_000,
              spec=MetricSpec.sum("parts_produced"), label="output"),
]

board = GlobalScorecard(site_kpis, period_column="period", registry=registry)

board.evaluate(targets)            # long: facility x metric
board.facility_summary(targets)    # wide: one row per site, score and rank
board.fleet_summary(targets)       # one row per metric at fleet level
board.trend(targets)               # movement between two periods
board.evaluate(targets, by="region")
```

---

## Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `evaluate(targets, by=...)` | Value, attainment, gap and status per node and metric | `DataFrame` |
| `facility_summary(targets)` | One row per site with weighted score and rank | `DataFrame` |
| `fleet_summary(targets)` | Fleet value per metric plus `facilities_at_risk` | `DataFrame` |
| `trend(targets, base, compare)` | Direction-aware movement between two periods | `DataFrame` |
| `KpiTarget.status(value)` / `.attainment(value)` | Single-value evaluation | `str` / `float` |

---

## Tips & Hints

!!! warning "A green fleet number can hide two red plants"
    `fleet_summary` reports `facilities_at_risk` next to the fleet value for
    exactly this reason — it is the most common way a consolidated scorecard
    misleads.

!!! tip "Attainment is direction-aware"
    For a lower-is-better metric attainment is `target / value`, so 100 % always
    means on plan and above 100 % always means better than plan.

!!! info "Related modules"
    - [Facility Benchmark](facility-benchmark.md) — relative standing rather than target attainment
    - [Variance Bridge](variance-bridge.md) — plan variance per site

---

## See Also

- [Global Reporting Guide](../../guides/global-reporting.md)
- [API Reference](../../reference/ts_shape/reporting/scorecard.md)
