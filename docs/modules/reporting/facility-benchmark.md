# FacilityBenchmark

> Rank sites, describe the fleet spread, and size the gap in units rather than points.

**Module:** `ts_shape.reporting.benchmark`
**Guide:** [Global Reporting Guide](../../guides/global-reporting.md)

---

## When to Use

Use for the comparison half of an operations review: where each plant stands, what separates the top quartile, what closing the gap is worth, and which sites are erratic rather than merely low. Every method takes `higher_is_better`, because half a plant scorecard is better when lower.

---

## Quick Example

```python
from ts_shape.reporting import FacilityBenchmark

bench = FacilityBenchmark(site_kpis, period_column="period")

bench.rank("oee_pct")
bench.rank("scrap_rate_pct", higher_is_better=False)
bench.quartiles("oee_pct")

# Gap to the top quartile, expressed in the exposure's units:
bench.opportunity("oee_pct", exposure_column="planned_minutes")

bench.scorecard({"oee_pct": True, "scrap_rate_pct": False}, weights={"oee_pct": 2.0})
bench.stability("oee_pct")
bench.outliers("energy_per_part", method="iqr")
```

---

## Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `rank(metric)` | Rank, percentile, quartile, z-score, gap to best/median | `DataFrame` |
| `quartiles(metric)` | Fleet distribution, top-quartile threshold, spread | `DataFrame` |
| `opportunity(metric, exposure_column=...)` | Gap to a benchmark in exposure units | `DataFrame` |
| `scorecard(metrics, weights=...)` | 0-100 normalised scores and a composite | `DataFrame` |
| `stability(metric)` | Mean vs variability vs trend per site | `DataFrame` |
| `outliers(metric, method=...)` | Sites outside the fleet distribution | `DataFrame` |

---

## Tips & Hints

!!! tip "Size the prize in units"
    `opportunity` multiplies each site's gap by its own exposure, so the output
    is parts or minutes rather than percentage points — the form a capital
    request needs.

!!! warning "Collapsing periods with the default spec"
    Several rows per facility are collapsed with an unweighted mean unless a
    `spec` is passed. For periods with differing exposure, pass
    `MetricSpec.weighted(metric, weight=...)`.

!!! info "Related modules"
    - [Global Scorecard](global-scorecard.md) — targets and RAG status
    - [Variance Bridge](variance-bridge.md) — what changed, rather than who leads

---

## See Also

- [Global Reporting Guide](../../guides/global-reporting.md)
- [API Reference](../../reference/ts_shape/reporting/benchmark.md)
