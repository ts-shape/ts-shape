# FleetRollup

> Consolidate per-facility KPIs into regional and global figures.

**Module:** `ts_shape.reporting.rollup`
**Guide:** [Global Reporting Guide](../../guides/global-reporting.md)

---

## When to Use

Use to produce the numbers a global operations review runs off: the fleet total, the regional split, the per-site detail, and the stacked frame a dashboard drills through. It resolves `region` / `country` from the registry and deeper dimensions from an asset hierarchy, so a frame carrying only facility ids still rolls up correctly.

---

## Quick Example

```python
from ts_shape.reporting import FleetRollup, MetricSpec

rollup = FleetRollup(kpi_df, registry=registry)
specs = [MetricSpec.sum("parts_produced"),
         MetricSpec.weighted("oee_pct", weight="planned_minutes")]

rollup.total(specs)             # the headline number, with `contributors`
rollup.by_facility(specs)       # per site
rollup.by(["region"], specs)    # per region
rollup.consolidate(specs)       # global + region + site, stacked
rollup.contribution("parts_produced")   # Pareto across sites
rollup.pivot("oee_pct")                 # site x period matrix
```

---

## Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `total(specs)` | The single global row (or one per period) | `DataFrame` |
| `by_facility(specs, per_period=False)` | One row per site | `DataFrame` |
| `by_period(specs)` | One fleet row per period | `DataFrame` |
| `by(dimensions, specs)` | Arbitrary dimensions (`region`, `country`, hierarchy levels) | `DataFrame` |
| `consolidate(specs, levels=...)` | Global, regional and site rows in one long frame | `DataFrame` |
| `contribution(metric)` | Ranked share and cumulative share of a total | `DataFrame` |
| `pivot(metric)` | Facility-by-period matrix | `DataFrame` |

---

## Tips & Hints

!!! tip "Read `contributors` before quoting the number"
    Every aggregate reports how many facilities are behind it. A global figure
    computed over 34 of 37 plants looks exactly like one computed over all 37.

!!! warning "Inferred specs are for exploration"
    Omitting `specs` infers them from column names — sums for quantities,
    unweighted means for rate-like names. Declare specs explicitly for anything
    that ships to a stakeholder.

!!! info "Related modules"
    - [Metric Aggregation](metric-aggregation.md) — the semantics applied
    - [Reporting Coverage](reporting-coverage.md) — run it before the rollup
    - [Asset Hierarchy](asset-hierarchy.md) — sub-site dimensions

---

## See Also

- [Global Reporting Guide](../../guides/global-reporting.md)
- [API Reference](../../reference/ts_shape/reporting/rollup.md)
