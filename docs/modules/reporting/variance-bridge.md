# VarianceBridge

> Decompose a change in a fleet number: which sites moved it, and rate versus mix.

**Module:** `ts_shape.reporting.variance`
**Guide:** [Global Reporting Guide](../../guides/global-reporting.md)

---

## When to Use

Use when the review asks *why* the number moved. `facility_bridge` gives the waterfall of per-site contributions; `rate_bridge` answers the question that unweighted reporting cannot: the fleet rate fell although every plant improved, because volume shifted to a weaker site.

---

## Quick Example

```python
from ts_shape.reporting import VarianceBridge

bridge = VarianceBridge(kpi_df)

bridge.facility_bridge("parts_produced", base="2026-01", compare="2026-02")

bridge.rate_bridge("oee_pct", weight_column="planned_minutes",
                   base="2026-01", compare="2026-02")
# rate_effect + mix_effect + interaction == the observed change, exactly

bridge.plan_vs_actual("parts_produced", targets={"DE-STR": 120_000})
bridge.period_over_period("parts_produced")
```

---

## Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `facility_bridge(metric, base, compare)` | Per-site contribution to a fleet delta | `DataFrame` |
| `rate_bridge(metric, weight_column=...)` | Rate / mix / interaction split of a weighted rate | `DataFrame` |
| `plan_vs_actual(metric, targets=...)` | Variance and RAG status against plan | `DataFrame` |
| `period_over_period(metric)` | Fleet change from each period to the next | `DataFrame` |

---

## Tips & Hints

!!! tip "Mix is not performance"
    A site with no exposure in one period has no rate there. `rate_bridge`
    carries its rate across, so an entering or exiting plant lands entirely in
    `mix_effect` instead of masquerading as a performance change.

!!! warning "`facility_bridge` assumes additivity"
    Contributions only reconcile when the fleet figure is the sum of the site
    figures. Use `rate_bridge` for rates.

!!! info "Related modules"
    - [Fleet Rollup](fleet-rollup.md) — the aggregates being bridged
    - [Global Scorecard](global-scorecard.md) — the same movement against targets

---

## See Also

- [Global Reporting Guide](../../guides/global-reporting.md)
- [API Reference](../../reference/ts_shape/reporting/variance.md)
