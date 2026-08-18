# AssetHierarchy

> The ISA-95 equipment tree — region / site / area / line — and roll-up along it.

**Module:** `ts_shape.reporting.hierarchy`
**Guide:** [Global Reporting Guide](../../guides/global-reporting.md)

---

## When to Use

Use when KPIs live below site level and the report has to walk up the equipment tree, or when the mapping itself needs checking. A line assigned to two sites double-counts the enterprise total; a line with no site quietly disappears from every regional view.

---

## Quick Example

```python
from ts_shape.reporting import AssetHierarchy, MetricSpec

hierarchy = AssetHierarchy(mapping_df, levels=["region", "site", "area", "line"])
hierarchy.validate()          # multiple_parents / missing_parent / duplicate_leaf

by_site = hierarchy.roll_up(
    line_kpis,
    to_level="site",
    specs=[MetricSpec.sum("parts"),
           MetricSpec.weighted("oee_pct", weight="planned_minutes")],
    group_extra=["period"],
)

hierarchy.enrich(line_kpis)                       # attach ancestor columns
hierarchy.children("EMEA", level="region", child_level="line")
```

---

## Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `validate()` | Structural defects in the mapping | `DataFrame` |
| `enrich(df, on=...)` | Attach ancestor level columns to a KPI frame | `DataFrame` |
| `roll_up(df, to_level, specs)` | Aggregate up to a level, keeping its ancestry | `DataFrame` |
| `nodes(level)` / `children(...)` / `parent(...)` / `path(node)` | Navigate the tree | `list` / `str` / `dict` |
| `levels_up_to(level)` | Levels from coarsest down to a level | `list` |

---

## Tips & Hints

!!! warning "Two parents means double counting"
    `multiple_parents` in `validate()` is not cosmetic: that node's output is
    counted once per parent, so the enterprise total exceeds the sum of its
    plants.

!!! tip "Unmapped assets stay visible"
    `enrich` keeps rows whose node is absent from the hierarchy, with NaN
    ancestors. They surface as a `NaN` group in the rollup instead of vanishing
    from the report.

!!! info "Related modules"
    - [Fleet Rollup](fleet-rollup.md) — accepts a hierarchy for sub-site dimensions
    - [Metric Aggregation](metric-aggregation.md) — the semantics `roll_up` applies

---

## See Also

- [Global Reporting Guide](../../guides/global-reporting.md)
- [API Reference](../../reference/ts_shape/reporting/hierarchy.md)
