# MetricSpec & aggregate_metrics

> Declare how each metric survives aggregation: additive, weighted, or recomputed from components.

**Module:** `ts_shape.reporting.aggregation`
**Guide:** [Global Reporting Guide](../../guides/global-reporting.md)

---

## When to Use

Use wherever rows are collapsed to a coarser grain. Sums are easy; *rates* are where consolidated reports go wrong, and this is the module that makes the choice explicit and reviewable rather than implicit in a `groupby().mean()`.

---

## Quick Example

```python
from ts_shape.reporting import MetricSpec, aggregate_metrics

specs = [
    MetricSpec.sum("parts_produced"),
    MetricSpec.ratio("quality_pct", numerator="good",
                     denominator="parts_produced", scale=100.0),
    MetricSpec.weighted("oee_pct", weight="planned_minutes"),
]

aggregate_metrics(kpi_df, by=["region", "period"], specs=specs,
                  contributor_column="facility")

# Exploratory shortcut -- sums for quantities, weighted means for rate-like names:
from ts_shape.reporting import infer_specs
infer_specs(kpi_df, exclude=["facility", "period"], weight="planned_minutes")
```

---

## Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `MetricSpec.sum(col)` | Additive metric (counts, minutes, costs) | `MetricSpec` |
| `MetricSpec.weighted(col, weight=...)` | Rate weighted by its own exposure | `MetricSpec` |
| `MetricSpec.ratio(col, numerator=..., denominator=...)` | Rate recomputed from summed components | `MetricSpec` |
| `MetricSpec.mean(col)` | Unweighted mean (only for comparable metrics) | `MetricSpec` |
| `aggregate_metrics(df, by, specs)` | Apply the specs at a grouping grain | `DataFrame` |
| `infer_specs(df, weight=...)` | Heuristic specs from column names | `list[MetricSpec]` |

---

## Tips & Hints

!!! warning "The unweighted mean of a rate"
    One plant at 95 % for 30 minutes and two at 60 % for three shifts average to
    77.5 % unweighted and 60.2 % weighted by planned minutes. Only the second is
    the fleet's OEE.

!!! tip "Prefer `ratio` over `weighted`"
    When both components are available, `sum(good) / sum(total)` is exact; a
    weighted mean of the rate is an approximation of it.

!!! info "Zero denominators and missing weights"
    A zero or missing denominator yields NaN, never an infinity, and rows with a
    non-positive weight are excluded from a weighted mean.

---

## See Also

- [Global Reporting Guide](../../guides/global-reporting.md)
- [API Reference](../../reference/ts_shape/reporting/aggregation.md)
