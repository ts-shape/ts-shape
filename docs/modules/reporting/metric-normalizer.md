# MetricNormalizer

> Currency, scale and exposure normalisation so sites are compared on performance, not size.

**Module:** `ts_shape.reporting.normalize`
**Guide:** [Global Reporting Guide](../../guides/global-reporting.md)

---

## When to Use

Use before ranking or consolidating anything monetary or size-dependent. Costs sit in several currencies, energy in kWh at one site and MWh at another, and output in absolute units at plants an order of magnitude apart.

---

## Quick Example

```python
from ts_shape.reporting import MetricNormalizer

norm = MetricNormalizer(site_kpis, registry=registry)

eur = norm.convert_currency(["scrap_cost"], to_currency="EUR",
                            rates={"EUR": 1.0, "USD": 0.92, "CNY": 0.13})

# Period-dated rates, applied as-of each row's period:
eur = norm.convert_currency(["scrap_cost"], to_currency="EUR",
                            rates=fx_table, period_column="period")

MetricNormalizer(eur).per_unit(["scrap_cost_EUR"], denominator="parts_produced")
norm.per_capacity(["parts_produced"], days_column="days")
norm.index_to_baseline(["oee_pct"], baseline="best", group_by=["period"])
```

---

## Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `convert_currency(cols, to_currency=..., rates=...)` | One reporting currency, flat or dated rates | `DataFrame` |
| `per_unit(cols, denominator=...)` | Per part, per tonne, per anything | `DataFrame` |
| `per_hour(cols, hours_column=...)` | Per operating hour | `DataFrame` |
| `per_capacity(cols)` | Share of nameplate capacity from the registry | `DataFrame` |
| `rescale(cols, factor=...)` | Unit scaling (kWh to MWh, pieces to thousands) | `DataFrame` |
| `index_to_baseline(cols, baseline=...)` | Index against fleet median, best site or a value | `DataFrame` |

---

## Tips & Hints

!!! tip "Keep the source figures"
    `convert_currency` keeps the original columns and records the `fx_rate`
    applied to each row. A restated report that hides its inputs is not
    auditable.

!!! warning "Unmapped currencies become NaN"
    A currency with no rate converts to NaN, never silently to zero, and the
    missing codes are logged.

!!! info "Related modules"
    - [Facility Registry](facility-registry.md) — supplies currencies and capacities
    - [Facility Benchmark](facility-benchmark.md) — rank on the normalised columns

---

## See Also

- [Global Reporting Guide](../../guides/global-reporting.md)
- [API Reference](../../reference/ts_shape/reporting/normalize.md)
