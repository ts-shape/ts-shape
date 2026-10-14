# OperatorPerformanceEvents

> Track and compare operator performance from operator-ID and counter signals.

**Module:** `ts_shape.events.production.operator_performance`
**Guide:** [Reporting Guide](../../guides/reporting.md)

---

## When to Use

Use for fair operator performance comparison. Tracks production output, efficiency vs targets, and quality (FPY) per operator or team. Designed for shift leads and production managers who need to identify training needs, recognize top performers, and ensure balanced workload distribution.

---

## Quick Example

```python
from ts_shape.events.production.operator_performance import OperatorPerformanceEvents

tracker = OperatorPerformanceEvents(production_df)

# Parts produced per operator
by_operator = tracker.production_by_operator(
    operator_uuid="line1:operator",
    counter_uuid="line1:parts"
)

# Efficiency vs target
efficiency = tracker.operator_efficiency(
    operator_uuid="line1:operator",
    counter_uuid="line1:parts",
    target_per_shift=500
)

# First-pass yield per operator
quality = tracker.quality_by_operator(
    operator_uuid="line1:operator",
    ok_uuid="line1:ok",
    nok_uuid="line1:nok"
)
```

---

## Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `production_by_operator(operator_uuid, counter_uuid)` | Total parts produced per operator | `DataFrame` |
| `operator_efficiency(operator_uuid, counter_uuid, target_per_shift)` | Efficiency percentage vs shift target per operator | `DataFrame` |
| `quality_by_operator(operator_uuid, ok_uuid, nok_uuid)` | First-pass yield (FPY) per operator | `DataFrame` |
| `operator_comparison(operator_uuid, counter_uuid)` | Ranked comparison of all operators | `DataFrame` |

---

## Tips & Hints

!!! tip "Normalize for shift length and product mix"
    Raw output counts can be misleading if operators work different shift lengths or produce different product variants. Use efficiency metrics with appropriate targets per product type for fair comparisons.

!!! info "Related modules"
    - [Shift Reporting](shift-reporting.md) — production totals by shift
    - [Quality Tracking](quality-tracking.md) — detailed quality metrics
    - [Target Tracking](target-tracking.md) — generic target vs actual comparisons

---

## See Also

- [Reporting Guide](../../guides/reporting.md)
- [API Reference](../../reference/ts_shape/events/production/operator_performance.md)
