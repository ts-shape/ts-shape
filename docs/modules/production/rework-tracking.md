# ReworkEvents

> Track parts that require rework from rework counter and reason-code signals.

**Module:** `ts_shape.events.production.rework_tracking`
**Guide:** [Reporting Guide](../../guides/reporting.md)

---

## When to Use

Use to track rework events and their cost. Identifies which parts and reasons drive the most rework, enabling targeted process improvements. Especially valuable for plants where rework is a significant cost driver and management needs visibility into root causes and financial impact.

---

## Quick Example

```python
from ts_shape.events.production.rework_tracking import ReworkEvents

tracker = ReworkEvents(production_df)

# Rework counts per shift
by_shift = tracker.rework_by_shift(rework_uuid="line1:rework")

# Rework breakdown by reason code
by_reason = tracker.rework_by_reason(
    rework_uuid="line1:rework",
    reason_uuid="line1:rework_reason"
)

# Rework rate as percentage of total production
rate = tracker.rework_rate(
    rework_uuid="line1:rework",
    total_production_uuid="line1:parts"
)

# Monetary cost of rework by part type
rework_costs = {"PN-100": 12.50, "PN-200": 8.75, "PN-300": 22.00}
cost = tracker.rework_cost(
    rework_uuid="line1:rework",
    part_id_uuid="line1:part_id",
    rework_costs=rework_costs
)
```

---

## Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `rework_by_shift(rework_uuid)` | Rework count per shift | `DataFrame` with `date`, `shift`, `rework_count` |
| `rework_by_reason(rework_uuid, reason_uuid)` | Rework count grouped by reason code | `DataFrame` with `reason`, `rework_count`, `pct_of_total` |
| `rework_rate(rework_uuid, total_production_uuid)` | Rework rate as percentage of total production per shift | `DataFrame` with `date`, `shift`, `total_produced`, `rework_count`, `rework_rate_pct` |
| `rework_cost(rework_uuid, part_id_uuid, rework_costs)` | Monetary cost of rework by part number | `DataFrame` with `part_number`, `rework_count`, `cost_per_rework`, `total_cost` |
| `rework_trend(rework_uuid, window='1D')` | Rework counts per time period | `DataFrame` with `period`, `rework_count` |

All methods count rework across counter resets by default (`handle_resets=True`); `rework_by_shift`, `rework_by_reason`, `rework_cost` and `rework_trend` add a `resets` column. Pass `handle_resets=False` for the plain `max(0, last - first)` quantity.

---

## Tips & Hints

!!! tip "Combine with Pareto analysis"
    Sort `rework_by_reason` results by count descending to build a Pareto chart. Typically 20% of reason codes drive 80% of rework volume, giving you clear improvement targets.

!!! info "Related modules"
    - [Scrap Tracking](scrap-tracking.md) — track scrap/waste that cannot be reworked
    - [Quality Tracking](quality-tracking.md) — overall quality metrics including FPY
    - [Operator Performance](operator-performance.md) — quality by operator

---

## See Also

- [Reporting Guide](../../guides/reporting.md)
- [API Reference](../../reference/ts_shape/events/production/rework_tracking.md)
