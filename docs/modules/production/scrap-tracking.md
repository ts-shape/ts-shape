# ScrapTracking

> Track material scrap and waste from scrap and reason-code signals.

**Module:** `ts_shape.events.production.scrap_tracking`
**Guide:** [Reporting Guide](../../guides/reporting.md)

---

## When to Use

Use for scrap/waste tracking and cost accounting. Converts scrap quantities to monetary cost by material type for financial reporting. Essential for plants focused on material efficiency and sustainability goals, where understanding scrap drivers enables targeted waste reduction.

---

## Quick Example

```python
from ts_shape.events.production.scrap_tracking import ScrapTracking

tracker = ScrapTracking(production_df)

# Scrap quantity per shift (cumulative scrap signal in value_double by default)
by_shift = tracker.scrap_by_shift(scrap_uuid="line1:scrap")

# Scrap breakdown by reason code
by_reason = tracker.scrap_by_reason(
    scrap_uuid="line1:scrap",
    reason_uuid="line1:scrap_reason"
)

# Monetary cost of scrap; keys are the values of the part_id_uuid signal
material_costs = {"PN-100": 3.20, "PN-200": 5.80, "PN-300": 1.10}
cost = tracker.scrap_cost(
    scrap_uuid="line1:scrap",
    part_id_uuid="line1:part_id",
    material_costs=material_costs
)

# Daily scrap trend
trend = tracker.scrap_trend(scrap_uuid="line1:scrap", window="1D")
```

---

## Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `scrap_by_shift(scrap_uuid)` | Scrap quantity per shift | `DataFrame` with `date`, `shift`, `scrap_quantity` |
| `scrap_by_reason(scrap_uuid, reason_uuid)` | Scrap quantity grouped by reason code | `DataFrame` with `reason`, `scrap_quantity`, `pct_of_total` |
| `scrap_cost(scrap_uuid, part_id_uuid, material_costs)` | Monetary cost of scrap by part number / material type | `DataFrame` with `part_number`, `scrap_quantity`, `cost_per_unit`, `total_cost` |
| `scrap_trend(scrap_uuid, window='1D')` | Scrap quantity per time period | `DataFrame` with `period`, `scrap_quantity` |

All methods accept `handle_resets=True` for scrap counters that reset; the result then gains a `resets` column.

---

## Tips & Hints

!!! tip "Track scrap at the point of generation"
    Place scrap counters as close to the source as possible. If scrap is only counted at end-of-line inspection, you lose visibility into which station generated it. Per-station scrap signals enable root-cause analysis.

!!! info "Related modules"
    - [Rework Tracking](rework-tracking.md) — parts that can be reworked instead of scrapped
    - [Quality Tracking](quality-tracking.md) — overall quality metrics including scrap rates
    - [OEE Calculator](oee-calculator.md) — scrap feeds into the quality component of OEE

---

## See Also

- [Reporting Guide](../../guides/reporting.md)
- [API Reference](../../reference/ts_shape/events/production/scrap_tracking.md)
