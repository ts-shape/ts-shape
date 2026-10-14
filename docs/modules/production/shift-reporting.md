# ShiftReportEvents

> Simple shift-based production reporting from counter and part-ID signals.

**Module:** `ts_shape.events.production.shift_reporting`
**Guide:** [Production Guide](../../guides/production.md)

---

## When to Use

Use for daily shift performance reporting. Computes production quantities per shift and compares to targets and historical performance. Ideal for shift supervisors who need a quick view of how each shift performed relative to plan.

---

## Quick Example

```python
from ts_shape.events.production.shift_reporting import ShiftReportEvents

reporter = ShiftReportEvents(
    production_df,
    shift_definitions={
        "morning": ("06:00", "14:00"),
        "afternoon": ("14:00", "22:00"),
        "night": ("22:00", "06:00"),
    },
)

# Production totals per shift
shift_prod = reporter.shift_production(counter_uuid="line1:parts")

# Compare shifts over the last 7 days
comparison = reporter.shift_comparison(counter_uuid="line1:parts", days=7)

# Check actual vs targets (keys must match the shift names)
targets = {"morning": 500, "afternoon": 480, "night": 400}
result = reporter.shift_targets(counter_uuid="line1:parts", targets=targets)
```

---

## Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `shift_production(counter_uuid, part_id_uuid=None)` | Production totals per shift, optionally split by part ID | `DataFrame` with `date`, `shift`, `quantity` (plus `part_number` when `part_id_uuid` is given) |
| `shift_comparison(counter_uuid, days=7)` | Compare shift performance over a number of days | `DataFrame` with `shift`, `avg_quantity`, `min_quantity`, `max_quantity`, `std_quantity`, `days_count` |
| `shift_targets(counter_uuid, targets)` | Compare actual production against per-shift targets | `DataFrame` with `date`, `shift`, `actual`, `target`, `variance`, `achievement_pct` |
| `best_and_worst_shifts(counter_uuid, days=30)` | Identify the five best and five worst shifts over a period | `DataFrame` with `start`, `end`, `duration_seconds`, `rank_type` (`"best"`/`"worst"`), `date`, `shift`, `quantity`. `as_frame=False` returns the deprecated `dict` form |

Without `shift_definitions`, the default shifts are `shift_1` (06:00-14:00), `shift_2` (14:00-22:00) and `shift_3` (22:00-06:00).

---

## Tips & Hints

!!! tip "Define shift boundaries consistently"
    Pass `shift_definitions` that match your plant's actual shift calendar. Misaligned boundaries will split production counts incorrectly.

!!! info "Related modules"
    - [Target Tracking](target-tracking.md) — generic target vs actual comparison
    - [Shift Handover](shift-handover.md) — full handover reports combining production, quality, and downtime
    - [Period Summary](period-summary.md) — roll up shift data into weekly/monthly summaries

---

## See Also

- [Production Guide](../../guides/production.md)
- [API Reference](../../reference/ts_shape/events/production/shift_reporting.md)
