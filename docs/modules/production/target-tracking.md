# TargetTracking

> Compare any production metric to targets — every plant has daily/shift targets.

**Module:** `ts_shape.events.production.target_tracking`
**Guide:** [Reporting Guide](../../guides/reporting.md)

---

## When to Use

Use as a generic target vs actual comparison module. Works with any production metric (output, quality, OEE) against per-shift or daily targets. This is the go-to module when management sets numeric goals and you need to report achievement rates over time.

---

## Quick Example

```python
from ts_shape.events.production.target_tracking import TargetTracking

tracker = TargetTracking(
    production_df,
    shift_definitions={
        "morning": ("06:00", "14:00"),
        "afternoon": ("14:00", "22:00"),
        "night": ("22:00", "06:00"),
    },
)

# Compare actual production to shift targets (keys must match the shift names)
shift_targets = {"morning": 500, "afternoon": 480, "night": 400}
comparison = tracker.compare_to_target(
    metric_uuid="line1:parts",
    targets=shift_targets
)

# Daily achievement summary
summary = tracker.target_achievement_summary(
    metric_uuid="line1:parts",
    daily_target=1400
)

# How often do we hit the daily target?
hit_rate = tracker.target_hit_rate(
    metric_uuid="line1:parts",
    daily_target=1400
)
```

---

## Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `compare_to_target(metric_uuid, targets)` | Actual vs target per shift with variance and percentage | `DataFrame` with `date`, `shift`, `actual`, `target`, `variance`, `achievement_pct`, `status` |
| `target_achievement_summary(metric_uuid, daily_target)` | Daily achievement percentage over the analysis period | `DataFrame` with `date`, `actual`, `target`, `variance`, `achievement_pct`, `status` |
| `target_hit_rate(metric_uuid, daily_target)` | Share of days where the daily target was met or exceeded | `dict` with `total_days`, `days_on_target`, `hit_rate_pct`, `avg_achievement_pct` |

`compare_to_target` and `target_achievement_summary` accept `handle_resets=True` for counters that reset; the result then gains a `resets` column. Without `shift_definitions`, shifts are named `shift_1`, `shift_2`, `shift_3`.

---

## Tips & Hints

!!! tip "Adjust targets for product mix"
    A flat daily target works when producing a single product. If your line runs multiple variants with different cycle times, pass product-weighted targets to get meaningful achievement rates.

!!! info "Related modules"
    - [Shift Reporting](shift-reporting.md) — shift-level production with target comparison
    - [Period Summary](period-summary.md) — roll up target achievement into weekly/monthly views
    - [Operator Performance](operator-performance.md) — operator efficiency vs targets

---

## See Also

- [Reporting Guide](../../guides/reporting.md)
- [API Reference](../../reference/ts_shape/events/production/target_tracking.md)
