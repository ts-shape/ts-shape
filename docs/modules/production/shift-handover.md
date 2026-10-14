# ShiftHandoverReport

> Generate automated shift handover reports combining production, quality, and downtime.

**Module:** `ts_shape.events.production.shift_handover`
**Guide:** [Reporting Guide](../../guides/reporting.md)

---

## When to Use

Use to generate automated shift handover reports for shift meetings. Combines production, quality, and downtime data into a single summary with issue highlighting. Replaces manual handover notes with a structured, data-driven report that ensures nothing is missed between shifts.

---

## Quick Example

```python
from ts_shape.events.production.shift_handover import ShiftHandoverReport

report = ShiftHandoverReport(production_df)

# Generate from raw signals (one row per shift for the latest date)
targets = {"shift_1": 450, "shift_2": 450, "shift_3": 400}
handover = report.generate_report(
    counter_uuid="line1:parts",
    ok_counter_uuid="line1:ok",
    nok_counter_uuid="line1:nok",
    state_uuid="line1:state",
    targets=targets,
    report_date="2024-01-15",
)

# Highlight issues that need attention (list of dicts)
thresholds = {"production_achievement_pct": 95.0, "quality_pct": 98.0, "availability_pct": 90.0}
issues = report.highlight_issues(report_df=handover, thresholds=thresholds)

# Or build from pre-computed shift-level DataFrames
handover = ShiftHandoverReport.from_shift_data(
    production_df=shift_production,   # e.g. ShiftReporting.shift_production()
    quality_df=shift_quality,         # e.g. QualityTracking.nok_by_shift()
    downtime_df=shift_downtime,       # e.g. DowntimeTracking.downtime_by_shift()
    targets=targets,
)
```

---

## Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `from_shift_data(production_df, quality_df=None, downtime_df=None, targets=None, report_date=None)` | Build report from pre-computed shift-level DataFrames (static method) | `DataFrame` |
| `generate_report(counter_uuid, ok_counter_uuid, nok_counter_uuid, state_uuid, targets=None)` | Generate report from raw signal UUIDs | `DataFrame` |
| `highlight_issues(report_df=None, thresholds=None)` | Identify metrics below thresholds, from a report DataFrame or from raw signal UUIDs | `list[dict]` with `shift`, `metric`, `value`, `threshold`, `severity` |

Both report methods return one row per shift with `date`, `shift`, `production`, `production_target`, `production_achievement_pct`, `ok_parts`, `nok_parts`, `quality_pct`, `availability_pct`, `downtime_minutes`. `highlight_issues` defaults to thresholds of `production_achievement_pct=95`, `quality_pct=98`, `availability_pct=90`; `severity` is `"warning"` within 5% of the threshold, otherwise `"critical"`.

---

## Tips & Hints

!!! tip "Customize thresholds per line"
    Different production lines have different normal operating ranges. Set per-line thresholds in `highlight_issues` so that alerts are meaningful and not ignored due to false positives.

!!! info "Related modules"
    - [Shift Reporting](shift-reporting.md) — production totals feeding into handover reports
    - [Quality Tracking](quality-tracking.md) — quality data for the handover
    - [Downtime Tracking](downtime.md) — downtime data for the handover

---

## See Also

- [Reporting Guide](../../guides/reporting.md)
- [API Reference](../../reference/ts_shape/events/production/shift_handover.md)
