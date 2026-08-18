# FacilityCalendar

> Local production days, per-site shifts and DST-correct reporting periods.

**Module:** `ts_shape.reporting.calendar`
**Guide:** [Global Reporting Guide](../../guides/global-reporting.md)

---

## When to Use

Use whenever UTC timeseries has to become a comparable *local* day, week or shift at each plant. Grouping UTC timestamps by calendar date books an evening shift on the wrong day in Asia and cuts a night shift in half in the Americas; a fixed UTC offset breaks twice a year at the DST boundary.

---

## Quick Example

```python
from ts_shape.reporting import FacilityCalendar

calendar = FacilityCalendar(registry)

calendar.localize(readings)          # + local_time, timezone, utc_offset_hours
calendar.assign_shifts(readings)     # + shift, production_date (per-site pattern)
calendar.local_periods(readings, freq="D")   # + period, period_start_utc/_end_utc

# What *should* have been reported -- the coverage denominator:
calendar.period_grid("2026-01-01", "2026-01-31", working_days_only=True)

# Query one plant's local day out of a UTC-indexed store:
start, end = calendar.utc_window("CN-SHA", "2026-03-01", "2026-03-02")
```

---

## Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `localize(df)` | Each row's timestamp in its facility's local wall clock | `DataFrame` |
| `assign_shifts(df)` | Per-site shift name and production day | `DataFrame` |
| `local_periods(df, freq)` | Local reporting period plus its exact UTC bounds | `DataFrame` |
| `period_grid(start, end, freq)` | Expected facility x period calendar | `DataFrame` |
| `utc_window(facility, start, end)` | UTC window for a site-local window | `tuple` |

---

## Tips & Hints

!!! warning "DST is not a rounding error"
    `period_grid` returns a 23-hour local day at spring-forward and a 25-hour day
    at fall-back. Any fixed-offset approximation shifts a whole shift across the
    boundary on those days.

!!! tip "Keep the UTC bounds"
    `local_periods(..., with_utc_bounds=True)` records the exact window each
    consolidated figure came from. Without it a restatement cannot be
    reproduced against the raw data.

!!! info "Related modules"
    - [Facility Registry](facility-registry.md) — supplies timezones and shift patterns
    - [Reporting Coverage](reporting-coverage.md) — consumes `period_grid` as the expected calendar
    - [Shift Reporting](../production/shift-reporting.md) — the single-site equivalent

---

## See Also

- [Global Reporting Guide](../../guides/global-reporting.md)
- [API Reference](../../reference/ts_shape/reporting/calendar.md)
