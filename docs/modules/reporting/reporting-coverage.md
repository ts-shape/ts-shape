# ReportingCoverage

> Did every facility actually report? The precondition for any global figure.

**Module:** `ts_shape.reporting.coverage`
**Guide:** [Global Reporting Guide](../../guides/global-reporting.md)

---

## When to Use

Run it before every consolidation. A total computed over 34 of 37 plants looks exactly like one computed over all 37 — it is simply lower, and nothing in the frame says why. Silent gaps are the most common defect in cross-site reporting and the hardest to spot after the fact.

---

## Quick Example

```python
from ts_shape.reporting import FacilityCalendar, ReportingCoverage

expected = FacilityCalendar(registry).period_grid("2026-01-01", "2026-01-31")
coverage = ReportingCoverage(kpi_df, value_columns=["parts_produced", "oee_pct"])

coverage.summary(expected)        # one row to publish beside the headline
coverage.completeness(expected)   # per site: complete / partial / silent
coverage.missing(expected)        # the gaps, one row each
coverage.null_report()            # null rate per site and metric
coverage.freshness(as_of="2026-02-01", max_age_days=2)
```

---

## Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `summary(expected)` | Fleet coverage in one row | `DataFrame` |
| `completeness(expected)` | Per-site completeness and status | `DataFrame` |
| `missing(expected)` | Every facility/period combination that never arrived | `DataFrame` |
| `null_report()` | Null rate per facility and metric | `DataFrame` |
| `freshness(as_of=..., max_age_days=...)` | Last reported period and staleness | `DataFrame` |

---

## Tips & Hints

!!! warning "Pass a real expected calendar"
    Without one the expected grid is the cross product of what was *observed*.
    That finds a site that skipped a period, but never one that reported nothing
    at all. Use
    [`FacilityCalendar.period_grid`](facility-calendar.md).

!!! tip "A column that is always NaN at one site"
    `null_report` shows it as 100 %. That is usually a signal that was never
    mapped at that plant, not a production problem.

!!! info "Related modules"
    - [Facility Calendar](facility-calendar.md) — builds the expected calendar
    - [Fleet Rollup](fleet-rollup.md) — its `contributors` count is the same check, after the fact

---

## See Also

- [Global Reporting Guide](../../guides/global-reporting.md)
- [API Reference](../../reference/ts_shape/reporting/coverage.md)
