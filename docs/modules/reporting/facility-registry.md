# Facility & FacilityRegistry

> Site master data — timezone, region, currency, shift pattern — and its validation.

**Module:** `ts_shape.reporting.facility`
**Guide:** [Global Reporting Guide](../../guides/global-reporting.md)

---

## When to Use

Use as the single source of truth for the attributes a consolidated report groups, converts and localises on. Every other class in the reporting layer reads timezones, regions and currencies from here rather than guessing, and `validate()` turns silent master-data defects into an explicit, testable frame.

---

## Quick Example

```python
from ts_shape.reporting import Facility, FacilityRegistry

registry = FacilityRegistry([
    Facility("DE-STR", name="Stuttgart", timezone="Europe/Berlin",
             region="EMEA", country="DE", currency="EUR", capacity_per_day=1200),
    Facility("US-DTW", name="Detroit", timezone="America/Detroit",
             region="AMER", country="US", currency="USD",
             shifts={"day": ("07:00", "19:00"), "night": ("19:00", "07:00")}),
])

registry.validate()                       # empty == sound
registry.attach(kpi_df, columns=["region", "currency"])
registry.filter(region="EMEA").ids

# Master data usually lives in a table already:
registry = FacilityRegistry.from_frame(plants_df)
```

---

## Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `FacilityRegistry.from_frame(df)` | Build a registry from a master-data table | `FacilityRegistry` |
| `validate()` | Report timezone, region, currency and shift-pattern defects | `DataFrame` |
| `to_frame()` | The registry as one row per site | `DataFrame` |
| `attach(df, columns=...)` | Left-join master-data columns onto a KPI frame | `DataFrame` |
| `regions()` | Region name to the site ids it contains | `dict` |
| `filter(region=..., country=...)` | A narrowed registry | `FacilityRegistry` |
| `Facility.utc_offset_hours(at)` | DST-aware UTC offset at a moment | `float` |

---

## Tips & Hints

!!! tip "Treat `validate()` as a gate"
    Assert `registry.validate().empty` in the job that builds the report. An
    unresolvable timezone silently lands a plant in UTC; a missing region lands
    it under `unassigned` in every regional view.

!!! warning "Overlapping and gapped shift patterns"
    A shift pattern that overlaps counts readings twice; one with a hole assigns
    them to `"unknown"` and drops them from shift reports. `validate()` reports
    both as `overlapping_shifts` / `shift_gap`.

!!! info "Related modules"
    - [Facility Calendar](facility-calendar.md) — uses the timezones and shifts
    - [Fleet Rollup](fleet-rollup.md) — uses the regions
    - [Metric Normalizer](metric-normalizer.md) — uses the currencies and capacities

---

## See Also

- [Global Reporting Guide](../../guides/global-reporting.md)
- [API Reference](../../reference/ts_shape/reporting/facility.md)
