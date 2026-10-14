# PeriodSummary

> Aggregate daily metrics into weekly/monthly summaries.

**Module:** `ts_shape.events.production.period_summary`
**Guide:** [Reporting Guide](../../guides/reporting.md)

---

## When to Use

Use to roll up daily production data into weekly or monthly summaries. Enables period-over-period comparison for management reporting. Ideal for generating the weekly and monthly KPI reports that plant management and corporate teams rely on for capacity planning and performance reviews.

---

## Quick Example

```python
from ts_shape.events.production.period_summary import PeriodSummary

summarizer = PeriodSummary(production_df)

# Weekly summary from raw signals
weekly = summarizer.weekly_summary(
    counter_uuid="line1:parts",
    ok_counter_uuid="line1:ok",
    nok_counter_uuid="line1:nok",
)

# Monthly summary from raw signals
monthly = summarizer.monthly_summary(counter_uuid="line1:parts")

# Compare two specific periods
comparison = summarizer.compare_periods(
    counter_uuid="line1:parts",
    period1=("2024-01-01", "2024-03-31"),
    period2=("2024-04-01", "2024-06-30")
)

# Or roll up an existing daily DataFrame (must have a "date" column)
monthly_from_df = PeriodSummary.from_daily_data(
    daily_df=daily_metrics,
    freq="MS",
)
```

---

## Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `from_daily_data(daily_df, freq='W', date_column='date')` | Roll up a daily DataFrame to weekly (`'W'`) or monthly (`'MS'`) frequency | `DataFrame` with `start`, `end`, `days` plus aggregated metric columns |
| `weekly_summary(counter_uuid, ok_counter_uuid=None, nok_counter_uuid=None)` | Weekly production summary from raw signals | `DataFrame` with `week_start`, `week_end`, `total_production`, `daily_avg`, `production_days` (plus `ok_parts`, `nok_parts`, `quality_pct` when OK/NOK counters are given) |
| `monthly_summary(counter_uuid, ok_counter_uuid=None, nok_counter_uuid=None)` | Monthly production summary from raw signals | `DataFrame` with `year`, `month`, `total_production`, `daily_avg`, `production_days` (plus quality columns as above) |
| `compare_periods(counter_uuid, period1, period2)` | Side-by-side comparison of two date ranges | `DataFrame` with `metric`, `period1_value`, `period2_value`, `change`, `change_pct` |

---

## Tips & Hints

!!! tip "Align periods to calendar boundaries"
    Use ISO week boundaries (Monday start) or calendar month boundaries for consistency. Partial weeks or months at the edges of your date range can skew averages and make period comparisons misleading.

!!! info "Related modules"
    - [Shift Reporting](shift-reporting.md) — shift-level data that feeds into daily and period summaries
    - [Target Tracking](target-tracking.md) — compare period summaries against targets
    - [OEE Calculator](oee-calculator.md) — period-level OEE trends

---

## See Also

- [Reporting Guide](../../guides/reporting.md)
- [API Reference](../../reference/ts_shape/events/production/period_summary.md)
