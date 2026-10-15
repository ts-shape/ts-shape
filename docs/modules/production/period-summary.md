# PeriodSummaryEvents

Weekly and monthly production summaries and a comparison of two date ranges from a production counter, or a weekly/monthly roll-up of any daily metrics table.

`ts_shape.events.production.period_summary` · [Production guide](../../guides/production.md) · [API reference](../../reference/ts_shape/events/production/period_summary.md)

## Example

```python
import numpy as np
import pandas as pd
from ts_shape.events.production.period_summary import PeriodSummaryEvents

t = pd.date_range("2024-01-01", periods=60 * 144, freq="10min")  # 60 days
df = pd.DataFrame({"systime": t, "uuid": "line1:parts", "value_integer": np.arange(len(t)) * 3})

summary = PeriodSummaryEvents(df)
print(summary.weekly_summary(counter_uuid="line1:parts").head())
monthly = summary.monthly_summary(counter_uuid="line1:parts")
comparison = summary.compare_periods(counter_uuid="line1:parts",
                                     period1=("2024-01-01", "2024-01-31"),
                                     period2=("2024-02-01", "2024-02-29"))
```

## Methods

| Method | Returns |
|---|---|
| `weekly_summary(counter_uuid, ok_counter_uuid=None, nok_counter_uuid=None)` | per week: `week_start`, `week_end`, `total_production`, `daily_avg`, `production_days` (+ `ok_parts`, `nok_parts`, `quality_pct`, `resets`) |
| `monthly_summary(counter_uuid, ok_counter_uuid=None, nok_counter_uuid=None)` | per month: `year`, `month`, same metrics |
| `compare_periods(counter_uuid, period1, period2)` | one row per metric: `metric`, `period1_value`, `period2_value`, `change`, `change_pct` |
| `from_daily_data(daily_df, window='W', date_column='date', aggregations=None)` (static) | `start`, `end`, `days` and the aggregated numeric columns |

## Notes

- Weeks run Monday to Monday midnight; months are calendar months. Partial weeks or months at the edges of the data lower the averages.
- `from_daily_data` takes any daily table with a `date` column (for example `QualityEvents.daily_quality_summary()` merged with `DowntimeEvents.availability_trend(window="1D")`, or `OEEEvents.oee()` with `date_column="start"`). `window='W'` gives Monday-to-Sunday weeks, `'W-<DAY>'` weeks ending on that day, `'MS'` calendar months.
- Each numeric column has a declared roll-up rule. Counts, durations and energy are summed. A ratio KPI is recomputed from the period's summed basis columns (ISO 22400-2), so the daily tables the library produces roll up exactly:

  | Column | Period value |
  |---|---|
  | `quality_pct`, `first_pass_yield_pct` | 100 × Σgood / Σtotal (`good_parts`/`ok_parts`, `total_parts` or ok + nok) |
  | `nok_rate_pct` | 100 × Σnok / Σtotal |
  | `availability_pct` | 100 × Σ`run_seconds` / Σ`planned_seconds` (OEE), Σuptime / Σ(uptime + downtime) (downtime), Σ`operating_hours` / Σ(operating + repair) (reliability) |
  | `performance_pct` | 100 × Σ`actual_parts` / Σ`ideal_parts` |
  | `oee_pct` | availability × performance × quality of the period |
  | `mttr_hours`, `mtbf_hours` | Σ`repair_hours` / Σ`failures`, Σ`operating_hours` / Σ`failures` |
  | `mtbf_iso22400_hours`, `failure_rate_per_hour` | Σoperating / (Σfailures + 1), Σfailures / Σoperating |
  | `enpi`, `sec`, `efficiency` | Σ`energy_kwh` / Σ`units_produced`, Σ`total_energy` / Σ`total_output`, Σ`units` / Σ`energy` (ISO 50006) |

  Without the basis columns a ratio is the mean weighted by its denominator when that column is present (e.g. MTTR by `failures`, quality by `total_parts`, which is exact), availability, OEE and other `pct` columns by the rows' `duration_seconds`, and otherwise the plain mean with a `DataQualityWarning` naming the columns. Other ratio-like columns (`avg`, `mean`, `rate`, `ratio`, cycle times) are averaged the same way, never summed. A numeric column that matches no rule is dropped with a `DataQualityWarning`; `aggregations={"col": "sum" | "mean" | "min" | "max" | "first" | "last" | "drop"}` sets the rule per column.
- Counters are reset-aware by default (`handle_resets=True`); increments are booked to the day of the reading that reports them.
