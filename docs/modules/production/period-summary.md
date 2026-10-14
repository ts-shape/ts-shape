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
| `from_daily_data(daily_df, window='W', date_column='date')` (static) | `start`, `end`, `days` and the aggregated numeric columns |

## Notes

- Weeks run Monday to Monday midnight; months are calendar months. Partial weeks or months at the edges of the data lower the averages.
- `from_daily_data` takes any daily table with a `date` column (for example `QualityEvents.daily_quality_summary()` merged with `DowntimeEvents.availability_trend(window="1D")`). Columns whose name contains `pct` are averaged; all other numeric columns are summed. `window='W'` gives Monday-to-Sunday weeks, `'W-<DAY>'` weeks ending on that day, `'MS'` calendar months.
- Counters are reset-aware by default (`handle_resets=True`); increments are booked to the day of the reading that reports them.
