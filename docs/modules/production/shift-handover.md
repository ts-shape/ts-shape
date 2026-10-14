# ShiftHandoverEvents

A shift handover report (production against target, quality, availability, downtime; one row per shift) and the list of metrics below threshold, from raw signals or from shift tables computed by other modules.

`ts_shape.events.production.shift_handover` · [Production guide](../../guides/production.md) · [API reference](../../reference/ts_shape/events/production/shift_handover.md)

## Example

```python
import numpy as np
import pandas as pd
from ts_shape.events.production.shift_handover import ShiftHandoverEvents

t = pd.date_range("2024-01-15 06:00", periods=1440, freq="1min")
n = np.arange(1440)
ints = {"line1:parts": n, "line1:ok": n - n // 40, "line1:nok": n // 40}
df = pd.concat([pd.DataFrame({"systime": t, "uuid": u, "value_integer": v}) for u, v in ints.items()]
               + [pd.DataFrame({"systime": t, "uuid": "line1:running", "value_bool": n % 60 < 50})],
               ignore_index=True)

report = ShiftHandoverEvents(df)
handover = report.generate_report(counter_uuid="line1:parts", ok_counter_uuid="line1:ok",
                                  nok_counter_uuid="line1:nok", state_uuid="line1:running",
                                  targets={"shift_1": 450, "shift_2": 450, "shift_3": 400})
print(report.highlight_issues(report_df=handover))
```

## Methods

| Method | Returns |
|---|---|
| `generate_report(counter_uuid, ok_counter_uuid, nok_counter_uuid, state_uuid, targets=None, report_date=None)` | one row per shift of `report_date` (default: latest date) |
| `from_shift_data(production_df, quality_df=None, downtime_df=None, targets=None, report_date=None)` (static) | the same report from `ShiftReportEvents.shift_production()`, `QualityEvents.nok_by_shift()` and `DowntimeEvents.downtime_by_shift()` outputs |
| `highlight_issues(report_df=None, thresholds=None)` | one row per metric below threshold: `date`, `shift`, `metric`, `value`, `threshold`, `severity` |

Report columns: `date`, `shift`, `production`, `production_target`, `production_achievement_pct`, `ok_parts`, `nok_parts`, `quality_pct`, `availability_pct`, `downtime_seconds`, `downtime_minutes` (+ `resets` from `generate_report`).

## Notes

- The run state may be a string signal (`running_value='Running'` in `value_string`) or a boolean run signal (`value_bool`, True = running); the boolean is used when the state rows carry no `value_string`.
- `highlight_issues` default thresholds: `production_achievement_pct=95`, `quality_pct=98`, `availability_pct=90`. `severity` is `"warning"` within 5 % of the threshold, otherwise `"critical"`. Without `report_df` it builds the report from the UUID arguments. `as_frame=False` returns the deprecated `list[dict]`.
- Counters are reset-aware by default; increments are computed over the whole signal before `report_date` is selected and booked to the shift of the reading that reports them.
- Default shifts are `shift_1` 06-14, `shift_2` 14-22, `shift_3` 22-06; `targets` keys must match the shift names. A night shift belongs to its start date.
