# StatisticalProcessControlEvents

Shewhart control-chart rules (the eight Nelson tests, including the Western Electric zone rules), dynamic control limits and CUSUM shift detection. Control limits come from reference rows of one signal (`tolerance_uuid`); the rules are checked on the rows of another (`actual_uuid`).

`ts_shape.events.quality.statistical_process_control` · [Quality guide](../../guides/quality.md) · [API reference](../../reference/ts_shape/events/quality/statistical_process_control.md)

## Example

```python
import pandas as pd
import ts_shape
from ts_shape.events.quality.statistical_process_control import StatisticalProcessControlEvents

df = pd.concat([
    ts_shape.make_timeseries(["limit:temp"], n_points=200, seed=1),  # in-control reference
    ts_shape.make_timeseries(["sensor:temp"], n_points=500, drift=3.0, n_outliers=3, seed=2),
], ignore_index=True)

spc = StatisticalProcessControlEvents(df, value_column="value_double", tolerance_uuid="limit:temp",
                                      actual_uuid="sensor:temp", event_uuid="spc:temp")
violations = spc.detect_violations()
print(violations["triggered_rule"].value_counts())
shifts = spc.detect_cusum_shifts(k=0.5, h=5.0)
```

## Methods

| Method | Returns |
|---|---|
| `control_limits()` | center line and ±1σ/2σ/3σ limits from the reference rows |
| `detect_violations(selected_rules=None, include_severity=True)` | one row per violating point: `systime`, value, `triggered_rule`, `severity` |
| `detect_rule_violations(selected_rules=None)` | one row per point and rule: `systime`, `value`, `rule`, `severity` |
| `dynamic_control_limits(method='moving_range', window_size=20)` | rolling limits per `systime`: `mean`, `1sigma_upper` … `3sigma_lower`; `method='ewma'` uses an EWMA with span `window_size` |
| `detect_cusum_shifts(target=None, k=0.5, h=5.0)` | `systime`, value, `cusum_high`, `cusum_low`, `shift_direction`, `severity` |
| `interpret_violations(violations_df)` | adds `interpretation`, `meaning`, `recommendation` |
| `rule_1(df, limits)` … `rule_8(df, limits)` | the violating rows with a boolean rule column; the input is not modified |

## Notes

- The rules follow Nelson (1984): rule 1 a point beyond 3σ; rule 2 nine points on one side of the center line; rule 3 six points steadily rising or falling; rule 4 fourteen points alternating up and down; rule 5 two of three beyond 2σ on one side; rule 6 four of five beyond 1σ on one side; rule 7 fifteen points within 1σ; rule 8 eight points with none within 1σ. Points beyond the next zone count for rules 5 and 6. An unchanged value breaks the runs of rules 3 and 4.
- Severity per rule: rule 1 `critical`, rule 5 `high`, rules 2, 3, 4, 6 `medium`, rules 7 and 8 `low`. `detect_violations` reports a point that violates several rules once, with the first selected rule.
- CUSUM: `k` and `h` are in multiples of the reference σ; `target` defaults to the reference mean. It is skipped (empty result) when the reference has fewer than two rows or zero spread.
- `selected_rules` takes names such as `["rule_1", "rule_5"]`.
