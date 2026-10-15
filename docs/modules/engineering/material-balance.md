# MaterialBalanceEvents

Mass, energy or flow balance per window: sum of input signals against sum of output signals, the imbalance trend, sustained imbalance events, and each signal's share of the totals.

`ts_shape.events.engineering.material_balance` · [Engineering guide](../../guides/engineering.md) · [API reference](../../reference/ts_shape/events/engineering/material_balance.md)

## Example

```python
import pandas as pd
import ts_shape
from ts_shape.events.engineering.material_balance import MaterialBalanceEvents

df = pd.concat([
    ts_shape.make_timeseries(["feed_1"], n_points=1440, freq="1min", baseline=60.0, seed=1),
    ts_shape.make_timeseries(["feed_2"], n_points=1440, freq="1min", baseline=40.0, seed=2),
    ts_shape.make_timeseries(["product"], n_points=1440, freq="1min", baseline=95.0, drift=-15.0, seed=3),
    ts_shape.make_timeseries(["waste"], n_points=1440, freq="1min", baseline=5.0, seed=4),
], ignore_index=True)

mb = MaterialBalanceEvents(df, input_uuids=["feed_1", "feed_2"], output_uuids=["product", "waste"])
result = mb.balance_check(window="1h", tolerance_pct=5.0).tail()
print(result[["start", "total_input", "total_output", "imbalance_pct", "balanced"]])
events = mb.detect_balance_exceedance(window="1h", tolerance_pct=5.0, min_duration="2h")
shares = mb.contribution_breakdown(window="1h")
```

## Methods

All arguments are keyword-only.

| Method | Returns (per window) |
|---|---|
| `balance_check(window='1h', tolerance_pct=5.0)` | `total_input`, `total_output`, `imbalance`, `imbalance_pct`, `balanced` |
| `imbalance_trend(window='1h')` | `imbalance_pct`, `rolling_avg_imbalance` (last three windows), `trend_direction` |
| `detect_balance_exceedance(window='1h', tolerance_pct=5.0, min_duration='2h')` | runs of unbalanced windows: `avg_imbalance_pct`, `max_imbalance_pct`, `likely_cause` |
| `contribution_breakdown(window='1h')` | per window and signal: `uuid`, `role`, `value`, `pct_of_total` |

## Notes

- `imbalance` = inputs − outputs; `imbalance_pct` = |imbalance| / inputs × 100. A window with zero input is balanced only if nothing flowed out (otherwise `inf`).
- Each flow is held until its next sample and averaged by time per window (`time_weighted=True`, default). `time_weighted=False` averages the samples, which is biased on irregular or change-only signals.
- A window in which a signal has no data has NaN totals and is not balanced, but `detect_balance_exceedance` does not report it and it ends a run, so a sensor outage is not read as an imbalance. `missing_as_zero=True` counts missing data as zero flow.
- `likely_cause`: `accumulation` when inputs exceed outputs in every window of the run, `leak_or_unmetered` when outputs exceed inputs throughout, else `measurement_error`.
