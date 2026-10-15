# DisturbanceRecoveryEvents

Process upsets on a signal (deviations from a rolling baseline or a setpoint), how long the process took to recover, how often upsets occur, and whether the process settled at a different level afterwards.

`ts_shape.events.engineering.disturbance_recovery` · [Engineering guide](../../guides/engineering.md) · [API reference](../../reference/ts_shape/events/engineering/disturbance_recovery.md)

## Example

```python
import ts_shape
from ts_shape.events.engineering.disturbance_recovery import DisturbanceRecoveryEvents

df = ts_shape.make_timeseries(["reactor_temp"], n_points=2880, freq="10s", baseline=180.0, noise=0.3)
df.loc[1500:1540, "value_double"] += 6.0  # upset of about 7 min

dr = DisturbanceRecoveryEvents(df, "reactor_temp")
result = dr.detect_disturbances(baseline_window="10min", n_sigma=3.0, min_duration="30s")
print(result[["start", "end", "peak_deviation", "direction"]])
recovery = dr.recovery_time(baseline_window="10min", recovery_pct=0.95, max_recovery="1h")
frequency = dr.disturbance_frequency(window="8h")
before_after = dr.before_after_comparison(comparison_window="5min")
```

## Methods

All arguments are keyword-only; each method also takes `baseline_window` and `n_sigma`.

| Method | Returns |
|---|---|
| `detect_disturbances(baseline_window='10min', n_sigma=3.0, min_duration='30s')` | one interval per upset: `peak_deviation`, `mean_deviation`, `direction`, `disturbance_type` |
| `recovery_time(recovery_pct=0.95, max_recovery='1h')` | per upset: `recovery_time_seconds`, `recovered`, `pre_disturbance_mean`, `post_recovery_mean`, `residual_offset` |
| `disturbance_frequency(window='8h')` | per window: `disturbance_count`, `total_disturbance_seconds`, `pct_time_disturbed`, `avg_recovery_seconds` |
| `before_after_comparison(comparison_window='5min')` | per upset: `pre_mean`, `post_mean`, `pre_std`, `post_std`, `mean_shift`, `variance_ratio` |

## Notes

- A disturbance is where the signal deviates from the rolling mean over `baseline_window` by more than `n_sigma` rolling standard deviations, for at least `min_duration`. With `setpoint_uuid=` in the constructor, the setpoint replaces the rolling baseline.
- Recovered means the deviation has fallen to within `1 − recovery_pct` of the peak deviation, searched up to `max_recovery` after the upset.
- `before_after_comparison` compares `comparison_window` just before the start and just after the end; upsets with fewer than two samples on either side are skipped.
