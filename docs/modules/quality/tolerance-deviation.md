# ToleranceDeviationEvents

Samples of a measurement outside its specification limits, grouped into events with deviation size and severity, and the capability indices Cp, Cpk, Pp, Ppk and Cpm. The limits are signals in the same frame, so they can change over time.

`ts_shape.events.quality.tolerance_deviation` · [Quality guide](../../guides/quality.md) · [API reference](../../reference/ts_shape/events/quality/tolerance_deviation.md)

## Example

```python
import pandas as pd
import ts_shape
from ts_shape.events.quality.tolerance_deviation import ToleranceDeviationEvents

df = pd.concat([
    ts_shape.make_timeseries(["shaft_diameter"], n_points=500, baseline=50.0, noise=0.3, n_outliers=4),
    ts_shape.make_timeseries(["shaft_diameter:usl"], n_points=1, baseline=51.0, noise=0),
    ts_shape.make_timeseries(["shaft_diameter:lsl"], n_points=1, baseline=49.0, noise=0),
], ignore_index=True)

tdev = ToleranceDeviationEvents(df, tolerance_column="value_double", actual_column="value_double",
                                actual_uuid="shaft_diameter", upper_tolerance_uuid="shaft_diameter:usl",
                                lower_tolerance_uuid="shaft_diameter:lsl", event_uuid="quality:tolerance")
print(tdev.detect_deviations()[["systime", "value_double", "deviation_abs", "severity"]])
capability = tdev.capability_indices(target_value=50.0)
```

## Methods

| Method | Returns |
|---|---|
| `detect_deviations()` | out-of-tolerance events: `systime`, the measured value, `deviation_abs`, `deviation_pct`, `severity` (`minor`/`major`/`critical`) |
| `capability_indices(target_value=None, sigma_method='moving_range')` | one row: `Cp`, `Cpk`, `Pp`, `Ppk`, `Cpm`, `mean`, `std`, `std_within`, `usl`, `lsl`, `target`, `sample_count` |

## Notes

- Each sample is checked against the latest limit values at that time (`tolerance_lag` shifts the limit timestamps). Out-of-tolerance samples closer than `time_threshold` (default `5min`) form one event, reported as its first and last row; a single-sample event is one row.
- Severity: `minor` within `warning_threshold` (default 0.8) × the tolerance range, `major` outside the limits, `critical` when the deviation exceeds twice the tolerance range.
- Two-sided spec: pass both `upper_tolerance_uuid` and `lower_tolerance_uuid`. With only one the spec is one-sided: Cp/Pp are NaN, Cpk/Ppk are CPU or CPL, and `deviation_pct` is relative to the single limit.
- Cp/Cpk use the within σ̂ = MR̄ / 1.128 of consecutive measurements (`std_within`); Pp/Ppk use the overall sample standard deviation (`std`), as in the AIAG SPC manual. `sigma_method="std"` computes Cp/Cpk from the overall σ (Cp = Pp, the behavior before October 2026). `Cpm = (USL − LSL) / (6·sqrt(σ² + (mean − target)²))` with the overall σ is computed only with `target_value`; otherwise `target` reports the spec midpoint.
- `actual_uuid` is required. The single `tolerance_uuid` (compared through `compare_func`) is deprecated; a single upper limit is `upper_tolerance_uuid` alone. `capability_indices(..., as_frame=False)` returns the deprecated dict.
- Measurements without spread (constant values, a single measurement or only missing values) give NaN capability indices and a `DataQualityWarning`, as in `CapabilityTrendingEvents`.
