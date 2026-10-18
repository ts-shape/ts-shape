# CalibrationValidationEvents

Checks whether an inline sensor still reads true: linearity and a fitted calibration curve from a multi-point check, and per-window bias, drift and stability against a reference value or a reference sensor.

`ts_shape.events.quality.calibration_validation` · [Quality guide](../../guides/quality.md) · [API reference](../../reference/ts_shape/events/quality/calibration_validation.md)

## Example

```python
import ts_shape
from ts_shape.events.quality.calibration_validation import CalibrationValidationEvents

points = {0.0: [0.02, 0.01, 0.03], 50.0: [50.10, 50.08, 50.12], 100.0: [100.25, 100.21, 100.27]}
print(CalibrationValidationEvents.linearity_assessment(points))
curve = CalibrationValidationEvents.calibration_curve(points, degree=2)

df = ts_shape.make_timeseries(["ft201", "ft201_ref"], n_points=5000, freq="1min", noise=0.2)
df.loc[df["uuid"] == "ft201", "value_double"] += 0.6  # biased sensor
cal = CalibrationValidationEvents(df, "ft201", reference_uuid="ft201_ref")
recal = cal.detect_recalibration_need(window="1D", max_bias=0.5, max_drift_rate=0.1)
stability = cal.calibration_stability(window="8h", lookback=10)
```

## Methods

| Method | Returns |
|---|---|
| `linearity_assessment(calibration_points)` (static) | per reference point: `reference`, `measured_mean`, `measured_std`, `bias`, `linearity_error` |
| `calibration_curve(calibration_points, degree=1)` (static) | per polynomial coefficient: `coefficient_index`, `power`, `coefficient_value`, `r_squared`, `max_residual` |
| `detect_recalibration_need(window='1D', max_bias=None, max_drift_rate=None)` | per window: `start`, `end`, `duration_seconds`, `bias`, `drift_rate`, `recalibration_needed`, `reason` |
| `calibration_stability(window='8h', lookback=10)` | per window: `start`, `end`, `duration_seconds`, `bias_mean`, `bias_std`, `bias_trend`, `stability_score` |

## Notes

- The static methods take `{reference_value: [measured, ...]}` and need no DataFrame.
- The window methods need a reference: `reference_value` (a constant, e.g. a check standard) or `reference_uuid` (that signal's mean in the same window). Without one they return an empty frame. Windows with fewer than 2 samples or no reference data are skipped. `window` is a fixed length (`"8h"`) or a calendar frequency (`"1D"` a calendar day, `"W"`, `"MS"`); the reference is averaged over the window's true bounds.
- `bias` = window mean − reference. `drift_rate` is the bias change from the previous window (0 for the first).
- `linearity_error` is each point's residual from a straight line fitted through all points (at least 2 points).
- `calibration_curve` fits the instrument response, measured = f(reference) (invert it to correct readings). The coefficients come from `numpy.polyfit`, highest power first; `power` is the exponent each one multiplies (degree used − `coefficient_index`), and the degree is capped at number of points − 1.
- `stability_score` starts at 100 and deducts up to 50 for the bias magnitude, 30 for its spread and 20 for its trend (slope over the `lookback` windows). The penalties are in signal units, so scores are only comparable between signals of similar scale.
