# CalibrationValidationEvents

> Check an inline sensor's calibration: linearity across its range, a fitted calibration curve, and when bias or drift says it is time to recalibrate.

**Module:** `ts_shape.events.quality.calibration_validation`  
**Guide:** [Quality Control & SPC](../../guides/quality.md)

---

## When to Use

Use when you need evidence that a measuring instrument still reads true:

- after a multi-point calibration check, to quantify bias and linearity error at each reference point and fit a calibration curve;
- in production, to compare the sensor against a reference (a fixed check-standard value or a second, trusted sensor) window by window and flag when bias or drift crosses your limits;
- to score how stable the bias has been over the last few windows.

---

## Quick Example

```python
from ts_shape.events.quality.calibration_validation import CalibrationValidationEvents

# Multi-point check: reference value -> repeated readings
points = {
    0.0: [0.02, 0.01, 0.03],
    50.0: [50.10, 50.08, 50.12],
    100.0: [100.25, 100.21, 100.27],
}
CalibrationValidationEvents.linearity_assessment(points)
CalibrationValidationEvents.calibration_curve(points, degree=2)

# In production: compare against a trusted reference sensor
cal = CalibrationValidationEvents(
    df,
    "line2:flowmeter_ft201",
    reference_uuid="line2:flowmeter_ref_ft201b",
)
recal = cal.detect_recalibration_need(window="1D", max_bias=0.5, max_drift_rate=0.1)
stability = cal.calibration_stability(window="8h", lookback=10)
```

---

## Key Methods

| Method | Shape | Returns |
|---|---|---|
| `linearity_assessment(calibration_points)` | static | one row per reference point: `reference`, `measured_mean`, `measured_std`, `bias`, `linearity_error` |
| `calibration_curve(calibration_points, degree)` | static | one row per polynomial coefficient: `coefficient_index`, `coefficient_value`, `r_squared`, `max_residual` |
| `detect_recalibration_need(window, max_bias, max_drift_rate)` | summary | per window: `window_start`, `bias`, `drift_rate`, `recalibration_needed`, `reason` |
| `calibration_stability(window, lookback)` | summary | per window: `window_start`, `bias_mean`, `bias_std`, `bias_trend`, `stability_score` |

## Notes

- `linearity_assessment` and `calibration_curve` are static methods: they take a dict `{reference_value: [measured, ...]}` and need no DataFrame.
- The production methods need a reference: either `reference_value` (a constant, e.g. a check standard) or `reference_uuid` (the mean of that signal in the same window). Without one they return an empty frame. Windows with fewer than 2 samples or no reference data are skipped.
- `bias` = window mean − reference. `drift_rate` is the bias change from the previous window (0 for the first).
- `linearity_error` is the residual of each point's mean from a straight line fitted through all points (needs at least 2 points).
- `calibration_curve` coefficients come from `numpy.polyfit`, highest power first; the degree is capped at number of points − 1.
- `stability_score` starts at 100 and deducts up to 50 for the bias magnitude, 30 for its spread and 20 for its trend (slope over the `lookback` windows). The penalties are in signal units, so the score is only comparable between signals of a similar scale.
