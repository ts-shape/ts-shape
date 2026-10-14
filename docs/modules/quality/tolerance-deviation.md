# ToleranceDeviationEvents

> Processes DataFrame data for comparing tolerance and actual values with severity classification and process capability indices.

**Module:** `ts_shape.events.quality.tolerance_deviation`  
**Guide:** [Quality Control & SPC](../../guides/quality.md)

---

## When to Use

Use when you have process signals with defined upper/lower specification limits and need to track deviations with severity classification. Computes Cp/Cpk/Pp/Ppk indices for process capability assessment. Requires long-format data with the actual measurements under `actual_uuid` and the specification limits under `upper_tolerance_uuid`/`lower_tolerance_uuid` (or a single one-sided upper limit under `tolerance_uuid`).

---

## Quick Example

```python
from ts_shape.events.quality.tolerance_deviation import ToleranceDeviationEvents

tdev = ToleranceDeviationEvents(
    df,
    tolerance_column="value_double",
    actual_column="value_double",
    actual_uuid="shaft_diameter",
    upper_tolerance_uuid="shaft_diameter:usl",
    lower_tolerance_uuid="shaft_diameter:lsl",
    event_uuid="quality:tolerance_violation",
    time_threshold="5min",
)

# Apply tolerance checks with severity classification
events = tdev.detect_deviations()
print(events[["systime", "deviation_abs", "deviation_pct", "severity"]].head())

# Compute process capability indices against the latest spec limits
capability = tdev.capability_indices(target_value=50.0)
print(f"Cpk: {capability['Cpk']:.2f}, Ppk: {capability['Ppk']:.2f}")
```

---

## Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `detect_deviations()` | Apply tolerance checks, group out-of-tolerance samples within `time_threshold` into events (first and last row per event; a single-sample event is one row; the actual value column keeps the measured value) | DataFrame with `systime`, `uuid`, `deviation_abs`, `deviation_pct`, `severity` (`minor`, `major`, `critical`) |
| `capability_indices(target_value=None)` | Calculate Cp, Cpk, Pp, Ppk capability indices | Dictionary with `Cp`, `Cpk`, `Pp`, `Ppk`, `mean`, `std`, `usl`, `lsl`, `target` (plus `Cpm` when `target_value` is given) |

---

## Tips & Hints

!!! tip "Use separate upper and lower limits for two-sided specs"
    With a single `tolerance_uuid` the limit is treated as a one-sided upper spec: Cp/Pp are undefined (NaN) and Cpk/Ppk are the one-sided upper index. Pass `upper_tolerance_uuid` and `lower_tolerance_uuid` for full two-sided capability. Cp/Cpk measure distance to the spec limits. σ is the overall sample standard deviation (the data has no subgroups), so Cp/Cpk equal Pp/Ppk. When `target_value` is given, the Taguchi index `Cpm = Cp / sqrt(1 + ((mean - target) / σ)²)` is added; otherwise `target` is only reported back (the spec midpoint).

!!! info "Related modules"
    - [SPC Rules](spc.md) — Western Electric rule-based monitoring with control limits
    - [Capability Trending](capability-trending.md) — track Cp/Cpk over rolling time windows
    - [Sensor Drift](sensor-drift.md) — detect calibration drift that can cause systematic deviations

---

## See Also

- [Quality Control & SPC Guide](../../guides/quality.md) — narrative overview
- [API Reference](../../reference/ts_shape/events/quality/tolerance_deviation.md) — full parameter docs
