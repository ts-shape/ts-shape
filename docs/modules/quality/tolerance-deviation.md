# ToleranceDeviationEvents

> Processes DataFrame data for comparing tolerance and actual values with severity classification and process capability indices.

**Module:** `ts_shape.events.quality.tolerance_deviation`  
**Guide:** [Quality Control & SPC](../../guides/quality.md)

---

## When to Use

Use when you have process signals with defined upper/lower specification limits and need to track deviations with severity classification. Computes Cp/Cpk/Pp/Ppk indices for process capability assessment. Requires long-format data with the actual measurements under `actual_uuid` and the specification limits under `upper_tolerance_uuid` and/or `lower_tolerance_uuid` (one of them alone is a one-sided spec).

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

!!! tip "Two-sided and one-sided specs"
    Pass `upper_tolerance_uuid` and `lower_tolerance_uuid` for a two-sided spec. With only one of them the spec is one-sided: Cp/Pp are undefined (NaN), Cpk/Ppk are the one-sided index (CPU or CPL), and `deviation_pct` is relative to the magnitude of the single limit. Cp/Cpk measure distance to the spec limits. σ is the overall sample standard deviation (the data has no subgroups), so Cp/Cpk equal Pp/Ppk. When `target_value` is given, the Taguchi index `Cpm = Cp / sqrt(1 + ((mean - target) / σ)²)` is added; otherwise `target` is only reported back (the spec midpoint).

!!! warning "`tolerance_uuid` is deprecated"
    The single `tolerance_uuid` (compared through `compare_func`) emits a `DeprecationWarning` and will be removed in the next major release. A single upper limit is `upper_tolerance_uuid` alone.

!!! info "Related modules"
    - [SPC Rules](spc.md) — Western Electric rule-based monitoring with control limits
    - [Capability Trending](capability-trending.md) — track Cp/Cpk over rolling time windows
    - [Sensor Drift](sensor-drift.md) — detect calibration drift that can cause systematic deviations

---

## See Also

- [Quality Control & SPC Guide](../../guides/quality.md) — narrative overview
- [API Reference](../../reference/ts_shape/events/quality/tolerance_deviation.md) — full parameter docs
