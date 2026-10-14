# StatisticalProcessControlEvents

> Applies SPC rules (Western Electric Rules) to a DataFrame for event detection using control limit UUIDs and actual value UUIDs.

**Module:** `ts_shape.events.quality.statistical_process_control`  
**Guide:** [Quality Control & SPC](../../guides/quality.md)

---

## When to Use

Use for real-time quality monitoring when you have a process signal and corresponding control limits. Ideal for detecting shifts, trends, and patterns that indicate a process is going out of control per the 8 Western Electric rules. Requires long-format data with the monitored values under `actual_uuid` and reference values under `tolerance_uuid`, from which the mean and 1/2/3 sigma control limits are calculated.

---

## Quick Example

```python
from ts_shape.events.quality.statistical_process_control import StatisticalProcessControlEvents

# Control limits are computed from the "limit:temp" reference rows;
# the rules are checked against the "sensor:temp" rows.
spc = StatisticalProcessControlEvents(
    df,
    value_column="value_double",
    tolerance_uuid="limit:temp",
    actual_uuid="sensor:temp",
    event_uuid="spc:temp",
)

# Run the full SPC pipeline (calculates limits + applies all rules)
result = spc.detect_violations(include_severity=True)

# Or apply specific rules only
limits = spc.control_limits()
violations = spc.detect_rule_violations(selected_rules=["rule_1", "rule_2", "rule_5"])
summary = spc.interpret_violations(violations)
```

---

## Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `control_limits()` | Calculate mean and 1/2/3 sigma limits | DataFrame with control limits |
| `dynamic_control_limits(method='moving_range', window_size=20)` | Adaptive control limits for non-stationary processes | DataFrame with dynamic limits |
| `detect_rule_violations(selected_rules=None)` | Apply selected Western Electric rules (e.g. `["rule_1", "rule_3"]`) efficiently | DataFrame of violations with `rule` and `severity` |
| `detect_cusum_shifts(target=None, k=0.5, h=5.0)` | CUSUM shift detection for small sustained shifts | DataFrame with `systime`, `value`, `cusum_high`, `cusum_low`, `shift_detected`, `shift_direction` |
| `interpret_violations(violations_df)` | Add human-readable interpretations to violations | DataFrame with added `interpretation`, `meaning`, `recommendation` |
| `process(selected_rules=None, include_severity=False)` | Full SPC pipeline: limits + rules | Violation events with `systime`, value column, `uuid` (plus `rule`, `severity` when `include_severity=True`) |
| `rule_1` through `rule_8` | Individual Western Electric rule checks | DataFrame with single rule flags |

---

## Tips & Hints

!!! tip "Start with the process() method"
    The `process()` method runs the complete pipeline and is the easiest entry point. Use individual rules or `detect_rule_violations(selected_rules=...)` only when you need to tune which rules apply to your specific process.

!!! info "Related modules"
    - [Tolerance Deviation](tolerance-deviation.md) — specification-limit-based deviation tracking with Cp/Cpk
    - [Capability Trending](capability-trending.md) — track capability indices over time
    - [Outlier Detection](outlier-detection.md) — statistical outlier detection as a preprocessing step

---

## See Also

- [Quality Control & SPC Guide](../../guides/quality.md) — narrative overview
- [API Reference](../../reference/ts_shape/events/quality/statistical_process_control.md) — full parameter docs
