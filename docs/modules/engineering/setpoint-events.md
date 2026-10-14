# SetpointChangeEvents

> Detect step/ramp changes on a setpoint signal and compute follow-up KPIs like time-to-settle and overshoot.

**Module:** `ts_shape.events.engineering.setpoint_events`  
**Guide:** [Process Engineering](../../guides/engineering.md)

---

## When to Use

Use when analyzing PID or setpoint-based control systems. Detects recipe changes and computes ISA/control-engineering KPIs like settling time, rise time, and overshoot. Ideal for evaluating controller tuning and recipe transition performance.

---

## Quick Example

```python
from ts_shape.events.engineering.setpoint_events import SetpointChangeEvents

detector = SetpointChangeEvents(
    df=process_data,
    uuid="setpoint_temperature_01"
)

# Detect all setpoint step changes
steps = detector.detect_setpoint_steps(min_delta=5.0, min_hold="30s")

# Measure how quickly the actual signal settles after each change
settling = detector.time_to_settle(
    actual_uuid="actual_temperature_01",
    tol=0.5,
    settle_pct=0.02
)

# Get comprehensive control quality metrics
quality = detector.control_quality_metrics(
    actual_uuid="actual_temperature_01",
    tol=0.5
)
```

---

## Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `detect_setpoint_steps(min_delta, min_hold='0s')` | Point events at setpoint changes | DataFrame of step events |
| `detect_setpoint_ramps(min_rate, min_duration='0s')` | Interval events for ramps | DataFrame of ramp intervals |
| `detect_setpoint_changes(min_delta=0.0, min_rate=None)` | Unified step+ramp table | DataFrame of all changes |
| `time_to_settle(actual_uuid, tol=0.0, settle_pct=None)` | Settling time after each change | DataFrame with settling times |
| `overshoot_metrics(actual_uuid, window='10min')` | Overshoot/undershoot per change | DataFrame with overshoot stats |
| `rise_time(actual_uuid, start_pct=0.1, end_pct=0.9)` | Rise time (10-90% by default) | DataFrame with rise times |
| `decay_rate(actual_uuid, lookahead='10min')` | Exponential decay rate | DataFrame with decay constants |
| `oscillation_frequency(actual_uuid, window='10min')` | Oscillation frequency | DataFrame with frequency estimates |
| `control_quality_metrics(actual_uuid, tol=0.0)` | Comprehensive control quality | DataFrame with combined KPIs |

---

## Tips & Hints

!!! tip "Combine steps and ramps for full coverage"
    Use `detect_setpoint_changes()` instead of calling steps and ramps separately. This gives you a unified table that captures both instantaneous recipe changes and gradual ramp profiles.

!!! info "Related modules"
    - [Control Loop Health](control-loop-health.md) — ongoing loop monitoring between setpoint changes
    - [Signal Comparison](signal-comparison.md) — setpoint vs actual divergence detection
    - [Steady State Detection](steady-state.md) — identify when the process has settled

---

## See Also

- [Process Engineering Guide](../../guides/engineering.md)
- [API Reference](../../reference/ts_shape/events/engineering/setpoint_events.md)
