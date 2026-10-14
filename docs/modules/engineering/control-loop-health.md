# ControlLoopHealthEvents

> Continuously assess PID/control loop health from setpoint + actual pairs.

**Module:** `ts_shape.events.engineering.control_loop_health`  
**Guide:** [Process Engineering](../../guides/engineering.md)

---

## When to Use

Use for ongoing PID loop health monitoring. Detects oscillation, valve saturation, and degrading control performance. Complements setpoint events by monitoring between setpoint changes — while setpoint events evaluate transitions, this module evaluates steady-state control quality.

---

## Quick Example

```python
from ts_shape.events.engineering.control_loop_health import ControlLoopHealthEvents

monitor = ControlLoopHealthEvents(
    dataframe=loop_data,
    setpoint_uuid="flow_setpoint_01",
    actual_uuid="flow_actual_01",
    output_uuid="flow_valve_output_01",  # optional; needed for output_saturation()
)

# Compute error integrals (IAE, ISE, ITAE, bias) per hour
integrals = monitor.error_integrals(window="1h")

# Detect oscillation in the control error
oscillations = monitor.detect_oscillation(
    window="5min",
    min_crossings=4
)

# Check for valve saturation (empty result without output_uuid)
saturation = monitor.output_saturation(
    high_limit=100.0,
    low_limit=0.0,
    window="1h"
)

# Generate an 8-hour shift-level report card
report = monitor.loop_health_summary(window="8h")
```

---

## Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `error_integrals(window='1h', max_gap=None)` | IAE/ISE/ITAE/bias per window | DataFrame: `start`, `iae`, `ise`, `itae`, `bias`, `sample_count` |
| `detect_oscillation(window='5min', min_crossings=4)` | Error oscillation detection | DataFrame of oscillation intervals (`start`, `end`, `crossing_count`, `estimated_period_seconds`, ...) |
| `output_saturation(high_limit=100.0, low_limit=0.0, window='1h')` | Valve/output saturation | DataFrame: `start`, `pct_time_at_high`, `pct_time_at_low`, `pct_time_saturated`, `longest_saturation_seconds` (requires `output_uuid`) |
| `loop_health_summary(window='8h')` | Shift-level report card | DataFrame: `start`, `iae`, `bias`, `oscillation_count`, `pct_saturated`, `health_grade` |

---

## Tips & Hints

!!! tip "Use IAE trends for early tuning degradation"
    A rising IAE trend across shifts — even when all values are within spec — signals that the loop is slowly losing control authority. Catch this early with `error_integrals()` before operators notice sluggish response or oscillation.

!!! info "Related modules"
    - [Setpoint Events](setpoint-events.md) — evaluate control performance during setpoint transitions
    - [Signal Comparison](signal-comparison.md) — detect divergence between setpoint and actual
    - [Disturbance Recovery](disturbance-recovery.md) — measure recovery after external upsets

---

## See Also

- [Process Engineering Guide](../../guides/engineering.md)
- [API Reference](../../reference/ts_shape/events/engineering/control_loop_health.md)
