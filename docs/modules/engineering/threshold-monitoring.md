# ThresholdMonitoringEvents

> Multi-level threshold monitoring with hysteresis for numeric signals.

**Module:** `ts_shape.events.engineering.threshold_monitoring`  
**Guide:** [Process Engineering](../../guides/engineering.md)

---

## When to Use

Use for continuous monitoring of process limits. Multi-level thresholds (warning/alarm/critical) with hysteresis prevent chattering around limit boundaries. Ideal for real-time alerting pipelines and shift-level compliance reporting.

---

## Quick Example

```python
from ts_shape.events.engineering.threshold_monitoring import ThresholdMonitoringEvents

monitor = ThresholdMonitoringEvents(
    dataframe=process_data,
    signal_uuid="reactor_pressure_01",
)

# Define warning, alarm, and critical levels (name -> threshold)
levels = {"warning": 85.0, "alarm": 92.0, "critical": 98.0}
exceedances = monitor.multi_level_threshold(levels, direction="above")

# Hysteresis-based alarm to avoid chattering
alarms = monitor.threshold_with_hysteresis(high=90.0, low=85.0)

# Track cumulative time above a limit per hour
time_above = monitor.time_above_threshold(threshold=85.0, window="1h")
```

---

## Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `multi_level_threshold(levels, direction='above')` | Multi-level threshold detection (`levels` is a dict of name to value) | DataFrame: `start`, `end`, `duration`, `level`, `peak_value` |
| `threshold_with_hysteresis(high, low)` | Alarm with hysteresis band | DataFrame: `start`, `end`, `duration`, `peak_value` |
| `time_above_threshold(threshold, window='1h')` | Cumulative time above per window | DataFrame: `start`, `time_above`, `pct_above`, `exceedance_count` |
| `threshold_exceedance_trend(threshold, window='1D')` | Exceedance trend over time | DataFrame: `start`, `exceedance_count`, `total_duration`, `trend_direction` |

---

## Tips & Hints

!!! tip "Always use hysteresis for alarms"
    Raw threshold crossings produce noisy on/off chatter when a signal hovers near the limit. Use `threshold_with_hysteresis()` with a dead-band gap of at least 2-5% of the operating range to get clean alarm intervals.

!!! info "Related modules"
    - [Rate of Change](rate-of-change.md) — catch rapid approaches to a limit before the threshold is breached
    - [Operating Range](operating-range.md) — understand the full operating envelope, not just the limits
    - [Process Stability Index](process-stability.md) — roll threshold violations into an overall stability score

---

## See Also

- [Process Engineering Guide](../../guides/engineering.md)
- [API Reference](../../reference/ts_shape/events/engineering/threshold_monitoring.md)
