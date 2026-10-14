# RateOfChangeEvents

> Detect rapid changes and step jumps in a numeric signal.

**Module:** `ts_shape.events.engineering.rate_of_change`  
**Guide:** [Process Engineering](../../guides/engineering.md)

---

## When to Use

Use for detecting sudden process upsets, valve jumps, or rapid thermal changes. Complements threshold monitoring by catching the rate of change, not just the absolute level. Especially useful as an early-warning mechanism before a threshold is actually breached.

---

## Quick Example

```python
from ts_shape.events.engineering.rate_of_change import RateOfChangeEvents

detector = RateOfChangeEvents(
    dataframe=process_data,
    signal_uuid="flow_rate_01",
)

# Detect intervals where the rate exceeds 0.5 units/second
rapid = detector.detect_rapid_change(threshold=0.5, window="1min")

# Get rate statistics per hour for trend reporting
stats = detector.rate_statistics(window="1h")

# Find sudden step jumps (large delta in short time)
steps = detector.detect_step_changes(min_delta=20.0, max_duration="5s")
```

---

## Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `detect_rapid_change(threshold, window='1min')` | Intervals where \|rate\| (units/second) exceeds `threshold` | DataFrame: `start`, `end`, `max_rate`, `direction` |
| `rate_statistics(window='1h')` | Rate statistics per window | DataFrame: `start`, `mean_rate`, `std_rate`, `max_rate`, `min_rate` |
| `detect_step_changes(min_delta, max_duration='5s')` | Sudden value jumps | DataFrame: `time`, `value_before`, `value_after`, `delta`, `duration` |

---

## Tips & Hints

!!! tip "Pair with threshold monitoring for defense-in-depth"
    A rapid rate of change often precedes a threshold breach. Use `detect_rapid_change()` as an early warning, and `ThresholdMonitoringEvents` as the definitive alarm. This two-layer approach gives operators more lead time.

!!! info "Related modules"
    - [Threshold Monitoring](threshold-monitoring.md) — absolute-level alarms that complement rate detection
    - [Setpoint Events](setpoint-events.md) — distinguish intentional setpoint steps from unexpected jumps
    - [Disturbance Recovery](disturbance-recovery.md) — characterize recovery after a detected upset

---

## See Also

- [Process Engineering Guide](../../guides/engineering.md)
- [API Reference](../../reference/ts_shape/events/engineering/rate_of_change.md)
