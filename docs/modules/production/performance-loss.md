# PerformanceLossEvents

> Track performance and speed losses against target cycle times.

**Module:** `ts_shape.events.production.performance_loss`
**Guide:** [Reporting Guide](../../guides/reporting.md)

---

## When to Use

Use to identify hidden speed losses where the machine runs but slower than target. The "performance" component of OEE, showing where actual cycle time exceeds ideal. This module helps production engineers find periods of reduced speed that may not trigger alarms but still erode throughput over time.

---

## Quick Example

```python
from ts_shape.events.production.performance_loss import PerformanceLossEvents

tracker = PerformanceLossEvents(production_df)

# Performance percentage per shift (cycle_uuid is a part counter or cycle trigger)
perf = tracker.performance_by_shift(
    cycle_uuid="line1:parts",
    target_cycle_time=4.5  # seconds
)

# Find periods where performance dropped below 90%
slow = tracker.slow_periods(
    cycle_uuid="line1:parts",
    target_cycle_time=4.5,
    threshold_pct=90.0
)

# Daily trend of performance percentage
trend = tracker.performance_trend(
    cycle_uuid="line1:parts",
    target_cycle_time=4.5,
    window="1D"
)
```

---

## Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `performance_by_shift(cycle_uuid, target_cycle_time, max_cycle_gap=None)` | Performance percentage per shift (ideal / actual) | `DataFrame` with `date`, `shift`, `actual_parts`, `avg_cycle_time_s`, `target_cycle_time_s`, `performance_pct`, `loss_minutes` |
| `slow_periods(cycle_uuid, target_cycle_time, threshold_pct=90.0, window='1h', max_cycle_gap=None)` | Time windows where performance fell below threshold | `DataFrame` with `start`, `end`, `actual_parts`, `avg_cycle_time_s`, `performance_pct`, `loss_minutes` |
| `performance_trend(cycle_uuid, target_cycle_time, window='1D', max_cycle_gap=None)` | Performance percentage per time period | `DataFrame` with `period`, `actual_parts`, `avg_cycle_time_s`, `performance_pct`, `loss_minutes` |

---

## Tips & Hints

!!! tip "Exclude planned slow-downs"
    If your process has intentional reduced-speed modes (e.g., warm-up, cool-down), filter those periods out before analysis. Otherwise they will appear as false performance losses.

!!! info "Related modules"
    - [Cycle Time Tracking](cycle-time.md) — raw cycle time analysis
    - [OEE Calculator](oee-calculator.md) — performance feeds into the OEE calculation
    - [Micro-Stop Detection](micro-stops.md) — very short stops that also reduce effective performance

---

## See Also

- [Reporting Guide](../../guides/reporting.md)
- [API Reference](../../reference/ts_shape/events/production/performance_loss.md)
