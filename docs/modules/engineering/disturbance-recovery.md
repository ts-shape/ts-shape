# DisturbanceRecoveryEvents

> Detect external upsets hitting a process signal and measure recovery.

**Module:** `ts_shape.events.engineering.disturbance_recovery`  
**Guide:** [Process Engineering](../../guides/engineering.md)

---

## When to Use

Use to detect and characterize external process upsets (material changes, ambient disturbances, upstream failures) and measure how quickly the process recovers. Critical for root cause analysis and for quantifying process robustness against unplanned events.

---

## Quick Example

```python
from ts_shape.events.engineering.disturbance_recovery import DisturbanceRecoveryEvents

analyzer = DisturbanceRecoveryEvents(
    dataframe=process_data,
    signal_uuid="reactor_temperature_01",
)

# Detect disturbances exceeding 3 sigma from baseline
disturbances = analyzer.detect_disturbances(
    baseline_window="10min",
    threshold_sigma=3.0
)

# Measure recovery time for each disturbance
recovery = analyzer.recovery_time(
    baseline_window="10min",
    recovery_pct=0.95
)

# Count disturbances per 8-hour shift
frequency = analyzer.disturbance_frequency(window="8h")

# Compare process behavior before vs after each upset
comparison = analyzer.before_after_comparison(baseline_window="10min")
```

---

## Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `detect_disturbances(baseline_window='10min', threshold_sigma=3.0, min_duration='30s')` | Detect process upsets | DataFrame of disturbance intervals (`start`, `end`, `peak_deviation`, `direction`, `duration_seconds`, ...) |
| `recovery_time(baseline_window='10min', threshold_sigma=3.0, recovery_pct=0.95, max_recovery='1h')` | Recovery time per disturbance | DataFrame: `start`, `end`, `recovery_time_seconds`, `recovered`, `residual_offset`, ... |
| `disturbance_frequency(window='8h')` | Count disturbances per window | DataFrame: `start`, `disturbance_count`, `total_disturbance_seconds`, `pct_time_disturbed`, `avg_recovery_seconds` |
| `before_after_comparison(baseline_window='10min', comparison_window='5min')` | Before vs after statistics | DataFrame: `start`, `pre_mean`, `post_mean`, `mean_shift`, `variance_ratio`, ... |

---

## Tips & Hints

!!! tip "Correlate disturbances with upstream events"
    After detecting disturbances, use `before_after_comparison()` to quantify the impact, then cross-reference timestamps with upstream signals or operator logs to identify root causes. Pair with `disturbance_frequency()` to spot recurring patterns.

!!! info "Related modules"
    - [Rate of Change](rate-of-change.md) — detect the rapid change that characterizes the onset of a disturbance
    - [Control Loop Health](control-loop-health.md) — assess how well the control loop handles the upset
    - [Process Stability Index](process-stability.md) — disturbances directly impact the stability score

---

## See Also

- [Process Engineering Guide](../../guides/engineering.md)
- [API Reference](../../reference/ts_shape/events/engineering/disturbance_recovery.md)
