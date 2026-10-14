# SignalComparisonEvents

> Compare two related signals (e.g. setpoint vs actual) and detect divergence.

**Module:** `ts_shape.events.engineering.signal_comparison`  
**Guide:** [Process Engineering](../../guides/engineering.md)

---

## When to Use

Use to compare setpoint vs actual, sensor A vs sensor B, or any paired signals. Detects when signals that should track each other start to diverge. Valuable for sensor validation, control loop monitoring, and redundancy checks.

---

## Quick Example

```python
from ts_shape.events.engineering.signal_comparison import SignalComparisonEvents

# process_data holds both the reference and the actual signal
comparator = SignalComparisonEvents(
    dataframe=process_data,
    reference_uuid="pressure_setpoint_01",
)

# Detect when actual diverges from setpoint beyond tolerance
divergence = comparator.detect_divergence(
    actual_uuid="pressure_actual_01",
    tolerance=2.0,
    min_duration="1min"
)

# Per-hour deviation statistics
dev_stats = comparator.deviation_statistics(
    actual_uuid="pressure_actual_01",
    window="1h"
)

# Rolling correlation to detect decoupling
correlation = comparator.correlation_windows(
    actual_uuid="pressure_actual_01",
    window="1h"
)
```

---

## Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `detect_divergence(actual_uuid, tolerance, min_duration='1min')` | Divergence intervals | DataFrame of divergence intervals (`start`, `end`, `max_deviation`, `mean_deviation`, `direction`) |
| `deviation_statistics(actual_uuid, window='1h')` | Per-window deviation stats | DataFrame: `start`, `mae`, `max_error`, `rmse`, `bias` |
| `tracking_error_trend(actual_uuid, window='1D')` | Deviation trend over time | DataFrame: `start`, `mae`, `trend_slope`, `trend_direction` |
| `correlation_windows(actual_uuid, window='1h')` | Per-window correlation | DataFrame: `start`, `correlation`, `sample_count` |

---

## Tips & Hints

!!! tip "Use correlation to catch slow drift"
    `detect_divergence()` catches absolute deviations, but two signals can drift apart slowly while staying within tolerance. Monitor `correlation_windows()` over days — a declining correlation trend signals degrading sensor or actuator performance.

!!! info "Related modules"
    - [Setpoint Events](setpoint-events.md) — detect setpoint changes that naturally cause temporary divergence
    - [Control Loop Health](control-loop-health.md) — deeper loop diagnostics when divergence is detected
    - [Material Balance](material-balance.md) — compare input vs output signals for balance checks

---

## See Also

- [Process Engineering Guide](../../guides/engineering.md)
- [API Reference](../../reference/ts_shape/events/engineering/signal_comparison.md)
