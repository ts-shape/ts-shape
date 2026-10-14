# MaterialBalanceEvents

> Check whether inputs and outputs balance (mass, energy, flow) per time window.

**Module:** `ts_shape.events.engineering.material_balance`  
**Guide:** [Process Engineering](../../guides/engineering.md)

---

## When to Use

Use as a fundamental daily engineering check for any process. Validates conservation of mass/energy across inputs and outputs. Imbalances indicate leaks, measurement errors, or unaccounted streams. Essential for process accounting, yield tracking, and regulatory compliance.

---

## Quick Example

```python
from ts_shape.events.engineering.material_balance import MaterialBalanceEvents

checker = MaterialBalanceEvents(
    dataframe=flow_data,
    input_uuids=["feed_flow_01", "feed_flow_02"],
    output_uuids=["product_flow_01", "waste_flow_01"],
)

# Check balance within 5% tolerance per hour
balance = checker.balance_check(window="1h", tolerance_pct=5.0)

# Track imbalance trend over time
trend = checker.imbalance_trend(window="1h")

# Detect sustained imbalance periods (at least 2 hours)
exceedances = checker.detect_balance_exceedance(
    window="1h",
    tolerance_pct=5.0,
    min_duration="2h",
)

# Break down each signal's contribution to the imbalance
breakdown = checker.contribution_breakdown(window="1h")
```

---

## Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `balance_check(window='1h', tolerance_pct=5.0)` | Balance pass/fail per window | DataFrame: `start`, `total_input`, `total_output`, `imbalance`, `imbalance_pct`, `balanced` |
| `imbalance_trend(window='1h')` | Imbalance trend over time | DataFrame: `start`, `imbalance_pct`, `rolling_avg_imbalance`, `trend_direction` |
| `detect_balance_exceedance(window='1h', tolerance_pct=5.0, min_duration='2h')` | Sustained imbalance intervals | DataFrame of exceedance intervals (`start`, `end`, `avg_imbalance_pct`, `duration_seconds`, ...) |
| `contribution_breakdown(window='1h')` | Signal-level contributions | DataFrame: `start`, `uuid`, `role`, `value`, `pct_of_total` |

---

## Tips & Hints

!!! tip "Start with contribution_breakdown when imbalance is detected"
    When `balance_check()` fails, use `contribution_breakdown()` to identify which specific input or output stream is responsible. This narrows the investigation to one sensor or pipe instead of chasing the entire process.

!!! note "Flows are averaged by time; missing data stays unknown"
    Each flow is held until its next sample and averaged by time per window (pass `time_weighted=False` to average samples). A window in which a signal has no data has NaN totals and is not balanced, but `detect_balance_exceedance()` does not report it, so a sensor outage is not read as an imbalance. Pass `missing_as_zero=True` to count missing data as zero flow.

!!! info "Related modules"
    - [Signal Comparison](signal-comparison.md) — compare individual input/output signal pairs
    - [Threshold Monitoring](threshold-monitoring.md) — set alarms on the imbalance metric itself
    - [Process Windows](process-windows.md) — windowed statistics for the individual streams

---

## See Also

- [Process Engineering Guide](../../guides/engineering.md)
- [API Reference](../../reference/ts_shape/events/engineering/material_balance.md)
