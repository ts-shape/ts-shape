# OperatingRangeEvents

> Analyze the operating envelope of a signal — what ranges it operates in and when the operating point shifts.

**Module:** `ts_shape.events.engineering.operating_range`  
**Guide:** [Process Engineering](../../guides/engineering.md)

---

## When to Use

Use for operating envelope analysis. Tracks what ranges your process actually operates in and detects regime changes that may indicate product changeovers or process shifts. Useful for building golden-run profiles and detecting off-spec operation.

---

## Quick Example

```python
from ts_shape.events.engineering.operating_range import OperatingRangeEvents

analyzer = OperatingRangeEvents(
    dataframe=process_data,
    signal_uuid="extruder_speed_01",
)

# Get the operating envelope per hour
envelope = analyzer.operating_envelope(window="1h")

# Detect when the operating point shifts significantly
shifts = analyzer.detect_regime_change(
    window="1h",
    shift_threshold=2.0
)

# Percentage of time in a target operating range
in_range = analyzer.time_in_range(
    lower=180.0,
    upper=220.0,
    window="1h"
)

# Value distribution histogram
distribution = analyzer.value_distribution(n_bins=10)
```

---

## Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `operating_envelope(window='1h')` | Per-window min/p5/mean/p95/max envelope | DataFrame: `start`, `min_value`, `p5`, `mean_value`, `p95`, `max_value`, `range_width` |
| `detect_regime_change(window='1h', shift_threshold=2.0)` | Operating point shifts | DataFrame of regime changes (`start`, `end`, `prev_mean`, `new_mean`, `shift_magnitude`) |
| `time_in_range(lower, upper, window='1h')` | Percentage of time in target range per window | DataFrame: `start`, `time_in_range_pct`, `time_below_pct`, `time_above_pct` |
| `value_distribution(n_bins=10)` | Value histogram | DataFrame: `bin_lower`, `bin_upper`, `count`, `pct`, `cumulative_pct` |

---

## Tips & Hints

!!! tip "Use time_in_range for OEE-style metrics"
    `time_in_range()` gives you the fraction of each window spent within spec. Aggregate this over a shift or day to get a process-level "in-spec percentage" that complements OEE availability and quality metrics.

!!! info "Related modules"
    - [Steady State Detection](steady-state.md) — filter to steady-state before analyzing the operating envelope
    - [Threshold Monitoring](threshold-monitoring.md) — alarm when the envelope breaches hard limits
    - [Process Windows](process-windows.md) — windowed statistics for shift-level reporting

---

## See Also

- [Process Engineering Guide](../../guides/engineering.md)
- [API Reference](../../reference/ts_shape/events/engineering/operating_range.md)
