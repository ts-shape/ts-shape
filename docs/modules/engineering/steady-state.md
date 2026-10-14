# SteadyStateDetectionEvents

> Identify when a process signal has settled into a steady operating state versus transient/dynamic periods.

**Module:** `ts_shape.events.engineering.steady_state_detection`  
**Guide:** [Process Engineering](../../guides/engineering.md)

---

## When to Use

Use to segment process data into steady-state vs transient periods. Essential for valid statistical analysis — SPC and capability indices are only meaningful on steady-state data. Also useful for filtering out startup, shutdown, and transition periods before computing KPIs.

---

## Quick Example

```python
from ts_shape.events.engineering.steady_state_detection import SteadyStateDetectionEvents

detector = SteadyStateDetectionEvents(
    df=process_data,
    uuid="reactor_temperature_01"
)

# Detect steady-state intervals
steady = detector.detect_steady_state(
    window="5min",
    std_threshold=1.0,
    min_duration="10min"
)

# Get transient periods (inverse of steady state)
transients = detector.detect_transient_periods(
    window="5min",
    std_threshold=1.0
)

# Identify distinct operating bands within steady state
bands = detector.steady_state_value_bands(window="5min")
```

---

## Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `detect_steady_state(window='5min', std_threshold=1.0, min_duration='10min')` | Steady-state intervals | DataFrame of steady intervals |
| `detect_transient_periods(window='5min', std_threshold=1.0)` | Transient/dynamic intervals | DataFrame of transient intervals |
| `steady_state_statistics(window='5min')` | Summary statistics | DataFrame with steady-state stats |
| `steady_state_value_bands(window='5min')` | Distinct operating bands | DataFrame of value bands |

---

## Tips & Hints

!!! tip "Tune std_threshold to your process noise"
    The default `std_threshold=1.0` works for many signals, but noisy sensors may need a higher value. Start by looking at `steady_state_statistics()` output and adjust until the detected intervals match your engineering judgment.

!!! info "Related modules"
    - [Process Windows](process-windows.md) — windowed statistics that pair well with steady-state filtering
    - [Setpoint Events](setpoint-events.md) — detect the setpoint changes that cause transient periods
    - [Operating Range](operating-range.md) — analyze the envelope within steady-state periods

---

## See Also

- [Process Engineering Guide](../../guides/engineering.md)
- [API Reference](../../reference/ts_shape/events/engineering/steady_state_detection.md)
