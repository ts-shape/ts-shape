# WarmUpCoolDownEvents

> Detect and characterize warm-up and cool-down curves for ovens, extruders, molds, hydraulic systems.

**Module:** `ts_shape.events.engineering.warmup_analysis`  
**Guide:** [Process Engineering](../../guides/engineering.md)

---

## When to Use

Use for thermal equipment (ovens, extruders, molds) where warm-up and cool-down phases impact production planning and energy consumption. Tracks warm-up consistency over time to detect degrading heating elements or insulation, and measures time-to-target for scheduling.

---

## Quick Example

```python
from ts_shape.events.engineering.warmup_analysis import WarmUpCoolDownEvents

analyzer = WarmUpCoolDownEvents(
    dataframe=oven_data,
    signal_uuid="oven_temperature_zone1",
)

# Detect warm-up intervals (rising at least 50 deg over 1+ min)
warmups = analyzer.detect_warmup(min_rise=50.0, min_duration="1min")

# Detect cool-down intervals
cooldowns = analyzer.detect_cooldown(min_fall=30.0, min_duration="1min")

# Check consistency of warm-up curves across days
consistency = analyzer.warmup_consistency(
    min_rise=50.0,
    min_duration="1min"
)

# Time from each rising ramp start to 180 °C operating temperature
time_to_temp = analyzer.time_to_target(
    target_value=180.0,
    direction="rising"
)
```

---

## Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `detect_warmup(min_rise, min_duration='1min', max_pause=0)` | Warm-up interval detection | DataFrame: `start`, `end`, `start_value`, `end_value`, `rise`, `duration_seconds`, `avg_rate`, ... |
| `detect_cooldown(min_fall, min_duration='1min', max_pause=0)` | Cool-down interval detection | DataFrame: `start`, `end`, `start_value`, `end_value`, `fall`, `duration_seconds`, `avg_rate`, ... |
| `warmup_consistency(min_rise, min_duration='1min')` | Warm-up curve consistency | DataFrame: `warmup_index`, `start`, `duration_seconds`, `avg_rate`, `deviation_from_median_duration` |
| `time_to_target(target_value, direction='rising')` | Time to reach a target value | DataFrame: `start`, `target_reached_at`, `time_to_target_seconds`, `overshoot` |

---

## Tips & Hints

!!! tip "Track warm-up consistency for predictive maintenance"
    A gradually increasing time-to-target across weeks indicates degrading heating elements, fouled heat exchangers, or deteriorating insulation. Plot `warmup_consistency()` output over time to set up early maintenance alerts.

!!! info "Related modules"
    - [Startup Detection](startup-detection.md) — detect the initial startup event that triggers a warm-up
    - [Rate of Change](rate-of-change.md) — monitor the ramp rate during warm-up
    - [Steady State Detection](steady-state.md) — detect when the warm-up phase ends and steady state begins

---

## See Also

- [Process Engineering Guide](../../guides/engineering.md)
- [API Reference](../../reference/ts_shape/events/engineering/warmup_analysis.md)
