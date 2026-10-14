# ProcessWindowEvents

> Analyze time-windowed process statistics for shift reports, SPC context, and trend monitoring.

**Module:** `ts_shape.events.engineering.process_window`  
**Guide:** [Process Engineering](../../guides/engineering.md)

---

## When to Use

Use as a building block for shift-level process monitoring. Answers "how is my process doing this hour/shift?" with windowed statistics and shift detection. Provides the statistical foundation for SPC charts, shift handover reports, and trend dashboards.

---

## Quick Example

```python
from ts_shape.events.engineering.process_window import ProcessWindowEvents

analyzer = ProcessWindowEvents(
    dataframe=process_data,
    signal_uuid="mixer_torque_01",
)

# Descriptive statistics per 1-hour window
stats = analyzer.windowed_statistics(window="1h")

# Detect windows where the mean has shifted vs. the previous window
shifts = analyzer.detect_mean_shift(window="1h", sensitivity=2.0)

# Detect windows where variance has changed
var_changes = analyzer.detect_variance_change(
    window="1h",
    ratio_threshold=2.0
)

# Compare each window mean to the overall baseline
comparison = analyzer.window_comparison(window="1h")
```

---

## Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `windowed_statistics(window='1h')` | Per-window descriptive stats | DataFrame: `start`, `count`, `mean`, `std`, `min`, `max`, `median`, `p25`, `p75`, `range` |
| `detect_mean_shift(window='1h', sensitivity=2.0)` | Mean shift detection | DataFrame of shift events (`start`, `end`, `prev_mean`, `current_mean`, `shift_sigma`) |
| `detect_variance_change(window='1h', ratio_threshold=2.0)` | Variance change detection | DataFrame of variance events (`start`, `end`, `prev_std`, `current_std`, `variance_ratio`) |
| `window_comparison(window='1h')` | Compare windows to baseline | DataFrame: `start`, `mean`, `z_score_vs_global`, `is_anomalous` |

---

## Tips & Hints

!!! tip "Align windows to shift boundaries"
    Use `window='8h'` for 8-hour shifts or `window='12h'` for 12-hour shifts. This ensures each window maps to exactly one shift, making handover reports straightforward. Combine with `detect_mean_shift()` to flag shifts that deviate from normal.

!!! info "Related modules"
    - [Steady State Detection](steady-state.md) — filter to steady-state before computing window statistics
    - [Process Stability Index](process-stability.md) — roll window statistics into a single stability score
    - [Operating Range](operating-range.md) — envelope analysis that complements windowed stats

---

## See Also

- [Process Engineering Guide](../../guides/engineering.md)
- [API Reference](../../reference/ts_shape/events/engineering/process_window.md)
