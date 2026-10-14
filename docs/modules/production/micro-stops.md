# MicroStopEvents

> Detect brief idle intervals that individually seem harmless but accumulate into significant availability losses.

**Module:** `ts_shape.events.production.micro_stop_detection`
**Guide:** [OEE Analytics](../../guides/oee-analytics.md)

---

## When to Use

Use to uncover hidden availability losses. Micro-stops (under 30s) are often invisible in standard reports but can add up to significant lost time. Common causes include sensor misfeeds, part jams, and minor alignment issues that operators clear without logging a downtime event.

---

## Quick Example

```python
from ts_shape.events.production.micro_stop_detection import MicroStopEvents
import pandas as pd
import numpy as np

np.random.seed(42)
# Simulate a machine with frequent brief stops (long format, boolean run signal)
running = []
for _ in range(100):
    running.extend([True] * np.random.randint(10, 30))
    running.extend([False] * np.random.randint(1, 4))  # 1-3 second stops
df = pd.DataFrame({
    "systime": pd.date_range("2024-01-01 06:00", periods=len(running), freq="1s"),
    "uuid": "line1:running",
    "value_bool": running,
})

ms = MicroStopEvents(df, run_state_uuid="line1:running")
stops = ms.detect_micro_stops(max_duration="30s")
frequency = ms.micro_stop_frequency(window="15min", max_duration="30s")
impact = ms.micro_stop_impact(window="15min", max_duration="30s")
patterns = ms.micro_stop_patterns(hour_grouping=True)
print(f"Found {len(stops)} micro-stops")
print(impact.head())
```

---

## Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `detect_micro_stops(max_duration='30s', min_duration='0s')` | Find all idle intervals shorter than the specified threshold | DataFrame with `start`, `end`, `duration`, `preceding_run_duration` |
| `micro_stop_frequency(window='1h', max_duration='30s')` | Count micro-stops per time window to identify problem periods | DataFrame with `start`, `count`, `total_lost_time`, `pct_of_window` |
| `micro_stop_impact(window='1h', max_duration='30s')` | Calculate time lost to micro-stops vs run time per window | DataFrame with `start`, `total_run_time`, `total_micro_stop_time`, `availability_loss_pct` |
| `micro_stop_patterns(hour_grouping=True, max_duration='30s')` | Group micro-stops by hour of day (or by 8-hour shift block when `hour_grouping=False`) to find recurring patterns | DataFrame with `hour` (or `shift`), `avg_count`, `avg_lost_time` |

---

## Tips & Hints

!!! tip "Adjust max_duration to your process"
    The 30-second default works for high-speed discrete manufacturing. For slower processes (e.g., assembly), consider increasing to 60-120 seconds. The key is to capture stops below your standard downtime reporting threshold. Note that `max_duration` is passed to each method separately.

!!! tip "Change-encoded signals"
    Intervals close at the next transition by default (`interval_end="next_change"`), so a run signal that only logs a row on each state change is measured correctly. With `interval_end="last_sample"` every interval of such a signal has zero duration and no lost time is reported.

!!! info "Related modules"
    - [Machine State](machine-state.md) — provides the run/idle signal that micro-stop detection analyzes
    - [OEE Calculator](oee-calculator.md) — micro-stops appear as availability losses in OEE
    - [Duty Cycle](duty-cycle.md) — excessive cycling patterns may correlate with micro-stop clusters

---

## See Also

- [OEE Analytics Guide](../../guides/oee-analytics.md)
- [API Reference](../../reference/ts_shape/events/production/micro_stop_detection.md)
