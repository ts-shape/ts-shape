# MachineStateEvents

> Detect run/idle transitions and intervals from a boolean state signal.

**Module:** `ts_shape.events.production.machine_state`
**Guide:** [Production Monitoring](../../guides/production.md)

---

## When to Use

Use as the foundational production module — machine state drives downtime, OEE, and throughput analysis. Requires a boolean run/idle signal from the PLC. Every production analytics workflow typically starts with machine state detection before layering on throughput, quality, or downtime analysis.

---

## Quick Example

```python
from ts_shape.events.production.machine_state import MachineStateEvents
import pandas as pd

# Long format: a boolean run signal (True = running)
df = pd.DataFrame({
    "systime": pd.date_range("2024-01-01 06:00", periods=120, freq="30s"),
    "uuid": "line1:running",
    "value_bool": [True] * 20 + [False] * 5 + [True] * 40 + [False] * 10 + [True] * 45,
})

events = MachineStateEvents(df, run_state_uuid="line1:running")
intervals = events.detect_run_idle(min_duration="1min")
transitions = events.transition_events()
quality = events.state_quality_metrics()
print(intervals.head())
```

---

## Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `detect_run_idle(min_duration='0s', interval_end=None, data_end=None)` | Identify contiguous run and idle intervals, optionally filtering by minimum duration | DataFrame with `start`, `end`, `uuid`, `source_uuid`, `is_delta`, `state` (`run`/`idle`), `duration_seconds` |
| `transition_events()` | Generate point events at each run-to-idle or idle-to-run transition | DataFrame with `systime`, `uuid`, `source_uuid`, `is_delta`, `transition` (`idle_to_run`/`run_to_idle`), `time_since_last_transition_seconds` |
| `detect_rapid_transitions(threshold='5s', min_count=3)` | Flag suspicious rapid state changes that may indicate sensor noise or PLC faults | DataFrame with `start`, `end`, `transition_count`, `duration_seconds` |
| `state_quality_metrics(interval_end=None)` | Compute state-data quality metrics: transitions, average run/idle durations, gaps, rapid transitions (`interval_end=None` resolves as in `detect_run_idle`; `run_idle_ratio` is NaN when there is no idle time) | One-row `DataFrame` with `start`, `end`, `duration_seconds`, `uuid`, `source_uuid`, `total_transitions`, `avg_run_duration`, `avg_idle_duration`, `run_idle_ratio`, `data_gaps_detected`, `rapid_transitions_detected`. `as_frame=False` returns the deprecated dict |

---

## Tips & Hints

!!! tip "Filter short idle gaps"
    Use `min_duration` in `detect_run_idle()` to ignore brief sensor glitches — a 2-5 second threshold works well for most PLCs.

!!! tip "Change-encoded signals"
    Intervals end at the next transition by default (`interval_end=None` resolves to `"next_change"` without specs), so a PLC that only logs a row on each state change is measured correctly. `interval_end="last_sample"` ends each interval at its own last sample instead.

!!! info "Related modules"
    - [Downtime Tracking](downtime.md) — categorizes idle intervals by reason code
    - [Micro-Stop Detection](micro-stops.md) — finds brief idles that accumulate into losses
    - [OEE Calculator](oee-calculator.md) — uses machine state for the availability component

---

## See Also

- [Production Monitoring Guide](../../guides/production.md)
- [API Reference](../../reference/ts_shape/events/production/machine_state.md)
