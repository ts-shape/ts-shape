# FlowConstraintEvents

> Detect blocked and starved conditions between upstream/downstream stations.

**Module:** `ts_shape.events.production.flow_constraints`
**Guide:** [Production Monitoring](../../guides/production.md)

---

## When to Use

Use for multi-station line analysis. Identifies where material flow is constrained — blocked (downstream can't keep up) or starved (upstream can't supply). Understanding flow constraints is critical for line balancing and identifying the true bottleneck versus symptomatic idle time.

---

## Quick Example

```python
from ts_shape.events.production.flow_constraints import FlowConstraintEvents
import pandas as pd

# Long format: one boolean run signal per station
times = pd.date_range("2024-01-01 06:00", periods=240, freq="30s")
df = pd.concat([
    pd.DataFrame({"systime": times, "uuid": "line1:st1:run",
                  "value_bool": [True] * 110 + [False] * 30 + [True] * 100}),
    pd.DataFrame({"systime": times, "uuid": "line1:st2:run",
                  "value_bool": [True] * 50 + [False] * 60 + [True] * 130}),
], ignore_index=True)

roles = {"upstream_run": "line1:st1:run", "downstream_run": "line1:st2:run"}
fc = FlowConstraintEvents(df)
blocked = fc.blocked_events(roles=roles, tolerance="1s", min_duration="1min")
starved = fc.starved_events(roles=roles, tolerance="1s", min_duration="1min")
analytics = fc.flow_constraint_analytics(roles=roles, tolerance="1s")
print(blocked.head())
print(analytics[["constraint_type", "event_count", "total_duration_seconds"]])
```

---

## Key Methods

`roles` maps `"upstream_run"` and `"downstream_run"` to the UUIDs of the two stations' boolean run signals (`value_bool`). All methods take keyword-only arguments.

| Method | Purpose | Returns |
|--------|---------|---------|
| `blocked_events(roles=..., tolerance='200ms', min_duration='0s', interval_end='next_change')` | Detect intervals where upstream is running but downstream is stopped (blocked condition) | DataFrame with `start`, `end`, `uuid`, `source_uuid`, `is_delta`, `type`, `time_alignment_quality`, `duration`, `severity` |
| `starved_events(roles=..., tolerance='200ms', min_duration='0s', interval_end='next_change')` | Detect intervals where downstream is running while upstream is stopped (starved condition) | Same columns as `blocked_events` |
| `flow_constraint_analytics(roles=..., warning_threshold='5s', critical_threshold='30s')` | Combined analytics for blocked and starved events, with severity classification | `DataFrame` with one row per `constraint_type` (`blocked`, `starved`): `start`, `end`, `duration_seconds`, `uuid`, `source_uuid`, `event_count`, `total_duration_seconds`, `avg_duration_seconds`, `minor_count`, `moderate_count`, `severe_count`, `time_alignment_quality`. `as_frame=False` returns the deprecated dict (`blocked_events`, `starved_events`, `summary`) |

---

## Tips & Hints

!!! note "What \"starved\" means here"
    Both detectors only see the two run signals. `starved_events` reports the *onset* of starvation: downstream still running while upstream is stopped, so downstream drains its buffer. Once downstream runs dry and stops too, both stations are stopped and the interval is no longer reported — a station idle for lack of input cannot be told apart from any other joint stop without a buffer or part-presence signal.

!!! note "Where an interval ends"
    By default (`interval_end="next_change"`) a blocked or starved interval ends at the first sample where the condition no longer holds, like the other state detectors: blocked from 10 s with the release at 30 s lasts 20 s, and a single-sample blockage lasts until the next sample. `interval_end="last_sample"` ends it at the last sample where the condition holds (the earlier behaviour; a single-sample event then lasts 0 s).

!!! tip "Use tolerance to handle signal jitter"
    Set `tolerance` to account for PLC scan-time differences between stations — 200ms to 1s is typical. Without tolerance, you may get false blocked/starved detections from timing mismatches.

!!! info "Related modules"
    - [Bottleneck Detection](bottleneck-detection.md) — identifies which station constrains the line
    - [Machine State](machine-state.md) — provides the per-station run/idle signals used as input
    - [Line Throughput](line-throughput.md) — measures the throughput impact of flow constraints

---

## See Also

- [Production Monitoring Guide](../../guides/production.md)
- [API Reference](../../reference/ts_shape/events/production/flow_constraints.md)
