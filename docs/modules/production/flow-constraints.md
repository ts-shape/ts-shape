# FlowConstraintEvents

Blocked and starved intervals between an upstream and a downstream station, from the two stations' boolean run signals.

`ts_shape.events.production.flow_constraints` · [Production guide](../../guides/production.md) · [API reference](../../reference/ts_shape/events/production/flow_constraints.md)

## Example

```python
import pandas as pd
from ts_shape.events.production.flow_constraints import FlowConstraintEvents

t = pd.date_range("2024-01-01 06:00", periods=240, freq="30s")
df = pd.concat([
    pd.DataFrame({"systime": t, "uuid": "line1:st1:run", "value_bool": [True] * 110 + [False] * 30 + [True] * 100}),
    pd.DataFrame({"systime": t, "uuid": "line1:st2:run", "value_bool": [True] * 50 + [False] * 60 + [True] * 130}),
], ignore_index=True)

roles = {"upstream_run": "line1:st1:run", "downstream_run": "line1:st2:run"}
fc = FlowConstraintEvents(df)
blocked = fc.blocked_events(roles=roles, tolerance="1s", min_duration="1min")
result = fc.flow_constraint_analytics(roles=roles, tolerance="1s")
print(result[["constraint_type", "event_count", "total_duration_seconds", "severe_count"]])
```

## Methods

All arguments are keyword-only; `roles` maps `"upstream_run"` and `"downstream_run"` to the two `value_bool` signals.

| Method | Returns |
|---|---|
| `blocked_events(roles, tolerance='200ms', min_duration='0s')` | upstream running, downstream stopped: `start`, `end`, `type`, `duration`, `severity`, `time_alignment_quality` |
| `starved_events(roles, tolerance='200ms', min_duration='0s')` | downstream running, upstream stopped; same columns |
| `flow_constraint_analytics(roles, warning_threshold='5s', critical_threshold='30s')` | one row per `constraint_type` (`blocked`, `starved`): `event_count`, `total_duration_seconds`, `avg_duration_seconds`, `minor_count`, `moderate_count`, `severe_count` |

## Notes

- Only the two run signals are used, so `starved_events` reports the onset of starvation: downstream still running while upstream is stopped. Once downstream also stops, both are stopped and nothing is reported; without a buffer or part-presence signal that state cannot be told apart from any other joint stop.
- With `interval_end="next_change"` (default) an interval ends at the first sample where the condition no longer holds: blocked from 10 s with the release at 30 s lasts 20 s, and a single-sample blockage lasts until the next sample. `interval_end="last_sample"` ends it at the last sample where the condition holds (a single-sample event then lasts 0 s).
- `tolerance` (or `tolerance_before` / `tolerance_after`) aligns the two signals in time; set it to cover PLC scan-time differences between stations (200 ms to 1 s is typical) to avoid false events.
- In `flow_constraint_analytics` an event shorter than `warning_threshold` is minor, shorter than `critical_threshold` moderate, otherwise severe.
- `flow_constraint_analytics(..., as_frame=False)` returns the deprecated dict (`blocked_events`, `starved_events`, `summary`).
