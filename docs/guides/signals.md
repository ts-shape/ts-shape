# Signal semantics and profiling

Most wrong numbers in industrial time-series analysis come from misreading
what a signal is:

| Mistake | Effect |
|---|---|
| Counter treated as a level (`last - first`) | production lost at every counter reset |
| Interval closed at its last sample | zero-length intervals on change-encoded data; one period short on sampled data |
| `resample().mean()` on irregular data | averages weighted by sample count, not time |
| Value held across a logging outage | the outage is booked as run time |
| `merge_asof(direction="nearest")` | a sample picks up a setpoint that was not set yet |
| Fixed 24 h days, naive timestamps | wrong DST days, results shifted by the UTC offset |

`ts_shape.signals` addresses this: you describe each signal once as a
`SignalSpec`, and the primitives and detectors aggregate it accordingly.

The examples on this page use this frame: a part counter that resets every 200
parts, a boolean run state and a temperature.

```python
import numpy as np
import pandas as pd
import ts_shape

t = pd.date_range("2024-06-03 06:00", periods=480, freq="1min", tz="UTC")
df = pd.concat([
    pd.DataFrame({"systime": t, "uuid": "line1:parts", "value_integer": np.arange(480) % 200}),
    pd.DataFrame({"systime": t, "uuid": "line1:state", "value_bool": (np.arange(480) // 37) % 2 == 0}),
    pd.DataFrame({"systime": t, "uuid": "oven:temp", "value_double": 180 + np.random.default_rng(0).normal(0, 1, 480)}),
], ignore_index=True)
```

## Profile first

```python
report = ts_shape.profile(df)
print(report.summary())   # here: one info issue, two counter resets on line1:parts
report.issues()           # error / warning / info, most severe first
report.to_frame()         # one row per signal: kind, encoding, period, gaps, resets, ...
```

The profile flags unparseable or duplicate timestamps, rows out of time order,
missing values, a uuid spread over several `value_*` columns, logging gaps,
counter resets, change-encoded signals and signals that look stuck.

## The SignalSpec fields

| Field | Meaning |
|---|---|
| `kind` | `continuous` (measured level), `counter` (cumulative, may reset), `state` (held discrete value), `event` (sparse occurrences) |
| `encoding` | `sampled` (regular rows, values repeat) or `change_encoded` (one row per change; the value holds until the next row) |
| `value_column` | the `value_*` column that carries the signal |
| `nominal_period` | typical time between samples |
| `max_gap` | longest span a value may be held; longer spans are treated as no data |
| `tz` | timezone of `systime` |
| `reset_policy` | counters: `reset_to_zero` (a drop restarts the count) or `none` |
| `unit` | engineering unit (informational) |

## Infer specs, then override

```python
from ts_shape.signals import infer_specs

specs = infer_specs(df, overrides={
    "line1:parts": {"unit": "pcs"},
    "line1:state": {"max_gap": "15min"},   # your logging-outage threshold
})
specs.to_frame()
specs["line1:state"].notes   # why each field was inferred
```

Inference cannot know everything. A change-encoded signal never gets an
inferred `max_gap`, because a value may legitimately hold for hours; set it to
your logging-outage threshold.

Specs can also be kept as YAML next to the data (needs `ts-shape[yaml]`) and
loaded with `SignalSpecs.from_yaml("signals.yaml")`:

```yaml
line1:state:
  kind: state
  value_column: value_bool
  max_gap: 15min
meter:main:
  kind: counter
  value_column: value_double
  unit: kWh
```

## Primitives

```python
from ts_shape.signals.primitives import (
    intervals, increments, per_window, windows, shift_windows,
)

intervals(df, specs["line1:state"])            # held states, closed at the next change, split at gaps
increments(df, specs["line1:parts"])           # reset-aware production per reading
per_window(df, specs["line1:parts"], "1h")     # parts per hour, nothing lost at resets or boundaries
per_window(df, specs["oven:temp"], "15min")    # time-weighted mean, never across a gap
windows("2024-03-30", "2024-04-02", "1D", tz="Europe/Berlin")   # days of 24 h, 23 h, 24 h
shift_windows("2024-03-30", "2024-03-31", tz="Europe/Berlin")   # the DST night shift is 7 h
```

`align(left, left_spec, right, right_spec)` attaches to each sample of one
signal the value of another signal that was in force at that time, for
example the setpoint at each process-value sample.

## Detectors that accept specs

Where a detector accepts `specs=`, its defaults follow the signal. An explicit
argument always wins over the spec. Detectors are being moved onto specs pack
by pack (see the [roadmap](../roadmap.md)).

```python
ts_shape.MachineStateEvents(df, "line1:state", specs=specs).detect_run_idle()
ts_shape.DowntimeEvents(df, specs=specs).downtime_by_shift("line1:state")
```
