# Signal semantics & profiling

Industrial time-series analytics fail silently when the code misreads what a
signal *is*. The same few mistakes cause most wrong numbers:

| Mistake | Effect |
|---|---|
| Counter treated as a level (`last - first`) | production lost at every PLC restart / counter reset |
| Interval closed at its last sample | zero-length intervals on change-encoded data; one period short on sampled data |
| `resample().mean()` on irregular data | averages weighted by sample count, not time |
| Value held across a logging outage | outage booked as run time / uptime |
| `merge_asof(direction="nearest")` | a sample picks up a setpoint that was not set yet |
| Fixed 24 h days / naive timestamps | wrong DST days, results shifted by the UTC offset |

`ts_shape.signals` fixes this at the root. You describe each signal once, as a
**`SignalSpec`**. The **primitives** then do the right thing for that kind of
signal.

## 1. Profile first

```python
import ts_shape

report = ts_shape.profile(df)
print(report.summary())
report.issues()        # error / warning / info, most severe first
report.to_frame()      # one row per signal: kind, encoding, period, gaps, resets, ...
```

The profile flags these problems:
- unparseable or duplicate timestamps;
- rows out of time order;
- missing values;
- uuids spread over several `value_*` columns;
- logging gaps;
- counter resets;
- change-encoded signals;
- signals that look stuck.

## 2. The semantic model

| Field | Meaning |
|---|---|
| `kind` | `continuous` (measured level), `counter` (cumulative, may reset), `state` (held discrete value), `event` (sparse occurrences) |
| `encoding` | `sampled` (regular rows, values repeat) or `change_encoded` (one row per change; value holds until the next row) |
| `value_column` | which `value_*` column carries the signal |
| `nominal_period` | typical time between samples |
| `max_gap` | longest span a value may be held; longer spans are *no data* |
| `tz` | timezone of `systime` |
| `reset_policy` | counters: `reset_to_zero` (a drop restarts the count) or `none` |
| `unit` | engineering unit (informational) |

## 3. Infer, then override what the data cannot tell

```python
from ts_shape.signals import infer_specs

specs = infer_specs(df, overrides={
    "line1:parts":  {"unit": "pcs"},
    "line1:state":  {"max_gap": "15min"},   # outage threshold of a change-encoded signal
    "meter:main":   {"kind": "counter", "unit": "kWh"},
})
specs.to_frame()
specs["line1:state"].notes   # why each field was inferred
```

Inference is a starting point, not a guarantee. A change-encoded signal never
gets an inferred `max_gap`, because a value may legitimately hold for hours.
Set it to your logging-outage threshold.

Specs can live next to your data as YAML:

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

```python
from ts_shape.signals import SignalSpecs
specs = SignalSpecs.from_yaml("signals.yaml")
```

## 4. Primitives

```python
from ts_shape.signals.primitives import (
    intervals, increments, per_window, align, windows, shift_windows,
)

intervals(df, specs["line1:state"])              # held states, closed at the next change, split at gaps
increments(df, specs["line1:parts"])             # reset-aware production per reading
per_window(df, specs["line1:parts"], "1h")       # parts per hour, nothing lost at boundaries
per_window(df, specs["oven:temp"], "15min")      # time-weighted mean, never across a gap
align(pv_df, specs["pv"], sp_df, specs["sp"])    # setpoint in force at each PV sample
windows("2024-03-30", "2024-04-02", "1D", tz="Europe/Berlin")   # 24 h, 23 h, 24 h
shift_windows("2024-03-30", "2024-03-31", tz="Europe/Berlin")   # night shift = 7 h on DST
```

## 5. Detectors that understand specs

Detectors are moving onto specs, pack by pack (see the [roadmap](../roadmap.md)).
Where a detector accepts `specs=`, its defaults follow the signal:

```python
ts_shape.MachineStateEvents(df, "line1:state", specs=specs).detect_run_idle()
ts_shape.DowntimeEvents(df, specs=specs).downtime_by_shift("line1:state")
```

An explicit argument always wins over the spec.
