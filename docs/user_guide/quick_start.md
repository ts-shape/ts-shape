# Quick start

This page walks through one analysis end to end. Every block runs as is, in
order, after `pip install ts-shape`.

## 1. Get data in the standard format

ts-shape works on a long-format DataFrame: one row per sample, with a
timestamp (`systime`), a signal identifier (`uuid`) and the value in one of
`value_double`, `value_integer`, `value_bool` or `value_string`. See
[Data model](../concept.md) for the full schema.

`make_timeseries` generates such a frame. With your own data, use a
[loader](../guides/loading.md) or build the frame with pandas.

```python
import pandas as pd
import ts_shape

temp = ts_shape.make_timeseries(["oven:temp"], n_points=1440, freq="1min", n_outliers=4)
running = ts_shape.make_timeseries(
    ["oven:running"], n_points=1440, freq="1min", value_column="value_bool"
)
df = pd.concat([temp, running], ignore_index=True)
```

## 2. Profile it

`profile` reports what each signal is (continuous, counter, state, event) and
what is wrong with it: gaps, duplicate timestamps, counter resets, stuck values.

```python
report = ts_shape.profile(df)
print(report.summary())
report.to_frame()      # one row per signal
```

## 3. Filter and describe

```python
from ts_shape.transform.filter.numeric_filter import DoubleFilter
from ts_shape.features.stats.numeric_stats import NumericStatistics

temp = df[df["uuid"] == "oven:temp"]
temp = DoubleFilter.filter_value_double_between(temp, "value_double", min_value=0, max_value=200)
NumericStatistics.summary_as_dataframe(temp, "value_double")
```

## 4. Detect events

Detectors are classes named `...Events`. You build one on a DataFrame and call
its methods; each method returns a DataFrame.

```python
outliers = ts_shape.OutlierDetectionEvents(temp, value_column="value_double")
spikes = outliers.detect_outliers_zscore(threshold=4.0)

state = ts_shape.MachineStateEvents(df, run_state_uuid="oven:running")
intervals = state.detect_run_idle(min_duration="5min")
intervals[["start", "end", "duration_seconds", "state"]].head()
```

`ts_shape.list_detectors("events.production")` lists the detector classes of a
pack.

## 5. Build an event log

Any detector output can be normalized into one OCEL 2.0 event log, and from
there exported in the column layout process-mining tools such as pm4py expect.

```python
from ts_shape.eventlog import concat, to_event_log, to_event_log_xes

log = concat(
    to_event_log(spikes, detector="OutlierDetectionEvents.detect_outliers_zscore"),
    to_event_log(intervals, detector="MachineStateEvents.detect_run_idle"),
)
xes = to_event_log_xes(log, case_object_type="asset")
xes[["case:concept:name", "concept:name", "time:timestamp"]].head()
```

## Next steps

- [Data model and layers](../concept.md)
- [Production and OEE](../guides/production.md), [Quality and SPC](../guides/quality.md)
- [Pipelines](../guides/pipeline-builder.md): put these steps into one reusable definition
