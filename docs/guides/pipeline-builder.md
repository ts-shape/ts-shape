# Pipelines

`ts_shape.Pipeline` chains transforms and detectors into one definition that
you can run on many DataFrames, print, and debug step by step. Use it when the
same sequence of steps runs on several datasets or on a schedule.

```python
import pandas as pd
import ts_shape
from ts_shape import Pipeline
from ts_shape.transform.calculator.numeric_calc import IntegerCalc
from ts_shape.events.quality.outlier_detection import OutlierDetectionEvents

df = ts_shape.make_timeseries(["sensor:temp"], n_points=1000, n_outliers=4)

pipe = (
    Pipeline(name="sensor-quality")
    .transform(IntegerCalc, "scale_column", value_column="value_double", factor=0.1)
    .detect(OutlierDetectionEvents, "detect_outliers_zscore",
            name="outliers", value_column="value_double", threshold=3.0)
)

result = pipe.run(df)
result.data                 # the signal after all transforms
result.events["outliers"]   # the detector output
result.to_event_log()       # all detector outputs as one OCEL event log
```

## Step kinds

The pipeline is linear. You declare the role of each step:

| Method | Effect |
|---|---|
| `.source(...)` | optional, first step only: loads the first DataFrame |
| `.transform(...)` | the output replaces the working DataFrame |
| `.detect(...)` | the output is stored in `result.events[name]`; the working DataFrame is unchanged |

Step names are unique: a second step with an explicit name that is taken
raises `ValueError`, while a transform without `name=` gets a `_2`, `_3`
suffix. `"input"` is reserved for the input frame in `run_steps()`. Each
transform works on a copy of the working DataFrame, so a transform that
changes its input in place never changes the DataFrame passed to `run()`.

## Step targets

A step is either a plain callable `df -> df`:

```python
pipe = Pipeline(name="select").transform(
    lambda d: d[d["uuid"].isin(["sensor:temp"])], name="select_signals"
)
```

or a `(class, "method")` pair. A classmethod or staticmethod (the filters,
`IntegerCalc`) is called on the class; for an instance method (detectors,
`DataHarmonizer`) the class is instantiated first. Keyword arguments are
passed flat and routed to the constructor or the method by parameter name. An
unknown keyword raises a `ValueError` listing what both accept.

## Loading inside the pipeline

With a `.source(...)` step the pipeline loads its own data and is run as
`run()`, with no argument. The source can be a loader classmethod, a
`(class, "method")` pair or a plain callable:

```python
pipe = (
    Pipeline(name="quality-from-source")
    .source(ts_shape.make_timeseries, uuids=["sensor:temp"], n_points=500, n_outliers=3)
    .detect(OutlierDetectionEvents, "detect_outliers_zscore",
            name="outliers", value_column="value_double", threshold=3.0)
)
result = pipe.run()
```

For files, use for example
`.source(ParquetLoader, "load_by_time_range", base_path=..., start=..., end=...)`.
There is at most one source step and it must come first. Passing a DataFrame
to `run()` of a pipeline with a source raises `TypeError`; a loader error is
raised as a `RuntimeError` naming step 0.

## Steps that need two DataFrames

Any keyword value can be a sentinel: `"$input"` is the DataFrame passed to
`run()`, `"$prev"` the current working DataFrame. If the keyword names the
step's own DataFrame parameter (such as `dataframe`), the sentinel replaces the
automatically passed working frame. Only strings are compared with the
sentinels, so arrays, Series and DataFrames can be passed as keyword values;
write `"$$input"` for the literal string `"$input"`.

The recipe below cuts process signals into order segments and computes a
feature table with one row per order and 15-minute window:

```python
from ts_shape.features.segment_analysis.segment_extractor import SegmentExtractor
from ts_shape.features.segment_analysis.segment_processor import SegmentProcessor
from ts_shape.features.segment_analysis.time_windowed_features import TimeWindowedFeatureTable

process = ts_shape.make_timeseries(["temperature", "pressure"], n_points=180, freq="1min")
orders = pd.DataFrame({
    "systime": pd.to_datetime(["2025-01-01 00:00", "2025-01-01 01:00", "2025-01-01 02:00"]),
    "uuid": "order_number",
    "value_string": ["Order-A", "Order-B", "Order-C"],
})
data = pd.concat([process, orders], ignore_index=True)

features = (
    Pipeline(name="feature-table")
    .transform(SegmentExtractor, "extract_time_ranges", segment_uuid="order_number")
    .transform(SegmentProcessor, "apply_ranges",
               dataframe="$input", time_ranges="$prev", target_uuids=["temperature", "pressure"])
    .transform(TimeWindowedFeatureTable, "compute", window="15min", metrics=["mean", "std", "max"])
)
table = features.run(data).data   # time_window, segment_value, temperature__mean, ...
```

[Cycles and segments](feature-extraction.md) explains the segment classes.

## Inspecting and debugging

```python
print(features.describe())      # every step with its arguments, without running

steps = features.run_steps(data)
steps["input"]                  # the original frame
steps["extract_time_ranges"]    # the output of each step, by step name
```

An exception inside a step names the failing step (index and name). A step
that does not return a DataFrame raises `TypeError`.

## Event-log export

`PipelineResult.to_event_log()` normalizes the output of every `.detect` step
into one canonical event log (see [Event log](eventlog/index.md)). Pass
`concat=False` for a dict with one log per step.
