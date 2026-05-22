# Feature-Table Pipeline

> From raw timeseries to ML-ready feature tables in a single reusable chain.

**Signals needed:**

| Role | UUID example | Type | Description |
|------|-------------|------|-------------|
| Order signal | `order_number` | `value_string` | Categorical signal that changes when a new order/batch/recipe starts |
| Process param 1 | `temperature` | `value_double` | Numeric measurement (any process variable) |
| Process param 2 | `pressure` | `value_double` | Numeric measurement |
| Process param 3 | `speed` | `value_double` | Numeric measurement |

**Modules used:** [Pipeline](../reference/ts_shape/pipeline.md) | [DateTimeFilter](../reference/ts_shape/transform/filter/datetime_filter.md) | [DoubleFilter](../reference/ts_shape/transform/filter/numeric_filter.md) | [DataHarmonizer](../reference/ts_shape/transform/harmonization.md) | [SegmentExtractor](../reference/ts_shape/features/segment_analysis/segment_extractor.md) | [SegmentProcessor](../reference/ts_shape/features/segment_analysis/segment_processor.md) | [TimeWindowedFeatureTable](../reference/ts_shape/features/segment_analysis/time_windowed_features.md)

---

## Prerequisites

```python
# -- The only things you customize --
PROCESS_UUIDS = ['temperature', 'pressure', 'speed']
ORDER_UUID = 'order_number'

START = '2024-01-01'
END   = '2024-01-31'

FREQ    = '1min'                          # time window for features
METRICS = ['mean', 'std', 'min', 'max']   # statistical metrics per window
```

!!! info "New to `Pipeline`?"
    Read the [Pipeline guide](../guides/pipeline-builder.md) first — it explains
    `.transform` vs `.detect` steps, sentinels (`$prev`, `$input`), and the
    debugging tools.

---

## Step 1: Build the pipeline

A `Pipeline` is defined once and run on any DataFrame. Every step here is a
`.transform` — each one's output replaces the working signal.

```python
from ts_shape import Pipeline
from ts_shape.transform.filter.numeric_filter import DoubleFilter
from ts_shape.transform.filter.datetime_filter import DateTimeFilter
from ts_shape.transform.harmonization import DataHarmonizer
from ts_shape.features.segment_analysis.segment_extractor import SegmentExtractor
from ts_shape.features.segment_analysis.segment_processor import SegmentProcessor
from ts_shape.features.segment_analysis.time_windowed_features import TimeWindowedFeatureTable

pipe = (
    Pipeline(name="feature-table")

    # 1. Trim to time window
    .transform(DateTimeFilter, "filter_between_datetimes",
               start_datetime=START, end_datetime=END)

    # 2. Remove rows with NaN in value_double
    .transform(DoubleFilter, "filter_nan_value_double")

    # 3. Keep only process signals (drop the order signal for numeric steps)
    .transform(lambda df: df[df['uuid'].isin(PROCESS_UUIDS)],
               name='select_process_signals')

    # 4. Resample to a uniform 1-second grid (DataHarmonizer is instantiated)
    .transform(DataHarmonizer, "resample_to_uniform", freq='1s')

    # 5. Extract time ranges from the order signal (uses the original data)
    .transform(SegmentExtractor, "extract_time_ranges",
               dataframe='$input', segment_uuid=ORDER_UUID)

    # 6. Apply those ranges to the process data
    .transform(SegmentProcessor, "apply_ranges",
               dataframe='$input', time_ranges='$prev',
               target_uuids=PROCESS_UUIDS)

    # 7. Compute the feature table
    .transform(TimeWindowedFeatureTable, "compute",
               freq=FREQ, metrics=METRICS)
)
```

A `(class, "method")` step works for both stateless classmethods (the filters,
`SegmentExtractor`, `TimeWindowedFeatureTable`) and stateful classes
(`DataHarmonizer`) — the pipeline inspects the class and does the right thing.
The `$input` / `$prev` sentinels wire the original frame and the previous
step's output into steps that need a second DataFrame.

---

## Step 2: Preview with `describe()`

```python
print(pipe.describe())
```

```
Pipeline 'feature-table' (7 steps):
  0. [transform] filter_between_datetimes  start_datetime='2024-01-01', end_datetime='2024-01-31'
  1. [transform] filter_nan_value_double
  2. [transform] select_process_signals
  3. [transform] resample_to_uniform  freq='1s'
  4. [transform] extract_time_ranges  dataframe='$input', segment_uuid='order_number'
  5. [transform] apply_ranges  dataframe='$input', time_ranges='$prev', target_uuids=['temperature', 'pressure', 'speed']
  6. [transform] compute  freq='1min', metrics=['mean', 'std', 'min', 'max']
```

---

## Step 3: Run

```python
result = pipe.run(df)          # reusable — call .run() on any DataFrame

feature_table = result.data
print(f"Feature table: {feature_table.shape[0]} rows x {feature_table.shape[1]} cols")
print(feature_table.head())
```

```
Feature table: 90 rows x 14 cols

  time_window          segment_value  temperature__mean  temperature__std  pressure__mean  ...
  2024-01-01 00:00:00  Order-A        50.12              1.87              100.34          ...
  2024-01-01 00:01:00  Order-A        50.08              1.91              100.28          ...
```

Each row is one time window; columns follow the pattern `{uuid}__{metric}`.

---

## Step 4: Debug with `run_steps()`

To inspect every intermediate DataFrame, use `run_steps()` instead of `run()`:

```python
intermediates = pipe.run_steps(df)

for name, step_df in intermediates.items():
    print(f"{name:30s} -> {step_df.shape[0]:>6} rows x {step_df.shape[1]} cols")
```

```
input                          ->   4800 rows x 4 cols
filter_between_datetimes        ->   4800 rows x 4 cols
filter_nan_value_double         ->   3600 rows x 4 cols
select_process_signals          ->   3600 rows x 4 cols
resample_to_uniform             ->   3600 rows x 4 cols
extract_time_ranges             ->      3 rows x 5 cols
apply_ranges                    ->   3600 rows x 6 cols
compute                         ->     90 rows x 14 cols
```

---

## Step 5: Add a detector branch

Because `Pipeline` also runs detectors, you can score the cleaned signal in the
same pass. A `.detect` step stores its output in `result.events` and leaves the
feature-table channel untouched:

```python
from ts_shape.events.quality.outlier_detection import OutlierDetectionEvents

pipe.detect(OutlierDetectionEvents, "detect_outliers_zscore",
            name="outliers", value_column="value_double", threshold=3.0)

result = pipe.run(df)
result.data                # the feature table
result.events["outliers"]  # outlier events on the cleaned signal
result.to_event_log()      # detector output as a canonical OCEL event log
```

---

## Results

| Output | Description | Typical shape |
|--------|-------------|---------------|
| `result.data` | Feature table: one row per time window, columns = `{uuid}__{metric}` | 90 rows x 14 cols |
| `result.events` | Detector outputs keyed by step name | one entry per `.detect` |
| `pipe.run_steps(df)` | Dict of DataFrames for every pipeline step | 8 entries |

---

## Next Steps

- [Pipeline](../guides/pipeline-builder.md) — Step types, sentinels, and debugging tools
- [Feature Extraction](../guides/feature-extraction.md) — Cycles vs segments (manual approach)
- [Quality & SPC](quality-spc.md) — Apply SPC rules and capability analysis to your feature table
- [Process Engineering](process-engineering.md) — Correlate features with setpoint changes
