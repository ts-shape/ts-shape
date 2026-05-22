# Quality & SPC Pipeline

> From Azure Blob measurement data to outlier detection, SPC rule checks, tolerance analysis, and capability trending — in one reusable `Pipeline`.

**Signals needed:**

| Role | UUID example | Type | Description |
|------|-------------|------|-------------|
| Measurement | `temperature_actual` | `value_double` | Process measurement (temperature, pressure, dimension, etc.) |
| Upper spec limit | `temperature_usl` | `value_double` | Upper specification limit (or provide as fixed value) |
| Lower spec limit | `temperature_lsl` | `value_double` | Lower specification limit (or provide as fixed value) |

**Modules used:** [Pipeline](../reference/ts_shape/pipeline.md) | [AzureBlobParquetLoader](../reference/ts_shape/loader/timeseries/azure_blob_loader.md) | [MetadataJsonLoader](../reference/ts_shape/loader/metadata/metadata_json_loader.md) | [ContextEnricher](../reference/ts_shape/loader/context/context_enricher.md) | [DataHarmonizer](../reference/ts_shape/transform/harmonization.md) | [DoubleFilter](../reference/ts_shape/transform/filter/numeric_filter.md) | [SignalQualityEvents](../reference/ts_shape/events/quality/signal_quality.md) | [OutlierDetectionEvents](../reference/ts_shape/events/quality/outlier_detection.md) | [StatisticalProcessControlRuleBased](../reference/ts_shape/events/quality/statistical_process_control.md) | [ToleranceDeviationEvents](../reference/ts_shape/events/quality/tolerance_deviation.md) | [CapabilityTrendingEvents](../reference/ts_shape/events/quality/capability_trending.md)

---

## Prerequisites

```python
# -- The only things you customize --
AZURE_CONNECTION = "DefaultEndpointsProtocol=https;AccountName=...;AccountKey=..."
CONTAINER = "timeseries-data"

UUID_LIST = [
    "temperature_actual",   # double: process measurement
    "temperature_usl",      # double: upper spec limit (if stored as signal)
    "temperature_lsl",      # double: lower spec limit (if stored as signal)
]

START = "2024-06-01"
END   = "2024-06-08"

METADATA_PATH = "config/signal_metadata.json"

UPPER_SPEC = 105.0   # engineering specification
LOWER_SPEC = 95.0
```

!!! info "New to `Pipeline`?"
    Read the [Pipeline guide](../guides/pipeline-builder.md) first — it explains
    `.transform` vs `.detect` steps and the debugging tools used below.

---

## Step 1: Load the data

The Azure loader produces the DataFrame the pipeline runs on, so it stays
outside the pipeline. Metadata is loaded the same way.

```python
from ts_shape.loader.timeseries.azure_blob_loader import AzureBlobParquetLoader
from ts_shape.loader.metadata.metadata_json_loader import MetadataJsonLoader

loader = AzureBlobParquetLoader(
    connection_string=AZURE_CONNECTION,
    container_name=CONTAINER,
)
df = loader.load_files_by_time_range_and_uuids(
    start_timestamp=START,
    end_timestamp=END,
    uuid_list=UUID_LIST,
)

meta_df = MetadataJsonLoader.from_file(METADATA_PATH).to_df()

print(f"Loaded {len(df):,} rows, {df['uuid'].nunique()} signals")
```

---

## Step 2: Build the pipeline

One `Pipeline` captures the whole workflow. `.transform` steps clean the
signal; every `.detect` step branches off a quality table.

```python
from ts_shape import Pipeline
from ts_shape.loader.context.context_enricher import ContextEnricher
from ts_shape.transform.filter.numeric_filter import DoubleFilter
from ts_shape.transform.harmonization import DataHarmonizer
from ts_shape.events.quality.signal_quality import SignalQualityEvents
from ts_shape.events.quality.outlier_detection import OutlierDetectionEvents
from ts_shape.events.quality.statistical_process_control import (
    StatisticalProcessControlRuleBased,
)
from ts_shape.events.quality.tolerance_deviation import ToleranceDeviationEvents
from ts_shape.events.quality.capability_trending import CapabilityTrendingEvents

pipe = (
    Pipeline(name="quality-spc")

    # -- clean the signal --
    .transform(lambda df: ContextEnricher(df).enrich_with_metadata(
        meta_df, columns=["description", "unit"]),
        name="enrich_metadata")
    .transform(DoubleFilter, "filter_nan_value_double",
               column_name="value_double")
    .detect(DataHarmonizer, "detect_gaps", name="gaps", threshold="10s")
    .transform(DataHarmonizer, "fill_gaps", strategy="interpolate",
               max_gap="30s")

    # -- signal quality diagnostics --
    .detect(SignalQualityEvents, "detect_missing_data", name="missing_data",
            signal_uuid="temperature_actual", expected_freq="1s",
            tolerance_factor=2.0)
    .detect(SignalQualityEvents, "sampling_regularity", name="regularity",
            signal_uuid="temperature_actual", window="1h")
    .detect(SignalQualityEvents, "data_completeness", name="completeness",
            signal_uuid="temperature_actual", window="1h", expected_freq="1s")

    # -- outlier detection --
    .detect(OutlierDetectionEvents, "detect_outliers_zscore",
            name="outliers_zscore", value_column="value_double", threshold=3.0)
    .detect(OutlierDetectionEvents, "detect_outliers_iqr",
            name="outliers_iqr", value_column="value_double",
            threshold=(1.5, 1.5))

    # -- SPC rule checks --
    .detect(StatisticalProcessControlRuleBased, "calculate_control_limits",
            name="control_limits", value_column="value_double",
            tolerance_uuid="temperature_usl",
            actual_uuid="temperature_actual", event_uuid="quality:spc")
    .detect(StatisticalProcessControlRuleBased,
            "calculate_dynamic_control_limits", name="dynamic_limits",
            value_column="value_double", tolerance_uuid="temperature_usl",
            actual_uuid="temperature_actual", event_uuid="quality:spc",
            window=20)
    .detect(StatisticalProcessControlRuleBased, "process", name="violations",
            value_column="value_double", tolerance_uuid="temperature_usl",
            actual_uuid="temperature_actual", event_uuid="quality:spc",
            include_severity=True)

    # -- tolerance & capability --
    .detect(ToleranceDeviationEvents, "process_and_group_data_with_events",
            name="tolerance_deviations", tolerance_column="value_double",
            actual_column="value_double", actual_uuid="temperature_actual",
            event_uuid="quality:tolerance",
            upper_tolerance_uuid="temperature_usl",
            lower_tolerance_uuid="temperature_lsl", warning_threshold=0.8)
    .detect(CapabilityTrendingEvents, "capability_over_time",
            name="capability_over_time", signal_uuid="temperature_actual",
            upper_spec=UPPER_SPEC, lower_spec=LOWER_SPEC, window="4h")
    .detect(CapabilityTrendingEvents, "detect_capability_drop",
            name="capability_drops", signal_uuid="temperature_actual",
            upper_spec=UPPER_SPEC, lower_spec=LOWER_SPEC, window="4h",
            min_cpk=1.33)
    .detect(CapabilityTrendingEvents, "capability_forecast",
            name="capability_forecast", signal_uuid="temperature_actual",
            upper_spec=UPPER_SPEC, lower_spec=LOWER_SPEC, window="4h",
            horizon=12, threshold=1.0)
)
```

Each detector's constructor and method keyword arguments are passed flat —
the pipeline routes them by name. The SPC `process` step applies the Western
Electric rules; `include_severity=True` adds the `rule` and `severity` columns.

!!! tip "Choose the right outlier method"
    - **Z-score**: normally distributed signals (most process measurements)
    - **IQR**: skewed data (flow rates, energy consumption)

!!! warning "Low completeness = unreliable SPC"
    Inspect the `completeness` result first. If completeness drops below 90%,
    SPC calculations become unreliable — investigate the data source before
    trusting the control charts.

---

## Step 3: Preview with `describe()`

```python
print(pipe.describe())
```

```
Pipeline 'quality-spc' (16 steps):
  0. [transform] enrich_metadata
  1. [transform] filter_nan_value_double  column_name='value_double'
  2. [detect   ] gaps  threshold='10s'
  3. [transform] fill_gaps  strategy='interpolate', max_gap='30s'
  4. [detect   ] missing_data  signal_uuid='temperature_actual', expected_freq='1s', tolerance_factor=2.0
  5. [detect   ] regularity  signal_uuid='temperature_actual', window='1h'
  6. [detect   ] completeness  signal_uuid='temperature_actual', window='1h', expected_freq='1s'
  7. [detect   ] outliers_zscore  value_column='value_double', threshold=3.0
  8. [detect   ] outliers_iqr  value_column='value_double', threshold=(1.5, 1.5)
  9. [detect   ] control_limits  value_column='value_double', tolerance_uuid='temperature_usl', actual_uuid='temperature_actual', event_uuid='quality:spc'
  10. [detect   ] dynamic_limits  value_column='value_double', tolerance_uuid='temperature_usl', actual_uuid='temperature_actual', event_uuid='quality:spc', window=20
  11. [detect   ] violations  value_column='value_double', tolerance_uuid='temperature_usl', actual_uuid='temperature_actual', event_uuid='quality:spc', include_severity=True
  12. [detect   ] tolerance_deviations  tolerance_column='value_double', actual_column='value_double', actual_uuid='temperature_actual', event_uuid='quality:tolerance', upper_tolerance_uuid='temperature_usl', lower_tolerance_uuid='temperature_lsl', warning_threshold=0.8
  13. [detect   ] capability_over_time  signal_uuid='temperature_actual', upper_spec=105.0, lower_spec=95.0, window='4h'
  14. [detect   ] capability_drops  signal_uuid='temperature_actual', upper_spec=105.0, lower_spec=95.0, window='4h', min_cpk=1.33
  15. [detect   ] capability_forecast  signal_uuid='temperature_actual', upper_spec=105.0, lower_spec=95.0, window='4h', horizon=12, threshold=1.0
```

---

## Step 4: Run

```python
result = pipe.run(df)          # reusable — call .run() on any DataFrame

print(f"Z-score outliers: {len(result.events['outliers_zscore'])}")
print(f"SPC violations:   {len(result.events['violations'])}")

print(result.events["capability_over_time"])   # Cp/Cpk per 4h window
print(result.events["capability_drops"])       # windows with Cpk < 1.33
```

`result.data` holds the cleaned signal after `fill_gaps`; every quality table
is keyed by its step name in `result.events`.

---

## Step 5: Debug with `run_steps()`

To inspect every intermediate DataFrame, use `run_steps()` instead of `run()`:

```python
intermediates = pipe.run_steps(df)

for name, step_df in intermediates.items():
    print(f"{name:22s} -> {step_df.shape[0]:>6} rows x {step_df.shape[1]} cols")
```

---

## Results

| `result.events` key | Description | Use case |
|---------------------|-------------|----------|
| `outliers_zscore` / `outliers_iqr` | Detected outlier events | Immediate investigation |
| `violations` | SPC rule violations (Western Electric) | Control chart alerts |
| `control_limits` / `dynamic_limits` | Static and adaptive control limits | Control charts |
| `tolerance_deviations` | Out-of-tolerance measurements | Quality escape prevention |
| `capability_over_time` | Cp/Cpk per window | Capability monitoring |
| `capability_drops` | Capability degradation alerts | Predictive quality |
| `capability_forecast` | Cpk trend extrapolation | Maintenance planning |
| `missing_data` / `regularity` / `completeness` | Signal quality diagnostics | Data trust check |

---

## Next Steps

- Correlate outlier timestamps with [Downtime Pareto](downtime-pareto.md) to find root causes
- Feed capability data into [OEE Dashboard](oee-dashboard.md) quality component
- Use [Process Engineering](process-engineering.md) to correlate quality issues with setpoint changes
