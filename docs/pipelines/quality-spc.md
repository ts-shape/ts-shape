# Quality & SPC Pipeline

> From Azure Blob measurement data to outlier detection, SPC rule checks, tolerance analysis, and capability trending (Cp/Cpk).

**Signals needed:**

| Role | UUID example | Type | Description |
|------|-------------|------|-------------|
| Measurement | `temperature_actual` | `value_double` | Process measurement (temperature, pressure, dimension, etc.) |
| Upper spec limit | `temperature_usl` | `value_double` | Upper specification limit (or provide as fixed value) |
| Lower spec limit | `temperature_lsl` | `value_double` | Lower specification limit (or provide as fixed value) |

**Modules used:** [AzureBlobParquetLoader](../reference/ts_shape/loader/timeseries/azure_blob_loader.md) | [MetadataJsonLoader](../reference/ts_shape/loader/metadata/metadata_json_loader.md) | [ContextEnricher](../reference/ts_shape/loader/context/context_enricher.md) | [DataHarmonizer](../reference/ts_shape/transform/harmonization.md) | [SignalQualityEvents](../reference/ts_shape/events/quality/signal_quality.md) | [DoubleFilter](../reference/ts_shape/transform/filter/numeric_filter.md) | [OutlierDetectionEvents](../reference/ts_shape/events/quality/outlier_detection.md) | [StatisticalProcessControlRuleBased](../reference/ts_shape/events/quality/statistical_process_control.md) | [ToleranceDeviationEvents](../reference/ts_shape/events/quality/tolerance_deviation.md) | [CapabilityTrendingEvents](../reference/ts_shape/events/quality/capability_trending.md)

---

## Prerequisites

```python
AZURE_CONNECTION = "DefaultEndpointsProtocol=https;AccountName=...;AccountKey=..."
CONTAINER = "timeseries-data"

# Measurement signal + spec limits
UUID_LIST = [
    "temperature_actual",   # double: process measurement
    "temperature_usl",      # double: upper spec limit (if stored as signal)
    "temperature_lsl",      # double: lower spec limit (if stored as signal)
]

# Or use fixed spec limits instead of UUID signals:
UPPER_SPEC = 105.0   # engineering specification
LOWER_SPEC = 95.0

START = "2024-06-01"
END   = "2024-06-08"

METADATA_PATH = "config/signal_metadata.json"
```

---

## Step 1: Load Data from Azure

```python
from ts_shape.loader.timeseries.azure_blob_loader import AzureBlobParquetLoader

loader = AzureBlobParquetLoader(
    connection_string=AZURE_CONNECTION,
    container_name=CONTAINER,
)

df = loader.load_files_by_time_range_and_uuids(
    start_timestamp=START,
    end_timestamp=END,
    uuid_list=UUID_LIST,
)

print(f"Loaded {len(df):,} rows, {df['uuid'].nunique()} signals")
```

---

## Step 2: Enrich with Metadata & Tolerances

```python
from ts_shape.loader.metadata.metadata_json_loader import MetadataJsonLoader
from ts_shape.loader.context.context_enricher import ContextEnricher

meta = MetadataJsonLoader.from_file(METADATA_PATH)
enricher = ContextEnricher(df)

# Add signal descriptions and units
df = enricher.enrich_with_metadata(meta.to_df(), columns=["description", "unit"])

# If tolerances are in metadata, add them directly
# enricher = ContextEnricher(df)
# df = enricher.enrich_with_tolerances(
#     tolerance_df,
#     low_col="low_limit",
#     high_col="high_limit",
# )
```

---

## Step 3: Validate Signal Quality

```python
from ts_shape.events.quality.signal_quality import SignalQualityEvents

sq = SignalQualityEvents(df, signal_uuid="temperature_actual")

# Check for missing data
missing = sq.detect_missing_data(expected_freq="1s", tolerance_factor=2.0)
print(f"Data gaps found: {len(missing)}")
if not missing.empty:
    print(missing[["start", "end", "gap_duration"]].head())

# Check sampling regularity
regularity = sq.sampling_regularity(window="1h")
print(regularity.head())

# Check data completeness
completeness = sq.data_completeness(expected_freq="1s", window="1h")
print(f"Average completeness: {completeness['completeness_pct'].mean():.1f}%")
```

!!! warning "Low completeness = unreliable SPC"
    If completeness drops below 90%, SPC calculations become unreliable. Investigate data source issues before running control charts.

---

## Step 4: Filter and Clean

```python
from ts_shape.transform.filter.numeric_filter import DoubleFilter
from ts_shape.transform.harmonization import DataHarmonizer

# Remove NaN values
df_clean = DoubleFilter.filter_nan_value_double(df, column_name="value_double")

# Detect and fill small gaps
harmonizer = DataHarmonizer(df_clean, value_column="value_double")
gaps = harmonizer.detect_gaps(threshold="10s")
if not gaps.empty:
    df_clean = harmonizer.fill_gaps(strategy="interpolate", max_gap="30s")

print(f"Clean data: {len(df_clean):,} rows")
```

---

## Step 5: Outlier Detection

```python
from ts_shape.events.quality.outlier_detection import OutlierDetectionEvents

detector = OutlierDetectionEvents(
    df_clean,
    value_column="value_double",
    event_uuid="quality:outlier",
)

# Z-score for normally distributed signals
outliers_z = detector.detect_outliers_zscore(threshold=3.0)
print(f"Z-score outliers: {len(outliers_z)}")

# IQR for skewed distributions
outliers_iqr = detector.detect_outliers_iqr(threshold=(1.5, 1.5))
print(f"IQR outliers: {len(outliers_iqr)}")
```

!!! tip "Choose the right detection method"
    - **Z-score**: Best for normally distributed signals (most process measurements)
    - **IQR**: Better for skewed data (flow rates, energy consumption)
    - **MAD**: Robust against extreme outliers (sensor spikes)
    - **Isolation Forest**: Complex multivariate patterns

---

## Step 6: SPC Rule Checks

```python
from ts_shape.events.quality.statistical_process_control import StatisticalProcessControlRuleBased

spc = StatisticalProcessControlRuleBased(
    dataframe=df_clean,
    value_column="value_double",
    tolerance_uuid="temperature_usl",    # UUID for tolerance signal
    actual_uuid="temperature_actual",    # UUID for measurement signal
    event_uuid="quality:spc_violation",
)

# Calculate control limits
limits = spc.calculate_control_limits()
print("Control limits:")
print(limits)

# Dynamic control limits (adapts over time)
dynamic_limits = spc.calculate_dynamic_control_limits(
    method="moving_range",
    window=20,
)

# Detect SPC rule violations (Western Electric Rules)
violations = spc.detect_violations()
print(f"SPC violations: {len(violations)}")
if not violations.empty:
    print(violations[["systime", "value_double", "rule", "severity"]].head())
```

---

## Step 7: Tolerance Deviation Analysis

```python
from ts_shape.events.quality.tolerance_deviation import ToleranceDeviationEvents

tolerance = ToleranceDeviationEvents(
    dataframe=df_clean,
    tolerance_column="value_double",
    actual_column="value_double",
    actual_uuid="temperature_actual",
    event_uuid="quality:tolerance",
    upper_tolerance_uuid="temperature_usl",
    lower_tolerance_uuid="temperature_lsl",
    warning_threshold=0.8,   # warn at 80% of tolerance band
)

# Detect out-of-tolerance events
deviations = tolerance.detect_tolerance_deviations()
print(f"Out-of-tolerance events: {len(deviations)}")

# Process capability indices
capability = tolerance.calculate_capability()
print(f"Cp: {capability['Cp']:.3f}, Cpk: {capability['Cpk']:.3f}")
```

---

## Step 8: Capability Trending

```python
from ts_shape.events.quality.capability_trending import CapabilityTrendingEvents

cap_trend = CapabilityTrendingEvents(
    dataframe=df_clean,
    signal_uuid="temperature_actual",
    upper_spec=UPPER_SPEC,
    lower_spec=LOWER_SPEC,
)

# Cp/Cpk over rolling time windows
capability_over_time = cap_trend.capability_over_time(window="4h")
print(capability_over_time.head())

# Alert on capability drops
drops = cap_trend.detect_capability_drop(threshold=1.33, window="4h")
print(f"Capability drops (Cpk < 1.33): {len(drops)}")

# Forecast: when will Cpk breach threshold?
forecast = cap_trend.capability_forecast(
    threshold=1.0,
    window="4h",
    horizon_windows=12,
)
print(forecast)
```

---

## Results

| Output | Description | Use case |
|--------|-------------|----------|
| `outliers_z` / `outliers_iqr` | Detected outlier events | Immediate investigation |
| `violations` | SPC rule violations (Western Electric) | Control chart alerts |
| `deviations` | Out-of-tolerance measurements | Quality escape prevention |
| `capability_over_time` | Cp/Cpk per window | Capability monitoring |
| `drops` | Capability degradation alerts | Predictive quality |
| `forecast` | Cpk trend extrapolation | Maintenance planning |

---

## Next Steps

- Correlate outlier timestamps with [Downtime Pareto](downtime-pareto.md) to find root causes
- Feed capability data into [OEE Dashboard](oee-dashboard.md) quality component
- Use [Process Engineering](process-engineering.md) to correlate quality issues with setpoint changes
