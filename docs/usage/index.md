# Usage Examples

!!! note "These examples have moved to the new **[Guides](../guides/index.md)** section"
    The guides are organized by topic — [Data Acquisition](../guides/loading.md), [Signal Conditioning](../guides/transforms.md), [Quality Control](../guides/quality.md), [Production Monitoring](../guides/production.md), [OEE](../guides/oee-analytics.md), [Traceability](../guides/traceability.md), [Engineering](../guides/engineering.md), and [Reporting](../guides/reporting.md). This page is kept for backward compatibility.

Practical examples for common timeseries data tasks.

## Loading Data

### From Parquet Files

```python
from ts_shape.loader.timeseries.parquet_loader import ParquetLoader

# Load all parquet files from a directory
df = ParquetLoader.load_all_files("data/sensors/")

# Preview the data
print(df.head())
#          uuid                   systime  value_double
# 0  temperature  2024-01-01 00:00:00+00:00         23.5
# 1  temperature  2024-01-01 00:01:00+00:00         23.7
# 2     pressure  2024-01-01 00:00:00+00:00       1013.2
```

### From Azure Blob Storage (Parquet)

Three authentication methods are supported. Pick whichever matches your setup.

#### Connect with a SAS URL (simplest)

```python
from ts_shape.loader.timeseries.azure_blob_loader import AzureBlobParquetLoader

# SAS URL — no container_name needed, it's embedded in the URL
loader = AzureBlobParquetLoader(
    sas_url="https://myaccount.blob.core.windows.net/timeseries?sv=2021-06-08&st=...&se=...&sr=c&sp=rl&sig=...",
    prefix="parquet/",         # optional path prefix to narrow listing
)
```

#### Connect with a connection string

```python
loader = AzureBlobParquetLoader(
    connection_string="DefaultEndpointsProtocol=https;AccountName=...;AccountKey=...",
    container_name="timeseries",
    prefix="parquet/",
)
```

#### Connect with AAD credential (DefaultAzureCredential)

```python
from azure.identity import DefaultAzureCredential

loader = AzureBlobParquetLoader(
    account_url="https://myaccount.blob.core.windows.net",
    container_name="timeseries",
    credential=DefaultAzureCredential(),
    prefix="parquet/",
)
```

#### Explore the container structure

```python
# List folders and files to understand the layout
structure = loader.list_structure(limit=20)
print("Folders:", structure["folders"])
print("Files:",   structure["files"])
# Folders: ['parquet/2024/01/15/08/', 'parquet/2024/01/15/09/', ...]
# Files:   ['parquet/2024/01/15/08/temperature.parquet', ...]
```

#### Load all parquet files

```python
df = loader.load_all_files()
print(df.head())
```

#### Load by time range

```python
# Requires time-structured folders: prefix/YYYY/MM/DD/HH/
df = loader.load_by_time_range("2024-01-15 08:00", "2024-01-15 12:00")
```

#### Load by time range and specific UUIDs

```python
df = loader.load_files_by_time_range_and_uuids(
    start_timestamp="2024-01-15 08:00",
    end_timestamp="2024-01-15 12:00",
    uuid_list=["temperature", "pressure", "humidity"],
)
```

#### Stream results (low memory)

```python
# Yields (blob_name, DataFrame) one at a time
for blob_name, chunk_df in loader.stream_by_time_range("2024-01-15 08:00", "2024-01-15 12:00"):
    print(f"{blob_name}: {len(chunk_df)} rows")
    process(chunk_df)
```

### From Azure Blob Storage (Any File Type)

Use `AzureBlobFlexibleFileLoader` for non-parquet files (CSV, JSON, XML, etc.)
under the same time-structured layout.

```python
from ts_shape.loader.timeseries.azure_blob_loader import AzureBlobFlexibleFileLoader

# Same three auth methods are supported (sas_url, connection_string, AAD)
loader = AzureBlobFlexibleFileLoader(
    sas_url="https://myaccount.blob.core.windows.net/rawdata?sv=...&sig=...",
    prefix="incoming/",
)

# List blob names for a time range, filtering by extension
names = loader.list_files_by_time_range(
    start_timestamp="2024-01-15 08:00",
    end_timestamp="2024-01-15 12:00",
    extensions=[".csv", ".json"],
)
print(names)
# ['incoming/2024/01/15/08/report.csv', 'incoming/2024/01/15/09/data.json', ...]

# Download and auto-parse files (CSV → DataFrame, JSON → dict, etc.)
results = loader.fetch_files_by_time_range(
    start_timestamp="2024-01-15 08:00",
    end_timestamp="2024-01-15 12:00",
    extensions=[".csv"],
    parse=True,
)
for blob_name, parsed in results.items():
    print(f"{blob_name}: {type(parsed)}")
```

### From S3-Compatible Storage

```python
from ts_shape.loader.timeseries.s3proxy_parquet_loader import S3ProxyParquetLoader

loader = S3ProxyParquetLoader(
    endpoint_url="https://s3.example.com",
    bucket="data-lake",
    prefix="timeseries/"
)

df = loader.fetch_data_as_dataframe()
```

### Loading Metadata

```python
from ts_shape.loader.metadata.metadata_json_loader import MetadataLoader

# Load signal metadata from JSON
meta = MetadataLoader("config/signals.json").to_df()

print(meta)
#          uuid         label    unit
# 0  temperature  Temperature  Celsius
# 1     pressure     Pressure     hPa
```

---

## Combining Data

### Merge Timeseries with Metadata

```python
from ts_shape.loader.combine.integrator import DataIntegratorHybrid

# Combine timeseries with signal metadata
combined = DataIntegratorHybrid.combine_data(
    timeseries_sources=[ts_df],
    metadata_sources=[meta_df],
    join_key="uuid",
    merge_how="left"
)

print(combined.head())
#          uuid                   systime  value_double        label     unit
# 0  temperature  2024-01-01 00:00:00+00:00         23.5  Temperature  Celsius
```

### Filter by Specific Signals

```python
# Only load specific UUIDs
combined = DataIntegratorHybrid.combine_data(
    timeseries_sources=[ts_df],
    metadata_sources=[meta_df],
    uuids=["temperature", "humidity"],
    join_key="uuid"
)
```

---

## Filtering Data

### By Numeric Range

```python
from ts_shape.transform.filter.numeric_filter import NumericFilter

# Keep values between 0 and 100
df = NumericFilter.filter_value_in_range(df, "value_double", min_value=0, max_value=100)

# Remove null values
df = NumericFilter.filter_not_null(df, "value_double")

# Keep values above threshold
df = NumericFilter.filter_greater_than(df, "value_double", threshold=50)
```

### By Time Range

```python
from ts_shape.transform.filter.datetime_filter import DateTimeFilter

# Filter to specific date range
df = DateTimeFilter.filter_between(
    df,
    column="systime",
    start_date="2024-01-01",
    end_date="2024-01-31"
)

# Keep only data after a date
df = DateTimeFilter.filter_after(df, "systime", "2024-06-01")

# Filter by hour of day (e.g., business hours)
df = DateTimeFilter.filter_by_hour_range(df, "systime", start_hour=9, end_hour=17)
```

### By String Pattern

```python
from ts_shape.transform.filter.string_filter import StringFilter

# Filter by exact match
df = StringFilter.filter_equals(df, "uuid", "temperature")

# Filter by pattern
df = StringFilter.filter_contains(df, "uuid", "sensor_")

# Filter by list of values
df = StringFilter.filter_in_list(df, "uuid", ["temp_1", "temp_2", "temp_3"])
```

### By Boolean Flag

```python
from ts_shape.transform.filter.boolean_filter import IsDeltaFilter

# Keep only delta values
df = IsDeltaFilter.filter_is_delta_true(df)

# Keep only absolute values
df = IsDeltaFilter.filter_is_delta_false(df)
```

---

## Transforming Data

### Timezone Conversion

```python
from ts_shape.transform.time_functions.timezone_shift import TimezoneShift
from ts_shape.transform.time_functions.timestamp_converter import TimestampConverter

# Convert Unix timestamps to datetime
df = TimestampConverter.convert_to_datetime(
    df,
    columns=["systime"],
    unit="ns",
    timezone="UTC"
)

# Shift to local timezone
df = TimezoneShift.shift_timezone(
    df,
    time_column="systime",
    input_timezone="UTC",
    target_timezone="Europe/Berlin"
)
```

### Numeric Calculations

```python
from ts_shape.transform.calculator.numeric_calc import NumericCalc

# Add rolling average
df = NumericCalc.add_rolling_mean(df, "value_double", window=10)

# Add difference from previous value
df = NumericCalc.add_diff(df, "value_double")

# Normalize values (0-1 scale)
df = NumericCalc.normalize(df, "value_double")
```

---

## Computing Statistics

### Numeric Statistics

```python
from ts_shape.features.stats.numeric_stats import NumericStatistics

stats = NumericStatistics(df, "value_double")

print(f"Count: {stats.count()}")
print(f"Mean: {stats.mean():.2f}")
print(f"Std: {stats.std():.2f}")
print(f"Min: {stats.min():.2f}")
print(f"Max: {stats.max():.2f}")
print(f"Median: {stats.median():.2f}")

# Get percentiles
print(f"P95: {stats.percentile(95):.2f}")
print(f"P99: {stats.percentile(99):.2f}")
```

### Time Coverage Statistics

```python
from ts_shape.features.stats.timestamp_stats import TimestampStatistics

time_stats = TimestampStatistics(df, "systime")

print(f"First: {time_stats.first()}")
print(f"Last: {time_stats.last()}")
print(f"Duration: {time_stats.duration()}")
print(f"Count: {time_stats.count()}")
```

### String Value Counts

```python
from ts_shape.features.stats.string_stats import StringStatistics

str_stats = StringStatistics(df, "uuid")

# Get value frequency
print(str_stats.value_counts())
#          uuid  count
# 0  temperature   1440
# 1     pressure   1440
# 2     humidity   1440
```

---

## Detecting Events

### Outlier Detection

```python
from ts_shape.events.quality.outlier_detection import OutlierDetection

# Z-score based outliers (values > 3 std from mean)
outliers = OutlierDetection.detect_zscore_outliers(
    df,
    column="value_double",
    threshold=3.0
)

print(f"Found {len(outliers)} outliers")

# IQR-based outliers
outliers = OutlierDetection.detect_iqr_outliers(
    df,
    column="value_double",
    multiplier=1.5
)
```

### Statistical Process Control

```python
from ts_shape.events.quality.statistical_process_control import StatisticalProcessControl

spc = StatisticalProcessControl(df, value_column="value_double")

# Detect control limit violations
violations = spc.detect_control_violations(
    ucl=100,  # Upper control limit
    lcl=0     # Lower control limit
)

# Detect Western Electric rules violations
we_violations = spc.detect_western_electric_rules()
```

### Tolerance Deviations

```python
from ts_shape.events.quality.tolerance_deviation import ToleranceDeviation

# Find values outside specification limits
deviations = ToleranceDeviation.detect_out_of_tolerance(
    df,
    column="value_double",
    upper_limit=100,
    lower_limit=0
)
```

---

## Complete Pipeline Example

```python
import pandas as pd
from ts_shape.loader.timeseries.parquet_loader import ParquetLoader
from ts_shape.loader.metadata.metadata_json_loader import MetadataLoader
from ts_shape.loader.combine.integrator import DataIntegratorHybrid
from ts_shape.transform.filter.datetime_filter import DateTimeFilter
from ts_shape.transform.filter.numeric_filter import NumericFilter
from ts_shape.features.stats.numeric_stats import NumericStatistics
from ts_shape.events.quality.outlier_detection import OutlierDetection

# 1. Load data
print("Loading data...")
ts_df = ParquetLoader.load_all_files("data/sensors/")
meta_df = MetadataLoader("config/signals.json").to_df()

# 2. Combine with metadata
print("Combining with metadata...")
df = DataIntegratorHybrid.combine_data(
    timeseries_sources=[ts_df],
    metadata_sources=[meta_df],
    join_key="uuid"
)
print(f"  Total records: {len(df)}")

# 3. Filter to analysis period
print("Filtering...")
df = DateTimeFilter.filter_between(df, "systime", "2024-01-01", "2024-03-31")
df = NumericFilter.filter_not_null(df, "value_double")
print(f"  After filtering: {len(df)}")

# 4. Detect and remove outliers
print("Detecting outliers...")
outliers = OutlierDetection.detect_zscore_outliers(df, "value_double", threshold=3.0)
clean_df = df[~df.index.isin(outliers.index)]
print(f"  Outliers removed: {len(outliers)}")

# 5. Compute statistics per signal
print("\nStatistics by signal:")
for uuid in clean_df["uuid"].unique():
    signal_df = clean_df[clean_df["uuid"] == uuid]
    stats = NumericStatistics(signal_df, "value_double")
    print(f"  {uuid}:")
    print(f"    Count: {stats.count()}")
    print(f"    Mean: {stats.mean():.2f}")
    print(f"    Std: {stats.std():.2f}")
    print(f"    Range: [{stats.min():.2f}, {stats.max():.2f}]")

# 6. Export results
clean_df.to_parquet("output/clean_data.parquet")
print("\nExported to output/clean_data.parquet")
```

Output:
```
Loading data...
Combining with metadata...
  Total records: 125000
Filtering...
  After filtering: 98500
Detecting outliers...
  Outliers removed: 127

Statistics by signal:
  temperature:
    Count: 32850
    Mean: 23.45
    Std: 2.31
    Range: [18.20, 28.70]
  pressure:
    Count: 32800
    Mean: 1013.25
    Std: 5.67
    Range: [995.00, 1030.00]
  humidity:
    Count: 32723
    Mean: 65.30
    Std: 12.45
    Range: [35.00, 95.00]

Exported to output/clean_data.parquet
```

---

## Production Traceability

### Part Production Tracking

Track production quantities by part number with time-based aggregation.

```python
from ts_shape.events.production.part_tracking import PartProductionTracking

tracker = PartProductionTracking(df)

# Production by part with hourly windows
hourly = tracker.production_by_part(
    part_id_uuid='part_number_signal',
    counter_uuid='counter_signal',
    window='1h'
)
#     window_start         part_number  quantity  first_count  last_count
# 0   2024-01-01 08:00:00  PART_A       150       1000        1150

# Daily production summary
daily = tracker.daily_production_summary(
    part_id_uuid='part_number_signal',
    counter_uuid='counter_signal'
)

# Total production for date range
totals = tracker.production_totals(
    part_id_uuid='part_number_signal',
    counter_uuid='counter_signal',
    start_date='2024-01-01',
    end_date='2024-01-31'
)
```

### Quality Tracking (NOK/Scrap)

Track defective parts, first-pass yield, and defect reasons.

```python
from ts_shape.events.production.quality_tracking import QualityTracking

tracker = QualityTracking(df, shift_definitions={
    "day": ("06:00", "14:00"),
    "afternoon": ("14:00", "22:00"),
    "night": ("22:00", "06:00"),
})

# NOK parts per shift with first-pass yield
shift_quality = tracker.nok_by_shift(
    ok_counter_uuid='good_parts',
    nok_counter_uuid='bad_parts'
)
#     date        shift    ok_parts  nok_parts  nok_rate_pct  first_pass_yield_pct
# 0   2024-01-01  day      450       12         2.6           97.4

# Quality by part number
part_quality = tracker.quality_by_part(
    ok_counter_uuid='good_parts',
    nok_counter_uuid='bad_parts',
    part_id_uuid='part_number'
)

# Pareto analysis of defect reasons
reasons = tracker.nok_by_reason(
    nok_counter_uuid='bad_parts',
    defect_reason_uuid='defect_code'
)
```

### Cycle Time Tracking

Analyze cycle times with trend detection and slow cycle identification.

```python
from ts_shape.events.production.cycle_time_tracking import CycleTimeTracking

tracker = CycleTimeTracking(df)

# Cycle times per part
cycles = tracker.cycle_time_by_part(
    part_id_uuid='part_number_signal',
    cycle_trigger_uuid='cycle_complete_signal'
)

# Statistics by part (min, avg, max, std, median)
stats = tracker.cycle_time_statistics(
    part_id_uuid='part_number_signal',
    cycle_trigger_uuid='cycle_complete_signal'
)

# Detect slow cycles (>1.5x median)
slow = tracker.detect_slow_cycles(
    part_id_uuid='part_number_signal',
    cycle_trigger_uuid='cycle_complete_signal',
    threshold_factor=1.5
)

# Trend analysis for specific part
trend = tracker.cycle_time_trend(
    part_id_uuid='part_number_signal',
    cycle_trigger_uuid='cycle_complete_signal',
    part_number='PART_A',
    window_size=20
)
```

### Downtime Tracking

Track machine downtime by shift, reason, and availability trends.

```python
from ts_shape.events.production.downtime_tracking import DowntimeTracking

tracker = DowntimeTracking(df)

# Downtime per shift with availability
shift_downtime = tracker.downtime_by_shift(
    state_uuid='machine_state',
    running_value='Running'
)
#     date        shift    downtime_minutes  uptime_minutes  availability_pct
# 0   2024-01-01  shift_1  45.2             434.8           90.6

# Downtime by reason (Pareto analysis)
reasons = tracker.downtime_by_reason(
    state_uuid='machine_state',
    reason_uuid='downtime_reason',
    stopped_value='Stopped'
)

# Top 5 downtime reasons
top_reasons = tracker.top_downtime_reasons(
    state_uuid='machine_state',
    reason_uuid='downtime_reason',
    top_n=5
)

# Availability trend over time
trend = tracker.availability_trend(
    state_uuid='machine_state',
    running_value='Running',
    window='1D'
)
```

### Shift Reporting

Compare shift performance and track against targets.

```python
from ts_shape.events.production.shift_reporting import ShiftReporting

reporter = ShiftReporting(df)

# Production per shift
shift_prod = reporter.shift_production(
    counter_uuid='counter_signal',
    part_id_uuid='part_number_signal'
)

# Compare shifts (last 7 days)
comparison = reporter.shift_comparison(counter_uuid='counter_signal', days=7)

# Track against targets
targets = reporter.shift_targets(
    counter_uuid='counter_signal',
    targets={'shift_1': 450, 'shift_2': 450, 'shift_3': 400}
)

# Best and worst shifts
results = reporter.best_and_worst_shifts(counter_uuid='counter_signal')
```

### Machine State Events

Detect run/idle intervals and state transitions.

```python
from ts_shape.events.production.machine_state import MachineStateEvents

state = MachineStateEvents(df, run_state_uuid='machine_running')

# Run/idle intervals with minimum duration
intervals = state.detect_run_idle(min_duration='30s')

# State transitions
transitions = state.transition_events()

# Detect rapid state changes (suspicious)
rapid = state.detect_rapid_transitions(threshold='5s', min_count=3)

# Quality metrics
metrics = state.state_quality_metrics()
print(f"Run/Idle ratio: {metrics['run_idle_ratio']:.2f}")
```

### Changeover Detection

Detect product/recipe changes and compute changeover windows.

```python
from ts_shape.events.production.changeover import ChangeoverEvents

changeover = ChangeoverEvents(df)

# Detect changeovers
changes = changeover.detect_changeover(
    product_uuid='product_signal',
    min_hold='5m'
)

# Compute changeover windows (fixed duration)
windows = changeover.changeover_window(
    product_uuid='product_signal',
    until='fixed_window',
    config={'duration': '10m'}
)

# Compute changeover windows (stable band - waits for process stability)
windows = changeover.changeover_window(
    product_uuid='product_signal',
    until='stable_band',
    config={
        'metrics': [
            {'uuid': 'temperature', 'band': 2.0, 'hold': '2m'},
            {'uuid': 'pressure', 'band': 5.0, 'hold': '2m'},
        ],
        'reference_method': 'ewma'
    }
)
```

---

## Engineering Events

### Setpoint Change Analysis

Comprehensive setpoint change detection with control quality KPIs.

```python
from ts_shape.events.engineering.setpoint_events import SetpointChangeEvents

setpoint = SetpointChangeEvents(df, setpoint_uuid='temperature_setpoint')

# Detect step changes
steps = setpoint.detect_setpoint_steps(min_delta=5.0, min_hold='30s')

# Detect ramp changes
ramps = setpoint.detect_setpoint_ramps(min_rate=0.1, min_duration='10s')

# Time to settle
settling = setpoint.time_to_settle(
    actual_uuid='temperature_actual',
    tol=1.0,
    hold='10s',
    lookahead='5m'
)

# Overshoot/undershoot metrics
overshoot = setpoint.overshoot_metrics(actual_uuid='temperature_actual')

# Rise time (10% to 90%)
rise = setpoint.rise_time(actual_uuid='temperature_actual')

# Comprehensive control quality metrics (all-in-one)
quality = setpoint.control_quality_metrics(
    actual_uuid='temperature_actual',
    tol=1.0,
    hold='10s'
)
# Returns: t_settle, rise_time, overshoot, undershoot, oscillations, decay_rate
```

---

## Advanced Features

### Cycle Extraction

ts-shape includes powerful cycle detection capabilities for industrial processes.

```python
from ts_shape.features.cycles.cycles_extractor import CycleExtractor

# Initialize extractor with start/end signals
extractor = CycleExtractor(
    dataframe=df,
    start_uuid="cycle_start_signal",
    end_uuid="cycle_end_signal",
    value_change_threshold=0.1
)

# Get recommendations for best extraction method
suggestions = extractor.suggest_method()
print(f"Recommended: {suggestions['recommended_methods']}")
print(f"Reason: {suggestions['reasoning']}")

# Extract cycles using the recommended method
if 'process_persistent_cycle' in suggestions['recommended_methods']:
    cycles = extractor.process_persistent_cycle()
elif 'process_step_sequence' in suggestions['recommended_methods']:
    cycles = extractor.process_step_sequence(start_step=1, end_step=10)
else:
    cycles = extractor.process_value_change_cycle()

# Validate cycles
validated = extractor.validate_cycles(
    cycles,
    min_duration='1s',
    max_duration='1h'
)

# Detect and resolve overlapping cycles
clean_cycles = extractor.detect_overlapping_cycles(
    validated,
    resolve='keep_longest'
)

# Get extraction statistics
stats = extractor.get_extraction_stats()
print(f"Total: {stats['total_cycles']}, Complete: {stats['complete_cycles']}")
```

### Advanced Outlier Detection

Multiple robust outlier detection methods beyond basic z-score.

```python
from ts_shape.events.quality.outlier_detection import OutlierDetectionEvents

detector = OutlierDetectionEvents(
    dataframe=df,
    value_column="value_double",
    event_uuid="outlier_event",
    time_threshold="5min"
)

# Z-score method (standard)
outliers_zscore = detector.detect_outliers_zscore(threshold=3.0)

# IQR method (resistant to extreme values)
outliers_iqr = detector.detect_outliers_iqr(threshold=(1.5, 1.5))

# MAD method (most robust to outliers)
outliers_mad = detector.detect_outliers_mad(threshold=3.5)

# IsolationForest (machine learning based)
outliers_ml = detector.detect_outliers_isolation_forest(
    contamination=0.1,
    random_state=42
)

# All methods return a numeric severity column
print(outliers_mad[['systime', 'value_double', 'severity']])
```

### Statistical Process Control (SPC)

Full Western Electric Rules and CUSUM shift detection.

```python
from ts_shape.events.quality.statistical_process_control import StatisticalProcessControlRuleBased

spc = StatisticalProcessControlRuleBased(
    dataframe=df,
    value_column="value_double",
    tolerance_uuid="control_limits",
    actual_uuid="measurements",
    event_uuid="spc_violation"
)

# Calculate control limits
limits = spc.calculate_control_limits()
print(f"Mean: {limits['mean'][0]:.2f}")
print(f"UCL (3σ): {limits['3sigma_upper'][0]:.2f}")
print(f"LCL (3σ): {limits['3sigma_lower'][0]:.2f}")

# Dynamic control limits (adapts over time)
dynamic_limits = spc.calculate_dynamic_control_limits(
    method='ewma',  # or 'moving_range'
    window=20
)

# Apply Western Electric Rules (all 8 rules)
violations = spc.apply_rules_vectorized()

# Or select specific rules
violations = spc.apply_rules_vectorized(
    selected_rules=['rule_1', 'rule_2', 'rule_3']
)

# Get human-readable interpretations
interpreted = spc.interpret_violations(violations)
print(interpreted[['systime', 'rule', 'interpretation', 'recommendation']])

# CUSUM shift detection (sensitive to small shifts)
shifts = spc.detect_cusum_shifts(
    k=0.5,    # Slack parameter
    h=5.0     # Decision threshold
)
print(shifts[['systime', 'shift_direction', 'severity']])
```

### Process Capability Indices

Calculate Cp, Cpk, Pp, Ppk for quality assessment.

```python
from ts_shape.events.quality.tolerance_deviation import ToleranceDeviationEvents

# With separate upper/lower tolerances
tolerance_checker = ToleranceDeviationEvents(
    dataframe=df,
    tolerance_column="value_double",
    actual_column="value_double",
    upper_tolerance_uuid="upper_spec_limit",
    lower_tolerance_uuid="lower_spec_limit",
    actual_uuid="measurements",
    event_uuid="deviation_event",
    warning_threshold=0.8  # 80% of tolerance = warning zone
)

# Calculate capability indices
capability = tolerance_checker.compute_capability_indices()
print(f"Cp:  {capability['Cp']:.3f}")   # Potential capability
print(f"Cpk: {capability['Cpk']:.3f}")  # Actual capability
print(f"Mean: {capability['mean']:.2f}")
print(f"Std:  {capability['std']:.2f}")

# Interpret results
if capability['Cpk'] >= 1.33:
    print("Process is capable")
elif capability['Cpk'] >= 1.0:
    print("Process needs improvement")
else:
    print("Process is not capable")
```

### Custom Filtering with Query Syntax

Use pandas query syntax for flexible filtering.

```python
from ts_shape.transform.filter.custom_filter import CustomFilter

# Complex multi-condition filtering
df = CustomFilter.filter_custom_conditions(
    df,
    "value_double > 50 and value_double < 100 and uuid == 'temperature'"
)

# With computed expressions
df = CustomFilter.filter_custom_conditions(
    df,
    "value_double > value_double.mean() * 1.5"
)

# Multiple OR conditions
df = CustomFilter.filter_custom_conditions(
    df,
    "uuid == 'temp_1' or uuid == 'temp_2' or uuid == 'temp_3'"
)
```

### Lambda Processing

Apply custom transformations to columns.

```python
from ts_shape.transform.functions.lambda_func import LambdaProcessor
import numpy as np

# Apply custom transformations
df = LambdaProcessor.apply_function(
    df,
    "value_double",
    lambda x: np.log1p(x)  # Log transform
)

# Scale values
df = LambdaProcessor.apply_function(
    df,
    "value_double",
    lambda x: (x - x.mean()) / x.std()  # Z-score normalization
)

# Clip extreme values
df = LambdaProcessor.apply_function(
    df,
    "value_double",
    lambda x: np.clip(x, 0, 100)
)
```

---

## End-to-End Example: Multi-Machine Startup Heatup Analysis

A real-world pipeline that loads data for multiple machines from Azure,
maps each UUID to a human-readable machine name, detects heatup startups
per machine, and classifies whether each heatup started early, on time,
or late relative to the planned shift.

### Step 1 -- Define a machine registry

Map each UUID to a machine name and its detection parameters.
This replaces hard-coded UUIDs scattered through the code with a single
configuration dict that is easy to extend.

```python
import pandas as pd
from ts_shape.loader.timeseries.azure_blob_loader import AzureBlobParquetLoader
from ts_shape.transform.time_functions.timestamp_converter import TimestampConverter
from ts_shape.events.engineering.startup_events import StartupDetectionEvents

# UUID -> machine name + startup detection config
MACHINES = {
    "9cd63e77-36b4-47f6-bb27-ec27eaaf711d": {
        "name": "Curing Oven SO_17 - Temp",
        "threshold": 60.0,
        "hysteresis": (100.0, 30.0),
        "min_above": "90s",
        "shift_start": "06:00",
        "heatup_offset_min": 30,
    },
    "afe57364-05c3-43cd-a469-7b51e782006e": {
        "name": "Curing Oven SO_17 - ConvSpeed",
        "threshold": 5.0,
        "hysteresis": None,
        "min_above": "60s",
        "shift_start": "06:00",
        "heatup_offset_min": 30,
    },
    # add more machines / signals here ...
}
```

### Step 2 -- Connect and load all UUIDs in one call

```python
loader = AzureBlobParquetLoader(
    connection_string="DefaultEndpointsProtocol=https;AccountName=...;AccountKey=...",
    container_name="timeseries",
    prefix="data",
    max_workers=4,
)

# Or connect with a SAS URL instead:
# loader = AzureBlobParquetLoader(
#     sas_url="https://myaccount.blob.core.windows.net/timeseries?sv=...&sig=...",
#     prefix="data",
# )

# Fetch all machine UUIDs at once -- one round trip
df = loader.load_files_by_time_range_and_uuids(
    start_timestamp="2026-01-01 00:00",
    end_timestamp="2026-03-06 08:00",
    uuid_list=list(MACHINES.keys()),
)
print(f"Loaded {len(df)} rows across {df['uuid'].nunique()} signal(s)")
```

### Step 3 -- Convert timestamps to local timezone

```python
df = TimestampConverter.convert_to_datetime(
    dataframe=df,
    columns=["systime"],
    timezone="Europe/Bucharest",
)
```

### Step 4 -- Loop per machine: detect startups and classify timing

```python
results = []

for uuid, cfg in MACHINES.items():
    # StartupDetectionEvents filters to target_uuid internally
    detector = StartupDetectionEvents(
        dataframe=df,
        target_uuid=uuid,
        value_column="value_double",
        time_column="systime",
    )

    events = detector.detect_startup_by_threshold(
        threshold=cfg["threshold"],
        hysteresis=cfg["hysteresis"],
        min_above=cfg["min_above"],
    )

    if events.empty:
        print(f"  {cfg['name']}: no startups detected")
        continue

    # Build result with machine name and shift-relative timing
    out = events[["start", "end"]].copy()
    out["machine"]  = cfg["name"]
    out["uuid"]     = uuid
    out["start"]    = pd.to_datetime(out["start"]).dt.tz_localize(None)
    out["end"]      = pd.to_datetime(out["end"]).dt.tz_localize(None)
    out["date"]     = out["start"].dt.date

    out["shift_start"] = pd.to_datetime(
        out["date"].astype(str) + " " + cfg["shift_start"]
    )
    offset = pd.Timedelta(minutes=cfg["heatup_offset_min"])
    out["theoretical_heatup"] = out["shift_start"] - offset
    out["diff_minutes"] = (
        (out["start"] - out["theoretical_heatup"]).dt.total_seconds() / 60
    ).round(1)
    out["classification"] = pd.cut(
        out["diff_minutes"],
        bins=[-float("inf"), -5, 5, float("inf")],
        labels=["early", "on_time", "late"],
    )
    out["started_before_theoretical"] = out["diff_minutes"] < 0
    results.append(out)

df_all = pd.concat(results, ignore_index=True)
```

### Step 5 -- View results grouped by machine

```python
for machine_name, group in df_all.groupby("machine"):
    print(f"\n{'=' * 60}")
    print(f"  {machine_name}")
    print(f"{'=' * 60}")
    print(
        group[["date", "start", "theoretical_heatup", "diff_minutes", "classification"]]
        .to_string(index=False)
    )

# Output:
# ============================================================
#   Curing Oven SO_17 - ConvSpeed
# ============================================================
#        date               start   theoretical_heatup  diff_minutes classification
#  2026-01-02 2026-01-02 05:28:00  2026-01-02 05:30:00          -2.0        on_time
#  2026-01-03 2026-01-03 05:15:00  2026-01-03 05:30:00         -15.0          early
#
# ============================================================
#   Curing Oven SO_17 - Temp
# ============================================================
#        date               start   theoretical_heatup  diff_minutes classification
#  2026-01-02 2026-01-02 05:32:00  2026-01-02 05:30:00           2.0        on_time
#  2026-01-03 2026-01-03 05:42:00  2026-01-03 05:30:00          12.0           late
```

### Step 6 -- Summary statistics per machine

```python
summary = (
    df_all
    .groupby("machine")
    .agg(
        total_startups=("date", "count"),
        early=("classification", lambda s: (s == "early").sum()),
        on_time=("classification", lambda s: (s == "on_time").sum()),
        late=("classification", lambda s: (s == "late").sum()),
        avg_diff_min=("diff_minutes", "mean"),
    )
)
summary["early_pct"] = (summary["early"] / summary["total_startups"] * 100).round(1)
print(summary)
```

---

### Alternative detection methods

The examples above use threshold-based detection. The same loop pattern works
with any `StartupDetectionEvents` method -- just swap the detection call inside
the loop.

#### Slope-based (for signals where the absolute value varies)

```python
events = detector.detect_startup_by_slope(
    min_slope=0.5,         # units/second
    min_duration="20s",
)
```

#### Adaptive (auto-adjusts threshold from recent baseline)

```python
events = detector.detect_startup_adaptive(
    baseline_window="1h",
    sensitivity=2.0,       # threshold = mean + 2 * std
    min_above="10s",
)
```

#### Multi-signal (require speed AND temperature to rise together)

```python
events = detector.detect_startup_multi_signal(
    signals={
        "afe57364-05c3-43cd-a469-7b51e782006e": {
            "method": "threshold", "threshold": 5.0, "min_above": "60s"
        },
        "9cd63e77-36b4-47f6-bb27-ec27eaaf711d": {
            "method": "threshold", "threshold": 60.0, "min_above": "90s"
        },
    },
    logic="all",
    time_tolerance="30s",
)
```

#### Startup quality assessment

```python
quality = detector.assess_startup_quality(events)
print(quality[["start", "end", "duration", "smoothness_score", "stability_score"]])
```

#### Failed startup detection

```python
failed = detector.detect_failed_startups(
    threshold=60.0,
    min_rise_duration="5s",
    max_completion_time="5m",
    completion_threshold=120.0,
    required_stability="10s",
)
```

#### Startup phase tracking

```python
phases = detector.track_startup_phases(
    phases=[
        {"name": "preheat",   "condition": "threshold", "threshold": 40.0},
        {"name": "ramp_up",   "condition": "range",     "lower": 40.0, "upper": 100.0},
        {"name": "operating", "condition": "threshold", "threshold": 100.0},
    ],
    min_phase_duration="5s",
)
print(phases[["phase_name", "start", "end", "duration", "completed"]])
```

---

## Next Steps

- [Concept Guide](../concept.md) - Understand the architecture
- [API Reference](../reference/index.md) - Full API documentation
- [Contributing](../contributing.md) - Contribute to ts-shape
