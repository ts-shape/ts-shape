# OEE Dashboard Pipeline

> From Azure Blob timeseries to daily OEE breakdown by shift — availability, performance, and quality.

**Signals needed:**

| Role | UUID example | Type | Description |
|------|-------------|------|-------------|
| Machine state | `machine_run_state` | `value_bool` | True = running, False = idle |
| Part counter | `part_counter` | `value_integer` | Monotonic produced-parts counter |
| Total counter | `total_counter` | `value_integer` | Total parts (good + bad) |
| Reject counter | `reject_counter` | `value_integer` | Rejected parts counter |

**Modules used:** [AzureBlobParquetLoader](../reference/ts_shape/loader/timeseries/azure_blob_loader.md) | [MetadataJsonLoader](../reference/ts_shape/loader/metadata/metadata_json_loader.md) | [ContextEnricher](../reference/ts_shape/loader/context/context_enricher.md) | [DataHarmonizer](../reference/ts_shape/transform/harmonization.md) | [MachineStateEvents](../reference/ts_shape/events/production/machine_state.md) | [OEECalculator](../reference/ts_shape/events/production/oee_calculator.md) | [ShiftReporting](../reference/ts_shape/events/production/shift_reporting.md)

---

## Prerequisites

```python
# -- The only things you customize --
AZURE_CONNECTION = "DefaultEndpointsProtocol=https;AccountName=...;AccountKey=..."
CONTAINER = "timeseries-data"

UUID_LIST = [
    "machine_run_state",   # bool: True = running
    "part_counter",        # int: monotonic part counter
    "total_counter",       # int: total parts produced
    "reject_counter",      # int: rejected parts
]

START = "2024-06-01"
END   = "2024-06-08"

METADATA_PATH = "config/signal_metadata.json"

SHIFT_DEFINITIONS = {
    "day":       ("06:00", "14:00"),
    "afternoon": ("14:00", "22:00"),
    "night":     ("22:00", "06:00"),
}
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
print(f"Time range: {df['systime'].min()} to {df['systime'].max()}")
```

!!! tip "Check your data shape"
    Expect a long-format DataFrame with columns: `systime`, `uuid`, `value_bool`, `value_integer`, `value_double`, `value_string`, `is_delta`. Each row is one signal sample.

---

## Step 2: Enrich with Metadata

```python
from ts_shape.loader.metadata.metadata_json_loader import MetadataJsonLoader
from ts_shape.loader.context.context_enricher import ContextEnricher

# Load signal metadata (descriptions, units, areas)
meta = MetadataJsonLoader.from_file(METADATA_PATH)
meta_df = meta.to_df()

# Enrich timeseries with metadata columns
enricher = ContextEnricher(df)
df = enricher.enrich_with_metadata(
    meta_df,
    columns=["description", "unit", "area"],
)

print(df[["uuid", "description", "unit"]].drop_duplicates())
```

!!! info "Why enrich?"
    Metadata enrichment attaches human-readable signal names, engineering units, and plant area codes. Downstream reports use these for labeling instead of raw UUIDs.

---

## Step 3: Validate Data Quality

```python
from ts_shape.transform.harmonization import DataHarmonizer

harmonizer = DataHarmonizer(df, value_column="value_double")

# Check for time gaps per signal
gaps = harmonizer.detect_gaps(threshold="60s")
print(f"Gaps found: {len(gaps)}")
if not gaps.empty:
    print(gaps.groupby("uuid")["gap_duration"].agg(["count", "max"]))
```

!!! warning "Handle gaps before analysis"
    Gaps in the machine state signal directly affect availability calculations. If gaps are large (> 5 minutes), investigate the data source before proceeding.

```python
# Fill small gaps (under 2 minutes) with forward-fill
df_clean = harmonizer.fill_gaps(
    strategy="ffill",
    max_gap="120s",
)
```

---

## Step 4: Detect Machine States

```python
from ts_shape.events.production.machine_state import MachineStateEvents

machine = MachineStateEvents(
    dataframe=df,
    run_state_uuid="machine_run_state",
)

# Get run/idle intervals (ignore glitches under 5 seconds)
intervals = machine.detect_run_idle(min_duration="5s")
print(f"Run intervals: {(intervals['state'] == 'run').sum()}")
print(f"Idle intervals: {(intervals['state'] == 'idle').sum()}")

# Check for signal noise
metrics = machine.state_quality_metrics()
print(f"Coverage: {metrics['coverage_pct']:.1f}%")
print(f"Rapid transitions: {metrics.get('rapid_transition_count', 0)}")
```

---

## Step 5: Calculate OEE

```python
from ts_shape.events.production.oee_calculator import OEECalculator

oee = OEECalculator(df)

# Individual components
availability = oee.calculate_availability("machine_run_state")
performance = oee.calculate_performance(
    "part_counter",
    ideal_cycle_time=30.0,       # seconds per part (from engineering spec)
    run_state_uuid="machine_run_state",
)
quality = oee.calculate_quality("total_counter", "reject_counter")

print("--- Availability ---")
print(availability)
print("\n--- Performance ---")
print(performance)
print("\n--- Quality ---")
print(quality)
```

```python
# Combined daily OEE
daily_oee = oee.calculate_oee(
    run_state_uuid="machine_run_state",
    counter_uuid="part_counter",
    ideal_cycle_time=30.0,
    total_uuid="total_counter",
    reject_uuid="reject_counter",
)

print("\n--- Daily OEE ---")
print(daily_oee)
# Columns: start, end, duration_seconds, availability, performance, quality, oee
```

---

## Step 6: Shift Reports

```python
from ts_shape.events.production.shift_reporting import ShiftReporting

reporter = ShiftReporting(df, shift_definitions=SHIFT_DEFINITIONS)

# Production per shift
shift_prod = reporter.shift_production(counter_uuid="part_counter")
print(shift_prod)

# Compare shifts over the week
comparison = reporter.shift_comparison(counter_uuid="part_counter", days=7)
print(comparison)

# Check against targets
targets = {"day": 500, "afternoon": 480, "night": 450}
target_results = reporter.shift_targets(
    counter_uuid="part_counter",
    targets=targets,
)
print(target_results)
```

---

## Results

At the end of this pipeline you have:

| Output | Description | Merge key |
|--------|-------------|-----------|
| `daily_oee` | Daily OEE with A/P/Q breakdown | `start` (midnight per day) |
| `shift_prod` | Production quantity per shift | `date`, `shift` |
| `comparison` | Cross-shift performance comparison | `shift` |
| `target_results` | Target vs actual per shift | `date`, `shift` |
| `intervals` | Run/idle intervals with durations | timestamp range |

These DataFrames can be exported to CSV, fed into a dashboard tool, or merged with outputs from other pipelines (e.g., [Downtime Pareto](downtime-pareto.md) for root cause analysis).

---

## Next Steps

- Combine with [Downtime Pareto](downtime-pareto.md) to understand *why* availability drops
- Add [Quality & SPC](quality-spc.md) to break down the quality component by defect type
- Use [Cycle Time Analysis](cycle-time-analysis.md) to investigate performance losses
