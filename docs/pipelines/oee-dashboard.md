# OEE Dashboard Pipeline

> From Azure Blob timeseries to daily OEE breakdown by shift — availability, performance, and quality — in one reusable `Pipeline`.

**Signals needed:**

| Role | UUID example | Type | Description |
|------|-------------|------|-------------|
| Machine state | `machine_run_state` | `value_bool` | True = running, False = idle |
| Part counter | `part_counter` | `value_integer` | Monotonic produced-parts counter |
| Total counter | `total_counter` | `value_integer` | Total parts (good + bad) |
| Reject counter | `reject_counter` | `value_integer` | Rejected parts counter |

**Modules used:** [Pipeline](../reference/ts_shape/pipeline.md) | [AzureBlobParquetLoader](../reference/ts_shape/loader/timeseries/azure_blob_loader.md) | [MetadataJsonLoader](../reference/ts_shape/loader/metadata/metadata_json_loader.md) | [ContextEnricher](../reference/ts_shape/loader/context/context_enricher.md) | [DataHarmonizer](../reference/ts_shape/transform/harmonization.md) | [MachineStateEvents](../reference/ts_shape/events/production/machine_state.md) | [OEECalculator](../reference/ts_shape/events/production/oee_calculator.md) | [ShiftReporting](../reference/ts_shape/events/production/shift_reporting.md)

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

IDEAL_CYCLE_TIME = 30.0   # seconds per part (from engineering spec)

SHIFT_DEFINITIONS = {
    "day":       ("06:00", "14:00"),
    "afternoon": ("14:00", "22:00"),
    "night":     ("22:00", "06:00"),
}
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

!!! tip "Check your data shape"
    Expect a long-format DataFrame with columns: `systime`, `uuid`, `value_bool`, `value_integer`, `value_double`, `value_string`, `is_delta`. Each row is one signal sample.

---

## Step 2: Build the pipeline

One `Pipeline` captures the whole workflow. `.transform` steps clean the
signal; every `.detect` step branches off a KPI table and leaves the signal
untouched.

```python
from ts_shape import Pipeline
from ts_shape.loader.context.context_enricher import ContextEnricher
from ts_shape.transform.harmonization import DataHarmonizer
from ts_shape.events.production.machine_state import MachineStateEvents
from ts_shape.events.production.oee_calculator import OEECalculator
from ts_shape.events.production.shift_reporting import ShiftReporting

pipe = (
    Pipeline(name="oee-dashboard")

    # -- clean the signal --
    .transform(lambda df: ContextEnricher(df).enrich_with_metadata(
        meta_df, columns=["description", "unit", "area"]),
        name="enrich_metadata")
    .detect(DataHarmonizer, "detect_gaps", name="gaps", threshold="60s")
    .transform(DataHarmonizer, "fill_gaps", strategy="ffill", max_gap="120s")

    # -- machine state --
    .detect(MachineStateEvents, "detect_run_idle", name="intervals",
            run_state_uuid="machine_run_state", min_duration="5s")

    # -- OEE components --
    .detect(OEECalculator, "calculate_availability", name="availability",
            run_state_uuid="machine_run_state")
    .detect(OEECalculator, "calculate_performance", name="performance",
            counter_uuid="part_counter", ideal_cycle_time=IDEAL_CYCLE_TIME,
            run_state_uuid="machine_run_state")
    .detect(OEECalculator, "calculate_quality", name="quality",
            total_uuid="total_counter", reject_uuid="reject_counter")
    .detect(OEECalculator, "calculate_oee", name="daily_oee",
            run_state_uuid="machine_run_state", counter_uuid="part_counter",
            ideal_cycle_time=IDEAL_CYCLE_TIME, total_uuid="total_counter",
            reject_uuid="reject_counter")

    # -- shift reports --
    .detect(ShiftReporting, "shift_production", name="shift_prod",
            shift_definitions=SHIFT_DEFINITIONS, counter_uuid="part_counter")
    .detect(ShiftReporting, "shift_comparison", name="shift_comparison",
            shift_definitions=SHIFT_DEFINITIONS, counter_uuid="part_counter",
            days=7)
    .detect(ShiftReporting, "shift_targets", name="shift_targets",
            shift_definitions=SHIFT_DEFINITIONS, counter_uuid="part_counter",
            targets={"day": 500, "afternoon": 480, "night": 450})
)
```

Keyword arguments are routed automatically: `run_state_uuid` reaches the
detector method, `shift_definitions` reaches the `ShiftReporting` constructor.
`detect_gaps` is a `.detect` step — it reports gaps without changing the
signal — while `fill_gaps` is a `.transform` that every later step builds on.

!!! warning "Handle gaps before analysis"
    Gaps in the machine state signal directly affect availability. Inspect the
    `gaps` result first; if gaps are large (> 5 minutes), investigate the data
    source before trusting the OEE numbers.

---

## Step 3: Preview with `describe()`

```python
print(pipe.describe())
```

```
Pipeline 'oee-dashboard' (11 steps):
  0. [transform] enrich_metadata
  1. [detect   ] gaps  threshold='60s'
  2. [transform] fill_gaps  strategy='ffill', max_gap='120s'
  3. [detect   ] intervals  run_state_uuid='machine_run_state', min_duration='5s'
  4. [detect   ] availability  run_state_uuid='machine_run_state'
  5. [detect   ] performance  counter_uuid='part_counter', ideal_cycle_time=30.0, run_state_uuid='machine_run_state'
  6. [detect   ] quality  total_uuid='total_counter', reject_uuid='reject_counter'
  7. [detect   ] daily_oee  run_state_uuid='machine_run_state', counter_uuid='part_counter', ideal_cycle_time=30.0, total_uuid='total_counter', reject_uuid='reject_counter'
  8. [detect   ] shift_prod  shift_definitions={'day': ('06:00', '14:00'), 'afternoon': ('14:00', '22:00'), 'night': ('22:00', '06:00')}, counter_uuid='part_counter'
  9. [detect   ] shift_comparison  shift_definitions={'day': ('06:00', '14:00'), 'afternoon': ('14:00', '22:00'), 'night': ('22:00', '06:00')}, counter_uuid='part_counter', days=7
  10. [detect   ] shift_targets  shift_definitions={'day': ('06:00', '14:00'), 'afternoon': ('14:00', '22:00'), 'night': ('22:00', '06:00')}, counter_uuid='part_counter', targets={'day': 500, 'afternoon': 480, 'night': 450}
```

---

## Step 4: Run

```python
result = pipe.run(df)          # reusable — call .run() on any DataFrame

print(result.events["daily_oee"])
# Columns: start, end, duration_seconds, availability, performance, quality, oee

print(result.events["shift_prod"])      # production per shift
print(result.events["shift_targets"])   # target vs actual per shift
```

`result.data` holds the cleaned signal after `fill_gaps`; every KPI table is
keyed by its step name in `result.events`.

---

## Step 5: Debug with `run_steps()`

To inspect every intermediate DataFrame, use `run_steps()` instead of `run()`:

```python
intermediates = pipe.run_steps(df)

for name, step_df in intermediates.items():
    print(f"{name:20s} -> {step_df.shape[0]:>6} rows x {step_df.shape[1]} cols")
```

---

## Results

| `result.events` key | Description | Merge key |
|---------------------|-------------|-----------|
| `daily_oee` | Daily OEE with A/P/Q breakdown | `start` (midnight per day) |
| `availability` / `performance` / `quality` | Individual OEE components | `start` |
| `shift_prod` | Production quantity per shift | `date`, `shift` |
| `shift_comparison` | Cross-shift performance comparison | `shift` |
| `shift_targets` | Target vs actual per shift | `date`, `shift` |
| `intervals` | Run/idle intervals with durations | timestamp range |
| `gaps` | Detected time gaps per signal | — |

These DataFrames can be exported to CSV, fed into a dashboard tool, or merged
with outputs from other pipelines (e.g., [Downtime Pareto](downtime-pareto.md)
for root cause analysis).

---

## Next Steps

- Combine with [Downtime Pareto](downtime-pareto.md) to understand *why* availability drops
- Add [Quality & SPC](quality-spc.md) to break down the quality component by defect type
- Use [Cycle Time Analysis](cycle-time-analysis.md) to investigate performance losses
