# Cycle Time Analysis Pipeline

> From Azure Blob timeseries to cycle time statistics, slow cycle detection, trend analysis, and cycle comparison — in one reusable `Pipeline`.

**Signals needed:**

| Role | UUID example | Type | Description |
|------|-------------|------|-------------|
| Cycle trigger | `cycle_complete` | `value_bool` | Rising edge (False to True) marks cycle end |
| Part number | `part_number_signal` | `value_string` | Current part type being produced |
| Machine state | `machine_run_state` | `value_bool` | True = running (optional, for filtering) |

**Modules used:** [Pipeline](../reference/ts_shape/pipeline.md) | [AzureBlobParquetLoader](../reference/ts_shape/loader/timeseries/azure_blob_loader.md) | [MetadataJsonLoader](../reference/ts_shape/loader/metadata/metadata_json_loader.md) | [ContextEnricher](../reference/ts_shape/loader/context/context_enricher.md) | [DataHarmonizer](../reference/ts_shape/transform/harmonization.md) | [CycleTimeTracking](../reference/ts_shape/events/production/cycle_time_tracking.md) | [CycleExtractor](../reference/ts_shape/features/cycles/cycles_extractor.md) | [CycleDataProcessor](../reference/ts_shape/features/cycles/cycle_processor.md)

---

## Prerequisites

```python
# -- The only things you customize --
AZURE_CONNECTION = "DefaultEndpointsProtocol=https;AccountName=...;AccountKey=..."
CONTAINER = "timeseries-data"

UUID_LIST = [
    "cycle_complete",       # bool: rising edge = cycle end
    "part_number_signal",   # string: current part type
    "machine_run_state",    # bool: machine running (optional)
]

START = "2024-06-01"
END   = "2024-06-08"

METADATA_PATH = "config/signal_metadata.json"

TRIGGER_UUID = "cycle_complete"
PART_UUID    = "part_number_signal"
TREND_PART   = "PART_A"        # part type to trend

# A cycle_uuid (from a prior extraction run) used as the comparison reference
REFERENCE_CYCLE_UUID = "a1b2c3d4-0000-0000-0000-000000000000"
```

!!! info "New to `Pipeline`?"
    Read the [Pipeline guide](../guides/pipeline-builder.md) first — it explains
    `.transform` vs `.detect` steps, the `$input` sentinel, and the debugging
    tools used below.

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

The pipeline runs in two stages on one signal. The `CycleTimeTracking`
detectors run first against the enriched raw signal. Then `extract_cycles` —
a `.transform` — reshapes the working signal into a validated cycle table,
which the final `compare_cycles` step consumes.

`extract_cycles` wraps `CycleExtractor` in a small `df -> df` helper so the
same extractor instance both builds and validates the cycles.

```python
from ts_shape import Pipeline
from ts_shape.loader.context.context_enricher import ContextEnricher
from ts_shape.transform.harmonization import DataHarmonizer
from ts_shape.events.production.cycle_time_tracking import CycleTimeTracking
from ts_shape.features.cycles.cycles_extractor import CycleExtractor
from ts_shape.features.cycles.cycle_processor import CycleDataProcessor


def extract_cycles(df):
    extractor = CycleExtractor(
        df[df["uuid"] == TRIGGER_UUID].copy(), start_uuid=TRIGGER_UUID
    )
    cycles = extractor.process_trigger_cycle()
    return extractor.validate_cycles(
        cycles, min_duration="10s", max_duration="10min"
    )


pipe = (
    Pipeline(name="cycle-time")

    # -- clean the signal --
    .transform(lambda df: ContextEnricher(df).enrich_with_metadata(
        meta_df, columns=["description", "unit"]),
        name="enrich_metadata")
    .detect(DataHarmonizer, "detect_gaps", name="gaps", threshold="30s")

    # -- cycle time analysis on the raw signal --
    .detect(CycleTimeTracking, "cycle_time_by_part", name="cycles",
            part_id_uuid=PART_UUID, cycle_trigger_uuid=TRIGGER_UUID)
    .detect(CycleTimeTracking, "cycle_time_statistics", name="stats",
            part_id_uuid=PART_UUID, cycle_trigger_uuid=TRIGGER_UUID)
    .detect(CycleTimeTracking, "detect_slow_cycles", name="slow_cycles",
            part_id_uuid=PART_UUID, cycle_trigger_uuid=TRIGGER_UUID,
            threshold_factor=1.5)
    .detect(CycleTimeTracking, "cycle_time_trend", name="trend",
            part_id_uuid=PART_UUID, cycle_trigger_uuid=TRIGGER_UUID,
            part_number=TREND_PART, window_size=20)

    # -- extract cycles, then compare against a reference --
    .transform(extract_cycles, name="extract_cycles")
    .detect(CycleDataProcessor, "compare_cycles", name="comparison",
            values_df="$input", reference_cycle_uuid=REFERENCE_CYCLE_UUID)
)
```

The `compare_cycles` step runs after `extract_cycles`, so the working signal
is the validated cycle table — passed to the `CycleDataProcessor` constructor
as `cycles_df`. The `$input` sentinel feeds the original raw DataFrame in as
`values_df`.

!!! warning "Cycle trigger gaps"
    Missing samples in the cycle trigger signal create phantom long cycles.
    Inspect the `gaps` result before trusting the cycle statistics.

---

## Step 3: Preview with `describe()`

```python
print(pipe.describe())
```

```
Pipeline 'cycle-time' (8 steps):
  0. [transform] enrich_metadata
  1. [detect   ] gaps  threshold='30s'
  2. [detect   ] cycles  part_id_uuid='part_number_signal', cycle_trigger_uuid='cycle_complete'
  3. [detect   ] stats  part_id_uuid='part_number_signal', cycle_trigger_uuid='cycle_complete'
  4. [detect   ] slow_cycles  part_id_uuid='part_number_signal', cycle_trigger_uuid='cycle_complete', threshold_factor=1.5
  5. [detect   ] trend  part_id_uuid='part_number_signal', cycle_trigger_uuid='cycle_complete', part_number='PART_A', window_size=20
  6. [transform] extract_cycles
  7. [detect   ] comparison  values_df='$input', reference_cycle_uuid='a1b2c3d4-0000-0000-0000-000000000000'
```

---

## Step 4: Run

```python
result = pipe.run(df)          # reusable — call .run() on any DataFrame

print(result.events["stats"])        # mean/median/std per part type
print(result.events["slow_cycles"])  # cycles exceeding 1.5x median
print(result.events["trend"])        # rolling-window trend for PART_A
```

`result.data` holds the validated cycle table (the output of `extract_cycles`);
every analysis table is keyed by its step name in `result.events`.

!!! tip "Pick a reference cycle"
    Run the pipeline once, inspect `result.data` for a representative cycle,
    and copy its `cycle_uuid` into `REFERENCE_CYCLE_UUID` to make `comparison`
    meaningful on the next run.

---

## Step 5: Debug with `run_steps()`

To inspect every intermediate DataFrame, use `run_steps()` instead of `run()`:

```python
intermediates = pipe.run_steps(df)

for name, step_df in intermediates.items():
    print(f"{name:18s} -> {step_df.shape[0]:>6} rows x {step_df.shape[1]} cols")
```

---

## Results

| `result.events` key | Description | Use case |
|---------------------|-------------|----------|
| `cycles` | Per-cycle times with part numbers | Raw cycle data |
| `stats` | Mean, median, std per part type | Capacity planning |
| `slow_cycles` | Cycles exceeding threshold | Loss investigation |
| `trend` | Rolling average + direction | Drift detection |
| `comparison` | Cycle-to-reference comparison | Quality benchmarking |
| `gaps` | Detected time gaps per signal | Data trust check |

---

## Next Steps

- Feed slow cycle timestamps into [Downtime Pareto](downtime-pareto.md) to correlate with machine stops
- Use cycle statistics to set the `ideal_cycle_time` parameter in [OEE Dashboard](oee-dashboard.md)
- Combine with [Quality & SPC](quality-spc.md) to correlate cycle time outliers with quality defects
