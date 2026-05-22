# Downtime Pareto Analysis Pipeline

> From Azure Blob timeseries to Pareto-ranked downtime reasons, shift-level comparison, and availability trends — in one reusable `Pipeline`.

**Signals needed:**

| Role | UUID example | Type | Description |
|------|-------------|------|-------------|
| Machine state | `machine_state_str` | `value_string` | "Running", "Stopped", "Idle" |
| Downtime reason | `downtime_reason` | `value_string` | Reason code (e.g., "Tool_Change", "Material_Shortage") |

**Modules used:** [Pipeline](../reference/ts_shape/pipeline.md) | [AzureBlobParquetLoader](../reference/ts_shape/loader/timeseries/azure_blob_loader.md) | [MetadataJsonLoader](../reference/ts_shape/loader/metadata/metadata_json_loader.md) | [ContextEnricher](../reference/ts_shape/loader/context/context_enricher.md) | [DataHarmonizer](../reference/ts_shape/transform/harmonization.md) | [DowntimeTracking](../reference/ts_shape/events/production/downtime_tracking.md)

---

## Prerequisites

```python
# -- The only things you customize --
AZURE_CONNECTION = "DefaultEndpointsProtocol=https;AccountName=...;AccountKey=..."
CONTAINER = "timeseries-data"

UUID_LIST = [
    "machine_state_str",   # string: Running/Stopped/Idle
    "downtime_reason",     # string: reason code
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

print(f"Loaded {len(df):,} rows")
print(f"Unique reason codes: {df[df['uuid'] == 'downtime_reason']['value_string'].nunique()}")
```

---

## Step 2: Build the pipeline

One `Pipeline` captures the whole workflow. `.transform` steps clean the
signal; every `.detect` step branches off a downtime table.

```python
from ts_shape import Pipeline
from ts_shape.loader.context.context_enricher import ContextEnricher
from ts_shape.transform.harmonization import DataHarmonizer
from ts_shape.events.production.downtime_tracking import DowntimeTracking

pipe = (
    Pipeline(name="downtime-pareto")

    # -- clean the signal --
    .transform(lambda df: ContextEnricher(df).enrich_with_metadata(
        meta_df, columns=["description", "area"]),
        name="enrich_metadata")
    .detect(DataHarmonizer, "detect_gaps", name="gaps", threshold="60s")
    .transform(DataHarmonizer, "fill_gaps", strategy="ffill", max_gap="120s")

    # -- downtime analysis --
    .detect(DowntimeTracking, "downtime_by_shift", name="shift_downtime",
            shift_definitions=SHIFT_DEFINITIONS,
            state_uuid="machine_state_str", running_value="Running")
    .detect(DowntimeTracking, "downtime_by_reason", name="reason_analysis",
            shift_definitions=SHIFT_DEFINITIONS,
            state_uuid="machine_state_str", reason_uuid="downtime_reason",
            stopped_value="Stopped")
    .detect(DowntimeTracking, "top_downtime_reasons", name="top_reasons",
            shift_definitions=SHIFT_DEFINITIONS,
            state_uuid="machine_state_str", reason_uuid="downtime_reason",
            top_n=5, stopped_value="Stopped")
    .detect(DowntimeTracking, "availability_trend", name="availability",
            shift_definitions=SHIFT_DEFINITIONS,
            state_uuid="machine_state_str", running_value="Running",
            window="1D")
)
```

`shift_definitions` is routed to the `DowntimeTracking` constructor; the
remaining keyword arguments reach each method.

!!! tip "State signal continuity"
    The machine state signal should be continuous (no gaps). Gaps are ambiguous
    — was the machine running or stopped? `fill_gaps` with `ffill` covers short
    gaps; inspect the `gaps` result for long ones.

---

## Step 3: Preview with `describe()`

```python
print(pipe.describe())
```

```
Pipeline 'downtime-pareto' (7 steps):
  0. [transform] enrich_metadata
  1. [detect   ] gaps  threshold='60s'
  2. [transform] fill_gaps  strategy='ffill', max_gap='120s'
  3. [detect   ] shift_downtime  shift_definitions={'day': ('06:00', '14:00'), 'afternoon': ('14:00', '22:00'), 'night': ('22:00', '06:00')}, state_uuid='machine_state_str', running_value='Running'
  4. [detect   ] reason_analysis  shift_definitions={'day': ('06:00', '14:00'), 'afternoon': ('14:00', '22:00'), 'night': ('22:00', '06:00')}, state_uuid='machine_state_str', reason_uuid='downtime_reason', stopped_value='Stopped'
  5. [detect   ] top_reasons  shift_definitions={'day': ('06:00', '14:00'), 'afternoon': ('14:00', '22:00'), 'night': ('22:00', '06:00')}, state_uuid='machine_state_str', reason_uuid='downtime_reason', top_n=5, stopped_value='Stopped'
  6. [detect   ] availability  shift_definitions={'day': ('06:00', '14:00'), 'afternoon': ('14:00', '22:00'), 'night': ('22:00', '06:00')}, state_uuid='machine_state_str', running_value='Running', window='1D'
```

---

## Step 4: Run

```python
result = pipe.run(df)          # reusable — call .run() on any DataFrame

print(result.events["top_reasons"])
# Columns: reason, total_minutes, occurrence_count, pct_of_total, cumulative_pct

print(result.events["shift_downtime"])   # downtime minutes per shift
print(result.events["availability"])     # daily availability trend
```

!!! info "The 80/20 rule"
    In most plants, 2-3 reason codes account for 80% of downtime. The
    `cumulative_pct` column in `top_reasons` shows where that line falls —
    focus improvement efforts there first.

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
| `shift_downtime` | Downtime minutes and availability per shift | `date`, `shift` |
| `reason_analysis` | Downtime broken down by reason code | `reason` |
| `top_reasons` | Pareto-ranked reasons with cumulative % | `reason` |
| `availability` | Daily availability trend | `period` |
| `gaps` | Detected time gaps per signal | — |

!!! tip "Merge with production data"
    Join `result.events["shift_downtime"]` with the [OEE Dashboard](oee-dashboard.md)
    `shift_prod` table on `[date, shift]` for a complete shift handover report:
    ```python
    report = oee_result.events["shift_prod"].merge(
        result.events["shift_downtime"], on=["date", "shift"])
    ```

---

## Next Steps

- Merge shift downtime with [OEE Dashboard](oee-dashboard.md) results for full shift reports
- Correlate top reasons with [Cycle Time Analysis](cycle-time-analysis.md) slow cycles
- Add [Quality & SPC](quality-spc.md) to check if downtime reasons correlate with quality issues
