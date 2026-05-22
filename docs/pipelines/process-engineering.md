# Process Engineering Pipeline

> From Azure Blob timeseries to setpoint adherence, startup detection, control loop health, and process stability scores — in one reusable `Pipeline`.

**Signals needed:**

| Role | UUID example | Type | Description |
|------|-------------|------|-------------|
| Setpoint | `temperature_setpoint` | `value_double` | Target value from recipe/PLC |
| Actual value | `temperature_actual` | `value_double` | Measured process value (PV) |
| Controller output | `temperature_output` | `value_double` | Control valve position / PID output |

**Modules used:** [Pipeline](../reference/ts_shape/pipeline.md) | [AzureBlobParquetLoader](../reference/ts_shape/loader/timeseries/azure_blob_loader.md) | [MetadataJsonLoader](../reference/ts_shape/loader/metadata/metadata_json_loader.md) | [ContextEnricher](../reference/ts_shape/loader/context/context_enricher.md) | [DataHarmonizer](../reference/ts_shape/transform/harmonization.md) | [SetpointChangeEvents](../reference/ts_shape/events/engineering/setpoint_events.md) | [StartupDetectionEvents](../reference/ts_shape/events/engineering/startup_events.md) | [SteadyStateDetectionEvents](../reference/ts_shape/events/engineering/steady_state_detection.md) | [ControlLoopHealthEvents](../reference/ts_shape/events/engineering/control_loop_health.md) | [ProcessStabilityIndex](../reference/ts_shape/events/engineering/process_stability_index.md)

---

## Prerequisites

```python
# -- The only things you customize --
AZURE_CONNECTION = "DefaultEndpointsProtocol=https;AccountName=...;AccountKey=..."
CONTAINER = "timeseries-data"

UUID_LIST = [
    "temperature_setpoint",   # double: target value
    "temperature_actual",     # double: process value (PV)
    "temperature_output",     # double: controller output
]

START = "2024-06-01"
END   = "2024-06-08"

METADATA_PATH = "config/signal_metadata.json"

# Process specifications
TARGET_VALUE = 100.0
UPPER_SPEC = 105.0
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
signal; every `.detect` step branches off an analysis table.

`ProcessStabilityIndex` takes a `target` argument, which collides with the
pipeline's own `target` parameter — so its three steps are wrapped in small
`df -> df` helper functions, the plain-callable step form.

```python
from ts_shape import Pipeline
from ts_shape.loader.context.context_enricher import ContextEnricher
from ts_shape.transform.harmonization import DataHarmonizer
from ts_shape.events.engineering.setpoint_events import SetpointChangeEvents
from ts_shape.events.engineering.startup_events import StartupDetectionEvents
from ts_shape.events.engineering.steady_state_detection import (
    SteadyStateDetectionEvents,
)
from ts_shape.events.engineering.control_loop_health import ControlLoopHealthEvents
from ts_shape.events.engineering.process_stability_index import ProcessStabilityIndex


def stability_scores(df):
    return ProcessStabilityIndex(
        df, signal_uuid="temperature_actual", target=TARGET_VALUE,
        upper_spec=UPPER_SPEC, lower_spec=LOWER_SPEC,
    ).stability_score(window="8h")


def stability_trend(df):
    return ProcessStabilityIndex(
        df, signal_uuid="temperature_actual", target=TARGET_VALUE,
        upper_spec=UPPER_SPEC, lower_spec=LOWER_SPEC,
    ).score_trend(window="8h")


def stability_worst_periods(df):
    return ProcessStabilityIndex(
        df, signal_uuid="temperature_actual", target=TARGET_VALUE,
        upper_spec=UPPER_SPEC, lower_spec=LOWER_SPEC,
    ).worst_periods(window="8h", n=3)


pipe = (
    Pipeline(name="process-engineering")

    # -- clean & harmonize the signal --
    .transform(lambda df: ContextEnricher(df).enrich_with_metadata(
        meta_df, columns=["description", "unit", "area"]),
        name="enrich_metadata")
    .detect(DataHarmonizer, "detect_gaps", name="gaps", threshold="10s")
    .transform(DataHarmonizer, "resample_to_uniform", freq="1s")
    .detect(DataHarmonizer, "align_asof", name="aligned",
            left_uuid="temperature_setpoint", right_uuid="temperature_actual",
            tolerance="2s", direction="nearest")

    # -- setpoint behaviour --
    .detect(SetpointChangeEvents, "detect_setpoint_steps", name="setpoint_steps",
            setpoint_uuid="temperature_setpoint", min_delta=1.0, min_hold="30s")
    .detect(SetpointChangeEvents, "time_to_settle", name="settling",
            setpoint_uuid="temperature_setpoint",
            actual_uuid="temperature_actual", settle_pct=0.02, hold="10s",
            lookahead="5min")
    .detect(SetpointChangeEvents, "overshoot_metrics", name="overshoot",
            setpoint_uuid="temperature_setpoint",
            actual_uuid="temperature_actual", window="5min")

    # -- startup & steady state --
    .detect(StartupDetectionEvents, "detect_startup_by_threshold",
            name="startups", target_uuid="temperature_actual",
            threshold=50.0, min_above="60s")
    .detect(SteadyStateDetectionEvents, "detect_steady_state",
            name="steady_intervals", signal_uuid="temperature_actual",
            window="60s", std_threshold=0.5, min_duration="120s")
    .detect(SteadyStateDetectionEvents, "detect_transient_periods",
            name="transients", signal_uuid="temperature_actual",
            window="60s", std_threshold=0.5)

    # -- control loop health --
    .detect(ControlLoopHealthEvents, "error_integrals", name="error_integrals",
            setpoint_uuid="temperature_setpoint",
            actual_uuid="temperature_actual",
            output_uuid="temperature_output", window="8h")
    .detect(ControlLoopHealthEvents, "detect_oscillation", name="oscillation",
            setpoint_uuid="temperature_setpoint",
            actual_uuid="temperature_actual",
            output_uuid="temperature_output", window="30min", min_crossings=6)
    .detect(ControlLoopHealthEvents, "output_saturation", name="saturation",
            setpoint_uuid="temperature_setpoint",
            actual_uuid="temperature_actual",
            output_uuid="temperature_output", high_limit=98.0, low_limit=2.0,
            window="8h")
    .detect(ControlLoopHealthEvents, "loop_health_summary", name="loop_health",
            setpoint_uuid="temperature_setpoint",
            actual_uuid="temperature_actual",
            output_uuid="temperature_output", window="8h")

    # -- process stability score --
    .detect(stability_scores, name="stability_scores")
    .detect(stability_trend, name="stability_trend")
    .detect(stability_worst_periods, name="worst_periods")
)
```

`resample_to_uniform` is a `.transform` — setpoint and actual signals arrive
at different rates, so every downstream detector runs on a clean uniform
grid. `detect_gaps` and `align_asof` are `.detect` steps: they produce
diagnostic tables without disturbing the working signal.

!!! tip "Why harmonize?"
    Setpoint changes only on a recipe switch; the PV updates every second.
    Resampling to a uniform grid ensures correct SP–PV alignment for control
    loop analysis.

---

## Step 3: Preview with `describe()`

```python
print(pipe.describe())
```

```
Pipeline 'process-engineering' (17 steps):
  0. [transform] enrich_metadata
  1. [detect   ] gaps  threshold='10s'
  2. [transform] resample_to_uniform  freq='1s'
  3. [detect   ] aligned  left_uuid='temperature_setpoint', right_uuid='temperature_actual', tolerance='2s', direction='nearest'
  4. [detect   ] setpoint_steps  setpoint_uuid='temperature_setpoint', min_delta=1.0, min_hold='30s'
  5. [detect   ] settling  setpoint_uuid='temperature_setpoint', actual_uuid='temperature_actual', settle_pct=0.02, hold='10s', lookahead='5min'
  6. [detect   ] overshoot  setpoint_uuid='temperature_setpoint', actual_uuid='temperature_actual', window='5min'
  7. [detect   ] startups  target_uuid='temperature_actual', threshold=50.0, min_above='60s'
  8. [detect   ] steady_intervals  signal_uuid='temperature_actual', window='60s', std_threshold=0.5, min_duration='120s'
  9. [detect   ] transients  signal_uuid='temperature_actual', window='60s', std_threshold=0.5
  10. [detect   ] error_integrals  setpoint_uuid='temperature_setpoint', actual_uuid='temperature_actual', output_uuid='temperature_output', window='8h'
  11. [detect   ] oscillation  setpoint_uuid='temperature_setpoint', actual_uuid='temperature_actual', output_uuid='temperature_output', window='30min', min_crossings=6
  12. [detect   ] saturation  setpoint_uuid='temperature_setpoint', actual_uuid='temperature_actual', output_uuid='temperature_output', high_limit=98.0, low_limit=2.0, window='8h'
  13. [detect   ] loop_health  setpoint_uuid='temperature_setpoint', actual_uuid='temperature_actual', output_uuid='temperature_output', window='8h'
  14. [detect   ] stability_scores
  15. [detect   ] stability_trend
  16. [detect   ] worst_periods
```

---

## Step 4: Run

```python
result = pipe.run(df)          # reusable — call .run() on any DataFrame

print(result.events["setpoint_steps"])    # detected setpoint changes
print(result.events["loop_health"])       # shift-level loop report card
print(result.events["stability_scores"])  # 0-100 stability score per shift
```

`result.data` holds the cleaned, uniform-grid signal; every analysis table is
keyed by its step name in `result.events`.

!!! info "Startup vs steady state"
    `startups` identifies *when* the process begins; `steady_intervals` finds
    *when* it stabilizes afterwards. Combine the two to exclude warm-up from
    your KPIs.

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

| `result.events` key | Description | Use case |
|---------------------|-------------|----------|
| `setpoint_steps` | Setpoint change events with magnitude | Recipe tracking |
| `settling` | Time-to-settle per setpoint change | Tuning assessment |
| `overshoot` | Overshoot / undershoot metrics per change | Control quality |
| `startups` | Equipment startup intervals | Startup optimization |
| `steady_intervals` / `transients` | Steady-state vs dynamic periods | Process efficiency |
| `error_integrals` | IAE/ISE/ITAE per window | Loop performance KPIs |
| `oscillation` / `saturation` | Oscillation and valve-saturation events | Tuning issues |
| `loop_health` | Shift-level loop report card | Daily loop health |
| `stability_scores` / `stability_trend` / `worst_periods` | 0-100 stability score, trend, worst windows | Daily process health |

---

## Next Steps

- Correlate setpoint changes with [Quality & SPC](quality-spc.md) to find which changes cause quality issues
- Use stability scores alongside [OEE Dashboard](oee-dashboard.md) for a complete production overview
- Feed startup times into [Cycle Time Analysis](cycle-time-analysis.md) to exclude warm-up from cycle statistics
