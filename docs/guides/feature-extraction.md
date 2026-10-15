# Cycles and segments

Two ways to cut a time series into pieces you can compare:

- **Cycles** come from a marker signal on the machine: a running bit, a trigger
  pulse, start and end events, a step number. `CycleExtractor` finds them,
  `CycleDataProcessor` attaches process data to them.
- **Segments** (periods) come from an identifier signal: an order number, a recipe,
  a batch id. `SegmentExtractor` finds them, `SegmentProcessor` cuts other signals
  into them.

Both work on long-format data (`systime`, `uuid`, `value_*`) and keep the time zone
of the input.

## Cycles

```python
import pandas as pd
from ts_shape.features.cycles.cycles_extractor import CycleExtractor
from ts_shape.features.cycles.cycle_processor import CycleDataProcessor

t = pd.date_range("2024-01-01 08:00", periods=12, freq="10s")
running = pd.DataFrame(
    {"systime": t, "uuid": "press:running",
     "value_bool": [0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]}
).astype({"value_bool": bool})
temp = pd.DataFrame({"systime": t, "uuid": "press:temp", "value_double": range(12)})

cycles = CycleExtractor(running, "press:running").process_persistent_cycle()
print(cycles[["cycle_start", "cycle_end", "is_complete"]])

proc = CycleDataProcessor(cycles, temp)
stats = proc.compute_cycle_statistics()
print(stats[["duration_seconds", "sample_count", "mean_value_double"]])
```

The first cycle runs 08:00:10–08:00:40 and holds temperatures 1, 2, 3 (mean 2.0).
The last cycle starts at 08:01:40 and has no end yet: it is reported with
`cycle_end` = NaT and `is_complete` = False, and gets no samples.

### Choosing the extraction method

| Method | Cycle | Use when |
|---|---|---|
| `process_persistent_cycle()` | rising edge → next falling edge of a bool | a "running" / "in cycle" bit |
| `process_trigger_cycle()` | rising edge → next rising edge | one pulse per cycle start; a signal that is never False is read as an event log (every True row is a trigger) |
| `process_separate_start_end_cycle()` | True events of `start_uuid` → True events of `end_uuid` | separate start and done signals |
| `process_step_sequence(start_step, end_step)` | entering `start_step` → entering `end_step` | a step-chain integer |
| `process_state_change_cycle()` | every sample is a boundary | each reading marks a new cycle |
| `process_value_change_cycle()` | a change larger than `value_change_threshold` is a boundary | a part number, recipe or setpoint that changes per cycle |

`suggest_method()` looks at the start signal and proposes a method.

### Rules every method follows

- Only the rows of `start_uuid` (and `end_uuid`) are read; other signals in the frame
  are ignored.
- An end closes the cycle open at that moment. At one timestamp the end is applied
  before the start, so back-to-back cycles can share a timestamp.
- A second start while a cycle is open aborts the open one: it stays in the result
  with `cycle_end` = NaT and `is_complete` = False. Ends without an open cycle are
  ignored and counted in `get_extraction_stats()["unmatched_ends"]`.
- A cycle that was already running at the first sample keeps its observed end but
  has `is_complete` = False (`partial_first_cycles` in the stats).
- A missing value keeps the previous state; it never starts or ends a cycle.
- `process_value_change_cycle` compares each sample with the previous one, so a slow
  drift in steps below the threshold never makes a boundary.
- `cycle_uuid` is deterministic: the same data gives the same ids in every run and
  in a longer query window, so a reference cycle id can be stored.

Output columns: `cycle_start`, `cycle_end`, `cycle_uuid`, `is_complete`, plus
`start`, `end`, `duration_seconds`, `uuid` (`"cycles:<method>"`) and `source_uuid`
(the start signal).

`validate_cycles(cycles, min_duration="1s", max_duration="1h")` adds `is_valid` and
`validation_issue` (`"too_short;"`, `"too_long;"` or `"incomplete_cycle"`).
`detect_overlapping_cycles(cycles, resolve="flag")` flags cycles that start before
another ends; `"keep_first"`, `"keep_last"` and `"keep_longest"` drop the clashing
ones.

### Data per cycle

`CycleDataProcessor(cycles, values)` assigns a sample to a cycle when
`cycle_start <= systime < cycle_end`. A sample at a shared boundary goes to the later
cycle only, so each sample is counted once. Matching is by time only: pass the
values of the machine the cycles belong to.

| Method | Returns |
|---|---|
| `merge_dataframes_by_cycle()` | the in-cycle samples with a `cycle_uuid` column |
| `split_by_cycle()` | `{cycle_uuid: samples}` in chronological order |
| `compute_cycle_statistics()` | one row per cycle: `duration_seconds`, `sample_count`, `value_count`, `mean_value_double`, `std_value_double` |
| `compare_cycles(reference_cycle_uuid, value_column)` | each cycle's mean and spread against a reference cycle |
| `identify_golden_cycles(value_column, method, top_n, target=None)` | the best cycle ids by `"low_variability"`, `"high_mean"` or `"target_value"` (needs `target`) |

## Segments (periods)

```python
import pandas as pd
from ts_shape.features.segment_analysis.segment_extractor import SegmentExtractor
from ts_shape.features.segment_analysis.segment_processor import SegmentProcessor

t = pd.date_range("2024-01-01 06:00", periods=6, freq="1h")
order = pd.DataFrame(
    {"systime": t[[0, 2, 4]], "uuid": "line:order", "value_string": ["A", "B", "C"]}
)
temp = pd.DataFrame({"systime": t, "uuid": "line:temp", "value_double": range(6)})
df = pd.concat([order, temp], ignore_index=True)

ranges = SegmentExtractor.extract_time_ranges(df, "line:order")
print(ranges[["segment_value", "start", "end"]])
cut = SegmentProcessor.apply_ranges(df, ranges, target_uuids=["line:temp"])
print(cut.groupby("segment_value")["value_double"].sum())
```

Order A covers 06:00–08:00 (temperatures 0 and 1), B 08:00–10:00 (2 and 3), and C,
the last order, runs to the end of the data at 11:00 (4 and 5).

### Rules

- One segment per run of equal values. A value that is sent again stays one segment;
  a value that comes back later is a new segment with a new `segment_index`.
- A segment ends where the next one starts (`interval_end="next_change"`, the
  default). The last segment ends at `data_end`, by default the last timestamp in the
  frame. With `interval_end="last_sample"` each segment ends at its own last sample.
- Missing ids (NaN / None) do not change the segment. Empty or whitespace ids mean
  "no order": the time until the next id belongs to no segment.
- `apply_ranges` cuts half-open, `start <= t < end`: a sample exactly on a boundary
  goes to the segment that starts there, and every row lands in at most one segment.
  The sample at the very end is included only for an end-inclusive segment (the last
  one, or `last_sample` segments). Rows before the first segment or in a gap are
  dropped.
- `min_duration` drops segments shorter than the given duration.

`SegmentProcessor.compute_metric_profiles(cut)` returns the descriptive statistics
of every signal per segment. `TimeWindowedFeatureTable` does the same per segment
and time window (`compute_long`) or as a wide feature table (`compute`).

For a continuous web (paper, steel strip, film) cut into pieces by a cutter, see
[`ContinuousProcessAlignmentEvents.segment_by_cut`](../modules/production/continuous-process-alignment.md).
