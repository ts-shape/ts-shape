# Data model and layers

ts-shape is a set of pandas functions and classes. There is no runtime, store
or scheduler: every step takes a DataFrame and returns one, so you can start
at any layer with data you already have.

## The input format

All signals live in one long-format DataFrame, one row per sample:

| Column | Type | Content |
|---|---|---|
| `systime` | datetime (tz-aware recommended) | sample timestamp |
| `uuid` | string | signal identifier, for example `line1:running` |
| `value_double` | float | measured values |
| `value_integer` | int | counters, codes |
| `value_bool` | bool | run states, flags |
| `value_string` | string | part numbers, order ids, state names |
| `is_delta` | bool, optional | whether the row is a change (delta) or a snapshot |

A signal uses one `value_*` column; the others stay empty. Not every column
has to be present, only `systime`, `uuid` and the value columns the
detector reads. Detectors take the column names as arguments
(`time_column`, `value_column`, ...), so other layouts work too.

Signal metadata (name, unit, configuration) is a second frame keyed by
`uuid`. [`DataIntegratorHybrid`](guides/loading.md#combining-timeseries-with-metadata)
joins the two.

What a signal *is* (a counter that resets, a change-encoded state, a sampled
measurement) changes how it must be aggregated. [Signal semantics and
profiling](guides/signals.md) describes how to declare this once as a
`SignalSpec` and let detectors follow it.

## Layers

| Layer | Package | Input | Output | Examples |
|---|---|---|---|---|
| Load | `ts_shape.loader` | parquet, S3, Azure Blob, Databricks, TimescaleDB, JSON | long-format frame | `ParquetLoader`, `AzureBlobParquetLoader`, `MetadataJsonLoader` |
| Transform | `ts_shape.transform` | long frame | long frame | `DoubleFilter`, `DateTimeFilter`, `TimezoneShift`, `DataHarmonizer` |
| Features | `ts_shape.features` | long frame | statistics, cycles, segments, feature tables | `NumericStatistics`, `CycleExtractor`, `SegmentExtractor` |
| Detect | `ts_shape.events` | long frame | event or KPI table | `MachineStateEvents`, `OEEEvents`, `OutlierDetectionEvents` |
| Event log | `ts_shape.eventlog` | detector output | OCEL 2.0 tables, XES-style frame | `to_event_log`, `to_event_log_xes`, `to_event_log_ocel` |

`ts_shape.Pipeline` chains transforms and detectors into one reusable
definition ([Pipelines](guides/pipeline-builder.md)).

## Detectors

Detectors are grouped into packs under `ts_shape.events`: `quality`,
`production`, `engineering`, `maintenance`, `energy`, `correlation`,
`supplychain` and `development`. Each detector is a class named `...Events`
that inherits from `ts_shape.utils.base.Base`. The base class:

- copies the input frame, converts the time column to datetime and sorts by it;
- converts nullable, pyarrow and text-number value columns to NumPy dtypes once;
- raises `ColumnNotFoundError` when the time or uuid column is missing, and
  warns with `DataQualityWarning` when a signal uuid you pass is not in the
  data.

You configure the detector in the constructor and call its methods:

```python
import ts_shape

df = ts_shape.make_timeseries(["line1:running"], n_points=500, freq="1min", value_column="value_bool")
state = ts_shape.MachineStateEvents(df, run_state_uuid="line1:running")
intervals = state.detect_run_idle(min_duration="2min")
```

`ts_shape.list_detectors()` lists every detector class, and
`ts_shape.list_detectors("events.quality")` one pack.

## Output shapes

Every detector method returns a DataFrame in one of four shapes:

| Shape | Required columns | Used for |
|---|---|---|
| point | `systime`, `uuid`, `source_uuid` | something that happened at an instant (an outlier, a rule violation) |
| interval | `start`, `end`, `duration_seconds`, `uuid`, `source_uuid` | something that lasted (a run, a downtime, a batch) |
| summary | `start`, `end`, `duration_seconds` | a value per time window (OEE per day, parts per shift) |
| static | none | a table not tied to time (statistics by reason, a fitted model) |

`uuid` is the event type the detector emits (for example `quality:outlier`)
and `source_uuid` the signal it was found on. Column naming, units and
error behavior are specified in [Naming and output
conventions](naming.md).

Because the shapes are fixed, any detector output can be normalized into
one event log. See [Event log](guides/eventlog/index.md).

## Extending

- A plain function `df -> df` can be a step in a [Pipeline](guides/pipeline-builder.md).
- A threshold or interval rule can be declared in YAML as a
  [lambda rule](guides/lambda-rules.md); it then works with the event log like
  a built-in detector.
- A new detector class subclasses `Base`, returns one of the shapes above and
  registers a label rule ([Adding a detector method](guides/eventlog/taxonomy.md#adding-a-new-detector-method)).

## Scope

ts-shape runs in memory on pandas. It is meant for batch analysis of
historian, data-lake and database extracts, not for stream processing or
low-latency control.
