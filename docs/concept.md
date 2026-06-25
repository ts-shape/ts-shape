# Concept

ts-shape is a lightweight toolkit for shaping timeseries data into analysis-ready DataFrames.

## Architecture

A layered, abstract view of the pipeline. The detection layer is intentionally pluggable — see [Lambda Rules](guides/lambda-rules.md) for the user-authored path.

```mermaid
flowchart TB
    subgraph IN["Sources"]
        S1[Time-series stores<br/><i>Parquet · S3/Azure · TimescaleDB</i>]
        S2[Object &amp; context<br/><i>batches · shifts · assets</i>]
    end
    subgraph LOAD["Load &amp; Enrich"]
        L1[Loaders]
        L2[Transforms · Features · Statistics]
    end
    subgraph DETECT["Detection Layer"]
        D1["Built-in detectors<br/>(290+ methods, 70+ classes)"]
        D2["Lambda rules<br/>(YAML / DSL)"]
        D3["Gen-AI authoring<br/><i>roadmap</i>"]
    end
    subgraph EVENTLOG["Canonical EventLog (OCEL 2.0)"]
        E1[Events]
        E2[Objects]
        E3[Relations]
    end
    subgraph OUT["Consumers"]
        O1[XES / pm4py]
        O2[OCEL viewers]
        O3[KPIs &amp; reports]
    end
    IN --> LOAD --> DETECT
    D1 --> EVENTLOG
    D2 --> EVENTLOG
    D3 -.-> D2
    EVENTLOG --> OUT
    style DETECT fill:#0f2a3d,stroke:#38bdf8,color:#e0f2fe
    style EVENTLOG fill:#3d2a0f,stroke:#fbbf24,color:#fef3c7
    style D3 stroke-dasharray: 4 3
```

## Full library architecture

For the prose explanation of each layer (input, output, when to use it)
see [Architecture](guides/architecture.md). For the interactive
full-screen graph of every package, class, and detector method see
[Architecture Map](guides/architecture-map.md).

[Open the interactive architecture map →](guides/architecture-map.md){ .md-button .md-button--primary }

## Core Principles

| Principle | Description |
|-----------|-------------|
| **DataFrame-First** | Every operation accepts and returns Pandas DataFrames |
| **Modular** | Use only what you need - all components are decoupled |
| **Composable** | Chain operations together like building blocks |
| **Consistent Schema** | Simple, predictable data structure |

## Data Model

### Timeseries DataFrame

| Column | Type | Description |
|--------|------|-------------|
| `uuid` | string | Signal/sensor identifier |
| `systime` | datetime | Timestamp (tz-aware recommended) |
| `value_double` | float | Numeric measurements |
| `value_integer` | int | Counter/integer values |
| `value_string` | string | Categorical data |
| `value_bool` | bool | Binary states |
| `is_delta` | bool | Delta vs absolute (optional) |

### Metadata DataFrame

| Column | Type | Description |
|--------|------|-------------|
| `uuid` | string | Signal identifier (join key) |
| `label` | string | Human-readable name |
| `unit` | string | Measurement unit |
| `config.*` | any | Additional configuration |

## Detector & module reference

The full, always-current catalogue of loaders, transforms, features, and
every detector class lives in the dedicated reference docs rather than
being duplicated here:

- **[Module Reference](modules/index.md)** — one hand-written page per detector
  (when to use it, quick example, key methods).
- **[API Reference](reference/index.md)** — signatures auto-generated from the source docstrings.
- **[Architecture Map](guides/architecture-map.md)** — interactive graph of every class and method.

From a REPL, `ts_shape.list_detectors("events.quality")` lists the same
catalogue programmatically.

## Pipeline Pattern

```python
# 1. LOAD
from ts_shape.loader.timeseries.parquet_loader import ParquetLoader
from ts_shape.loader.metadata.metadata_json_loader import MetadataLoader

ts_df = ParquetLoader.load_all_files("data/")
meta_df = MetadataLoader("config/signals.json").to_df()

# 2. COMBINE
from ts_shape.loader.combine.integrator import DataIntegratorHybrid

df = DataIntegratorHybrid.combine_data(
    timeseries_sources=[ts_df],
    metadata_sources=[meta_df],
    join_key="uuid"
)

# 3. TRANSFORM
from ts_shape.transform.filter.datetime_filter import DateTimeFilter
from ts_shape.transform.filter.numeric_filter import NumericFilter

df = DateTimeFilter.filter_after(df, "systime", "2024-01-01")
df = NumericFilter.filter_not_null(df, "value_double")

# 4. ANALYZE
from ts_shape.features.stats.numeric_stats import NumericStatistics
from ts_shape.events.quality.outlier_detection import OutlierDetection

stats = NumericStatistics(df, "value_double")
outliers = OutlierDetection.detect_zscore_outliers(df, "value_double", threshold=3.0)
```

## Design Decisions

### Why DataFrames?

- **Universal**: Understood by all data scientists
- **Ecosystem**: Works with matplotlib, scikit-learn, etc.
- **Debuggable**: Easy to inspect intermediate results
- **Exportable**: Save to CSV, parquet, database

### Why Modular?

- **Lightweight**: Import only what you need
- **Testable**: Each component works independently
- **Extensible**: Add custom modules easily
- **Maintainable**: Clear separation of concerns

### Why This Schema?

- **Flexible**: Not all columns required
- **Multi-type**: Handles numeric, string, boolean values
- **Joinable**: UUID enables metadata enrichment
- **Sparse-friendly**: Nulls are fine

## Extending ts-shape

### Custom Loader

```python
class MyDatabaseLoader:
    def __init__(self, connection: str):
        self.conn = connection

    def fetch_data_as_dataframe(self, start: str, end: str) -> pd.DataFrame:
        # Query database, return DataFrame with uuid, systime, value_*
        return df
```

### Custom Transform

```python
class MyFilter:
    @staticmethod
    def filter_business_hours(df: pd.DataFrame, column: str) -> pd.DataFrame:
        hours = pd.to_datetime(df[column]).dt.hour
        return df[(hours >= 9) & (hours < 17)]
```

### Custom Feature

```python
class MyMetrics:
    def __init__(self, df: pd.DataFrame, column: str):
        self.data = df[column].dropna()

    def coefficient_of_variation(self) -> float:
        return self.data.std() / self.data.mean()
```

## When to Use ts-shape

| Use Case | ts-shape? |
|----------|-----------|
| Load parquet/S3/Azure/DB into DataFrames | Yes |
| Filter and transform timeseries | Yes |
| Compute statistics on signals | Yes |
| Detect outliers and events | Yes |
| Real-time streaming | No (use Kafka/Flink) |
| Sub-millisecond latency | No (use specialized libs) |
| GPU acceleration | No (use cuDF/Rapids) |
