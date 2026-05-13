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
        D1["Built-in detectors<br/>(290 methods, 68 classes)"]
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

A more detailed map showing every top-level package, the key classes inside each, and how they hand data to one another. The five top rows mirror the abstract pipeline above; the bottom row is the canonical event log + exporters that every detector funnels into. For how a single detection actually flows through these layers, see [Event Handling — Visual Overview](guides/event-handling-flow.md).

```mermaid
flowchart TB
    subgraph LOADER["ts_shape.loader — data acquisition"]
        direction LR
        L_TS["<b>timeseries/</b><br/>ParquetLoader · S3Loader<br/>AzureBlobLoader · TimescaleLoader"]
        L_MD["<b>metadata/</b><br/>MetadataJsonLoader<br/>MetadataCsvLoader"]
        L_CX["<b>context/</b><br/>ContextLoader · BatchContextLoader"]
        L_CM["<b>combine/</b><br/>JoinLoaders · MergeOnTime"]
    end

    subgraph TRANSFORM["ts_shape.transform — signal conditioning"]
        direction LR
        T_FN["<b>functions/</b><br/>resample · interpolate · clip"]
        T_FT["<b>filter/</b><br/>RangeFilter · DeltaFilter<br/>NullFilter · UuidFilter"]
        T_CA["<b>calculator/</b><br/>UnitConvert · Derivatives<br/>SignalArithmetic"]
        T_TM["<b>time_functions/</b><br/>shift · align · gap_fill"]
        T_HM["harmonization.py"]
    end

    subgraph FEATURE["ts_shape.features — analytics"]
        direction LR
        F_ST["<b>stats/</b><br/>WindowedStats · RollingAgg"]
        F_TS["<b>time_stats/</b><br/>seasonal · trend · changepoint"]
        F_CY["<b>cycles/</b><br/>CycleExtractor · CycleStats"]
        F_SE["<b>segment_analysis/</b><br/>regime detection"]
        F_CR["<b>cross_signal.py</b> · <b>pattern_recognition.py</b>"]
    end

    subgraph CONTEXT["ts_shape.context"]
        CX["value_mapping.py — categorical lookups, code → label"]
    end

    subgraph EVENTS["ts_shape.events — detector library (290 methods, 68 classes)"]
        direction LR
        EV_Q["<b>quality/</b> · 11 classes<br/>OutlierDetection · SPC · ToleranceDeviation<br/>SensorDrift · GaugeRepeatability · ValueDistribution<br/>AnomalyClassification · CapabilityTrending<br/>DataGapAnalysis · MultiSensorValidation · SignalQuality"]
        EV_P["<b>production/</b> · 30 classes<br/>MachineState · OEECalculator · DowntimeTracking<br/>CycleTime · ChangeoverEvents · BatchTracking<br/>QualityTracking · ScrapTracking · ShiftReporting<br/>OrderTraceability · …"]
        EV_E["<b>engineering/</b> · 12 classes<br/>ControlLoopHealth · ProcessStability<br/>SetpointChange · StartupDetection · SteadyState<br/>ThresholdMonitoring · WarmUpCoolDown · …"]
        EV_M["<b>maintenance/</b> · 3 classes<br/>DegradationDetection<br/>FailurePrediction · VibrationAnalysis"]
        EV_EN["<b>energy/</b> · 5 classes<br/>EnergyConsumption · EnergyEfficiency<br/>CarbonIntensity · IdleEnergyDetection<br/>EnergyPerformanceIndicator"]
        EV_CO["<b>correlation/</b> · 2 classes<br/>SignalCorrelation · AnomalyCorrelation"]
        EV_SC["<b>supplychain/</b> · 3 classes<br/>DemandPattern · InventoryMonitoring<br/>LeadTimeAnalysis"]
    end

    subgraph EVENTLOG["ts_shape.eventlog — canonical OCEL 2.0 layer"]
        direction LR
        EL_M["<b>model.py</b><br/>EventLog (events · objects · relations)"]
        EL_S["<b>schema.py</b><br/>OCEL columns · STANDARD_ATTR_KEYS<br/>register_object_type · validate"]
        EL_T["<b>taxonomy.py</b><br/>LabelRule · REGISTRY (290+ entries)"]
        EL_A["<b>adapters.py</b> · <b>archetypes.py</b><br/>shape-driven adapter · ARCHETYPE_BY_METHOD"]
        EL_N["<b>normalizer.py</b><br/>to_event_log(df, detector=…)"]
        EL_FO["<b>flat.py · ocel.py · concat.py</b><br/>to_flat_df (XES) · to_ocel_tables · concat"]
        EL_LR["<b>lambda_rules/</b><br/>RuleSpec · TriggerSpec · LambdaDetector<br/>compile_expression (AST-restricted)<br/>register_lambda_rule · load_yaml<br/>run_backtest"]
    end

    subgraph UTILS["ts_shape.utils"]
        U["base.py — Base class (get_dataframe, common ctor)"]
    end

    subgraph CONSUMERS["Consumers"]
        direction LR
        C_XES["pm4py / Disco / Celonis<br/>(XES)"]
        C_OCEL["OCEL 2.0 viewers"]
        C_KPI["BI · dashboards · reports"]
    end

    LOADER --> TRANSFORM --> FEATURE
    FEATURE --> EVENTS
    CONTEXT -.enriches.-> EVENTS
    U -.base class.-> EVENTS
    EVENTS -.legacy DataFrames.-> EL_N
    EL_T --> EL_N
    EL_A --> EL_N
    EL_S --> EL_N
    EL_N --> EL_M
    EL_LR -.registers in.-> EL_T
    EL_LR -.runs through.-> EL_N
    EL_M --> EL_FO
    EL_FO --> CONSUMERS

    style LOADER fill:#0f2a3d,stroke:#38bdf8,color:#e0f2fe
    style TRANSFORM fill:#1a3a4a,stroke:#2dd4bf,color:#ccfbf1
    style FEATURE fill:#1a3a4a,stroke:#2dd4bf,color:#ccfbf1
    style CONTEXT fill:#1a3a4a,stroke:#2dd4bf,color:#ccfbf1
    style EVENTS fill:#3d2a0f,stroke:#f59e0b,color:#fef3c7
    style EVENTLOG fill:#3d2a0f,stroke:#fbbf24,color:#fef3c7
    style UTILS fill:#27272a,stroke:#a1a1aa,color:#e4e4e7
    style CONSUMERS fill:#14532d,stroke:#22c55e,color:#dcfce7
    style EL_LR stroke:#a78bfa,stroke-width:2px
```

**How to read it.**

- **Top to bottom** = data direction. A pipeline reads `loader → transform → feature → events → eventlog`.
- **`utils.base.Base`** is the lone shared parent for nearly every detector class (provides `__init__(dataframe, column_names, ...)` and `get_dataframe()`). Dashed arrow.
- **`eventlog/`** is the convergence point — every detector method maps to a `LabelRule` in `REGISTRY` and is normalized by the shape-driven adapter into the same `EventLog` schema.
- **`lambda_rules/`** (purple-bordered) coexists with the 290 built-in detectors. It registers dynamically in `REGISTRY`, then runs through `normalizer.to_event_log` exactly like a hand-coded class.
- **No detector class imports another detector class.** All cross-detector composition happens at the `EventLog` level via `concat()`.

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

## Module Reference

### Loaders

| Module | Source | Method |
|--------|--------|--------|
| `ParquetLoader` | Local/remote parquet | `load_all_files()` |
| `S3ProxyParquetLoader` | S3-compatible storage | `fetch_data_as_dataframe()` |
| `AzureBlobLoader` | Azure Blob containers | `fetch_data_as_dataframe()` |
| `TimescaleLoader` | TimescaleDB | `fetch_data_as_dataframe()` |
| `MetadataLoader` | JSON files | `to_df()` |

### Transforms

| Module | Purpose |
|--------|---------|
| `NumericFilter` | Filter by numeric range, null handling |
| `StringFilter` | Pattern matching, contains, regex |
| `DateTimeFilter` | Time range, weekday, hour filters |
| `BooleanFilter` | Flag-based row filtering |
| `CustomFilter` | Flexible pandas query syntax filtering |
| `NumericCalc` | Derived columns, calculations |
| `TimezoneShift` | Convert between timezones |
| `TimestampConverter` | Parse/format timestamps |
| `LambdaProcessor` | Apply custom functions to columns |

### Features

| Module | Output |
|--------|--------|
| `NumericStatistics` | min, max, mean, std, percentiles |
| `TimestampStats` | first, last, count, coverage |
| `StringStatistics` | value counts, cardinality |
| `CycleExtractor` | Cycle detection, validation, method suggestion |

### Events - Quality

| Module | Detection |
|--------|-----------|
| `OutlierDetection` | Z-score, IQR, MAD, IsolationForest |
| `StatisticalProcessControl` | Western Electric Rules, CUSUM shifts |
| `ToleranceDeviation` | Specification violations, Cp/Cpk indices |

### Events - Production (Event Detection)

| Module | Purpose |
|--------|---------|
| `MachineStateEvents` | Run/idle intervals, state transitions, rapid change detection |
| `LineThroughputEvents` | Parts per window, takt adherence, throughput trends |
| `ChangeoverEvents` | Product changeover detection, stability-based windows |
| `FlowConstraintEvents` | Blocked/starved detection between stations |

### Events - Production (Daily Tracking)

| Module | Purpose |
|--------|---------|
| `PartProductionTracking` | Production by part, daily summaries, totals |
| `CycleTimeTracking` | Cycle times, slow cycles, trends |
| `DowntimeTracking` | Downtime by shift/reason, availability |
| `QualityTracking` | NOK/scrap analysis, FPY, defect reasons |
| `ShiftReporting` | Shift production, targets, comparisons |

### Events - Production (OEE & Advanced)

| Module | Purpose |
|--------|---------|
| `OEECalculator` | Availability x Performance x Quality — daily OEE |
| `AlarmManagementEvents` | ISA-18.2 alarm analysis, chattering, standing alarms |
| `BatchTrackingEvents` | Batch detection, duration stats, yield, transition matrix |
| `BottleneckDetectionEvents` | Station utilization, shifting bottleneck detection |
| `MicroStopEvents` | Brief idle intervals, frequency, impact analysis |
| `DutyCycleEvents` | On/off patterns, excessive cycling detection |

### Events - Production (Traceability)

| Module | Purpose |
|--------|---------|
| `ValueTraceabilityEvents` | Single ID tracking across stations |
| `RoutingTraceabilityEvents` | ID + routing/state signal correlation, lead times |
| `MultiProcessTraceabilityEvents` | Parallel process lines with handover events |

### Events - Production (Performance & Targets)

| Module | Purpose |
|--------|---------|
| `PerformanceLossTracking` | Speed loss vs target cycle time per shift |
| `ScrapTracking` | Material waste by shift, reason, and monetary cost |
| `TargetTracking` | Actual vs target comparison, hit rate |
| `SetupTimeTracking` | SMED analysis — setup durations by product transition |
| `OperatorPerformanceTracking` | Operator output, efficiency, quality comparison |
| `ReworkTracking` | Rework rates by shift/reason, cost analysis |

### Events - Production (Reporting)

| Module | Purpose |
|--------|---------|
| `ShiftHandoverReport` | Auto-generated shift reports with issue highlighting |
| `PeriodSummary` | Weekly/monthly aggregation, period comparison |

### Events - Engineering

| Module | Purpose |
|--------|---------|
| `SetpointChangeEvents` | Step/ramp detection, settling, overshoot |
| `StartupDetectionEvents` | Threshold, slope, multi-signal startup detection |
| `ThresholdMonitoringEvents` | Multi-level thresholds with hysteresis |
| `RateOfChangeEvents` | Rapid change and step jump detection |
| `SteadyStateDetectionEvents` | Steady vs transient period segmentation |
| `SignalComparisonEvents` | Setpoint vs actual divergence |
| `OperatingRangeEvents` | Operating envelope and regime change |
| `WarmUpCoolDownEvents` | Warm-up/cool-down curve analysis |
| `ProcessWindowEvents` | Windowed statistics for shift monitoring |
| `ControlLoopHealthEvents` | PID loop health, oscillation, saturation |
| `DisturbanceRecoveryEvents` | Upset detection and recovery time |
| `MaterialBalanceEvents` | Input/output balance checks |
| `ProcessStabilityIndex` | Composite 0-100 stability score |

## Advanced Capabilities

### Quality & SPC

| Feature | Module | Method |
|---------|--------|--------|
| CUSUM Shift Detection | `StatisticalProcessControl` | `detect_cusum_shifts()` |
| Western Electric Rules | `StatisticalProcessControl` | `apply_rules_vectorized()` |
| Rule Interpretations | `StatisticalProcessControl` | `interpret_violations()` |
| Dynamic Control Limits | `StatisticalProcessControl` | `calculate_dynamic_control_limits()` |
| Process Capability (Cp/Cpk) | `ToleranceDeviation` | `compute_capability_indices()` |

### Outlier Detection Methods

| Method | Description | Best For |
|--------|-------------|----------|
| Z-score | Distance from mean in std units | Normal distributions |
| IQR | Interquartile range based | Skewed distributions |
| MAD | Median Absolute Deviation | Robust to extremes |
| IsolationForest | ML-based anomaly detection | Complex patterns |

### Cycle Analysis

| Feature | Method | Description |
|---------|--------|-------------|
| Method Suggestion | `suggest_method()` | Auto-detect best extraction method |
| Cycle Validation | `validate_cycles()` | Validate duration constraints |
| Overlap Detection | `detect_overlapping_cycles()` | Find and resolve overlaps |
| Extraction Stats | `get_extraction_stats()` | Track success rate |

### Production Tracking

| Feature | Module | Key Methods |
|---------|--------|-------------|
| Machine State | `MachineStateEvents` | `detect_run_idle()`, `transition_events()`, `state_quality_metrics()` |
| Line Throughput | `LineThroughputEvents` | `count_parts()`, `takt_adherence()`, `throughput_trends()` |
| Changeovers | `ChangeoverEvents` | `detect_changeover()`, `changeover_window()` |
| Flow Constraints | `FlowConstraintEvents` | `blocked_events()`, `starved_events()`, `flow_constraint_analytics()` |
| Part Tracking | `PartProductionTracking` | `production_by_part()`, `daily_production_summary()` |
| Cycle Times | `CycleTimeTracking` | `cycle_time_statistics()`, `detect_slow_cycles()`, `cycle_time_trend()` |
| Downtime | `DowntimeTracking` | `downtime_by_shift()`, `downtime_by_reason()`, `availability_trend()` |
| Quality/NOK | `QualityTracking` | `nok_by_shift()`, `quality_by_part()`, `nok_by_reason()` |
| Shift Reports | `ShiftReporting` | `shift_production()`, `shift_targets()`, `shift_comparison()` |

### OEE & Plant Analytics

| Feature | Module | Key Methods |
|---------|--------|-------------|
| OEE | `OEECalculator` | `calculate_oee()`, `calculate_availability()`, `calculate_performance()` |
| Alarms | `AlarmManagementEvents` | `alarm_frequency()`, `chattering_detection()`, `standing_alarms()` |
| Batches | `BatchTrackingEvents` | `detect_batches()`, `batch_duration_stats()`, `batch_yield()` |
| Bottlenecks | `BottleneckDetectionEvents` | `detect_bottleneck()`, `shifting_bottleneck()`, `station_utilization()` |
| Micro-Stops | `MicroStopEvents` | `detect_micro_stops()`, `micro_stop_frequency()`, `micro_stop_impact()` |
| Duty Cycles | `DutyCycleEvents` | `on_off_intervals()`, `duty_cycle_per_window()`, `excessive_cycling()` |

### Traceability

| Feature | Module | Key Methods |
|---------|--------|-------------|
| Value Trace | `ValueTraceabilityEvents` | `build_timeline()`, `lead_time()`, `station_statistics()` |
| Routing Trace | `RoutingTraceabilityEvents` | `build_routing_timeline()`, `lead_time()`, `routing_paths()` |
| Multi-Process | `MultiProcessTraceabilityEvents` | `build_timeline()`, `parallel_activity()`, `handover_log()` |

### Performance & Targets

| Feature | Module | Key Methods |
|---------|--------|-------------|
| Performance Loss | `PerformanceLossTracking` | `performance_by_shift()`, `slow_periods()`, `performance_trend()` |
| Scrap | `ScrapTracking` | `scrap_by_shift()`, `scrap_by_reason()`, `scrap_cost()` |
| Targets | `TargetTracking` | `compare_to_target()`, `target_achievement_summary()`, `target_hit_rate()` |
| Setup Time | `SetupTimeTracking` | `setup_durations()`, `setup_by_product()`, `setup_trend()` |
| Operator | `OperatorPerformanceTracking` | `production_by_operator()`, `operator_efficiency()`, `quality_by_operator()` |
| Rework | `ReworkTracking` | `rework_by_shift()`, `rework_by_reason()`, `rework_cost()` |
| Shift Handover | `ShiftHandoverReport` | `generate_report()`, `highlight_issues()`, `from_shift_data()` |
| Period Summary | `PeriodSummary` | `weekly_summary()`, `monthly_summary()`, `compare_periods()` |

### Control Quality KPIs

| Feature | Module | Method |
|---------|--------|--------|
| Time to Settle | `SetpointChangeEvents` | `time_to_settle()` |
| Rise Time | `SetpointChangeEvents` | `rise_time()` |
| Overshoot/Undershoot | `SetpointChangeEvents` | `overshoot_metrics()` |
| Oscillation Analysis | `SetpointChangeEvents` | `oscillation_frequency()` |
| Decay Rate | `SetpointChangeEvents` | `decay_rate()` |
| Comprehensive Metrics | `SetpointChangeEvents` | `control_quality_metrics()` |

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
