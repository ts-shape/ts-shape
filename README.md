# ts-shape | Timeseries Shaper

[![pypi version](https://img.shields.io/pypi/v/ts-shape.svg)](https://pypi.org/project/ts-shape/)
[![downloads](https://static.pepy.tech/badge/ts-shape/week)](https://pepy.tech/projects/ts-shape)
[![CI](https://github.com/ts-shape/ts-shape/actions/workflows/ci.yml/badge.svg)](https://github.com/ts-shape/ts-shape/actions/workflows/ci.yml)
[![docs](https://img.shields.io/badge/docs-mkdocs-708FCC.svg?style=flat)](https://ts-shape.github.io/ts-shape/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)

**ts-shape** is a composable, production-ready Python toolkit for loading, shaping, and analysing industrial timeseries data. Built for manufacturing and IoT, it follows a simple **DataFrame-in, DataFrame-out** philosophy across loaders, transforms, feature extractors, and event detectors.

---

## Key Features

- **Unified DataFrame workflow** -- Load timeseries + metadata, join on `uuid`, process.
- **Modular packs** -- Quality, Production, Engineering, Maintenance, Supply Chain, Energy, Correlation, and Development (product & process R&D) events.
- **Performance-aware** -- Vectorised ops, chunked DB reads, concurrent I/O.
- **Zero ML dependencies** -- Core uses only pandas, numpy, scipy.
- **Multi-source loaders** -- Parquet, S3, Azure Blob, TimescaleDB, REST APIs.

---

## Installation

```bash
pip install ts-shape

# Recommended: parquet engine
pip install pyarrow          # or: pip install fastparquet
```

Optional integrations:

| Integration | Install |
|-------------|---------|
| Azure Blob Storage | `pip install azure-storage-blob` |
| Azure AAD + management | `pip install azure-identity azure-mgmt-storage` |
| S3 proxy access | Included via `s3fs` |
| TimescaleDB / PostgreSQL | `pip install ts-shape[postgres]` or any SQLAlchemy-compatible driver |
| Unit conversion (`UnitConverter`) | `pip install ts-shape[units]` (pulls in `pint`) |

---

## Quick Start

This snippet runs as-is -- no data files, no setup:

```python
import ts_shape

# Generate a sample dataset (or load your own via pandas / a ts-shape loader)
df = ts_shape.make_timeseries(["sensor:temp"], n_points=2000, n_outliers=6)

# Discover what is available -- 70+ detectors across 8 packs
ts_shape.list_detectors("events.quality")

# Detect outliers
detector = ts_shape.OutlierDetectionEvents(df, value_column="value_double")
outliers = detector.detect_outliers_zscore(threshold=3.0)
print(outliers)
```

Every script under [`examples/`](examples/) is likewise self-contained -- run any
of them straight after install, e.g. `python examples/quality_events_demo.py`. See
[`examples/README.md`](examples/README.md) for the full, authoritative catalogue
(loaders, transforms, pipelines, and every event pack).

---

## Data Model

ts-shape works with a standardised timeseries DataFrame schema:

| Column | Type | Description |
|--------|------|-------------|
| `systime` | `datetime64[ns]` | Timestamp (sorted, tz-aware supported) |
| `uuid` | `str` | Signal identifier |
| `value_double` | `float64` | Numeric values |
| `value_integer` | `int64` | Integer values |
| `value_bool` | `bool` | Boolean values |
| `value_string` | `str` | String values |
| `is_delta` | `bool` | Change indicator |

All classes inherit from a common `Base` class that automatically detects time columns, converts to datetime, and sorts by timestamp.

---

## Architecture

```
ts_shape/
├── loader/              # Data Loading & Integration
│   ├── timeseries/      # Parquet, S3, Azure Blob, TimescaleDB, Energy API
│   ├── metadata/        # JSON, REST API, Database metadata
│   └── combine/         # DataIntegratorHybrid (merge timeseries + metadata)
│
├── transform/           # Data Transformation
│   ├── filter/          # Numeric, String, Boolean, DateTime, Custom filters
│   ├── calculator/      # Arithmetic operations (scale, offset, power, etc.)
│   ├── functions/       # Lambda/callable application
│   └── time_functions/  # Timestamp conversion, timezone operations
│
├── features/            # Feature Extraction
│   ├── stats/           # Numeric, String, Boolean, Timestamp statistics
│   ├── time_stats/      # Time-windowed aggregations
│   └── cycles/          # Cycle detection & processing (6 methods)
│
├── events/              # Event Detection (Domain Packs)
│   ├── quality/         # Outlier detection, SPC (8 rules), tolerance deviation
│   ├── production/      # OEE, machine state, throughput, shift, downtime, alarms, batches
│   ├── engineering/     # Setpoint changes, startup detection, control quality
│   ├── maintenance/     # Degradation, failure prediction, vibration analysis
│   ├── energy/          # Consumption analysis, efficiency tracking
│   ├── correlation/     # Signal correlation, anomaly co-occurrence
│   └── supplychain/     # Inventory monitoring, lead time, demand patterns
│
├── context/             # Value mapping (categorical codes → labels)
└── utils/               # Base class and shared utilities
```

---

## Packs Overview

### Quality Events
Detect anomalies and process deviations in sensor data.

```python
from ts_shape.events.quality.outlier_detection import OutlierDetectionEvents
from ts_shape.events.quality.statistical_process_control import StatisticalProcessControlRuleBased
from ts_shape.events.quality.tolerance_deviation import ToleranceDeviationEvents

# Outlier detection (Z-score, IQR, MAD, Isolation Forest)
outliers = OutlierDetectionEvents(df, value_column="value_double")
result = outliers.detect_outliers_zscore(threshold=3.0)

# Statistical Process Control -- 8 Western Electric rules
spc = StatisticalProcessControlRuleBased(
    df,
    value_column="value_double",
    tolerance_uuid="limit",
    actual_uuid="sensor",
    event_uuid="spc_violation",
)
violations = spc.process()

# Tolerance deviation with severity classification
tol = ToleranceDeviationEvents(
    df,
    tolerance_column="value_double",
    actual_column="value_double",
    actual_uuid="sensor",
    tolerance_uuid="limit",
    event_uuid="tolerance_deviation",
)
deviations = tol.process_and_group_data_with_events()
```

### Production Events
Track production performance, equipment states, and operational metrics.

```python
from ts_shape.events.production.machine_state import MachineStateEvents
from ts_shape.events.production.oee_calculator import OEECalculator
from ts_shape.events.production.shift_reporting import ShiftReporting
from ts_shape.events.production.alarm_management import AlarmManagementEvents
from ts_shape.events.production.batch_tracking import BatchTrackingEvents

# Machine state detection (run/idle intervals)
mse = MachineStateEvents(df, run_state_uuid="machine:running")
intervals = mse.detect_run_idle(min_duration="30s")

# OEE calculation (Availability x Performance x Quality)
oee = OEECalculator(df)
result = oee.calculate_oee(
    run_state_uuid="machine:state",
    counter_uuid="parts:count",
    ideal_cycle_time=10.0,
)

# Alarm analysis (ISA-18.2 style)
alarms = AlarmManagementEvents(df, alarm_uuid="alarm:overtemp")
chattering = alarms.chattering_detection(min_transitions=5, window="10m")

# Batch tracking
batches = BatchTrackingEvents(df, batch_uuid="batch:id")
batch_list = batches.detect_batches()
```

Line & flow analytics for industrial engineers:

```python
from ts_shape.events.production.line_balancing import LineBalancingEvents
from ts_shape.events.production.flow_metrics import FlowMetricsEvents

# Line balancing -- station loading, balance efficiency, Yamazumi
lb = LineBalancingEvents(df, station_uuids={"st1": "Station 1", "st2": "Station 2"})
balance = lb.balance_metrics(takt_time="55s", window="1h")
yamazumi = lb.yamazumi(demand=480, available_time="8h")

# Flow metrics -- WIP, throughput, lead time, Little's Law
flow = FlowMetricsEvents(df, entry_uuid="process:in", exit_uuid="process:out")
wip = flow.wip_over_time(window="1h")
summary = flow.flow_summary(value_add_seconds=120, window="1h")
```

Runtime accounting and unit conversion:

```python
from ts_shape.events.production.runtime_accounting import RuntimeAccountingEvents
from ts_shape.transform.calculator.unit_conversion import UnitConverter

# Operating-hours accounting -- run time, starts, longest run, hour-meter
rt = RuntimeAccountingEvents(df, run_uuid="machine:running")
summary = rt.runtime_summary()
meter = rt.operating_hours_meter(window="1h")

# Unit conversion -- backed by pint (pip install ts-shape[units])
UnitConverter.convert_value(100, "C", "F")            # 212.0
df_psi = UnitConverter.convert_column(df, "bar", "psi", column_name="value_double")
```

### Engineering Events
Analyse control system behaviour and setpoint responses.

```python
from ts_shape.events.engineering.setpoint_events import SetpointChangeEvents
from ts_shape.events.engineering.startup_events import StartupDetectionEvents

# Setpoint change detection + settling time + overshoot
sp = SetpointChangeEvents(df, setpoint_uuid="setpoint:temp")
steps = sp.detect_setpoint_steps(min_delta=2.0)
settle = sp.time_to_settle(actual_uuid="actual:temp", tol=0.5)
quality = sp.control_quality_metrics(actual_uuid="actual:temp")

# Startup detection
startup = StartupDetectionEvents(df, signal_uuid="motor:speed")
events = startup.detect_startup_by_threshold(threshold=100.0)
```

### Maintenance Events
Predictive maintenance through degradation detection and failure prediction.

```python
from ts_shape.events.maintenance.degradation_detection import DegradationDetectionEvents
from ts_shape.events.maintenance.failure_prediction import FailurePredictionEvents
from ts_shape.events.maintenance.vibration_analysis import VibrationAnalysisEvents

# Degradation detection (trend, variance, level shift, health score)
deg = DegradationDetectionEvents(df, signal_uuid="sensor:bearing_temp")
trends = deg.detect_trend_degradation(window="1h", direction="increasing")
health = deg.health_score(window="1h", baseline_window="24h")

# Remaining Useful Life estimation
fp = FailurePredictionEvents(df, signal_uuid="sensor:bearing_temp")
rul = fp.remaining_useful_life(degradation_rate=0.01, failure_threshold=120.0)

# Vibration analysis (RMS, crest factor, kurtosis)
vib = VibrationAnalysisEvents(df, signal_uuid="sensor:vibration")
indicators = vib.bearing_health_indicators(window="5m")
```

### Development Events (Product & Process R&D)
Designed for the activities that happen before commercial production: DOE runs,
design-space qualification, golden-batch comparison, recipe-phase adherence, and
outcome-driven critical-parameter ranking.

```python
from ts_shape.events.development import (
    DesignOfExperimentsEvents,
    DesignSpaceEvents,
    GoldenBatchDeviationEvents,
    RecipePhaseAdherenceEvents,
    CriticalParameterRankingEvents,
)

# Recover DOE run structure from a continuous trace
doe = DesignOfExperimentsEvents(df, factor_uuids=["factor:F1", "factor:F2"])
runs = doe.detect_runs(min_duration="5min", stability_tol=0.01)
effects = doe.compute_effects(response_uuid="response:Y", statistic="settled")

# Multivariate qualified operating window
ds = DesignSpaceEvents(qualification_df, cpp_uuids=["cpp:temp", "cpp:ph"]).fit_box()
excursions = ds.detect_excursions(operation_df)

# Golden-batch trajectory comparison (pointwise / area / DTW)
gb = GoldenBatchDeviationEvents(reference_df, signal_uuid="reactor:temp")
deviation = gb.compare(new_batch_df, mode="dtw")

# Recipe-phase pass/fail vs. a declarative spec
spec = {"hold": {"hold_value": (78.0, 82.0)}, "heat_up": {"ramp_rate_max": 0.5}}
rp = RecipePhaseAdherenceEvents(df, phase_uuid="phase:reactor",
                                value_uuid="temp:reactor", spec=spec)
phases = rp.evaluate()

# Rank candidate CPPs by their statistical link to a quality outcome
cpp = CriticalParameterRankingEvents(df)
drivers = cpp.top_drivers(per_run_df, candidate_columns=["x1", "x2", "x3"],
                          outcome_column="yield", method="spearman")
```

### Supply Chain Events
Monitor inventory, lead times, and demand patterns.

```python
from ts_shape.events.supplychain.inventory_monitoring import InventoryMonitoringEvents
from ts_shape.events.supplychain.lead_time_analysis import LeadTimeAnalysisEvents
from ts_shape.events.supplychain.demand_pattern import DemandPatternEvents

# Inventory monitoring with stockout prediction
inv = InventoryMonitoringEvents(df, level_uuid="inventory:raw_material")
low_stock = inv.detect_low_stock(min_level=100, hold="30m")
prediction = inv.stockout_prediction(consumption_rate_window="4h")

# Lead time analysis
lt = LeadTimeAnalysisEvents(df)
lead_times = lt.calculate_lead_times(order_uuid="order:placed", delivery_uuid="order:delivered")
anomalies = lt.detect_lead_time_anomalies(order_uuid="order:placed", delivery_uuid="order:delivered")

# Demand patterns and seasonality
demand = DemandPatternEvents(df, demand_uuid="demand:daily")
spikes = demand.detect_demand_spikes(threshold_factor=2.0)
seasonal = demand.seasonality_summary(period="1D")
```

### Loaders
Load data from multiple sources into the standard schema.

```python
from ts_shape.loader.timeseries.parquet_loader import ParquetLoader
from ts_shape.loader.timeseries.azure_blob_loader import AzureBlobParquetLoader
from ts_shape.loader.metadata.metadata_json_loader import MetadataJsonLoader
from ts_shape.loader.combine.integrator import DataIntegratorHybrid

# Load parquet files
df = ParquetLoader.load_all_files("/data/timeseries")
df_range = ParquetLoader.load_by_time_range("/data/timeseries", start, end)

# Load metadata and combine
meta = MetadataJsonLoader.from_file("metadata.json")
combined = DataIntegratorHybrid.combine_data(
    timeseries_sources=[df], metadata_sources=[meta.to_df()]
)
```

### Features & Statistics
Extract statistical features and detect cycles.

```python
from ts_shape.features.stats.numeric_stats import NumericStatistics
from ts_shape.features.stats.time_stats_numeric import TimeGroupedStatistics
from ts_shape.features.cycles.cycles_extractor import CycleExtractor

# Descriptive statistics
stats = NumericStatistics.summary_as_dict(df, "value_double")

# Time-windowed aggregations
hourly = TimeGroupedStatistics.calculate_statistic(
    df,
    time_column="systime",
    value_column="value_double",
    freq="1h",
    stat_method="mean",
)

# Cycle extraction (6 detection methods)
extractor = CycleExtractor(df, start_uuid="cycle:trigger")
cycles = extractor.process_persistent_cycle()
```

---

## Composing a Pipeline

`Pipeline` chains transforms and detectors into one reusable definition.
A `.transform(...)` step's output **replaces** the working signal; a
`.detect(...)` step's output is **stored** under a name, leaving the signal
untouched. The choice of `.transform` vs `.detect` is explicit — never inferred.

```python
from ts_shape import Pipeline
from ts_shape.transform.calculator.numeric_calc import IntegerCalc
from ts_shape.events.quality.outlier_detection import OutlierDetectionEvents

pipe = (
    Pipeline(name="sensor-quality")
    .transform(IntegerCalc, "scale_column", column_name="value_double", factor=0.1)
    .detect(OutlierDetectionEvents, "detect_outliers_zscore",
            name="outliers", value_column="value_double", threshold=3.0)
)

result = pipe.run(df)          # reusable: call .run() on many DataFrames
result.data                    # final transformed signal
result.events["outliers"]      # detector output
result.to_event_log()          # normalized, combined OCEL event log
```

An optional `.source(...)` step lets the pipeline load its own data, so the
whole definition -- *source → transform → detect* -- is self-contained and
reusable for scheduled jobs. A source step must be the first step; the pipeline
is then run with no DataFrame argument:

```python
from ts_shape import Pipeline
from ts_shape.loader.timeseries.parquet_loader import ParquetLoader

pipe = (
    Pipeline(name="quality-from-parquet")
    .source(ParquetLoader, "load_all_files", base_path="/data/timeseries")
    .detect(OutlierDetectionEvents, "detect_outliers_zscore",
            name="outliers", value_column="value_double", threshold=3.0)
)

result = pipe.run()            # no DataFrame -- the source produces it
```

`Pipeline` also supports `$input` / `$prev` sentinels for steps that need a
second DataFrame, and `run_steps()` to inspect every intermediate. See the
[Pipeline guide](https://ts-shape.github.io/ts-shape/guides/pipeline-builder/).

---

## Development

```bash
# Clone and install in development mode
git clone https://github.com/ts-shape/ts-shape.git
cd ts-shape
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=ts_shape --cov-report=term-missing

# Build documentation
pip install -r requirements-docs.txt
mkdocs serve
```

---

## CI/CD

The project uses GitHub Actions for continuous integration and deployment:

| Workflow | Trigger | Description |
|----------|---------|-------------|
| **CI** | Push / PR | Runs tests on Python 3.10, 3.11, 3.12 |
| **Release** | Push to `main` / Tag `v*` | Build docs, deploy to GitHub Pages, publish to PyPI |

Versioning is managed with `setuptools-scm` -- version numbers are derived automatically from git tags. To release:

```bash
git tag v0.2.0
git push origin v0.2.0
```

---

## Project Structure

```
ts-shape/
├── src/ts_shape/           # Library source code
├── tests/                  # pytest test suite (100+ tests)
├── examples/               # Runnable demo scripts
├── docs/                   # MkDocs documentation
├── .github/workflows/      # CI/CD pipelines
├── pyproject.toml          # Package configuration + auto-versioning
├── setup.py                # Legacy setup (delegates to pyproject.toml)
├── requirements.txt        # Runtime dependencies
└── requirements-docs.txt   # Documentation dependencies
```

---

## Contributing

Contributions are welcome! Please see `docs/contributing.md` for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Write tests for your changes
4. Ensure all tests pass (`pytest tests/ -v`)
5. Submit a pull request

---

## License

MIT -- see [LICENSE.txt](LICENSE.txt).

---

## Acknowledgments

Parts of this library were developed with the assistance of
[Claude](https://www.anthropic.com/claude), Anthropic's AI assistant. All code
has been reviewed and is maintained by the project authors.

---

## Links

- [Documentation](https://ts-shape.github.io/ts-shape/)
- [PyPI](https://pypi.org/project/ts-shape/)
- [GitHub](https://github.com/ts-shape/ts-shape)
- [Bug Tracker](https://github.com/ts-shape/ts-shape/issues)
