# Example scripts

The [`examples/`](https://github.com/ts-shape/ts-shape/tree/main/examples)
folder of the repository has one script per area of the library. Each script
generates its own synthetic data and runs without input files:

```bash
python examples/quality_events_demo.py
```

Some scripts use optional extras; install them with `pip install "ts-shape[all]"`
or `pip install -e ".[dev]"` in a clone.

## Detectors by pack

| Script | Shows |
|---|---|
| [`quality_events_demo.py`](https://github.com/ts-shape/ts-shape/blob/main/examples/quality_events_demo.py) | Outlier detection, SPC rules and tolerance deviation |
| [`production_events_demo.py`](https://github.com/ts-shape/ts-shape/blob/main/examples/production_events_demo.py) | Machine state, throughput, changeover, flow constraints, OEE, alarms, batches, downtime and shift reports |
| [`production_tracking_demo.py`](https://github.com/ts-shape/ts-shape/blob/main/examples/production_tracking_demo.py) | Parts, cycle times, downtime, quality and shift reports |
| [`line_flow_analytics_demo.py`](https://github.com/ts-shape/ts-shape/blob/main/examples/line_flow_analytics_demo.py) | Line balancing and takt, flow metrics and Little's law |
| [`unit_runtime_demo.py`](https://github.com/ts-shape/ts-shape/blob/main/examples/unit_runtime_demo.py) | Unit conversion (needs `ts-shape[units]`) and operating-hours accounting |
| [`setpoint_events_advanced_usage.py`](https://github.com/ts-shape/ts-shape/blob/main/examples/setpoint_events_advanced_usage.py) | Setpoint changes: noise filtering, percentage tolerance, settling and control-quality KPIs |
| [`maintenance_events_demo.py`](https://github.com/ts-shape/ts-shape/blob/main/examples/maintenance_events_demo.py) | Degradation, failure prediction and vibration analysis |
| [`energy_events_demo.py`](https://github.com/ts-shape/ts-shape/blob/main/examples/energy_events_demo.py) | Energy consumption and efficiency |
| [`correlation_events_demo.py`](https://github.com/ts-shape/ts-shape/blob/main/examples/correlation_events_demo.py) | Signal correlation and anomaly correlation |
| [`supplychain_events_demo.py`](https://github.com/ts-shape/ts-shape/blob/main/examples/supplychain_events_demo.py) | Inventory, lead time and demand patterns |

## Loading, transforms and features

| Script | Shows |
|---|---|
| [`loader_usage_demo.py`](https://github.com/ts-shape/ts-shape/blob/main/examples/loader_usage_demo.py) | Timeseries and metadata from parquet and JSON, combined on `uuid` |
| [`transform_operations_demo.py`](https://github.com/ts-shape/ts-shape/blob/main/examples/transform_operations_demo.py) | Filters, calculations and time-zone shifts |
| [`features_statistics_demo.py`](https://github.com/ts-shape/ts-shape/blob/main/examples/features_statistics_demo.py) | Numeric, string, boolean and timestamp statistics (including missing values), feature tables, time-grouped statistics and cycle processing |
| [`cycle_extractor_enhancements_demo.py`](https://github.com/ts-shape/ts-shape/blob/main/examples/cycle_extractor_enhancements_demo.py) | Cycle extraction, method suggestion and validation |
| [`context_enricher_demo.py`](https://github.com/ts-shape/ts-shape/blob/main/examples/context_enricher_demo.py) | Enriching signals with descriptions, units, limits and value mappings |
| [`pipeline_demo.py`](https://github.com/ts-shape/ts-shape/blob/main/examples/pipeline_demo.py) | Transforms and detectors chained in a `Pipeline` |

## Event log

| Script | Shows |
|---|---|
| [`eventlog_demo.py`](https://github.com/ts-shape/ts-shape/blob/main/examples/eventlog_demo.py) | Detector output as an OCEL 2.0 event log, exported XES-style and as OCEL tables |
| [`object_detection_demo.py`](https://github.com/ts-shape/ts-shape/blob/main/examples/object_detection_demo.py) | OCEL objects (batches, serials) derived from id signals |
| [`lambda_rules_demo.py`](https://github.com/ts-shape/ts-shape/blob/main/examples/lambda_rules_demo.py) | Detectors declared as lambda rules (with `lambda_rules_demo.yaml`) |
