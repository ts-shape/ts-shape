# ts-shape examples

Runnable demo scripts, one per area of the library. Every script is
**self-contained** — it generates its own synthetic data, so you can run
any of them straight after installing ts-shape, with no external files:

```bash
pip install -e ".[dev]"
python examples/quality_events_demo.py
```

## Detectors by pack

| Script | What it shows |
|--------|---------------|
| `quality_events_demo.py` | Quality pack — outlier detection, SPC, tolerance deviation |
| `production_events_demo.py` | Production pack — machine state, throughput, OEE, alarms, batches |
| `production_tracking_demo.py` | Daily production tracking — parts, cycles, downtime, shifts |
| `line_flow_analytics_demo.py` | Line balancing & takt, flow metrics & Little's Law |
| `unit_runtime_demo.py` | Unit conversion (pint) and runtime / operating-hours accounting |
| `maintenance_events_demo.py` | Maintenance pack — degradation, failure prediction, vibration |
| `energy_events_demo.py` | Energy pack — consumption, efficiency, carbon intensity, idle energy |
| `correlation_events_demo.py` | Correlation pack — signal and anomaly correlation |
| `supplychain_events_demo.py` | Supply-chain pack — inventory, lead time, demand patterns |
| `setpoint_events_advanced_usage.py` | Engineering pack — advanced `SetpointChangeEvents` usage |

## Pipeline stages

| Script | What it shows |
|--------|---------------|
| `loader_usage_demo.py` | Loading timeseries + metadata from Parquet / JSON |
| `transform_operations_demo.py` | Filtering, calculations, timezone shifts |
| `features_statistics_demo.py` | Feature tables and statistics |
| `statistics_demo.py` | Numeric / string / boolean / timestamp statistics |
| `cycle_extractor_enhancements_demo.py` | Cycle extraction and validation |
| `context_enricher_demo.py` | Enriching timeseries with categorical context |

## Event log (XES / OCEL)

| Script | What it shows |
|--------|---------------|
| `eventlog_demo.py` | Turning detector output into a canonical OCEL 2.0 event log |
| `lambda_rules_demo.py` | User-authored detectors via the lambda-rule DSL |
| `lambda_rules_demo.yaml` | Companion YAML rule set for `lambda_rules_demo.py` |

## See also

- [Architecture map](https://ts-shape.github.io/ts-shape/guides/architecture-map/) — every module, searchable
- [Guides](https://ts-shape.github.io/ts-shape/guides/) — topic-focused walkthroughs
- [API reference](https://ts-shape.github.io/ts-shape/reference/) — full reference docs
