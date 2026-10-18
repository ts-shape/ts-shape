# ts-shape

[![PyPI](https://img.shields.io/pypi/v/ts-shape.svg)](https://pypi.org/project/ts-shape/)
[![Python](https://img.shields.io/pypi/pyversions/ts-shape.svg)](https://pypi.org/project/ts-shape/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](license.md)

ts-shape is a pandas library for manufacturing and IIoT time series. It works on
long-format signal data (`systime`, `uuid`, `value_*`) and provides loaders,
filters, feature extraction and event detectors for production, quality,
engineering, maintenance, energy and supply-chain questions: OEE, downtime,
SPC rules, setpoint response, drift and more. Every detector takes a DataFrame
and returns a DataFrame, and any detector output can be turned into an
OCEL 2.0 or XES event log for process mining.

```bash
pip install ts-shape
```

```python
import ts_shape

# Synthetic data in the standard long format; use a loader or pandas for your own
df = ts_shape.make_timeseries(["press:temp"], n_points=2000, n_outliers=5)

print(ts_shape.profile(df).summary())   # signal kinds, gaps, resets, duplicates

outliers = ts_shape.OutlierDetectionEvents(df, value_column="value_double")
events = outliers.detect_outliers_zscore(n_sigma=4.0)

log = ts_shape.to_event_log(events, detector="OutlierDetectionEvents.detect_outliers_zscore")
print(log.events[["ocel:activity", "ocel:timestamp", "ts_shape:value"]])
```

## Where to go next

- [Installation](user_guide/installation.md) and [Quick start](user_guide/quick_start.md)
- [Data model and layers](concept.md): the input format, detectors and output shapes
- Guides: [loading](guides/loading.md), [transforms](guides/transforms.md),
  [cycles and segments](guides/feature-extraction.md),
  [object detection](guides/eventlog/objects.md),
  [production and OEE](guides/production.md), [quality and SPC](guides/quality.md),
  [process engineering](guides/engineering.md), [pipelines](guides/pipeline-builder.md),
  [event logs](guides/eventlog/index.md)
- [Module reference](modules/index.md): one page per detector class
- [API reference](reference/index.md): signatures and docstrings, generated from the source
