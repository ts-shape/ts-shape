# ts-shape

[![PyPI](https://img.shields.io/pypi/v/ts-shape.svg)](https://pypi.org/project/ts-shape/)
[![CI](https://github.com/ts-shape/ts-shape/actions/workflows/ci.yml/badge.svg)](https://github.com/ts-shape/ts-shape/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-mkdocs-708FCC.svg)](https://ts-shape.github.io/ts-shape/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/ts-shape/ts-shape/blob/main/LICENSE.txt)

ts-shape is a pandas library for manufacturing and IIoT time series. It works
on long-format signal data (`systime`, `uuid`, `value_double` /
`value_integer` / `value_bool` / `value_string`) and provides loaders, filters,
feature extraction and event detectors for production (OEE, downtime, shift
reports, traceability), quality (outliers, SPC rules, capability),
process engineering, maintenance, energy, supply chain and process
development. Every detector takes a DataFrame and returns a DataFrame, and
any detector output can be turned into an OCEL 2.0 or XES event log for
process mining.

## Install

```bash
pip install ts-shape
```

Python 3.11 or later. The core depends only on pandas, numpy and scipy.
Loader backends and a few detectors need extras:

| Extra | Installs | Needed for |
|---|---|---|
| `parquet` | pyarrow | reading parquet files |
| `s3` | s3fs, pyarrow | S3 loaders |
| `azure` | azure-storage-blob, pyarrow | Azure Blob loaders |
| `api` | requests | REST metadata and energy API loaders |
| `sql` | sqlalchemy | TimescaleDB and SQL metadata loaders |
| `postgres` | sqlalchemy, psycopg2-binary | `sql` plus the PostgreSQL driver |
| `yaml` | pyyaml | signal specs and lambda rules from YAML |
| `units` | pint | `UnitConverter` |
| `ml` | scikit-learn | Isolation Forest outlier detection |
| `all` | all of the above | |

```bash
pip install "ts-shape[parquet,azure]"
```

## Example

```python
import pandas as pd
import ts_shape
from ts_shape.eventlog import concat, to_event_log

# Synthetic data in the standard format; use a loader or pandas for your own
temp = ts_shape.make_timeseries(["oven:temp"], n_points=1440, freq="1min", n_outliers=4)
running = ts_shape.make_timeseries(["oven:running"], n_points=1440, freq="1min", value_column="value_bool")
df = pd.concat([temp, running], ignore_index=True)

print(ts_shape.profile(df).summary())   # signal kinds, gaps, counter resets, duplicates

outliers = ts_shape.OutlierDetectionEvents(temp, value_column="value_double").detect_outliers_zscore(threshold=4.0)
intervals = ts_shape.MachineStateEvents(df, run_state_uuid="oven:running").detect_run_idle(min_duration="5min")

log = concat(
    to_event_log(outliers, detector="OutlierDetectionEvents.detect_outliers_zscore"),
    to_event_log(intervals, detector="MachineStateEvents.detect_run_idle"),
)
print(log.events[["ocel:activity", "ocel:timestamp"]].head())
```

`ts_shape.list_detectors()` lists the detector classes. The
[`examples/`](https://github.com/ts-shape/ts-shape/tree/main/examples) folder
has one self-contained script per area of the library.

## Links

- Documentation: https://ts-shape.github.io/ts-shape/
  ([quick start](https://ts-shape.github.io/ts-shape/user_guide/quick_start/),
  [guides](https://ts-shape.github.io/ts-shape/guides/production/),
  [migration and deprecations](https://ts-shape.github.io/ts-shape/migration/))
- Source and issues: https://github.com/ts-shape/ts-shape
- Development setup and release process: [development guide](https://ts-shape.github.io/ts-shape/insiders/development/)
- License: MIT, see [LICENSE.txt](https://github.com/ts-shape/ts-shape/blob/main/LICENSE.txt)

Parts of this library were developed with the assistance of
[Claude](https://www.anthropic.com/claude), Anthropic's AI assistant. All code
is reviewed and maintained by the project authors.
