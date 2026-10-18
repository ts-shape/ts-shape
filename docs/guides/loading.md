# Loading data

Every loader returns a long-format DataFrame (`systime`, `uuid`, `value_*`).
If you already have such a frame, from pandas or any other source, skip this
page: nothing else in ts-shape requires a loader.

| Source | Class | Module | Extra |
|---|---|---|---|
| Local parquet files | `ParquetLoader` | `loader.timeseries.parquet_loader` | `parquet` |
| Azure Blob Storage (parquet) | `AzureBlobParquetLoader` | `loader.timeseries.azure_blob_loader` | `azure` |
| Azure Blob Storage (CSV, JSON, XML) | `AzureBlobFlexibleFileLoader` | `loader.timeseries.azure_blob_loader` | `azure` |
| S3-compatible storage | `S3ProxyDataAccess` | `loader.timeseries.s3proxy_parquet_loader` | `s3` |
| Databricks Unity Catalog Volume | `DatabricksUnityParquetLoader` | `loader.timeseries.databricks_unity_parquet_loader` | |
| Databricks via Spark | `DatabricksSparkParquetLoader` | `loader.timeseries.databricks_spark_parquet_loader` | pyspark on the cluster |
| TimescaleDB | `TimescaleDBDataAccess` | `loader.timeseries.timescale_loader` | `postgres` |
| Energy CSV on Azure Blob Storage | `AzureBlobEnergyLoader` | `loader.timeseries.azure_blob_energy_loader` | `azure` |
| Energy CSV on a Databricks Unity Catalog Volume | `DatabricksUnityEnergyLoader` | `loader.timeseries.databricks_unity_energy_loader` | |
| Energy REST API (JSON) | `EnergyAPILoader` | `loader.timeseries.energy_api_loader` | `api` |
| Signal metadata (JSON) | `MetadataJsonLoader` | `loader.metadata.metadata_json_loader` | |
| Signal metadata (Datadash REST API) | `DatapointAPI` | `loader.metadata.metadata_api_loader` | `api` |
| Signal metadata (PostgreSQL config repository) | `DatapointDB` | `loader.metadata.metadata_db_loader` | `postgres` |

## The hourly folder layout

The file-based loaders expect one file per signal and hour:
`<base>/YYYY/MM/DD/HH/<uuid>.parquet`. The example writes a small dataset in
that layout and reads it back.

```python
import tempfile
from pathlib import Path

import pandas as pd
import ts_shape
from ts_shape.loader.timeseries.parquet_loader import ParquetLoader

base = Path(tempfile.mkdtemp())
df = ts_shape.make_timeseries(["temperature", "pressure"], n_points=240, freq="1min", start="2024-01-15 06:00")
for (hour, uuid), part in df.groupby([df["systime"].dt.floor("h"), "uuid"]):
    folder = base / hour.strftime("%Y/%m/%d/%H")
    folder.mkdir(parents=True, exist_ok=True)
    part.to_parquet(folder / f"{uuid}.parquet", index=False)

everything = ParquetLoader.load_all_files(base)
temperature = ParquetLoader.load_files_by_time_range_and_uuids(
    base, pd.Timestamp("2024-01-15 07:00"), pd.Timestamp("2024-01-15 08:00"), ["temperature"]
)
```

`ParquetLoader` selects whole hour folders: the second call returns the 07:00
and 08:00 folders completely (07:00 to 08:59). Filter the rows afterwards if you
need an exact range. A start in the middle of an hour includes that hour's
folder. Folder hours are UTC: tz-aware bounds are converted to UTC, naive
bounds are taken as UTC (the same for `DatabricksSparkParquetLoader`). A file
matches a uuid only when its name is exactly `<uuid>.parquet`.

## Azure Blob Storage

Authenticate with a SAS URL, a connection string, or an AAD credential:

<!-- no-run -->

```python
from ts_shape.loader.timeseries.azure_blob_loader import AzureBlobParquetLoader

loader = AzureBlobParquetLoader(
    sas_url="https://myaccount.blob.core.windows.net/timeseries?sv=...&sig=...",
    prefix="parquet/",
)
# or: AzureBlobParquetLoader(connection_string="...", container_name="timeseries", prefix="parquet/")
# or: AzureBlobParquetLoader(account_url="https://myaccount.blob.core.windows.net",
#         container_name="timeseries", credential=DefaultAzureCredential(), prefix="parquet/")

loader.list_structure(limit=20)   # {"folders": [...], "files": [...]}

df = loader.load_files_by_time_range_and_uuids(
    start="2024-01-15 08:00",
    end="2024-01-15 12:00",
    uuid_list=["temperature", "pressure"],
)
```

- `load_by_time_range` reads all signals; `stream_by_time_range` yields
  `(blob_name, frame)` pairs one blob at a time.
- Rows outside `[start, end]` are dropped (`trim_to_range=True`, the default).
- A blob that cannot be downloaded or parsed is skipped, and one
  `DataQualityWarning` names every skipped blob. Pass `errors="raise"` to get a
  `LoaderError` instead. The Databricks Unity loader takes the same argument.

`AzureBlobFlexibleFileLoader` lists and downloads non-parquet files in the
same layout (`list_files_by_time_range`, `fetch_files_by_time_range(...,
parse=True)`).

## S3-compatible storage

<!-- no-run -->

```python
from ts_shape.loader.timeseries.s3proxy_parquet_loader import S3ProxyDataAccess

loader = S3ProxyDataAccess(
    start="2024-01-01 00:00:00",
    end="2024-01-31 23:59:59",
    uuids=["temperature", "pressure"],
    s3_config={
        "endpoint_url": "https://s3.example.com",
        "key": "...",
        "secret": "...",
        "use_ssl": True,
        "version_aware": False,
        "s3_path_base": "data-lake/timeseries/",
    },
)
df = loader.fetch_data_as_dataframe()
```

## Databricks

Two loaders read the hourly layout on Databricks. Choose by who does the scan.

`DatabricksUnityParquetLoader` reads a mounted Unity Catalog Volume on the
driver with pandas, with no Spark involved:

<!-- no-run -->

```python
from ts_shape.loader.timeseries.databricks_unity_parquet_loader import (
    DatabricksUnityParquetLoader,
)

loader = DatabricksUnityParquetLoader(
    catalog="main", schema="plant", volume="timeseries", prefix="parquet",
)
df = loader.load_by_time_range("2026-06-19 06:00", "2026-06-19 14:00")
```

`DatabricksSparkParquetLoader` lets the cluster's Spark session read the files
and converts to pandas at the end:

<!-- no-run -->

```python
from ts_shape.loader.timeseries.databricks_spark_parquet_loader import (
    DatabricksSparkParquetLoader,
)

# spark defaults to the active session on Databricks; pass it explicitly elsewhere
loader = DatabricksSparkParquetLoader("/Volumes/main/plant/timeseries", spark=spark)
df = loader.load_by_time_range_and_uuids(
    "2026-06-19 06:00", "2026-06-19 14:00",
    uuids=["ABC", "DEF"],
    columns=["systime", "uuid", "value_integer"],   # column pruning
    filter_expr="value_integer > 0",                # Spark SQL predicate
)
loader.build_paths("2026-06-19 06:00", "2026-06-19 08:00", ["ABC"])   # the files it would read
```

| Argument | Purpose |
|---|---|
| `uuids` | one uuid or a list; one path per hour and uuid |
| `columns` | columns to select |
| `filter_expr` | Spark SQL predicate applied with `.where` |
| `hour_pattern`, `file_template` | a different folder or file layout |
| `ignore_missing` | skip hours without a file (default `True`) |
| `path_exists` | callable (for example over `dbutils.fs`) to pre-filter paths |
| `as_pandas` | `False` returns the Spark DataFrame |

## TimescaleDB

`TimescaleDBDataAccess` queries a `telemetry` table with the standard columns:

<!-- no-run -->

```python
from ts_shape.loader.timeseries.timescale_loader import TimescaleDBDataAccess

loader = TimescaleDBDataAccess(
    start="2024-01-01 00:00:00",
    end="2024-01-02 00:00:00",
    uuids=["temperature", "pressure"],
    db_config={"db_user": "...", "db_pass": "...", "db_host": "...", "db_name": "..."},
)
df = loader.fetch_data_as_dataframe()
```

## Energy data

Three loaders read energy readings and return `systime`, `uuid`,
`value_double`, `is_delta`, with `systime` in UTC.

`AzureBlobEnergyLoader` (Azure Blob Storage, same authentication as
`AzureBlobParquetLoader`) and `DatabricksUnityEnergyLoader` (a mounted Unity
Catalog Volume, read with pandas on the driver) read one CSV file per series
and day, `csv/YYYY/MM/DD/<series_id>.csv`, plus series metadata in
`.meta/series.csv`. The `uuid` is the series id (the file name without
`.csv`).

<!-- no-run -->

```python
from ts_shape.loader.timeseries.databricks_unity_energy_loader import (
    DatabricksUnityEnergyLoader,
)

loader = DatabricksUnityEnergyLoader(catalog="main", schema="plant", volume="energy")
meta = loader.load_series_metadata()      # id, label_lvl1..4, description, unit, hierarchy_lvl1..6
df = loader.load_by_time_range("2026-06-19", "2026-06-20", series_ids=["M-001"])
```

Both also offer `load_by_series_ids`, `stream_by_time_range` (one CSV file
at a time, as `(series_id, frame)` pairs) and `list_series`; a file that cannot be read is skipped with a
`DataQualityWarning` (`errors="raise"` raises `LoaderError`).

`EnergyAPILoader` fetches JSON records from a REST endpoint and assigns one
`uuid` to every row:

<!-- no-run -->

```python
from ts_shape.loader.timeseries.energy_api_loader import EnergyAPILoader

loader = EnergyAPILoader("https://api.energy-provider.example/v1",
                         headers={"Authorization": "Bearer <token>"})
df = loader.fetch_data_as_dataframe(
    "/meters/readings",
    params={"meter_id": "M-001", "start": "2024-01-01", "end": "2024-01-02"},
    time_key="timestamp", value_key="value", uuid_value="meter:M-001",
)
```

## Signal metadata from Datadash and the config repository

`DatapointAPI` (Datadash REST API, bearer token) and `DatapointDB` (the
PostgreSQL config repository) collect the datapoints (`uuid`, `label`,
`config`) of the devices you name. Both fetch on construction, write one
`<device>_data_points.json` per device to `output_path` (default `"data"`),
and return the uuids per device with `get_all_uuids()`, for example as the
uuid list of `AzureBlobParquetLoader`. `filter_enabled=True` (the default)
keeps enabled datapoints only (`DatapointDB` also drops archived ones).

<!-- no-run -->

```python
from ts_shape.loader.metadata.metadata_db_loader import DatapointDB

db = DatapointDB(["Press 1"], db_user="...", db_pass="...", db_host="...")
uuids = db.get_all_uuids()["Press 1"]
```

## Combining timeseries with metadata

`MetadataJsonLoader` reads signal metadata (a `uuid`, `label` and `config` per
signal; `config` keys become columns). `DataIntegratorHybrid.combine_data`
joins it to the timeseries on `uuid`:

```python
from ts_shape.loader.metadata.metadata_json_loader import MetadataJsonLoader
from ts_shape.loader.combine.integrator import DataIntegratorHybrid

meta = MetadataJsonLoader.from_str("""{
  "uuid":   {"0": "temperature", "1": "pressure"},
  "label":  {"0": "Oven temperature", "1": "Line pressure"},
  "config": {"0": {"unit": "degC"}, "1": {"unit": "bar"}}
}""").to_df()      # MetadataJsonLoader.from_file("signals.json") for a file

combined = DataIntegratorHybrid.combine_data(
    timeseries_sources=[everything],
    metadata_sources=[meta],
    uuids=["temperature"],    # optional: keep only these signals
)
```

The `uuids` filter applies with or without metadata. Several metadata sources
may describe the same uuid with different columns; two different values for
the same uuid and column raise `ValueError`. Timeseries sources whose time
columns differ in timezone handling are converted to UTC (naive times read
as UTC) with a `DataQualityWarning`.

Next: [Transforms and statistics](transforms.md).
