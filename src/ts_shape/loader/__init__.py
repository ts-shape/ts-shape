"""Loaders

Load timeseries and metadata from various backends and combine them.

Classes:
- DataIntegratorHybrid: Combine timeseries and metadata from DataFrames or source objects.
  - combine_data: Merge sources on a join key with optional UUID filter.

- ParquetLoader: Read parquet files from local folder structures.
  - load_all_files: Load all parquet under a base path.
  - load_by_time_range: Load files within YYYY/MM/DD/HH path range.
  - load_by_uuid_list: Load files matching UUIDs in filenames.
  - load_files_by_time_range_and_uuids: Combine time range and UUID filters.

- S3ProxyDataAccess: Retrieve parquet via an S3-compatible proxy.
  - fetch_data_as_parquet: Save parquet files to a local folder structure.
  - fetch_data_as_dataframe: Return a combined DataFrame.

- AzureBlobParquetLoader: Load parquet from Azure Blob Storage.

- AzureBlobEnergyLoader: Load CSV energy timeseries and series metadata from Azure Blob.
  - load_series_metadata: Download .meta/series.csv as a DataFrame.
  - load_by_time_range: Load CSVs by date range, optional series filter.
  - load_by_series_ids: Load specific series by ID, optional date filter.
  - stream_by_time_range: Yield (series_id, DataFrame) incrementally.
  - list_series: List all series IDs available in the blob store.
  - load_all_files: Load all parquet under an optional prefix.
  - load_by_time_range: Load hourly folders between start and end.
  - load_files_by_time_range_and_uuids: Load per-hour per-UUID parquet files.
  - list_structure: List folders and files under a prefix.

- TimescaleDBDataAccess: Stream timeseries from TimescaleDB.
  - fetch_data_as_parquet: Partition-by-hour and write parquet.
  - fetch_data_as_dataframe: Return a combined DataFrame.

- MetadataJsonLoader: Ingest JSON metadata and flatten config.
  - from_file: Create from file.
  - from_str: Create from string.
  - to_df: Return DataFrame view.
  - head: Preview top rows.
  - get_by_uuid: Access row by UUID.
  - get_by_label: Access row by label.
  - join_with: Join with other DataFrames.
  - filter_by_uuid: Filter by UUID set.
  - filter_by_label: Filter by label set.
  - list_uuids: List UUIDs.
  - list_labels: List non-null labels.

- DatapointAPI: Retrieve datapoint metadata from a REST API.
  - get_all_uuids: UUIDs per device.
  - get_all_metadata: Metadata per device.
  - display_dataframe: Print DataFrames for devices.

- DatapointDB: Retrieve datapoint metadata from PostgreSQL.
  - get_all_uuids: UUIDs per device.
  - get_all_metadata: Metadata per device.
  - display_dataframe: Print DataFrames for devices.

- AzureBlobFlexibleFileLoader: Load arbitrary file types from Azure Blob Storage
  under time-structured folders.

- DatabricksUnityParquetLoader: Load canonical parquet files governed by
  Databricks Unity Catalog.

- DatabricksUnityEnergyLoader: Load CSV energy timeseries and series metadata
  governed by Unity Catalog.

- DatabricksSparkParquetLoader: Read the canonical hourly parquet layout through
  a Spark session.

- EnergyAPILoader: Load energy timeseries data from REST APIs.

- ContextEnricher: Enrich timeseries DataFrames with contextual metadata.
"""

import importlib as _importlib
from typing import TYPE_CHECKING

# Explicit imports for type checkers and IDEs only (so ``from ts_shape import X``
# is typed as X, not object); at runtime __getattr__ below imports lazily.
# tests/typing/test_lazy_exports.py keeps this block in sync with the map.
if TYPE_CHECKING:
    from ts_shape.loader.combine.integrator import (
        DataIntegratorHybrid as DataIntegratorHybrid,
    )
    from ts_shape.loader.context.context_enricher import (
        ContextEnricher as ContextEnricher,
    )
    from ts_shape.loader.metadata.metadata_api_loader import (
        DatapointAPI as DatapointAPI,
    )
    from ts_shape.loader.metadata.metadata_db_loader import DatapointDB as DatapointDB
    from ts_shape.loader.metadata.metadata_json_loader import (
        MetadataJsonLoader as MetadataJsonLoader,
    )
    from ts_shape.loader.timeseries.azure_blob_energy_loader import (
        AzureBlobEnergyLoader as AzureBlobEnergyLoader,
    )
    from ts_shape.loader.timeseries.azure_blob_loader import (
        AzureBlobFlexibleFileLoader as AzureBlobFlexibleFileLoader,
    )
    from ts_shape.loader.timeseries.azure_blob_loader import (
        AzureBlobParquetLoader as AzureBlobParquetLoader,
    )
    from ts_shape.loader.timeseries.databricks_spark_parquet_loader import (
        DatabricksSparkParquetLoader as DatabricksSparkParquetLoader,
    )
    from ts_shape.loader.timeseries.databricks_unity_energy_loader import (
        DatabricksUnityEnergyLoader as DatabricksUnityEnergyLoader,
    )
    from ts_shape.loader.timeseries.databricks_unity_parquet_loader import (
        DatabricksUnityParquetLoader as DatabricksUnityParquetLoader,
    )
    from ts_shape.loader.timeseries.energy_api_loader import (
        EnergyAPILoader as EnergyAPILoader,
    )
    from ts_shape.loader.timeseries.parquet_loader import ParquetLoader as ParquetLoader
    from ts_shape.loader.timeseries.s3proxy_parquet_loader import (
        S3ProxyDataAccess as S3ProxyDataAccess,
    )
    from ts_shape.loader.timeseries.timescale_loader import (
        TimescaleDBDataAccess as TimescaleDBDataAccess,
    )

# The classes listed above, importable from the pack. They are imported on
# first access, so a class whose optional dependency is missing only fails
# when it is used.
_CLASSES = {
    "AzureBlobEnergyLoader": "ts_shape.loader.timeseries.azure_blob_energy_loader",
    "AzureBlobFlexibleFileLoader": "ts_shape.loader.timeseries.azure_blob_loader",
    "AzureBlobParquetLoader": "ts_shape.loader.timeseries.azure_blob_loader",
    "ContextEnricher": "ts_shape.loader.context.context_enricher",
    "DataIntegratorHybrid": "ts_shape.loader.combine.integrator",
    "DatabricksSparkParquetLoader": (
        "ts_shape.loader.timeseries.databricks_spark_parquet_loader"
    ),
    "DatabricksUnityEnergyLoader": (
        "ts_shape.loader.timeseries.databricks_unity_energy_loader"
    ),
    "DatabricksUnityParquetLoader": (
        "ts_shape.loader.timeseries.databricks_unity_parquet_loader"
    ),
    "DatapointAPI": "ts_shape.loader.metadata.metadata_api_loader",
    "DatapointDB": "ts_shape.loader.metadata.metadata_db_loader",
    "EnergyAPILoader": "ts_shape.loader.timeseries.energy_api_loader",
    "MetadataJsonLoader": "ts_shape.loader.metadata.metadata_json_loader",
    "ParquetLoader": "ts_shape.loader.timeseries.parquet_loader",
    "S3ProxyDataAccess": "ts_shape.loader.timeseries.s3proxy_parquet_loader",
    "TimescaleDBDataAccess": "ts_shape.loader.timeseries.timescale_loader",
}

__all__ = sorted(_CLASSES)


def __getattr__(name: str) -> object:
    if name in _CLASSES:
        return getattr(_importlib.import_module(_CLASSES[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_CLASSES))
