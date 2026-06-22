"""Timeseries Loaders

Load timeseries data from parquet folders, S3-compatible stores, Azure Blob,
and TimescaleDB.

Classes:
- ParquetLoader: Read parquet files from local folder structures.
  - load_all_files: Load all parquet under a base path.
  - load_by_time_range: Load files within YYYY/MM/DD/HH path range.
  - load_by_uuid_list: Load files matching UUIDs in filenames.
  - load_files_by_time_range_and_uuids: Combine time range and UUID filters.

- S3ProxyDataAccess: Retrieve parquet via an S3-compatible proxy.
  - fetch_data_as_parquet: Save parquet files to a local folder structure.
  - fetch_data_as_dataframe: Return a combined DataFrame.

- AzureBlobParquetLoader: Load parquet from Azure Blob Storage.
  - load_all_files: Load all parquet under an optional prefix.
  - load_by_time_range: Load hourly folders between start and end.
  - stream_by_time_range: Yield (blob, DataFrame) incrementally.
  - load_files_by_time_range_and_uuids: Load per-hour per-UUID parquet files.
  - stream_files_by_time_range_and_uuids: Yield per-UUID frames incrementally.
  - list_structure: List folders and files under a prefix.

- DatabricksUnityParquetLoader: Load canonical parquet governed by Unity Catalog.
  Reads the same parquet files directly from a FUSE-mounted UC Volume
  (/Volumes/<catalog>/<schema>/<volume>/...), for use inside Databricks
  notebooks/pipelines -- no download, no SDK, low resource footprint.
  - load_all_files: Load all parquet under the Volume path (optional prefix).
  - load_by_time_range: Load only the hourly folders between start and end.
  - stream_by_time_range: Yield (path, DataFrame) incrementally (low memory).
  - load_files_by_time_range_and_uuids: Load per-hour per-UUID parquet files.
  - stream_files_by_time_range_and_uuids: Yield per-UUID frames incrementally.
  - list_structure: List folders and files under the Volume path.
  - fetch_data_as_dataframe: Combined DataFrame for Pipeline/DataIntegratorHybrid.

- AzureBlobEnergyLoader: Load CSV energy timeseries and series metadata from Azure Blob.
  - load_series_metadata: Download .meta/series.csv as a DataFrame.
  - load_by_time_range: Load CSVs by date range, optional series filter.
  - load_by_series_ids: Load specific series by ID, optional date filter.
  - stream_by_time_range: Yield (series_id, DataFrame) incrementally.
  - list_series: List all series IDs available in the blob store.

- DatabricksUnityEnergyLoader: Load CSV energy timeseries + metadata governed by Unity Catalog.
  Reads the same .meta/series.csv and csv/YYYY/MM/DD/<series_id>.csv files
  directly from a FUSE-mounted UC Volume, for use inside Databricks
  notebooks/pipelines -- no download, no SDK, low resource footprint.
  - load_series_metadata: Read .meta/series.csv as a DataFrame.
  - load_by_time_range: Load only the day folders between start and end.
  - load_by_series_ids: Load specific series by ID, optional date filter.
  - stream_by_time_range: Yield (series_id, DataFrame) incrementally (low memory).
  - list_series: List all series IDs available in the Volume.
  - fetch_data_as_dataframe: Combined DataFrame for Pipeline/DataIntegratorHybrid.

- AzureBlobFlexibleFileLoader: Load arbitrary file types from Azure Blob Storage.
  - list_files_by_time_range: List matching files (by extension) under hourly folders.
  - iter_file_names_by_time_range: Generator of names without downloading.
  - fetch_files_by_time_range: Download matching files as raw bytes or parsed objects.
  - stream_files_by_time_range: Stream (blob, bytes/parsed) incrementally.
  - fetch_files_by_time_range_and_basenames: Download by explicit basenames.
  - stream_files_by_time_range_and_basenames: Stream by explicit basenames.
  - register_parser/unregister_parser: Plug-in parser functions per file extension.

- TimescaleDBDataAccess: Stream timeseries from TimescaleDB.
  - fetch_data_as_parquet: Partition-by-hour and write parquet.
  - fetch_data_as_dataframe: Return a combined DataFrame.
"""
