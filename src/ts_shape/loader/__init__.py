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
"""
