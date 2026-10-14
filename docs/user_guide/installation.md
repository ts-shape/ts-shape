# Installation

ts-shape needs Python 3.11 or later. The core install depends only on pandas,
numpy and scipy.

```bash
pip install ts-shape
python -c "import ts_shape; print(ts_shape.__version__)"
```

## Optional extras

Loader backends and a few detectors need extra packages. A loader whose
backend is missing raises an `ImportError` that names the extra to install.

| Extra | Installs | Needed for |
|---|---|---|
| `parquet` | pyarrow | reading and writing parquet files |
| `s3` | s3fs, pyarrow | `S3ProxyDataAccess` and other S3 loaders |
| `azure` | azure-storage-blob, pyarrow | `AzureBlobParquetLoader`, `AzureBlobFlexibleFileLoader` |
| `api` | requests | REST metadata and energy API loaders |
| `sql` | sqlalchemy | TimescaleDB and SQL metadata loaders |
| `postgres` | sqlalchemy, psycopg2-binary | `sql` plus the PostgreSQL driver |
| `yaml` | pyyaml | `SignalSpecs.from_yaml`, `ts_shape.eventlog.load_yaml` |
| `units` | pint | `UnitConverter` |
| `ml` | scikit-learn | `OutlierDetectionEvents.detect_outliers_isolation_forest` |
| `all` | all of the above | |

```bash
pip install "ts-shape[parquet,azure]"
```

For Azure AAD authentication, also install `azure-identity`.

## From source

```bash
git clone https://github.com/ts-shape/ts-shape.git
cd ts-shape
pip install -e ".[dev]"   # editable install with test and lint tools
pytest
```

See [Development](../insiders/development.md) for linting, the docs build and
the release process.
