# Installation

## Requirements

- Python 3.10 or higher
- pandas >= 2.0.0
- numpy >= 1.24.0

## Install from PyPI

```bash
pip install ts-shape
```

## Install from Source

```bash
git clone https://github.com/ts-shape/ts-shape.git
cd ts-shape
pip install -e .
```

## Verify Installation

```python
import ts_shape
from ts_shape.transform.filter.numeric_filter import DoubleFilter

print("ts-shape installed successfully!")
```

## Optional Dependencies

The core install needs only pandas, numpy and scipy. Loader backends are
extras; a loader whose backend is missing raises an `ImportError` naming the
extra to install.

```bash
# Parquet support (recommended)
pip install pyarrow

pip install ts-shape[s3]        # S3 / S3-compatible proxy (s3fs)
pip install ts-shape[azure]     # Azure Blob Storage
pip install ts-shape[api]       # REST metadata / energy APIs (requests)
pip install ts-shape[sql]       # TimescaleDB / SQL metadata (sqlalchemy)
pip install ts-shape[postgres]  # sql + psycopg2 driver
pip install ts-shape[yaml]      # SignalSpecs.from_yaml
pip install ts-shape[all]       # everything above
```

## Development Installation

For contributing:

```bash
git clone https://github.com/ts-shape/ts-shape.git
cd ts-shape

# Install in development mode
pip install -e .

# Install dev dependencies (tests, linters, all loader backends)
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/
```

## Troubleshooting

**ImportError for ts_shape**
```bash
# Check your environment
which python
pip list | grep ts-shape
```

**pandas version conflict**
```bash
pip install --upgrade pandas
```

**Performance with large datasets**
```bash
pip install pyarrow  # Faster parquet loading
```

## Next Steps

- [Quick Start](quick_start.md) - Get started in 5 minutes
- [Guides](../guides/index.md) - Topic-focused guides from data acquisition to shift reports
- [Concept Guide](../concept.md) - Architecture overview
