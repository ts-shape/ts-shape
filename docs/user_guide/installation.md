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
from ts_shape.transform.filter.numeric_filter import NumericFilter

print("ts-shape installed successfully!")
```

## Optional Dependencies

Install extras for specific backends:

```bash
# Parquet support (recommended)
pip install pyarrow

# S3 storage
pip install s3fs

# Azure Blob storage
pip install azure-storage-blob

# TimescaleDB / PostgreSQL
pip install ts-shape[postgres]
```

## Development Installation

For contributing:

```bash
git clone https://github.com/ts-shape/ts-shape.git
cd ts-shape

# Install in development mode
pip install -e .

# Install dev dependencies
pip install pytest black flake8

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
