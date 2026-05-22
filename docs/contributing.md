# Contributing

Thank you for your interest in contributing to **ts-shape**! This page covers the licensing landscape of the project and its dependencies so that contributors can make informed decisions when adding or updating packages.

For development setup, building, testing, and publishing workflows, see the [Development Guide](insiders/development.md).

---

## Project License

ts-shape is released under the **MIT License**. See the full license text in [`LICENSE.txt`](../LICENSE.txt).

All contributions to this repository are accepted under the same MIT License.

---

## Third-Party Dependency Licenses

The table below lists every runtime dependency declared in `pyproject.toml`, its license, and what it is used for in ts-shape.

| Dependency | Version | License | Purpose |
|---|---|---|---|
| **pandas** | >= 2.1.0 | BSD-3-Clause | Core dataframe operations |
| **numpy** | >= 1.26.4 | BSD-3-Clause | Numerical computations |
| **scipy** | >= 1.13.1 | BSD-3-Clause | Scientific and statistical functions |
| **sqlalchemy** | >= 2.0.32 | MIT | Database connectivity (TimescaleDB loader) |
| **azure-storage-blob** | >= 12.19.1 | MIT | Azure Blob Storage loader |
| **s3fs** | >= 2024.10.0 | BSD-3-Clause | S3 file system access |
| **requests** | >= 2.32.3 | Apache-2.0 | HTTP requests |

### Optional Dependency Licenses

The packages below are **not installed by default**. They are declared as optional extras in `pyproject.toml` and are imported lazily via guarded `try/except ImportError` blocks, so they do not affect the license footprint of a base `pip install ts-shape`.

| Dependency | Version | License | Extra | Purpose |
|---|---|---|---|---|
| **pint** | >= 0.24 | BSD-3-Clause | `units` / `dev` | Engineering unit conversion (`UnitConverter`) |
| **scikit-learn** | >= 1.3.0 | BSD-3-Clause | `ml` | Optional ML-based detectors (e.g. Isolation Forest outliers) |
| **psycopg2-binary** | >= 2.9.9 | LGPL-3.0-or-later (with exceptions) | `postgres` | PostgreSQL / TimescaleDB driver |

`psycopg2-binary` is LGPL-licensed, but it is a separately installed, dynamically linked database driver — ts-shape neither bundles nor redistributes it, so its LGPL terms do not extend to ts-shape itself.

### License Compatibility

All dependencies are **compatible with the MIT License** and use permissive licenses:

- **BSD-3-Clause** (pandas, numpy, scipy, s3fs, pint, scikit-learn) — permissive, no restrictions beyond attribution.
- **MIT** (sqlalchemy, azure-storage-blob) — same terms as ts-shape itself.
- **Apache-2.0** (requests) — permissive, compatible with MIT distribution.

PostgreSQL connectivity is handled through SQLAlchemy, which supports any DB-API 2.0 compatible driver. Users can install the driver of their choice (e.g. `pip install ts-shape[postgres]` for psycopg2-binary, or install psycopg, pg8000, etc.).

### Guidelines for Adding New Dependencies

When proposing a new dependency, please consider:

1. **License compatibility** — MIT, BSD, and Apache-2.0 are preferred. Avoid GPL-licensed packages as they are incompatible with MIT distribution.
2. **Necessity** — prefer the standard library or existing dependencies where possible.
3. **Maintenance** — choose well-maintained packages with active communities.
4. **Update `requirements.in`** — add direct dependencies there, then run `python scripts/requirements.py compile` to pin versions (see [Development Guide](insiders/development.md#6-manage-requirements-pip-tools)).
