# Contributing

Contributions are accepted under the project's MIT license
([`LICENSE.txt`](https://github.com/ts-shape/ts-shape/blob/main/LICENSE.txt)).
Setup, tests, linting and the release process are described in
[Development](insiders/development.md); naming and output rules for new
detectors in [Naming and output conventions](naming.md).

## Dependencies and their licenses

Runtime dependencies (`dependencies` in `pyproject.toml`):

| Package | Version | License |
|---|---|---|
| pandas | >= 2.1.0 | BSD-3-Clause |
| numpy | >= 1.26.4 | BSD-3-Clause |
| scipy | >= 1.13.1 | BSD-3-Clause |

Optional extras, installed only on request and imported lazily:

| Package | Version | License | Extra |
|---|---|---|---|
| pyarrow | >= 15.0 | Apache-2.0 | `parquet`, `s3`, `azure` |
| s3fs | >= 2024.10.0 | BSD-3-Clause | `s3` |
| azure-storage-blob | >= 12.19.1 | MIT | `azure` |
| requests | >= 2.32.3 | Apache-2.0 | `api` |
| sqlalchemy | >= 2.0.32 | MIT | `sql`, `postgres` |
| psycopg2-binary | >= 2.9.9 | LGPL-3.0-or-later (with exceptions) | `postgres` |
| pyyaml | >= 6.0 | MIT | `yaml` |
| pint | >= 0.24 | BSD-3-Clause | `units` |
| scikit-learn | >= 1.3.0 | BSD-3-Clause | `ml` |

All are compatible with distributing ts-shape under MIT. psycopg2-binary is
LGPL, but it is a separately installed, dynamically loaded database driver
that ts-shape neither bundles nor redistributes. SQLAlchemy works with any
DB-API driver, so users can install psycopg, pg8000 or another driver instead.

## Adding a dependency

1. Prefer the standard library or an existing dependency.
2. Use MIT, BSD or Apache-2.0 licensed packages; GPL packages cannot be used.
3. Prefer an optional extra over a core dependency, and import it lazily.
4. Add it to `pyproject.toml` and run `python scripts/requirements.py compile`
   (see [Development](insiders/development.md#requirements-pip-tools)).
