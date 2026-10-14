# Development Guide

## 1. Install Package Locally

```bash
pip install -e .
```
> Installs your package in editable mode for local development.

---

## 2. Run Tests

```bash
pytest ./tests
```
> Executes all tests to ensure your code is working as expected.

---

## 2a. Lint and Type Check

```bash
black --check src/ tests/   # formatting
ruff check src/ tests/      # linting
mypy                        # type checking (config in pyproject.toml)
```

All three are blocking in CI.

### Incremental strict typing

`mypy` runs the whole tree with `disallow_untyped_defs`,
`disallow_incomplete_defs`, `warn_return_any` and `warn_unused_ignores` on
globally, so every new function needs full annotations.

A single **strict exemptions** override block in `pyproject.toml` lists the few
modules that do not pass those flags yet. To pay one off, delete its entry,
run `mypy` to see what it needs, and fix the reported errors (per-module
settings win over command-line flags, so the module must leave the list before
mypy reports on it).

pandas ships no type information here, so pandas results are `Any`. Where a
function returns one under a narrower declared type, annotate a local
(`out: np.ndarray = ...`) or use `typing.cast` with a comment saying why;
converting with `float()`/`int()` changes the returned scalar type.

---

## 3. Build Distribution Packages

```bash
python -m build
```
> Creates source and wheel distributions in the `dist/` directory.

---

## 4. Publish to PyPI

Releases are published by CI (next section); do not upload by hand. To check a
build locally:

```bash
python -m build
twine check --strict dist/*
```

The version comes from the git tags (`setuptools-scm`): a checkout 13 commits
after `0.0.62` builds `0.0.63.dev13`.

---

## 5. Releases (CI)

Two workflows run in GitHub Actions:

- `.github/workflows/ci.yml` runs on every pull request and every push to a
  branch other than `main`. It lints (black, ruff, mypy), builds the docs with
  `mkdocs build --strict`, runs the tests on Python 3.11 to 3.14, and builds
  the package (`twine check --strict`, no `0.0.0` version, the wheel imports).
- `.github/workflows/release.yml` runs on every push to `main`:
  1. it runs the full CI workflow;
  2. it decides the version (`.github/scripts/next_version.py`);
  3. it builds the sdist and wheel with that version and checks them;
  4. it publishes them to PyPI;
  5. only then it pushes the `X.Y.Z` tag and creates the GitHub release;
  6. it builds the docs and deploys them to GitHub Pages.

Every merge is a release. By default it is a patch release. To choose another
version, put a trailer in the message of any commit since the last release
(the most recent trailer wins):

```text
Release-Bump: minor      # patch | minor | major | none (no release)
Release-As: 0.1.0        # an exact version, higher than the last release
```

A manual run of the Release workflow (Actions → Release Pipeline → Run
workflow) takes the same choice as inputs.

Merges normally need no trailer: each one publishes the next patch version.
Renamed names keep working with a `DeprecationWarning` and are removed only in
a major release.

One-time repo settings: enable "Read and write permissions" for GitHub Actions
under Settings → Actions → General → Workflow permissions, and set the
`PYPI_API_TOKEN` secret (or configure PyPI Trusted Publishing for
`release.yml` and drop the token).

## 6. Manage Requirements (pip-tools)

Direct dependencies live in `pyproject.toml`; `requirements.txt` pins the core dependencies (Renovate refreshes it in CI).

Setup (once per environment):

```bash
python -m pip install --upgrade pip-tools
```

Compile/update pins:

```bash
# Compile pyproject.toml -> requirements.txt
python scripts/requirements.py compile

# Upgrade all pins to latest compatible versions
python scripts/requirements.py upgrade
```

Sync your virtualenv exactly to `requirements.txt` (adds/removes packages):

```bash
python scripts/requirements.py sync
```

Notes:
- Edit direct deps in `pyproject.toml` (not `requirements.txt`).
- `pip-sync` will uninstall anything not listed in `requirements.txt`.
