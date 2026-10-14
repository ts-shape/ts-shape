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

`mypy` runs the whole tree under a modest global baseline. Two override blocks
in `pyproject.toml` drive an incremental rollout toward full strictness:

- **Strict allowlist** — modules held to `disallow_untyped_defs`,
  `disallow_incomplete_defs`, `warn_return_any` and `warn_unused_ignores`. New
  code should land here.
- **Typing debt** — older modules quarantined with `ignore_errors` so the gate
  stays green.

To chip away at the debt, pick one module, remove it from the debt list, fix the
reported errors, then (ideally) add it to the strict allowlist:

```bash
# See what a quarantined module needs:
mypy src/ts_shape/features/cross_signal.py \
  --check-untyped-defs --disallow-untyped-defs --warn-return-any --warn-unused-ignores
```

---

## 3. Build Distribution Packages

```bash
python -m build
```
> Creates source and wheel distributions in the `dist/` directory.

---

## 4. Publish to PyPI

```bash
twine upload dist/* --verbose --skip-existing
```
> Uploads your package to [PyPI](https://pypi.org/).  
> **Tip:** Ensure your credentials are set up in `~/.pypirc`.

---

## 5. Automatic Version Bumping and Publishing (CI)

Two workflows run in GitHub Actions:

- `.github/workflows/ci.yml` runs on every push and pull request. It lints
  (black, ruff, mypy), builds the docs with `mkdocs build --strict`, and runs
  the tests on Python 3.11 to 3.14.
- `.github/workflows/release.yml` runs on every push to `main`:
  1. it runs the tests;
  2. it takes the latest `X.Y.Z` tag, increments the patch number, and pushes
     the new tag;
  3. it builds the docs and deploys them to GitHub Pages;
  4. it builds the sdist and wheel with that version (injected into
     `pyproject.toml`; locally `setuptools-scm` derives it from the tags);
  5. it publishes them to PyPI.

The bump is always a patch bump, whatever the commit message says. For a minor
or major release, push the new tag yourself before merging:

```bash
git tag -a 1.0.0 -m "Release 1.0.0"
git push origin 1.0.0
```

The next merge to `main` then releases `1.0.1`.

One-time repo setting required: enable "Read and write permissions" for GitHub
Actions under Settings → Actions → General → Workflow permissions.

## 6. Manage Requirements (pip-tools)

Keep only direct dependencies in `requirements.in` and compile pinned versions into `requirements.txt`.

Setup (once per environment):

```bash
python -m pip install --upgrade pip-tools
```

Compile/update pins:

```bash
# Compile requirements.in -> requirements.txt
python scripts/requirements.py compile

# Upgrade all pins to latest compatible versions
python scripts/requirements.py upgrade
```

Sync your virtualenv exactly to `requirements.txt` (adds/removes packages):

```bash
python scripts/requirements.py sync
```

Notes:
- Edit direct deps in `requirements.in` (not `requirements.txt`).
- `pip-sync` will uninstall anything not listed in `requirements.txt`.
