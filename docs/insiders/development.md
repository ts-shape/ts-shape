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

## 2b. Build the Documentation

```bash
pip install -e ".[docs]"
mkdocs build       # blocking in CI
mkdocs serve       # live preview on http://127.0.0.1:8000
```

The site is generated **from the source tree** — mkdocstrings reference pages, the
architecture graph, and `--8<--` snippet includes of the example scripts. A
renamed module, a nav entry pointing at a file that does not exist, or an example
script moved out from under `examples/` therefore breaks the docs build and
nothing else. CI builds the docs on every branch so that surfaces before merge
rather than in the release pipeline afterwards.

`mkdocs build --strict` is the goal but not yet the gate: the tree still carries
docstring-indentation warnings from griffe in a handful of older modules. Fixing
those (continuation lines in Google-style docstrings indented 6 spaces instead of
8) is a good self-contained contribution; once the count reaches zero, add
`--strict` to the CI step.

---

## 2c. Quality Gates at a Glance

Everything CI enforces, in the order it is cheapest to run locally:

| Gate | Command | Notes |
|------|---------|-------|
| Formatting | `black --check src/ tests/` | Pinned version — see `[project.optional-dependencies] dev`. |
| Linting | `ruff check src/ tests/` | pyupgrade + isort rules are enabled. |
| Types | `mypy` | Config in `pyproject.toml`; see the strict rollout above. |
| Tests | `pytest` | Coverage floor in `[tool.coverage.report] fail_under`; ratchet it up when it rises. |
| Docs | `mkdocs build` | See above. |

---

## 3. Build Distribution Packages

```bash
python setup.py sdist bdist_wheel
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

This repo is configured to auto-bump the version in `setup.py`, create a Git tag, and publish to PyPI on pushes to `main`.

- Workflow: `.github/workflows/auto_bump_version.yml`
- Publish on tags: `.github/workflows/pypi-packaging.yml` (triggers on `v*` tags)

Keep the version declaration in `setup.py` in this exact form (the trailing comma is fine):

```python
setuptools.setup(
    name="ts_shape",
    version = "0.0.0.24",
    # ...
)
```

On every push to `main`, the auto-bump workflow reads that line and increments it based on the latest commit message:

- Major: include `BREAKING CHANGE`, `#major`, or the short `!:` in the subject
- Minor: start the subject with `feat` or include `#minor`
- Patch: default for all other commits

Examples:

```text
feat: add new SPC rule 9
# => bumps 0.0.0.24 -> 0.1.0 and tags v0.1.0

fix: handle NaNs in StringFilter
# => bumps 0.1.0 -> 0.1.1 and tags v0.1.1

refactor!: remove deprecated API (BREAKING CHANGE)
# => bumps 0.1.1 -> 1.0.0 and tags v1.0.0
```

The workflow then:
- Commits the updated `setup.py` with `[skip ci]` to avoid loops
- Creates and pushes a tag `vX.Y.Z`
- The packaging workflow sees the tag and publishes the built artifacts to PyPI

One‑time repo setting required: enable “Read and write permissions” for GitHub Actions under
Settings → Actions → General → Workflow permissions.

---

## 6. Manage Requirements (pip‑tools)

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
