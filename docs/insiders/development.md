# Development

## Setup and tests

```bash
git clone https://github.com/ts-shape/ts-shape.git
cd ts-shape
pip install -e ".[dev]"     # editable install with all extras except postgres, and the dev tools
pytest
```

## Lint and type check

```bash
black --check src/ tests/   # formatting
ruff check src/ tests/      # linting
mypy                        # type checking (configured in pyproject.toml)
```

All three block CI. mypy checks `src/ts_shape` with
`disallow_untyped_defs`, `disallow_incomplete_defs`, `warn_return_any` and
`warn_unused_ignores` and no exempt modules, so every function needs full
annotations. pandas results are typed `Any`; where a function returns one
under a narrower declared type, annotate a local (`out: np.ndarray = ...`) or
use `typing.cast` with a comment saying why. Converting with `float()` or
`int()` changes the returned scalar type.

## Docs

```bash
pip install -e ".[docs]"
mkdocs serve
DISABLE_MKDOCS_2_WARNING=true mkdocs build --strict   # what CI runs
pytest tests/docs                                     # code blocks bind against real signatures
```

## Build

```bash
python -m build
twine check --strict dist/*
```

The version comes from the git tags (setuptools-scm): a checkout 13 commits
after `0.0.62` builds `0.0.63.dev13`. Releases are published by CI; do not
upload by hand.

## Releases

Two GitHub Actions workflows:

- `.github/workflows/ci.yml` runs on every pull request and on every push to a
  branch other than `main`. It lints (black, ruff, mypy), builds the docs with
  `mkdocs build --strict`, runs the tests on Python 3.11 to 3.14, and builds
  the package (`twine check --strict`, no `0.0.0` version, the wheel imports).
- `.github/workflows/release.yml` runs on every push to `main`:
  1. it runs the full CI workflow;
  2. it decides the version (`.github/scripts/next_version.py`);
  3. it builds and checks the sdist and wheel with that version;
  4. it publishes them to PyPI;
  5. only then it pushes the `X.Y.Z` tag and creates the GitHub release;
  6. it builds the docs and deploys them to GitHub Pages.

Every merge to `main` is a release, a patch release by default. To choose
another version, add a trailer to the message of any commit since the last
release (the most recent trailer wins):

```text
Release-Bump: minor      # patch | minor | major | none (no release)
Release-As: 0.1.0        # an exact version, higher than the last release
```

A manual run of the Release workflow (Actions, Release Pipeline, Run workflow)
takes the same choice as inputs.

One-time repository settings: allow "Read and write permissions" for GitHub
Actions (Settings, Actions, General, Workflow permissions) and set the
`PYPI_API_TOKEN` secret, or configure PyPI Trusted Publishing for
`release.yml` and drop the token.

## Dependencies

Dependencies are declared only in `pyproject.toml`: the core ones under
`[project].dependencies` (with minimum versions; CI tests those floors in the
`min-deps` job) and the optional ones as extras. There are no requirements
files; Renovate updates `pyproject.toml`, the pre-commit hooks and the
GitHub Actions directly.
