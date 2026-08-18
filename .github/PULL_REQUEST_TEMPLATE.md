## What & Why

<!-- What does this change, and what problem does it solve? Link any issue. -->

## Changes

<!-- The notable changes, one bullet each. Mention new public classes or methods
     by name so reviewers can find them. -->

-

## Behaviour Change

<!-- Delete the lines that do not apply. -->

- [ ] No behaviour change for existing callers (new code, or opt-in via a new keyword argument)
- [ ] Behaviour changes for existing callers — described below, with the reasoning
- [ ] Breaking API change — the migration path is documented in `docs/migration.md`

## Verification

<!-- What did you actually run? Paste the relevant result, not just a tick. -->

- [ ] `pytest` passes
- [ ] `black --check src/ tests/` and `ruff check src/ tests/` pass
- [ ] `mypy` passes
- [ ] New or changed behaviour is covered by a test that fails without the change

## Documentation

- [ ] Docstrings updated (Google style — Args / Returns / Raises)
- [ ] Guide or module page updated under `docs/`, if user-facing
- [ ] `CHANGES.txt` entry added, in the existing `add:` / `fix:` / `build:` style
