#!/usr/bin/env bash
set -u

pip install -e ".[dev]" pip-audit >/dev/null

echo "== pytest deprecations =="
pytest -W error::DeprecationWarning -W error::PendingDeprecationWarning --tb=line -q || true

echo
echo "== ruff pyupgrade (UP) =="
ruff check --select UP src/ tests/ || true

echo
echo "== mypy @deprecated (PEP 702) =="
mypy || true

echo
echo "== pip-audit =="
pip-audit || true

echo
echo "== outdated packages =="
pip list --outdated || true
