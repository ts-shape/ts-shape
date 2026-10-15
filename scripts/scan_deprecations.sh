#!/usr/bin/env bash
# Deprecation and vulnerability scan. CI's `scan` job runs this script; run it
# locally the same way.
#
#   pip install -e ".[dev]" pip-audit    # once; this script installs nothing
#   scripts/scan_deprecations.sh [OUTPUT_DIR]
#
# Reports go to OUTPUT_DIR (default: scan-results/):
#   deprecation-warnings.txt  tests run with DeprecationWarning as an error
#   mypy-full.txt             full mypy output
#   mypy-deprecated.txt       only the [deprecated] (PEP 702) findings
#   pip-audit.json/.txt       known vulnerabilities in the environment
#   outdated.json/.txt        installed packages with newer releases
#
# The scan is informational: it always exits 0.
set -u

out="${1:-scan-results}"
mkdir -p "$out"
py="${PYTHON:-python}"

echo "== pytest deprecations =="
"$py" -m pytest -p no:cacheprovider \
    -W error::DeprecationWarning -W error::PendingDeprecationWarning \
    --tb=line -q 2>&1 | tee "$out/deprecation-warnings.txt"

echo
echo "== mypy @deprecated (PEP 702) =="
"$py" -m mypy 2>&1 | tee "$out/mypy-full.txt" >/dev/null
grep '\[deprecated\]' "$out/mypy-full.txt" | tee "$out/mypy-deprecated.txt" || echo "(none)"

echo
echo "== pip-audit =="
if "$py" -m pip_audit --version >/dev/null 2>&1; then
    "$py" -m pip_audit --format=json --output "$out/pip-audit.json" || true
    "$py" -m pip_audit 2>&1 | tee "$out/pip-audit.txt"
else
    echo "pip-audit is not installed; skipped (pip install pip-audit)" | tee "$out/pip-audit.txt"
fi

echo
echo "== outdated packages =="
"$py" -m pip list --outdated --format=json > "$out/outdated.json" 2>/dev/null || true
"$py" -m pip list --outdated 2>&1 | tee "$out/outdated.txt"

exit 0
