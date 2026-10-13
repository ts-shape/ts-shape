"""Registry-wide invariants, pinned to a baseline (a ratchet).

See :mod:`tests.invariants.harness` for the checks. A status that differs
from ``baseline.json`` fails in *either* direction:

* ``ok`` -> ``fail``: a regression -- fix the detector.
* ``fail`` -> ``ok``: an improvement -- regenerate the baseline so it can
  never regress again::

      python -m tests.invariants.harness --update
"""

import pytest

from . import harness

_BASELINE = harness.load_baseline()
_ENTRIES = harness.entries()


def test_baseline_covers_registry():
    ids = {f"{c}.{m}" for c, m in _ENTRIES}
    assert set(_BASELINE) == ids, (
        "baseline.json is out of sync with the detector REGISTRY; run "
        "`python -m tests.invariants.harness --update`"
    )


def _optional_dependency(status: str) -> bool:
    return status.startswith("n/a: optional dependency")


@pytest.mark.parametrize(
    ("cls_name", "method"), _ENTRIES, ids=[f"{c}.{m}" for c, m in _ENTRIES]
)
def test_invariants_match_baseline(cls_name, method):
    key = f"{cls_name}.{method}"
    expected = _BASELINE.get(key, {})
    actual = harness.check(cls_name, method)
    for check in harness.CHECKS:
        want, got = expected.get(check), actual[check]
        if _optional_dependency(got) or (want and _optional_dependency(want)):
            continue
        assert got == want, f"{key} [{check}]: baseline {want!r}, now {got!r}. " + (
            "Regression: fix the detector."
            if want == "ok"
            else "Changed: if this is an improvement, run "
            "`python -m tests.invariants.harness --update`."
        )
