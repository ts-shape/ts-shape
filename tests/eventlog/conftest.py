"""Test fixtures local to ``tests/eventlog``.

* ``_eventlog_clean_overrides`` (autouse) — clears the
  ``ts_shape.eventlog.adapters._OVERRIDES`` registry before each test so
  fixtures registered in one test don't leak into the next.
"""
from __future__ import annotations

import pytest

from ts_shape.eventlog import adapters


@pytest.fixture(autouse=True)
def _eventlog_clean_overrides():
    """Snapshot/restore the override registry around every test."""
    snapshot = dict(adapters._OVERRIDES)
    try:
        yield
    finally:
        adapters._OVERRIDES.clear()
        adapters._OVERRIDES.update(snapshot)
