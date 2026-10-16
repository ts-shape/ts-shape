"""Importing a detector stays cheap: scipy.stats and scikit-learn load on use.

Each check runs in a fresh interpreter, since the test process has long
imported both.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import ts_shape

_SRC = str(Path(ts_shape.__file__).resolve().parent.parent)

#: Generous cap in seconds; the import takes about 0.4 s (pandas alone about
#: 0.3 s) and took 1.2-1.9 s while scipy.stats and scikit-learn were imported
#: eagerly.
IMPORT_TIME_CAP = 1.0


def _run(code: str) -> dict:
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": _SRC},
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


_IMPORT_OUTLIER = """
import json, sys, time
start = time.perf_counter()
from ts_shape import OutlierDetectionEvents
elapsed = time.perf_counter() - start
print(json.dumps({
    "seconds": elapsed,
    "sklearn": "sklearn" in sys.modules,
    "scipy_stats": "scipy.stats" in sys.modules,
}))
"""


def test_outlier_detection_import_skips_scipy_stats_and_sklearn():
    result = _run(_IMPORT_OUTLIER)
    assert not result["sklearn"]
    assert not result["scipy_stats"]


def test_outlier_detection_import_time():
    # Best of two runs, so one slow start on a busy machine does not fail.
    seconds = min(_run(_IMPORT_OUTLIER)["seconds"] for _ in range(2))
    assert seconds < IMPORT_TIME_CAP, f"import took {seconds:.2f} s"


def test_detector_packs_import_without_scipy_stats():
    code = """
import json, sys
import ts_shape.events.quality.outlier_detection
import ts_shape.events.quality.sensor_drift
import ts_shape.events.quality.capability_trending
import ts_shape.events.quality.value_distribution
import ts_shape.events.maintenance.reliability
import ts_shape.events.correlation.root_cause_correlation
print(json.dumps({"loaded": sorted(
    m for m in ("sklearn", "scipy.stats", "scipy.signal", "scipy.spatial")
    if m in sys.modules
)}))
"""
    assert _run(code)["loaded"] == []


def test_isolation_forest_still_works():
    import pytest

    pytest.importorskip("sklearn")
    df = ts_shape.make_timeseries(["temp"], n_outliers=3)
    out = ts_shape.OutlierDetectionEvents(
        df, "value_double"
    ).detect_outliers_isolation_forest(contamination=0.01)
    assert not out.empty


def test_isolation_forest_without_sklearn_names_the_extra(monkeypatch):
    import pytest

    monkeypatch.setitem(sys.modules, "sklearn", None)
    monkeypatch.setitem(sys.modules, "sklearn.ensemble", None)
    df = ts_shape.make_timeseries(["temp"], n_outliers=3)
    events = ts_shape.OutlierDetectionEvents(df, "value_double")
    with pytest.raises(ImportError, match=r"ts-shape\[ml\]"):
        events.detect_outliers_isolation_forest()
