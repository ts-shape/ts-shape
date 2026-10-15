"""Tests for the programmatic detector catalog."""

import ts_shape
from ts_shape.catalog import list_detectors


def test_list_detectors_returns_expected_columns():
    df = list_detectors()
    assert list(df.columns) == ["name", "category", "module"]
    assert len(df) > 50
    assert "OutlierDetectionEvents" in set(df["name"])


def test_list_detectors_category_filter():
    quality = list_detectors("events.quality")
    assert len(quality) > 0
    assert (quality["category"] == "events.quality").all()
    assert "OutlierDetectionEvents" in set(quality["name"])


def test_list_detectors_events_prefix_filter():
    events = list_detectors("events")
    assert (events["category"].str.startswith("events")).all()
    assert len(events) > len(list_detectors("events.quality"))


def test_list_detectors_is_sorted_and_unique():
    df = list_detectors()
    assert df["name"].is_unique
    assert list(df["category"]) == sorted(df["category"])


def test_list_detectors_exported_at_top_level():
    assert ts_shape.list_detectors is list_detectors


def _listed_by_import(name: str, module: str) -> bool:
    """The catalog rule decided by importing the class (the former algorithm)."""
    import importlib
    import inspect

    try:
        obj = getattr(importlib.import_module(module), name)
    except ImportError:  # optional backend missing: naming convention
        return name[:1].isupper() and not name.isupper()
    return inspect.isclass(obj) and not hasattr(obj, "__deprecated_alias_of__")


def test_list_detectors_matches_import_based_catalog():
    # list_detectors decides from names alone; importing every class must
    # give the same rows.
    from ts_shape import _LAZY
    from ts_shape.catalog import _category_for

    expected = sorted(
        (_category_for(module), name)
        for name, module in _LAZY.items()
        if _listed_by_import(name, module)
    )
    df = list_detectors()
    assert sorted(zip(df["category"], df["name"])) == expected
    assert list(zip(df["category"], df["name"])) == expected


def test_list_detectors_imports_no_detector_module():
    import os
    import subprocess
    import sys
    from pathlib import Path

    src = str(Path(ts_shape.__file__).resolve().parent.parent)
    env = {**os.environ, "PYTHONPATH": src}
    code = (
        "import sys, ts_shape; ts_shape.list_detectors();"
        "loaded = [m for m in sys.modules if m.startswith('ts_shape.events.')"
        " or m in ('sklearn', 'scipy.stats')];"
        "print(loaded)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    assert out.stdout.strip() == "[]", out.stdout
