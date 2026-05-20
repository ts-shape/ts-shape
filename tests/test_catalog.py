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
