"""Tests for the public synthetic-data generator."""

import pandas as pd  # type: ignore
import pytest

import ts_shape
from ts_shape.datasets import make_timeseries

_SCHEMA = (
    "systime",
    "uuid",
    "value_bool",
    "value_integer",
    "value_double",
    "value_string",
    "is_delta",
)


def test_make_timeseries_standard_schema():
    df = make_timeseries(["s1"], n_points=50)
    assert len(df) == 50
    for col in _SCHEMA:
        assert col in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["systime"])


def test_make_timeseries_multiple_uuids():
    df = make_timeseries(["a", "b"], n_points=100, n_outliers=3)
    assert set(df["uuid"]) == {"a", "b"}
    assert len(df) == 200


def test_make_timeseries_is_reproducible_with_seed():
    a = make_timeseries(["s"], n_points=20, seed=7)
    b = make_timeseries(["s"], n_points=20, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_make_timeseries_integer_and_bool_columns():
    di = make_timeseries(["s"], n_points=10, value_column="value_integer")
    assert pd.api.types.is_integer_dtype(di["value_integer"])
    db = make_timeseries(["s"], n_points=10, value_column="value_bool")
    assert pd.api.types.is_bool_dtype(db["value_bool"])


def test_make_timeseries_rejects_unsupported_value_column():
    with pytest.raises(ValueError):
        make_timeseries(["s"], value_column="value_string")


def test_make_timeseries_rejects_non_positive_n_points():
    with pytest.raises(ValueError):
        make_timeseries(["s"], n_points=0)


def test_make_timeseries_exported_at_top_level():
    assert ts_shape.make_timeseries is make_timeseries


def test_make_timeseries_feeds_a_detector():
    # The headline use case: generate data, run a detector, no setup.
    df = make_timeseries(["sensor:temp"], n_points=200, n_outliers=4)
    events = ts_shape.OutlierDetectionEvents(
        df, value_column="value_double"
    ).detect_outliers_zscore()
    assert "source_uuid" in events.columns
