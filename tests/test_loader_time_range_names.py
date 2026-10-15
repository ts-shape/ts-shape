"""Loader time ranges are ``start`` / ``end``; the old names still work."""

import inspect

import pytest

from ts_shape.loader.timeseries.databricks_spark_parquet_loader import (
    DatabricksSparkParquetLoader,
)
from ts_shape.loader.timeseries.parquet_loader import ParquetLoader
from ts_shape.utils.deprecation import deprecated_attribute


def test_parquet_loader_signature_uses_start_end():
    params = inspect.signature(ParquetLoader.load_by_time_range).parameters
    assert "start" in params and "end" in params
    assert "start_time" not in params


def test_old_keyword_names_warn_and_give_the_same_paths():
    loader = DatabricksSparkParquetLoader("/data", spark=object())
    new = loader.build_paths("2024-01-01 09:30", "2024-01-01 11:00", "u1")
    with pytest.warns(DeprecationWarning, match="_timestamp. is deprecated"):
        old = loader.build_paths(
            start_timestamp="2024-01-01 09:30",
            end_timestamp="2024-01-01 11:00",
            uuids="u1",
        )
    assert old == new


def test_deprecated_attribute_forwards_with_warning():
    class Range:
        start_timestamp = deprecated_attribute("start")

        def __init__(self):
            self.start = 1

    r = Range()
    with pytest.warns(DeprecationWarning, match=r"Range\.start_timestamp"):
        assert r.start_timestamp == 1
    with pytest.warns(DeprecationWarning):
        r.start_timestamp = 5
    assert r.start == 5
