"""Regression tests for loader path building, uuid matching and metadata input."""

from __future__ import annotations

import pandas as pd
import pytest

import ts_shape.loader as loader_pkg
from ts_shape.errors import DataQualityWarning
from ts_shape.loader._utils import hourly_slots, parse_utc_times, trim_to_time_range
from ts_shape.loader.metadata.metadata_json_loader import MetadataJsonLoader
from ts_shape.loader.timeseries.databricks_spark_parquet_loader import (
    DatabricksSparkParquetLoader,
)
from ts_shape.loader.timeseries.parquet_loader import ParquetLoader

# ---------- Databricks Spark path building ---------------------------------


def _hours(paths):
    return sorted({p.split("/base/")[1].rsplit("/", 1)[0] for p in paths})


def test_build_paths_floors_mid_hour_start():
    loader = DatabricksSparkParquetLoader("/base", spark=object())
    paths = loader.build_paths("2024-01-01 09:30", "2024-01-01 11:00", "u1")
    assert _hours(paths) == ["2024/01/01/09", "2024/01/01/10", "2024/01/01/11"]


def test_build_paths_uses_utc_folders_for_aware_bounds():
    loader = DatabricksSparkParquetLoader("/base", spark=object())
    paths = loader.build_paths(
        pd.Timestamp("2024-01-01 10:00", tz="Europe/Berlin"),
        pd.Timestamp("2024-01-01 11:00", tz="Europe/Berlin"),
        ["u1"],
    )
    assert _hours(paths) == ["2024/01/01/09", "2024/01/01/10"]


@pytest.mark.parametrize("uuids", ["", "  ", ["u1", ""]])
def test_build_paths_rejects_blank_uuid(uuids):
    loader = DatabricksSparkParquetLoader("/base", spark=object())
    with pytest.raises(ValueError, match="non-empty"):
        loader.build_paths("2024-01-01", "2024-01-01 01:00", uuids)


# ---------- ParquetLoader --------------------------------------------------


@pytest.fixture
def parquet_tree(tmp_path):
    rows = {
        ("2024/01/01/09", "temp"): 1,
        ("2024/01/01/09", "temp_raw"): 2,
        ("2024/01/01/10", "temp"): 3,
    }
    for (folder, uuid), value in rows.items():
        path = tmp_path / folder
        path.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"uuid": [uuid], "value_integer": [value]}).to_parquet(
            path / f"{uuid}.parquet"
        )
    return tmp_path


def test_uuid_match_is_exact(parquet_tree):
    out = ParquetLoader.load_by_uuid_list(str(parquet_tree), ["temp"])
    assert set(out["uuid"]) == {"temp"}
    assert sorted(out["value_integer"]) == [1, 3]


def test_time_range_includes_folder_of_mid_hour_start(parquet_tree):
    out = ParquetLoader.load_files_by_time_range_and_uuids(
        str(parquet_tree),
        pd.Timestamp("2024-01-01 09:30"),
        pd.Timestamp("2024-01-01 10:15"),
        ["temp"],
    )
    assert sorted(out["value_integer"]) == [1, 3]
    out = ParquetLoader.load_by_time_range(
        str(parquet_tree),
        pd.Timestamp("2024-01-01 10:30", tz="Europe/Berlin"),
        pd.Timestamp("2024-01-01 10:45", tz="Europe/Berlin"),
    )
    assert sorted(out["value_integer"]) == [1, 2]


def test_lazy_loader_map_lists_metadata_and_s3_loaders():
    for name in ("DatapointAPI", "DatapointDB", "S3ProxyDataAccess"):
        assert name in loader_pkg.__all__


# ---------- time bounds ----------------------------------------------------


def test_hourly_slots_with_mixed_aware_and_naive_bounds():
    slots = hourly_slots(pd.Timestamp("2024-01-01 09:30", tz="UTC"), "2024-01-01 11:00")
    assert len(slots) == 3
    assert str(slots.tz) == "UTC"


def test_trim_to_time_range_with_mixed_bounds():
    df = pd.DataFrame(
        {"systime": pd.date_range("2024-01-01", periods=4, freq="1h", tz="UTC")}
    )
    out = trim_to_time_range(
        df, "2024-01-01 01:00", pd.Timestamp("2024-01-01 02:00", tz="UTC")
    )
    assert len(out) == 2


def test_parse_utc_times_mixes_offsets_and_naive_strings():
    out = parse_utc_times(pd.Series(["2024-01-01T01:00:00+01:00", "2024-01-01 00:00"]))
    assert out.tolist() == [pd.Timestamp("2024-01-01", tz="UTC")] * 2


# ---------- MetadataJsonLoader ---------------------------------------------


def test_columnar_without_config_raises():
    with pytest.raises(ValueError, match="config"):
        MetadataJsonLoader({"uuid": ["a"], "label": ["A"]})


def test_strict_requires_labels():
    with pytest.raises(ValueError, match="no label"):
        MetadataJsonLoader([{"uuid": "a", "config": {}}], strict=True)
    loader = MetadataJsonLoader([{"uuid": "a", "config": {}}], strict=False)
    assert loader.list_uuids() == ["a"]


def test_non_dict_records_warn():
    with pytest.warns(DataQualityWarning, match="1 metadata record"):
        loader = MetadataJsonLoader(
            [{"uuid": "a", "label": "A", "config": {}}, "junk"], strict=False
        )
    assert loader.list_uuids() == ["a"]


def test_empty_metadata_strict_raises():
    with pytest.raises(ValueError, match="empty"):
        MetadataJsonLoader({}, strict=True)
    assert MetadataJsonLoader({}, strict=False).to_df().empty


def test_get_by_uuid_compares_as_text():
    loader = MetadataJsonLoader([{"uuid": 1, "label": "one", "config": {}}])
    assert loader.get_by_uuid("1")["label"] == "one"
    assert loader.get_by_uuid(1)["label"] == "one"
    assert loader.get_by_uuid("2") is None
