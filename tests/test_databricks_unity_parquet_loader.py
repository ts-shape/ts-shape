import warnings
from pathlib import Path

import pandas as pd  # type: ignore
import pytest

from ts_shape.errors import LoaderConfigWarning
from ts_shape.loader.timeseries.databricks_unity_parquet_loader import (
    DatabricksUnityCatalogParquetLoader,
)


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def _make_volume(tmp_path, prefix="parquet"):
    """Build a mini mounted UC Volume tree with the canonical hourly layout.

    Files are empty placeholders; tests monkeypatch ``pd.read_parquet`` so no
    parquet engine is required (matching test_parquet_loader.py).
    """
    root = tmp_path / "Volumes" / "main" / "plant" / "timeseries"
    base = root / prefix if prefix else root
    _touch(base / "2024/01/01/09/u1.parquet")
    _touch(base / "2024/01/01/10/u1.parquet")
    _touch(base / "2024/01/01/11/u2.parquet")
    return root


def _fake_read_by_stem(p, columns=None, filters=None):
    """Return a 1-row frame whose uuid is the file's stem (e.g. 'u1')."""
    return pd.DataFrame({"uuid": [Path(p).stem]})


def _loader(tmp_path, monkeypatch=None):
    root = _make_volume(tmp_path)
    loader = DatabricksUnityCatalogParquetLoader(volume_path=str(root), prefix="parquet")
    if monkeypatch is not None:
        monkeypatch.setattr(pd, "read_parquet", _fake_read_by_stem)
    return loader, root


def test_resolve_from_volume_path(tmp_path):
    loader, root = _loader(tmp_path)
    assert loader.base_path == root / "parquet"


def test_resolve_from_catalog_parts(tmp_path):
    root = _make_volume(tmp_path)
    loader = DatabricksUnityCatalogParquetLoader(
        catalog="main",
        schema="plant",
        volume="timeseries",
        prefix="parquet",
        base_path=str(tmp_path / "Volumes"),
    )
    assert loader.base_path == root / "parquet"


def test_missing_parts_raises_value_error():
    with pytest.raises(ValueError, match="volume_path"):
        DatabricksUnityCatalogParquetLoader(catalog="main", schema="plant")


def test_hour_dir_and_slots(tmp_path):
    loader, _ = _loader(tmp_path)
    hd = loader._hour_dir(pd.Timestamp("2024-01-01 09:05:00"))
    assert str(hd).endswith("parquet/2024/01/01/09")
    slots = list(loader._hourly_slots("2024-01-01 08:00:00", "2024-01-01 10:00:00"))
    assert len(slots) == 3


def test_load_all_and_list_structure(tmp_path, monkeypatch):
    loader, _ = _loader(tmp_path, monkeypatch)
    all_df = loader.load_all_files()
    assert set(all_df["uuid"]) == {"u1", "u2"}

    listed = loader.list_structure()
    assert all(f.endswith(".parquet") for f in listed["files"])
    assert len(listed["files"]) == 3
    assert all(p.endswith("/") for p in listed["folders"])


def test_load_by_time_range_visits_only_in_range_hours(tmp_path, monkeypatch):
    loader, _ = _loader(tmp_path)
    visited: list[str] = []

    def spy(p, columns=None, filters=None):
        visited.append(str(p))
        return _fake_read_by_stem(p)

    monkeypatch.setattr(pd, "read_parquet", spy)
    df = loader.load_by_time_range("2024-01-01 09:00:00", "2024-01-01 10:30:00")
    assert set(df["uuid"]) == {"u1"}
    # The 11:00 file must never have been read.
    assert all("/11/" not in v for v in visited)


def test_load_files_by_time_range_and_uuids(tmp_path, monkeypatch):
    loader, _ = _loader(tmp_path, monkeypatch)
    df = loader.load_files_by_time_range_and_uuids(
        "2024-01-01 09:00:00", "2024-01-01 11:00:00", ["u2"]
    )
    assert set(df["uuid"]) == {"u2"}


def test_load_files_by_uuids_unknown_warns(tmp_path, monkeypatch):
    loader, _ = _loader(tmp_path, monkeypatch)
    with pytest.warns(LoaderConfigWarning, match="No matching .parquet files"):
        df = loader.load_files_by_time_range_and_uuids(
            "2024-01-01 09:00:00", "2024-01-01 11:00:00", ["does-not-exist"]
        )
    assert df.empty


def test_stream_by_time_range_is_incremental(tmp_path, monkeypatch):
    loader, _ = _loader(tmp_path, monkeypatch)
    out = list(
        loader.stream_by_time_range("2024-01-01 09:00:00", "2024-01-01 11:00:00")
    )
    assert len(out) == 3
    assert all(isinstance(p, str) and isinstance(d, pd.DataFrame) for p, d in out)


def test_stream_by_time_range_empty_warns(tmp_path, monkeypatch):
    loader, _ = _loader(tmp_path, monkeypatch)
    with pytest.warns(LoaderConfigWarning, match="Yielded no DataFrames"):
        out = list(
            loader.stream_by_time_range("2030-01-01 00:00:00", "2030-01-01 00:00:00")
        )
    assert out == []


def test_pushdown_passes_columns_and_filters(tmp_path, monkeypatch):
    loader, _ = _loader(tmp_path)
    seen: dict = {}

    def fake_read_parquet(path, columns=None, filters=None):
        seen["columns"] = columns
        seen["filters"] = filters
        return pd.DataFrame({"uuid": ["u1"]})

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)
    df = loader.load_all_files(columns=["uuid"], filters=[("uuid", "==", "u1")])
    assert not df.empty
    assert seen["columns"] == ["uuid"]
    assert seen["filters"] == [("uuid", "==", "u1")]


def test_pushdown_defaults_none(tmp_path, monkeypatch):
    loader, _ = _loader(tmp_path)
    seen: dict = {}

    def fake_read_parquet(path, columns=None, filters=None):
        seen["columns"] = columns
        seen["filters"] = filters
        return pd.DataFrame({"uuid": ["u1"]})

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)
    df = loader.load_all_files()
    assert not df.empty
    assert seen["columns"] is None
    assert seen["filters"] is None


def test_fetch_data_as_dataframe_time_range(tmp_path, monkeypatch):
    loader, _ = _loader(tmp_path, monkeypatch)
    df = loader.fetch_data_as_dataframe("2024-01-01 09:00:00", "2024-01-01 10:30:00")
    assert set(df["uuid"]) == {"u1"}


def test_fetch_data_as_dataframe_all(tmp_path, monkeypatch):
    loader, _ = _loader(tmp_path, monkeypatch)
    df = loader.fetch_data_as_dataframe()
    assert set(df["uuid"]) == {"u1", "u2"}


def test_load_all_empty_warns(tmp_path):
    root = tmp_path / "Volumes" / "main" / "plant" / "empty"
    (root / "parquet").mkdir(parents=True)
    loader = DatabricksUnityCatalogParquetLoader(volume_path=str(root), prefix="parquet")
    with pytest.warns(LoaderConfigWarning, match="No .parquet files"):
        df = loader.load_all_files()
    assert df.empty


def test_load_by_time_range_outside_data_warns(tmp_path):
    loader, _ = _loader(tmp_path)
    with pytest.warns(LoaderConfigWarning, match="hour_pattern"):
        df = loader.load_by_time_range("2030-01-01 00:00:00", "2030-01-01 00:00:00")
    assert df.empty


def test_happy_load_emits_no_warning(tmp_path, monkeypatch):
    loader, _ = _loader(tmp_path, monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("error", LoaderConfigWarning)
        df = loader.load_by_time_range("2024-01-01 09:00:00", "2024-01-01 10:30:00")
    assert not df.empty


def test_validate_warns_on_missing_root(tmp_path):
    missing = tmp_path / "Volumes" / "main" / "plant" / "nope"
    with pytest.warns(LoaderConfigWarning, match="path does not exist"):
        DatabricksUnityCatalogParquetLoader(volume_path=str(missing), prefix="parquet")


def test_validate_false_is_silent_on_missing_root(tmp_path):
    missing = tmp_path / "Volumes" / "main" / "plant" / "nope"
    with warnings.catch_warnings():
        warnings.simplefilter("error", LoaderConfigWarning)
        DatabricksUnityCatalogParquetLoader(
            volume_path=str(missing), prefix="parquet", validate=False
        )


def test_filters_too_strict_warning_hint(tmp_path, monkeypatch):
    loader, _ = _loader(tmp_path)
    monkeypatch.setattr(
        pd,
        "read_parquet",
        lambda path, columns=None, filters=None: pd.DataFrame(),
    )
    with pytest.warns(LoaderConfigWarning, match="filters argument may be too strict"):
        df = loader.load_all_files(filters=[("value_double", ">", 1e9)])
    assert df.empty
