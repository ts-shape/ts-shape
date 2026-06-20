import warnings

import pandas as pd  # type: ignore
import pytest

from ts_shape.errors import LoaderConfigWarning
from ts_shape.loader.timeseries.databricks_unity_energy_loader import (
    DatabricksUnityEnergyLoader,
)

_CSV_TEXT = (
    "time,value\n"
    "2026-01-13T00:00:00Z,107.603588867188\n"
    "2026-01-13T00:15:00Z,184.093249511719\n"
    "2026-01-13T00:30:00Z,288.517919921875\n"
)

_META_TEXT = (
    "id\tlabel_lvl1\tlabel_lvl2\tlabel_lvl3\tlabel_lvl4\tdescription\tunit"
    "\thierarchy_lvl1\thierarchy_lvl2\thierarchy_lvl3"
    "\thierarchy_lvl4\thierarchy_lvl5\thierarchy_lvl6\n"
    "sensor_001\tBuilding A\tFloor 1\tLine 1\t\tMain meter\tkWh"
    "\tSite\tBuilding\tFloor\tLine\t\t\n"
)


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _make_volume(tmp_path, prefix=""):
    """Build a mini mounted UC Volume tree with the energy CSV layout."""
    root = tmp_path / "Volumes" / "main" / "plant" / "energy"
    base = root / prefix if prefix else root
    _write(base / ".meta/series.csv", _META_TEXT)
    _write(base / "csv/2026/01/13/sensor_001.csv", _CSV_TEXT)
    _write(base / "csv/2026/01/13/sensor_002.csv", _CSV_TEXT)
    _write(base / "csv/2026/01/14/sensor_001.csv", _CSV_TEXT)
    return root


def _loader(tmp_path, **kwargs):
    prefix = kwargs.pop("prefix", "")
    root = _make_volume(tmp_path, prefix=prefix)
    return (
        DatabricksUnityEnergyLoader(volume_path=str(root), prefix=prefix, **kwargs),
        root,
    )


# ── Init / resolution ─────────────────────────────────────────────────────────


def test_resolve_from_volume_path(tmp_path):
    loader, root = _loader(tmp_path)
    assert loader.base_path == root


def test_resolve_from_catalog_parts(tmp_path):
    root = _make_volume(tmp_path)
    loader = DatabricksUnityEnergyLoader(
        catalog="main",
        schema="plant",
        volume="energy",
        base_path=str(tmp_path / "Volumes"),
    )
    assert loader.base_path == root


def test_missing_parts_raises_value_error():
    with pytest.raises(ValueError, match="volume_path"):
        DatabricksUnityEnergyLoader(catalog="main", schema="plant")


def test_validate_warns_on_missing_root(tmp_path):
    missing = tmp_path / "Volumes" / "main" / "plant" / "nope"
    with pytest.warns(LoaderConfigWarning, match="path does not exist"):
        DatabricksUnityEnergyLoader(volume_path=str(missing))


def test_validate_false_is_silent(tmp_path):
    missing = tmp_path / "Volumes" / "main" / "plant" / "nope"
    with warnings.catch_warnings():
        warnings.simplefilter("error", LoaderConfigWarning)
        DatabricksUnityEnergyLoader(volume_path=str(missing), validate=False)


# ── Metadata ──────────────────────────────────────────────────────────────────


def test_load_series_metadata(tmp_path):
    loader, _ = _loader(tmp_path)
    df = loader.load_series_metadata()
    assert list(df.columns)[0] == "id"
    assert "hierarchy_lvl6" in df.columns
    assert df.iloc[0]["id"] == "sensor_001"
    assert df.iloc[0]["unit"] == "kWh"


def test_load_series_metadata_missing_returns_empty(tmp_path):
    root = tmp_path / "Volumes" / "main" / "plant" / "energy"
    (root / "csv").mkdir(parents=True)
    loader = DatabricksUnityEnergyLoader(volume_path=str(root))
    df = loader.load_series_metadata()
    assert df.empty
    assert "id" in df.columns


# ── load_by_time_range ────────────────────────────────────────────────────────


def test_load_by_time_range_standard_schema(tmp_path):
    loader, _ = _loader(tmp_path)
    df = loader.load_by_time_range("2026-01-13", "2026-01-13")
    assert list(df.columns) == ["systime", "uuid", "value_double", "is_delta"]
    # Two series on 2026-01-13, 3 rows each.
    assert len(df) == 6
    assert set(df["uuid"]) == {"sensor_001", "sensor_002"}


def test_load_by_time_range_filters_series(tmp_path):
    loader, _ = _loader(tmp_path)
    df = loader.load_by_time_range(
        "2026-01-13", "2026-01-13", series_ids=["sensor_001"]
    )
    assert set(df["uuid"]) == {"sensor_001"}
    assert len(df) == 3


def test_load_by_time_range_empty_returns_schema(tmp_path):
    loader, _ = _loader(tmp_path)
    df = loader.load_by_time_range("2030-01-01", "2030-01-01")
    assert df.empty
    assert list(df.columns) == ["systime", "uuid", "value_double", "is_delta"]


def test_load_by_time_range_visits_only_in_range_days(tmp_path, monkeypatch):
    loader, _ = _loader(tmp_path)
    visited: list[str] = []
    orig = loader._read_csv

    def spy(path):
        visited.append(str(path))
        return orig(path)

    monkeypatch.setattr(loader, "_read_csv", spy)
    loader.load_by_time_range("2026-01-13", "2026-01-13")
    # The 01/14 folder must never have been read.
    assert all("/14/" not in v for v in visited)


def test_values_are_numeric(tmp_path):
    loader, _ = _loader(tmp_path)
    df = loader.load_by_time_range("2026-01-13", "2026-01-13")
    assert df["value_double"].dtype == float
    assert df["value_double"].iloc[0] == pytest.approx(107.603588867188)


# ── load_by_series_ids ────────────────────────────────────────────────────────


def test_load_by_series_ids_with_dates(tmp_path):
    loader, _ = _loader(tmp_path)
    df = loader.load_by_series_ids(["sensor_001"], start="2026-01-13", end="2026-01-14")
    assert set(df["uuid"]) == {"sensor_001"}
    assert len(df) == 6  # two days


def test_load_by_series_ids_without_dates(tmp_path):
    loader, _ = _loader(tmp_path)
    df = loader.load_by_series_ids(["sensor_002"])
    assert set(df["uuid"]) == {"sensor_002"}


def test_load_by_series_ids_unknown_returns_empty(tmp_path):
    loader, _ = _loader(tmp_path)
    df = loader.load_by_series_ids(["unknown"])
    assert df.empty


# ── stream_by_time_range ──────────────────────────────────────────────────────


def test_stream_by_time_range_yields_tuples(tmp_path):
    loader, _ = _loader(tmp_path)
    results = list(loader.stream_by_time_range("2026-01-13", "2026-01-13"))
    assert len(results) == 2
    series_ids = {sid for sid, _ in results}
    assert series_ids == {"sensor_001", "sensor_002"}
    for _, df in results:
        assert list(df.columns) == ["systime", "uuid", "value_double", "is_delta"]


def test_stream_by_time_range_empty(tmp_path):
    loader, _ = _loader(tmp_path)
    results = list(loader.stream_by_time_range("2030-01-01", "2030-01-01"))
    assert results == []


# ── list_series ───────────────────────────────────────────────────────────────


def test_list_series_sorted_unique(tmp_path):
    loader, _ = _loader(tmp_path)
    assert loader.list_series() == ["sensor_001", "sensor_002"]


def test_list_series_empty(tmp_path):
    root = tmp_path / "Volumes" / "main" / "plant" / "energy"
    (root / "csv").mkdir(parents=True)
    loader = DatabricksUnityEnergyLoader(volume_path=str(root))
    assert loader.list_series() == []


# ── fetch_data_as_dataframe ───────────────────────────────────────────────────


def test_fetch_data_as_dataframe_time_range(tmp_path):
    loader, _ = _loader(tmp_path)
    df = loader.fetch_data_as_dataframe("2026-01-13", "2026-01-13")
    assert set(df["uuid"]) == {"sensor_001", "sensor_002"}


def test_fetch_data_as_dataframe_all(tmp_path):
    loader, _ = _loader(tmp_path)
    df = loader.fetch_data_as_dataframe()
    assert set(df["uuid"]) == {"sensor_001", "sensor_002"}
    assert len(df) == 9  # 3 files * 3 rows


# ── prefix handling ───────────────────────────────────────────────────────────


def test_prefix_subpath(tmp_path):
    loader, root = _loader(tmp_path, prefix="plantA")
    assert loader.base_path == root / "plantA"
    df = loader.load_by_time_range("2026-01-13", "2026-01-13")
    assert set(df["uuid"]) == {"sensor_001", "sensor_002"}


# ── _normalize_df edge cases ──────────────────────────────────────────────────


def test_normalize_df_bad_values_become_nan():
    raw = pd.DataFrame({"time": ["2026-01-13T00:00:00Z"], "value": ["not_a_number"]})
    out = DatabricksUnityEnergyLoader._normalize_df(raw, "s1")
    assert pd.isna(out["value_double"].iloc[0])


def test_normalize_df_missing_value_column():
    raw = pd.DataFrame({"time": ["2026-01-13T00:00:00Z"]})
    out = DatabricksUnityEnergyLoader._normalize_df(raw, "s1")
    assert out.empty
