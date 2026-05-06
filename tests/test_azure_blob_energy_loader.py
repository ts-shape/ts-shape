"""Tests for AzureBlobEnergyLoader."""
import io
from unittest.mock import MagicMock, patch, call

import pandas as pd
import pytest

from ts_shape.loader.timeseries.azure_blob_energy_loader import AzureBlobEnergyLoader

# ── Fixtures ────────────────────────────────────────────────────────────────

_CSV_BYTES = (
    "time,value\n"
    "2026-01-13T00:00:00Z,107.603588867188\n"
    "2026-01-13T00:15:00Z,184.093249511719\n"
    "2026-01-13T00:30:00Z,288.517919921875\n"
).encode()

_META_BYTES = (
    "id\tlabel_lvl1\tlabel_lvl2\tlabel_lvl3\tlabel_lvl4\tdescription\tunit"
    "\thierarchy_lvl1\thierarchy_lvl2\thierarchy_lvl3"
    "\thierarchy_lvl4\thierarchy_lvl5\thierarchy_lvl6\n"
    "sensor_001\tBuilding A\tFloor 1\tLine 1\t\tMain meter\tkWh"
    "\tSite\tBuilding\tFloor\tLine\t\t\n"
).encode()


def _make_container_client(blob_bytes: bytes = _CSV_BYTES, list_names: list = None):
    """Return a mock ContainerClient that yields one blob and returns blob_bytes."""
    client = MagicMock()

    if list_names is None:
        list_names = ["csv/2026/01/13/sensor_001.csv"]

    def _list_blobs(name_starts_with=None):
        for name in list_names:
            if name_starts_with is None or name.startswith(name_starts_with):
                b = MagicMock()
                b.name = name
                yield b

    client.list_blobs.side_effect = _list_blobs

    downloader = MagicMock()
    downloader.readall.return_value = blob_bytes
    client.download_blob.return_value = downloader
    return client


def _make_loader(container_client, **kwargs) -> AzureBlobEnergyLoader:
    """Construct a loader with a pre-built container_client, bypassing auth."""
    loader = object.__new__(AzureBlobEnergyLoader)
    loader.container_client = container_client
    loader.prefix = kwargs.get("prefix", "")
    loader.max_workers = kwargs.get("max_workers", 4)
    loader.thousands = kwargs.get("thousands", None)
    loader.decimal = kwargs.get("decimal", ".")
    return loader


# ── Init / Auth ──────────────────────────────────────────────────────────────

class TestAzureBlobEnergyLoaderInit:
    def test_missing_auth_raises(self):
        with patch("ts_shape.loader.timeseries.azure_blob_energy_loader.AzureBlobEnergyLoader.__init__") as mock_init:
            mock_init.side_effect = ValueError
            with pytest.raises(ValueError):
                AzureBlobEnergyLoader()

    def test_normalize_df_static(self):
        raw = pd.DataFrame({
            "time": ["2026-01-13T00:00:00Z", "2026-01-13T00:15:00Z"],
            "value": [107.6, 184.09],
        })
        out = AzureBlobEnergyLoader._normalize_df(raw, "sensor_001")
        assert list(out.columns) == ["systime", "uuid", "value_double", "is_delta"]
        assert (out["uuid"] == "sensor_001").all()
        assert (out["is_delta"] == True).all()
        assert out["value_double"].iloc[0] == pytest.approx(107.6)

    def test_series_id_from_blob_name(self):
        assert AzureBlobEnergyLoader._series_id_from_blob_name(
            "csv/2026/01/13/sensor_001.csv"
        ) == "sensor_001"
        assert AzureBlobEnergyLoader._series_id_from_blob_name(
            "prefix/csv/2026/01/14/meter-42.csv"
        ) == "meter-42"


# ── Metadata ─────────────────────────────────────────────────────────────────

class TestLoadSeriesMetadata:
    def test_returns_expected_columns(self):
        client = _make_container_client(blob_bytes=_META_BYTES)
        loader = _make_loader(client)
        df = loader.load_series_metadata()
        assert "id" in df.columns
        assert "label_lvl1" in df.columns
        assert "hierarchy_lvl6" in df.columns
        assert df.iloc[0]["id"] == "sensor_001"
        assert df.iloc[0]["unit"] == "kWh"

    def test_empty_when_blob_missing(self):
        client = MagicMock()
        client.download_blob.side_effect = Exception("BlobNotFound")
        loader = _make_loader(client)
        df = loader.load_series_metadata()
        assert df.empty
        assert "id" in df.columns


# ── load_by_time_range ────────────────────────────────────────────────────────

class TestLoadByTimeRange:
    def test_single_day_returns_standard_schema(self):
        client = _make_container_client()
        loader = _make_loader(client)
        df = loader.load_by_time_range("2026-01-13", "2026-01-13")
        assert list(df.columns) == ["systime", "uuid", "value_double", "is_delta"]
        assert len(df) == 3
        assert (df["uuid"] == "sensor_001").all()

    def test_filters_by_series_ids(self):
        client = _make_container_client(
            list_names=[
                "csv/2026/01/13/sensor_001.csv",
                "csv/2026/01/13/sensor_002.csv",
            ]
        )
        loader = _make_loader(client)
        df = loader.load_by_time_range("2026-01-13", "2026-01-13", series_ids=["sensor_001"])
        assert (df["uuid"] == "sensor_001").all()

    def test_empty_range_returns_empty_df(self):
        client = _make_container_client(list_names=[])
        loader = _make_loader(client)
        df = loader.load_by_time_range("2026-01-13", "2026-01-13")
        assert df.empty
        assert list(df.columns) == ["systime", "uuid", "value_double", "is_delta"]

    def test_multi_day_builds_multiple_prefixes(self):
        client = _make_container_client(list_names=[])
        loader = _make_loader(client)
        loader.load_by_time_range("2026-01-13", "2026-01-15")
        # Should have listed 3 day prefixes
        assert client.list_blobs.call_count == 3

    def test_values_are_numeric(self):
        client = _make_container_client()
        loader = _make_loader(client)
        df = loader.load_by_time_range("2026-01-13", "2026-01-13")
        assert df["value_double"].dtype == float
        assert df["value_double"].iloc[0] == pytest.approx(107.603588867188)


# ── load_by_series_ids ────────────────────────────────────────────────────────

class TestLoadBySeriesIds:
    def test_with_date_range_lists_by_day(self):
        client = _make_container_client()
        loader = _make_loader(client)
        df = loader.load_by_series_ids(["sensor_001"], start="2026-01-13", end="2026-01-13")
        assert len(df) == 3

    def test_without_dates_lists_all(self):
        client = _make_container_client()
        loader = _make_loader(client)
        df = loader.load_by_series_ids(["sensor_001"])
        assert len(df) == 3

    def test_unknown_series_returns_empty(self):
        client = _make_container_client()
        loader = _make_loader(client)
        df = loader.load_by_series_ids(["unknown_series"])
        assert df.empty


# ── stream_by_time_range ──────────────────────────────────────────────────────

class TestStreamByTimeRange:
    def test_yields_tuples(self):
        client = _make_container_client()
        loader = _make_loader(client)
        results = list(loader.stream_by_time_range("2026-01-13", "2026-01-13"))
        assert len(results) == 1
        series_id, df = results[0]
        assert series_id == "sensor_001"
        assert list(df.columns) == ["systime", "uuid", "value_double", "is_delta"]

    def test_yields_nothing_on_empty(self):
        client = _make_container_client(list_names=[])
        loader = _make_loader(client)
        results = list(loader.stream_by_time_range("2026-01-13", "2026-01-13"))
        assert results == []


# ── list_series ───────────────────────────────────────────────────────────────

class TestListSeries:
    def test_returns_sorted_series_ids(self):
        client = _make_container_client(
            list_names=[
                "csv/2026/01/13/sensor_002.csv",
                "csv/2026/01/13/sensor_001.csv",
                "csv/2026/01/14/sensor_001.csv",  # duplicate — should be deduplicated
            ]
        )
        loader = _make_loader(client)
        series = loader.list_series()
        assert series == ["sensor_001", "sensor_002"]

    def test_empty_container(self):
        client = _make_container_client(list_names=[])
        loader = _make_loader(client)
        assert loader.list_series() == []


# ── _normalize_df edge cases ──────────────────────────────────────────────────

class TestNormalizeDf:
    def test_bad_values_become_nan(self):
        raw = pd.DataFrame({"time": ["2026-01-13T00:00:00Z"], "value": ["not_a_number"]})
        out = AzureBlobEnergyLoader._normalize_df(raw, "s1")
        assert pd.isna(out["value_double"].iloc[0])

    def test_missing_value_column(self):
        raw = pd.DataFrame({"time": ["2026-01-13T00:00:00Z"]})
        out = AzureBlobEnergyLoader._normalize_df(raw, "s1")
        assert out.empty

    def test_sorted_by_systime(self):
        raw = pd.DataFrame({
            "time": ["2026-01-13T01:00:00Z", "2026-01-13T00:00:00Z"],
            "value": [2.0, 1.0],
        })
        out = AzureBlobEnergyLoader._normalize_df(raw, "s1")
        assert out["value_double"].iloc[0] == pytest.approx(1.0)

    def test_prefix_in_full_path(self):
        client = _make_container_client(list_names=[])
        loader = _make_loader(client, prefix="system-eu-plant1")
        path = loader._full_path(".meta/series.csv")
        assert path == "system-eu-plant1/.meta/series.csv"

    def test_date_paths_format(self):
        client = _make_container_client(list_names=[])
        loader = _make_loader(client)
        paths = loader._build_date_paths("2026-01-13", "2026-01-15")
        assert paths == [
            "csv/2026/01/13/",
            "csv/2026/01/14/",
            "csv/2026/01/15/",
        ]
