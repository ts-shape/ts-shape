"""Tests for EnergyAPILoader."""

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from ts_shape.loader.timeseries.energy_api_loader import EnergyAPILoader


@pytest.fixture
def loader():
    return EnergyAPILoader(
        base_url="https://api.example.com/v1", headers={"Authorization": "Bearer test"}
    )


def _mock_response(json_data, status_code=200):
    resp = MagicMock()
    resp.json.return_value = json_data
    resp.status_code = status_code
    resp.raise_for_status.return_value = None
    return resp


class TestFetchDataAsDataframe:
    @patch("ts_shape.loader.timeseries.energy_api_loader.requests.get")
    def test_basic_list_response(self, mock_get, loader):
        mock_get.return_value = _mock_response(
            [
                {"timestamp": "2024-01-01T08:00:00", "value": 100.5},
                {"timestamp": "2024-01-01T09:00:00", "value": 110.2},
            ]
        )
        df = loader.fetch_data_as_dataframe("/readings", uuid_value="meter:001")
        assert len(df) == 2
        assert list(df.columns) == ["systime", "uuid", "value_double", "is_delta"]
        assert df["uuid"].iloc[0] == "meter:001"
        assert df["value_double"].iloc[0] == 100.5
        assert df["is_delta"].all()

    @patch("ts_shape.loader.timeseries.energy_api_loader.requests.get")
    def test_data_root_key(self, mock_get, loader):
        mock_get.return_value = _mock_response(
            {
                "status": "ok",
                "readings": [
                    {"ts": "2024-01-01T10:00:00", "kw": 55.0},
                ],
            }
        )
        df = loader.fetch_data_as_dataframe(
            "/readings",
            time_key="ts",
            value_key="kw",
            uuid_value="meter:002",
            data_root="readings",
        )
        assert len(df) == 1
        assert df["value_double"].iloc[0] == 55.0

    @patch("ts_shape.loader.timeseries.energy_api_loader.requests.get")
    def test_empty_response(self, mock_get, loader):
        mock_get.return_value = _mock_response([])
        df = loader.fetch_data_as_dataframe("/readings")
        assert df.empty
        assert list(df.columns) == ["systime", "uuid", "value_double", "is_delta"]

    @patch("ts_shape.loader.timeseries.energy_api_loader.requests.get")
    def test_missing_time_key(self, mock_get, loader):
        mock_get.return_value = _mock_response([{"val": 1.0}])
        df = loader.fetch_data_as_dataframe("/readings", time_key="timestamp")
        assert df.empty

    @patch("ts_shape.loader.timeseries.energy_api_loader.requests.get")
    def test_missing_value_key(self, mock_get, loader):
        mock_get.return_value = _mock_response(
            [
                {"timestamp": "2024-01-01T08:00:00"},
            ]
        )
        df = loader.fetch_data_as_dataframe("/readings", value_key="nonexistent")
        assert len(df) == 1
        assert pd.isna(df["value_double"].iloc[0])

    @patch("ts_shape.loader.timeseries.energy_api_loader.requests.get")
    def test_sorted_output(self, mock_get, loader):
        mock_get.return_value = _mock_response(
            [
                {"timestamp": "2024-01-01T12:00:00", "value": 3.0},
                {"timestamp": "2024-01-01T08:00:00", "value": 1.0},
                {"timestamp": "2024-01-01T10:00:00", "value": 2.0},
            ]
        )
        df = loader.fetch_data_as_dataframe("/readings")
        assert df["value_double"].tolist() == [1.0, 2.0, 3.0]


class TestFetchMultipleMeters:
    @patch("ts_shape.loader.timeseries.energy_api_loader.requests.get")
    def test_multiple_meters(self, mock_get, loader):
        def side_effect(url, **kwargs):
            mid = kwargs.get("params", {}).get("meter_id", "")
            if mid == "A":
                return _mock_response(
                    [{"timestamp": "2024-01-01T08:00:00", "value": 10.0}]
                )
            else:
                return _mock_response(
                    [{"timestamp": "2024-01-01T09:00:00", "value": 20.0}]
                )

        mock_get.side_effect = side_effect
        df = loader.fetch_multiple_meters("/readings", ["A", "B"])
        assert len(df) == 2
        assert set(df["uuid"]) == {"energy:A", "energy:B"}

    @patch("ts_shape.loader.timeseries.energy_api_loader.requests.get")
    def test_empty_meters(self, mock_get, loader):
        mock_get.return_value = _mock_response([])
        df = loader.fetch_multiple_meters("/readings", ["A"])
        assert df.empty
        assert list(df.columns) == ["systime", "uuid", "value_double", "is_delta"]


class TestInit:
    def test_trailing_slash_stripped(self):
        loader = EnergyAPILoader(base_url="https://api.example.com/")
        assert loader.base_url == "https://api.example.com"

    def test_default_headers(self):
        loader = EnergyAPILoader(base_url="https://api.example.com")
        assert loader.headers == {}
        assert loader.timeout == 30
