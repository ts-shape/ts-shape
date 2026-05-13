"""Tests for the context enricher loader module."""

import pandas as pd
import pytest

from ts_shape.loader.context.context_enricher import ContextEnricher

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def timeseries_df():
    return pd.DataFrame(
        {
            "systime": pd.date_range("2024-01-01", periods=10, freq="1min"),
            "uuid": ["temp:sensor1"] * 5 + ["press:sensor2"] * 5,
            "value_double": [
                20.1,
                20.3,
                20.5,
                20.2,
                20.4,
                1.01,
                1.02,
                1.03,
                1.01,
                1.02,
            ],
            "value_string": [None] * 10,
            "is_delta": [True] * 10,
        }
    )


@pytest.fixture
def metadata_df():
    return pd.DataFrame(
        {
            "uuid": ["temp:sensor1", "press:sensor2"],
            "description": ["Temperature Sensor 1", "Pressure Sensor 2"],
            "unit": ["°C", "bar"],
            "area": ["zone_a", "zone_b"],
        }
    )


@pytest.fixture
def tolerance_df():
    return pd.DataFrame(
        {
            "uuid": ["temp:sensor1", "press:sensor2"],
            "low_limit": [18.0, 0.9],
            "high_limit": [25.0, 1.5],
        }
    )


@pytest.fixture
def mapping_df():
    return pd.DataFrame(
        {
            "uuid": ["state:machine1", "state:machine1", "state:machine1"],
            "raw_value": ["0", "1", "2"],
            "mapped_value": ["idle", "running", "error"],
        }
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestContextEnricher:

    def test_enrich_with_metadata(self, timeseries_df, metadata_df):
        enricher = ContextEnricher(timeseries_df)
        result = enricher.enrich_with_metadata(
            metadata_df, columns=["description", "unit"]
        )
        assert "description" in result.columns
        assert "unit" in result.columns
        assert (
            result.loc[result["uuid"] == "temp:sensor1", "description"].iloc[0]
            == "Temperature Sensor 1"
        )
        assert result.loc[result["uuid"] == "press:sensor2", "unit"].iloc[0] == "bar"

    def test_enrich_with_metadata_all_columns(self, timeseries_df, metadata_df):
        enricher = ContextEnricher(timeseries_df)
        result = enricher.enrich_with_metadata(metadata_df)
        assert "description" in result.columns
        assert "unit" in result.columns
        assert "area" in result.columns

    def test_enrich_with_metadata_missing_uuid(self, timeseries_df, metadata_df):
        # Add a UUID that's not in metadata
        extra = pd.DataFrame(
            {
                "systime": [pd.Timestamp("2024-01-01 00:10")],
                "uuid": ["unknown:sensor"],
                "value_double": [99.0],
                "value_string": [None],
                "is_delta": [True],
            }
        )
        df = pd.concat([timeseries_df, extra], ignore_index=True)
        enricher = ContextEnricher(df)
        result = enricher.enrich_with_metadata(metadata_df)
        # Unknown UUID should have NaN for metadata
        unknown_row = result[result["uuid"] == "unknown:sensor"]
        assert unknown_row["description"].isna().all()

    def test_enrich_with_tolerances(self, timeseries_df, tolerance_df):
        enricher = ContextEnricher(timeseries_df)
        result = enricher.enrich_with_tolerances(tolerance_df)
        assert "low_limit" in result.columns
        assert "high_limit" in result.columns
        temp_rows = result[result["uuid"] == "temp:sensor1"]
        assert temp_rows["low_limit"].iloc[0] == 18.0
        assert temp_rows["high_limit"].iloc[0] == 25.0

    def test_enrich_with_tolerances_missing_uuid(self, timeseries_df):
        # Empty tolerance table
        empty_tol = pd.DataFrame(columns=["uuid", "low_limit", "high_limit"])
        enricher = ContextEnricher(timeseries_df)
        result = enricher.enrich_with_tolerances(empty_tol)
        assert "low_limit" in result.columns
        assert result["low_limit"].isna().all()

    def test_enrich_with_mapping(self, mapping_df):
        ts_df = pd.DataFrame(
            {
                "systime": pd.date_range("2024-01-01", periods=6, freq="1min"),
                "uuid": ["state:machine1"] * 6,
                "value_double": [None] * 6,
                "value_string": ["0", "1", "1", "2", "0", "1"],
                "is_delta": [True] * 6,
            }
        )
        enricher = ContextEnricher(ts_df)
        result = enricher.enrich_with_mapping(mapping_df)
        assert "mapped_value" in result.columns
        assert result["mapped_value"].iloc[0] == "idle"
        assert result["mapped_value"].iloc[1] == "running"
        assert result["mapped_value"].iloc[3] == "error"

    def test_enrich_with_mapping_unknown_uuid(self, mapping_df):
        ts_df = pd.DataFrame(
            {
                "systime": pd.date_range("2024-01-01", periods=3, freq="1min"),
                "uuid": ["unknown:signal"] * 3,
                "value_double": [None] * 3,
                "value_string": ["0", "1", "2"],
                "is_delta": [True] * 3,
            }
        )
        enricher = ContextEnricher(ts_df)
        result = enricher.enrich_with_mapping(mapping_df)
        assert result["mapped_value"].isna().all()

    def test_get_enriched_dataframe(self, timeseries_df):
        enricher = ContextEnricher(timeseries_df)
        result = enricher.get_enriched_dataframe()
        assert len(result) == len(timeseries_df)

    def test_custom_uuid_column(self):
        df = pd.DataFrame(
            {
                "systime": pd.date_range("2024-01-01", periods=3, freq="1min"),
                "signal_id": ["sig:a", "sig:a", "sig:b"],
                "value_double": [1.0, 2.0, 3.0],
                "is_delta": [True] * 3,
            }
        )
        meta = pd.DataFrame(
            {
                "signal_id": ["sig:a", "sig:b"],
                "description": ["Signal A", "Signal B"],
            }
        )
        enricher = ContextEnricher(df, uuid_column="signal_id")
        result = enricher.enrich_with_metadata(meta, metadata_uuid_col="signal_id")
        assert "description" in result.columns
        assert (
            result.loc[result["signal_id"] == "sig:a", "description"].iloc[0]
            == "Signal A"
        )
