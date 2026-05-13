import pytest
import pandas as pd
import numpy as np
from ts_shape.features.segment_analysis.segment_extractor import SegmentExtractor


@pytest.fixture
def production_df():
    """Simulate production data with an order signal changing over time."""
    np.random.seed(42)
    n = 300
    times = pd.date_range("2024-01-01", periods=n, freq="1s")

    orders = ["Order-A"] * 100 + ["Order-B"] * 100 + ["Order-A"] * 100
    return pd.DataFrame(
        {
            "systime": times,
            "uuid": "order_number",
            "value_string": orders,
            "value_double": np.nan,
        }
    )


class TestExtractTimeRanges:
    def test_detects_all_segments(self, production_df):
        result = SegmentExtractor.extract_time_ranges(
            production_df, segment_uuid="order_number"
        )
        assert len(result) == 3

    def test_correct_values(self, production_df):
        result = SegmentExtractor.extract_time_ranges(
            production_df, segment_uuid="order_number"
        )
        assert result.iloc[0]["segment_value"] == "Order-A"
        assert result.iloc[1]["segment_value"] == "Order-B"
        assert result.iloc[2]["segment_value"] == "Order-A"

    def test_correct_columns(self, production_df):
        result = SegmentExtractor.extract_time_ranges(
            production_df, segment_uuid="order_number"
        )
        expected = {
            "segment_value",
            "segment_start",
            "segment_end",
            "segment_duration",
            "segment_index",
        }
        assert set(result.columns) == expected

    def test_sequential_indices(self, production_df):
        result = SegmentExtractor.extract_time_ranges(
            production_df, segment_uuid="order_number"
        )
        assert list(result["segment_index"]) == [0, 1, 2]

    def test_min_duration_filters(self, production_df):
        result = SegmentExtractor.extract_time_ranges(
            production_df,
            segment_uuid="order_number",
            min_duration="200s",
        )
        assert len(result) == 0

    def test_min_duration_keeps_long_segments(self, production_df):
        result = SegmentExtractor.extract_time_ranges(
            production_df,
            segment_uuid="order_number",
            min_duration="50s",
        )
        assert len(result) == 3

    def test_empty_uuid(self, production_df):
        result = SegmentExtractor.extract_time_ranges(
            production_df, segment_uuid="nonexistent"
        )
        assert len(result) == 0

    def test_integer_value_column(self):
        df = pd.DataFrame(
            {
                "systime": pd.date_range("2024-01-01", periods=10, freq="1s"),
                "uuid": "part_id",
                "value_integer": [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
            }
        )
        result = SegmentExtractor.extract_time_ranges(
            df, segment_uuid="part_id", value_column="value_integer"
        )
        assert len(result) == 3
        assert list(result["segment_value"]) == [1, 2, 3]

    def test_single_value_produces_one_segment(self):
        df = pd.DataFrame(
            {
                "systime": pd.date_range("2024-01-01", periods=5, freq="1s"),
                "uuid": "order",
                "value_string": ["X"] * 5,
            }
        )
        result = SegmentExtractor.extract_time_ranges(df, segment_uuid="order")
        assert len(result) == 1
        assert result.iloc[0]["segment_value"] == "X"

    def test_skips_empty_string_segments(self):
        df = pd.DataFrame(
            {
                "systime": pd.date_range("2024-01-01", periods=6, freq="1s"),
                "uuid": "order",
                "value_string": ["A", "A", "", "", "B", "B"],
            }
        )
        result = SegmentExtractor.extract_time_ranges(df, segment_uuid="order")
        assert len(result) == 2
        assert list(result["segment_value"]) == ["A", "B"]

    def test_duration_is_correct(self, production_df):
        result = SegmentExtractor.extract_time_ranges(
            production_df, segment_uuid="order_number"
        )
        for _, row in result.iterrows():
            assert row["segment_duration"] == row["segment_end"] - row["segment_start"]

    def test_nan_values_join_adjacent_segment(self):
        """NaN rows should be absorbed into the adjacent segment via ffill."""
        df = pd.DataFrame(
            {
                "systime": pd.date_range("2024-01-01", periods=8, freq="1s"),
                "uuid": "order",
                "value_string": ["A", "A", np.nan, np.nan, "B", "B", np.nan, "B"],
            }
        )
        result = SegmentExtractor.extract_time_ranges(df, segment_uuid="order")
        # NaN after A joins A, NaN after B joins B → 2 segments
        assert len(result) == 2
        assert list(result["segment_value"]) == ["A", "B"]

    def test_leading_nan_values_skipped(self):
        """NaN values at the start (before any real value) should be skipped."""
        df = pd.DataFrame(
            {
                "systime": pd.date_range("2024-01-01", periods=5, freq="1s"),
                "uuid": "order",
                "value_string": [np.nan, np.nan, "A", "A", "B"],
            }
        )
        result = SegmentExtractor.extract_time_ranges(df, segment_uuid="order")
        assert len(result) == 2
        assert list(result["segment_value"]) == ["A", "B"]

    def test_single_row_dataframe(self):
        df = pd.DataFrame(
            {
                "systime": [pd.Timestamp("2024-01-01")],
                "uuid": "order",
                "value_string": ["X"],
            }
        )
        result = SegmentExtractor.extract_time_ranges(df, segment_uuid="order")
        assert len(result) == 1
        assert result.iloc[0]["segment_duration"] == pd.Timedelta(0)
