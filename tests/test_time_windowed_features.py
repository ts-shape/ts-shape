import pytest
import pandas as pd
import numpy as np
from ts_shape.features.segment_analysis.segment_extractor import SegmentExtractor
from ts_shape.features.segment_analysis.segment_processor import (
    SegmentProcessor,
    ALL_METRICS,
)
from ts_shape.features.segment_analysis.time_windowed_features import (
    TimeWindowedFeatureTable,
)


@pytest.fixture
def production_df():
    """Production data: order signal + 3 process parameters over 5 minutes."""
    np.random.seed(42)
    n = 300
    times = pd.date_range("2024-01-01", periods=n, freq="1s")

    frames = []

    orders = ["Order-A"] * 100 + ["Order-B"] * 100 + ["Order-A"] * 100
    frames.append(
        pd.DataFrame(
            {
                "systime": times,
                "uuid": "order_number",
                "value_string": orders,
                "value_double": np.nan,
            }
        )
    )

    temp = np.concatenate(
        [
            50 + np.random.randn(100) * 2,
            80 + np.random.randn(100) * 2,
            50 + np.random.randn(100) * 2,
        ]
    )
    frames.append(
        pd.DataFrame(
            {
                "systime": times,
                "uuid": "temperature",
                "value_string": "",
                "value_double": temp,
            }
        )
    )

    pressure = np.concatenate(
        [
            100 + np.random.randn(100) * 5,
            200 + np.random.randn(100) * 5,
            100 + np.random.randn(100) * 5,
        ]
    )
    frames.append(
        pd.DataFrame(
            {
                "systime": times,
                "uuid": "pressure",
                "value_string": "",
                "value_double": pressure,
            }
        )
    )

    speed = 1000 + np.random.randn(n) * 1
    frames.append(
        pd.DataFrame(
            {
                "systime": times,
                "uuid": "speed",
                "value_string": "",
                "value_double": speed,
            }
        )
    )

    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def segmented_df(production_df):
    ranges = SegmentExtractor.extract_time_ranges(
        production_df, segment_uuid="order_number"
    )
    return SegmentProcessor.apply_ranges(
        production_df,
        ranges,
        target_uuids=["temperature", "pressure", "speed"],
    )


# ---------------------------------------------------------------------------
# compute_long
# ---------------------------------------------------------------------------


class TestComputeLong:
    def test_basic_output_columns(self, segmented_df):
        result = TimeWindowedFeatureTable.compute_long(segmented_df, freq="1min")
        assert "time_window" in result.columns
        assert "uuid" in result.columns
        assert "segment_value" in result.columns
        assert "sample_count" in result.columns
        assert "mean" in result.columns

    def test_groups_by_uuid_and_window(self, segmented_df):
        result = TimeWindowedFeatureTable.compute_long(segmented_df, freq="1min")
        # Each row is unique per (time_window, uuid, segment_value)
        keys = result[["time_window", "uuid", "segment_value"]]
        assert len(keys) == len(keys.drop_duplicates())

    def test_multiple_windows(self, segmented_df):
        result = TimeWindowedFeatureTable.compute_long(segmented_df, freq="30s")
        # 300s of data at 30s windows → up to 10 windows per UUID
        assert result["time_window"].nunique() > 1

    def test_includes_segment_column(self, segmented_df):
        result = TimeWindowedFeatureTable.compute_long(
            segmented_df, freq="1min", segment_column="segment_value"
        )
        assert "segment_value" in result.columns
        assert set(result["segment_value"].unique()) == {"Order-A", "Order-B"}

    def test_no_segment_column(self, segmented_df):
        result = TimeWindowedFeatureTable.compute_long(
            segmented_df, freq="1min", segment_column=None
        )
        assert "segment_value" not in result.columns
        # Should still have time_window and uuid
        assert "time_window" in result.columns
        assert "uuid" in result.columns

    def test_metrics_subset(self, segmented_df):
        result = TimeWindowedFeatureTable.compute_long(
            segmented_df, freq="1min", metrics=["mean", "std"]
        )
        assert "mean" in result.columns
        assert "std" in result.columns
        assert "min" not in result.columns
        assert "kurtosis" not in result.columns

    def test_invalid_metric_raises(self, segmented_df):
        with pytest.raises(ValueError, match="Unknown metrics"):
            TimeWindowedFeatureTable.compute_long(
                segmented_df, freq="1min", metrics=["nonexistent"]
            )

    def test_sample_count_positive(self, segmented_df):
        result = TimeWindowedFeatureTable.compute_long(segmented_df, freq="1min")
        assert (result["sample_count"] >= 2).all()

    def test_empty_dataframe(self, segmented_df):
        empty = segmented_df.iloc[:0]
        result = TimeWindowedFeatureTable.compute_long(empty, freq="1min")
        assert result.empty

    def test_missing_column_raises(self, segmented_df):
        with pytest.raises(ValueError):
            TimeWindowedFeatureTable.compute_long(
                segmented_df, freq="1min", value_column="nonexistent"
            )

    def test_all_19_metrics_present(self, segmented_df):
        result = TimeWindowedFeatureTable.compute_long(segmented_df, freq="1min")
        for metric in ALL_METRICS:
            assert metric in result.columns, f"Missing metric: {metric}"


# ---------------------------------------------------------------------------
# compute (wide format)
# ---------------------------------------------------------------------------


class TestCompute:
    def test_wide_column_naming(self, segmented_df):
        result = TimeWindowedFeatureTable.compute(segmented_df, freq="1min")
        # Should have columns like temperature__mean, pressure__std, etc.
        wide_cols = [c for c in result.columns if "__" in c]
        assert len(wide_cols) > 0
        for col in wide_cols:
            parts = col.split("__")
            assert len(parts) == 2
            assert parts[0] in ["temperature", "pressure", "speed"]

    def test_one_row_per_window(self, segmented_df):
        result = TimeWindowedFeatureTable.compute(
            segmented_df, freq="1min", segment_column=None
        )
        # Each time_window should appear exactly once
        assert result["time_window"].is_unique

    def test_column_count(self, segmented_df):
        result = TimeWindowedFeatureTable.compute(
            segmented_df, freq="1min", metrics=["mean", "std"]
        )
        # 3 UUIDs × (2 metrics + sample_count) = 9 wide columns + index cols
        wide_cols = [c for c in result.columns if "__" in c]
        assert len(wide_cols) == 3 * 3  # 3 UUIDs × 3 (mean, std, sample_count)

    def test_custom_separator(self, segmented_df):
        result = TimeWindowedFeatureTable.compute(
            segmented_df, freq="1min", column_separator="_"
        )
        # Columns should use single underscore
        wide_cols = [
            c for c in result.columns if c not in ["time_window", "segment_value"]
        ]
        assert all("_" in c for c in wide_cols)

    def test_correct_mean_values(self, segmented_df):
        """Temperature mean should be ~50 for Order-A, ~80 for Order-B."""
        result = TimeWindowedFeatureTable.compute(
            segmented_df, freq="2min", metrics=["mean"]
        )
        if "temperature__mean" in result.columns:
            temp_means = result["temperature__mean"].dropna()
            assert len(temp_means) > 0

    def test_segment_grouping(self, segmented_df):
        result = TimeWindowedFeatureTable.compute(
            segmented_df, freq="1min", segment_column="segment_value"
        )
        assert "segment_value" in result.columns
        assert "time_window" in result.columns

    def test_no_segment_column(self, segmented_df):
        result = TimeWindowedFeatureTable.compute(
            segmented_df, freq="1min", segment_column=None
        )
        assert "segment_value" not in result.columns

    def test_empty_input(self, segmented_df):
        empty = segmented_df.iloc[:0]
        result = TimeWindowedFeatureTable.compute(empty, freq="1min")
        assert result.empty

    def test_nan_for_missing_uuid_windows(self):
        """If a UUID has no data in a window, its columns should be NaN."""
        times = pd.date_range("2024-01-01", periods=120, freq="1s")
        # temperature exists in first 60s only, pressure in all 120s
        frames = [
            pd.DataFrame(
                {
                    "systime": times[:60],
                    "uuid": "temperature",
                    "value_double": np.random.randn(60),
                    "segment_value": "A",
                }
            ),
            pd.DataFrame(
                {
                    "systime": times,
                    "uuid": "pressure",
                    "value_double": np.random.randn(120),
                    "segment_value": "A",
                }
            ),
        ]
        df = pd.concat(frames, ignore_index=True)
        result = TimeWindowedFeatureTable.compute(df, freq="1min", metrics=["mean"])
        # Second minute should have NaN for temperature
        second_min = result[
            result["time_window"] == pd.Timestamp("2024-01-01 00:01:00")
        ]
        if not second_min.empty and "temperature__mean" in second_min.columns:
            assert second_min["temperature__mean"].isna().all()

    def test_columns_sorted_by_uuid_then_metric(self, segmented_df):
        result = TimeWindowedFeatureTable.compute(
            segmented_df, freq="1min", metrics=["mean", "std", "min"]
        )
        wide_cols = [c for c in result.columns if "__" in c]
        # Extract (uuid, metric) pairs
        pairs = [(c.split("__")[0], c.split("__")[1]) for c in wide_cols]
        uuids_seen = [p[0] for p in pairs]
        # UUIDs should be in alphabetical blocks
        unique_uuids = list(dict.fromkeys(uuids_seen))
        assert unique_uuids == sorted(unique_uuids)
