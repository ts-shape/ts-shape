import pytest
import pandas as pd
import numpy as np
from ts_shape.features.segment_analysis.segment_extractor import SegmentExtractor
from ts_shape.features.segment_analysis.segment_processor import SegmentProcessor


@pytest.fixture
def production_df():
    """Production data: order signal + 3 process parameters."""
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
def time_ranges(production_df):
    return SegmentExtractor.extract_time_ranges(
        production_df, segment_uuid="order_number"
    )


@pytest.fixture
def segmented_df(production_df, time_ranges):
    return SegmentProcessor.apply_ranges(
        production_df,
        time_ranges,
        target_uuids=["temperature", "pressure", "speed"],
    )


class TestApplyRanges:
    def test_filters_to_target_uuids(self, production_df, time_ranges):
        result = SegmentProcessor.apply_ranges(
            production_df,
            time_ranges,
            target_uuids=["temperature"],
        )
        assert result["uuid"].unique().tolist() == ["temperature"]

    def test_annotates_segment_value(self, segmented_df):
        assert "segment_value" in segmented_df.columns
        assert set(segmented_df["segment_value"].unique()) == {"Order-A", "Order-B"}

    def test_annotates_segment_index(self, segmented_df):
        assert "segment_index" in segmented_df.columns
        assert set(segmented_df["segment_index"].unique()) == {0, 1, 2}

    def test_all_uuids_present(self, segmented_df):
        assert set(segmented_df["uuid"].unique()) == {
            "temperature",
            "pressure",
            "speed",
        }

    def test_keeps_all_uuids_when_no_filter(self, production_df, time_ranges):
        result = SegmentProcessor.apply_ranges(production_df, time_ranges)
        # order_number itself is also included since no target_uuids filter
        assert "order_number" in result["uuid"].unique()

    def test_empty_ranges(self, production_df):
        empty_ranges = pd.DataFrame(
            columns=[
                "segment_value",
                "segment_start",
                "segment_end",
                "segment_duration",
                "segment_index",
            ]
        )
        result = SegmentProcessor.apply_ranges(production_df, empty_ranges)
        assert len(result) == 0


class TestComputeMetricProfiles:
    def test_by_segment_value(self, segmented_df):
        result = SegmentProcessor.compute_metric_profiles(segmented_df)
        # 3 UUIDs x 2 unique segment_values = 6 rows
        assert len(result) == 6
        assert "mean" in result.columns
        assert "std" in result.columns

    def test_by_segment_index(self, segmented_df):
        result = SegmentProcessor.compute_metric_profiles(
            segmented_df, group_column="segment_index"
        )
        # 3 UUIDs x 3 segments = 9 rows
        assert len(result) == 9

    def test_metrics_subset(self, segmented_df):
        result = SegmentProcessor.compute_metric_profiles(
            segmented_df, metrics=["mean", "std"]
        )
        metric_cols = [
            c
            for c in result.columns
            if c not in ("uuid", "segment_value", "sample_count")
        ]
        assert set(metric_cols) == {"mean", "std"}

    def test_correct_mean_values(self, segmented_df):
        result = SegmentProcessor.compute_metric_profiles(
            segmented_df, metrics=["mean"]
        )
        temp_a = result[
            (result["uuid"] == "temperature") & (result["segment_value"] == "Order-A")
        ]["mean"].values
        temp_b = result[
            (result["uuid"] == "temperature") & (result["segment_value"] == "Order-B")
        ]["mean"].values
        assert all(abs(v - 50) < 5 for v in temp_a)
        assert all(abs(v - 80) < 5 for v in temp_b)

    def test_sample_count_column(self, segmented_df):
        result = SegmentProcessor.compute_metric_profiles(segmented_df)
        assert "sample_count" in result.columns
        assert (result["sample_count"] > 0).all()

    def test_invalid_metric_raises(self, segmented_df):
        with pytest.raises(ValueError, match="Unknown metrics"):
            SegmentProcessor.compute_metric_profiles(
                segmented_df, metrics=["nonexistent"]
            )

    def test_single_uuid(self, production_df, time_ranges):
        segmented = SegmentProcessor.apply_ranges(
            production_df,
            time_ranges,
            target_uuids=["temperature"],
        )
        result = SegmentProcessor.compute_metric_profiles(segmented)
        assert result["uuid"].unique().tolist() == ["temperature"]
        assert len(result) == 2  # 1 UUID x 2 segment_values

    def test_overlapping_ranges_warns(self, production_df, caplog):
        """Overlapping time ranges should produce a warning."""
        import logging

        ranges = pd.DataFrame(
            {
                "segment_value": ["A", "B"],
                "segment_start": [
                    pd.Timestamp("2024-01-01 00:00:00"),
                    pd.Timestamp("2024-01-01 00:00:30"),
                ],
                "segment_end": [
                    pd.Timestamp("2024-01-01 00:01:00"),
                    pd.Timestamp("2024-01-01 00:01:30"),
                ],
                "segment_duration": [pd.Timedelta("60s"), pd.Timedelta("60s")],
                "segment_index": [0, 1],
            }
        )
        with caplog.at_level(logging.WARNING):
            SegmentProcessor.apply_ranges(production_df, ranges)
        assert "overlapping" in caplog.text.lower()
