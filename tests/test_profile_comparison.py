import pytest
import pandas as pd
import numpy as np
from ts_shape.features.segment_analysis.segment_extractor import SegmentExtractor
from ts_shape.features.segment_analysis.segment_processor import SegmentProcessor
from ts_shape.features.segment_analysis.profile_comparison import ProfileComparison


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
def profiles(production_df):
    """Metric profiles grouped by segment_value."""
    ranges = SegmentExtractor.extract_time_ranges(
        production_df, segment_uuid="order_number"
    )
    segmented = SegmentProcessor.apply_ranges(
        production_df,
        ranges,
        target_uuids=["temperature", "pressure", "speed"],
    )
    return SegmentProcessor.compute_metric_profiles(segmented)


@pytest.fixture
def profiles_by_index(production_df):
    """Metric profiles grouped by segment_index."""
    ranges = SegmentExtractor.extract_time_ranges(
        production_df, segment_uuid="order_number"
    )
    segmented = SegmentProcessor.apply_ranges(
        production_df,
        ranges,
        target_uuids=["temperature", "pressure", "speed"],
    )
    return SegmentProcessor.compute_metric_profiles(
        segmented, group_column="segment_index"
    )


class TestComputeDistanceMatrix:
    def test_compare_uuids(self, profiles):
        dm = ProfileComparison.compute_distance_matrix(profiles, group_column="uuid")
        assert dm.shape == (3, 3)
        np.testing.assert_array_almost_equal(np.diag(dm.values), 0.0)

    def test_compare_segments(self, profiles):
        dm = ProfileComparison.compute_distance_matrix(
            profiles, group_column="segment_value"
        )
        assert dm.shape == (2, 2)

    def test_symmetry(self, profiles):
        dm = ProfileComparison.compute_distance_matrix(profiles, group_column="uuid")
        np.testing.assert_array_almost_equal(dm.values, dm.values.T)

    def test_distance_metrics(self, profiles):
        for metric in ["euclidean", "cosine", "manhattan"]:
            dm = ProfileComparison.compute_distance_matrix(
                profiles, group_column="uuid", distance_metric=metric
            )
            assert dm.shape == (3, 3)

    def test_invalid_metric_raises(self, profiles):
        with pytest.raises(ValueError):
            ProfileComparison.compute_distance_matrix(
                profiles, group_column="uuid", distance_metric="invalid"
            )


class TestCluster:
    def test_n_clusters(self, profiles):
        dm = ProfileComparison.compute_distance_matrix(profiles, group_column="uuid")
        clusters = ProfileComparison.cluster(dm, n_clusters=2)
        assert len(clusters) == 3
        assert "cluster" in clusters.columns
        assert clusters["cluster"].nunique() <= 2

    def test_distance_threshold(self, profiles):
        dm = ProfileComparison.compute_distance_matrix(profiles, group_column="uuid")
        clusters = ProfileComparison.cluster(dm, distance_threshold=0.5)
        assert len(clusters) == 3


class TestFindSimilar:
    def test_basic(self, profiles):
        dm = ProfileComparison.compute_distance_matrix(profiles, group_column="uuid")
        result = ProfileComparison.find_similar(dm, target="temperature")
        assert len(result) == 2
        assert "distance" in result.columns
        assert "rank" in result.columns

    def test_invalid_target_raises(self, profiles):
        dm = ProfileComparison.compute_distance_matrix(profiles, group_column="uuid")
        with pytest.raises(ValueError):
            ProfileComparison.find_similar(dm, target="nonexistent")

    def test_sorted_by_distance(self, profiles):
        dm = ProfileComparison.compute_distance_matrix(profiles, group_column="uuid")
        result = ProfileComparison.find_similar(dm, target="temperature")
        assert list(result["distance"]) == sorted(result["distance"])


class TestDetectAnomalous:
    def test_output_columns(self, profiles):
        dm = ProfileComparison.compute_distance_matrix(profiles, group_column="uuid")
        result = ProfileComparison.detect_anomalous(dm)
        assert "anomaly_score" in result.columns
        assert "z_score" in result.columns
        assert "is_anomalous" in result.columns
        assert len(result) == 3


class TestDetectChanges:
    def test_by_segment_index(self, profiles_by_index):
        result = ProfileComparison.detect_changes(profiles_by_index)
        assert "change_score" in result.columns
        assert len(result) > 0

    def test_temperature_changes_more_than_speed(self, profiles_by_index):
        result = ProfileComparison.detect_changes(profiles_by_index)
        temp_max = result.loc[result["uuid"] == "temperature", "change_score"].max()
        speed_max = result.loc[result["uuid"] == "speed", "change_score"].max()
        assert temp_max > speed_max


class TestFindSimilarPairs:
    def test_output_columns(self, profiles):
        result = ProfileComparison.find_similar_pairs(profiles, top_k=3)
        assert "uuid_a" in result.columns
        assert "group_a" in result.columns
        assert "uuid_b" in result.columns
        assert "distance" in result.columns
        assert "rank" in result.columns

    def test_top_k_respected(self, profiles):
        result = ProfileComparison.find_similar_pairs(profiles, top_k=2)
        assert len(result) == 2

    def test_distances_sorted(self, profiles):
        result = ProfileComparison.find_similar_pairs(profiles, top_k=5)
        assert list(result["distance"]) == sorted(result["distance"])


class TestEdgeCases:
    def test_cluster_single_item(self):
        """cluster() with 1 item should return single cluster without crashing."""
        dm = pd.DataFrame([[0.0]], index=["only"], columns=["only"])
        result = ProfileComparison.cluster(dm, n_clusters=1)
        assert len(result) == 1
        assert result.iloc[0]["cluster"] == 1

    def test_cluster_two_items(self):
        dm = pd.DataFrame(
            [[0.0, 1.5], [1.5, 0.0]],
            index=["A", "B"],
            columns=["A", "B"],
        )
        result = ProfileComparison.cluster(dm, n_clusters=2)
        assert len(result) == 2
        assert result["cluster"].nunique() == 2

    def test_distance_matrix_single_group(self):
        """Single-group distance matrix should be 1x1 with value 0."""
        profiles = pd.DataFrame(
            {
                "uuid": ["A"],
                "segment_value": ["X"],
                "sample_count": [10],
                "mean": [5.0],
                "std": [1.0],
            }
        )
        dm = ProfileComparison.compute_distance_matrix(profiles, group_column="uuid")
        assert dm.shape == (1, 1)
        assert dm.iloc[0, 0] == 0.0

    def test_find_similar_single_item(self):
        dm = pd.DataFrame([[0.0]], index=["only"], columns=["only"])
        result = ProfileComparison.find_similar(dm, target="only", top_k=5)
        assert len(result) == 0

    def test_detect_anomalous_all_equal_distances(self):
        dm = pd.DataFrame(
            [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
            index=["A", "B", "C"],
            columns=["A", "B", "C"],
        )
        result = ProfileComparison.detect_anomalous(dm)
        assert not result["is_anomalous"].any()

    def test_find_similar_pairs_single_row(self):
        profiles = pd.DataFrame(
            {
                "uuid": ["A"],
                "segment_value": ["X"],
                "sample_count": [10],
                "mean": [5.0],
            }
        )
        result = ProfileComparison.find_similar_pairs(profiles, top_k=5)
        assert len(result) == 0

    def test_detect_changes_single_segment(self):
        """Single segment per UUID → no consecutive pairs → empty result."""
        profiles = pd.DataFrame(
            {
                "uuid": ["A", "B"],
                "segment_index": [0, 0],
                "sample_count": [10, 10],
                "mean": [5.0, 10.0],
                "std": [1.0, 2.0],
            }
        )
        result = ProfileComparison.detect_changes(profiles)
        assert len(result) == 0


class TestEndToEndWorkflow:
    def test_full_pipeline(self, production_df):
        """Complete workflow: extract → apply → profile → compare."""
        # 1. Extract
        ranges = SegmentExtractor.extract_time_ranges(
            production_df, segment_uuid="order_number"
        )
        assert len(ranges) == 3

        # 2. Apply
        segmented = SegmentProcessor.apply_ranges(
            production_df,
            ranges,
            target_uuids=["temperature", "pressure", "speed"],
        )
        assert "segment_value" in segmented.columns

        # 3. Profile
        profiles = SegmentProcessor.compute_metric_profiles(segmented)
        assert len(profiles) == 6

        # 4. Compare orders
        dm = ProfileComparison.compute_distance_matrix(
            profiles, group_column="segment_value"
        )
        assert dm.shape == (2, 2)
        assert dm.loc["Order-A", "Order-B"] > 0

        # 5. Compare UUIDs
        uuid_dm = ProfileComparison.compute_distance_matrix(
            profiles, group_column="uuid"
        )
        assert uuid_dm.shape == (3, 3)

        # 6. Cluster
        clusters = ProfileComparison.cluster(uuid_dm, n_clusters=2)
        assert len(clusters) == 3

        # 7. Anomaly
        anomalies = ProfileComparison.detect_anomalous(uuid_dm)
        assert len(anomalies) == 3
