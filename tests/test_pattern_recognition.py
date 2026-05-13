import pytest
import pandas as pd
import numpy as np
from ts_shape.features.pattern_recognition import PatternRecognition


@pytest.fixture
def motif_df():
    """Create a DataFrame with repeated patterns embedded in noise."""
    np.random.seed(42)
    n = 500
    series = np.random.randn(n) * 0.1
    # Insert the same pattern at positions 50, 200, 350
    pattern = np.sin(np.linspace(0, 2 * np.pi, 30))
    for start in [50, 200, 350]:
        series[start : start + 30] = pattern

    times = pd.date_range("2024-01-01", periods=n, freq="1s")
    return pd.DataFrame({"systime": times, "value_double": series})


@pytest.fixture
def discord_df():
    """Create a DataFrame with a single anomalous subsequence."""
    np.random.seed(42)
    n = 300
    series = np.sin(np.linspace(0, 20 * np.pi, n))
    # Inject a spike anomaly
    series[150:170] = 5.0 + np.random.randn(20) * 0.1

    times = pd.date_range("2024-01-01", periods=n, freq="1s")
    return pd.DataFrame({"systime": times, "value_double": series})


@pytest.fixture
def simple_df():
    """Simple short DataFrame for basic testing."""
    np.random.seed(42)
    n = 100
    times = pd.date_range("2024-01-01", periods=n, freq="1s")
    return pd.DataFrame(
        {
            "systime": times,
            "value_double": np.sin(np.linspace(0, 4 * np.pi, n)),
        }
    )


class TestDiscoverMotifs:
    def test_finds_repeated_patterns(self, motif_df):
        result = PatternRecognition.discover_motifs(motif_df, window_size=30, top_k=3)
        assert len(result) > 0
        # The top motif should have a low distance
        assert result.iloc[0]["distance"] < 1.0

    def test_top_k_limit(self, motif_df):
        result = PatternRecognition.discover_motifs(motif_df, window_size=30, top_k=2)
        assert len(result) <= 2

    def test_motif_indices_in_range(self, motif_df):
        result = PatternRecognition.discover_motifs(motif_df, window_size=30, top_k=3)
        for _, row in result.iterrows():
            assert 0 <= row["index_a"] < len(motif_df)
            assert 0 <= row["index_b"] < len(motif_df)


class TestDiscoverDiscords:
    def test_injected_anomaly(self, discord_df):
        result = PatternRecognition.discover_discords(
            discord_df, window_size=20, top_k=3
        )
        assert len(result) > 0
        # The top discord should be near the injected anomaly
        top_discord_idx = result.iloc[0]["start_index"]
        assert abs(top_discord_idx - 150) < 30

    def test_top_k_limit(self, discord_df):
        result = PatternRecognition.discover_discords(
            discord_df, window_size=20, top_k=1
        )
        assert len(result) == 1


class TestSimilaritySearch:
    def test_exact_match(self, simple_df):
        series = simple_df["value_double"].values
        query = series[20:30].copy()
        result = PatternRecognition.similarity_search(
            simple_df, query, top_k=1, normalize=True
        )
        assert len(result) == 1
        assert result.iloc[0]["dtw_distance"] < 1e-5

    def test_scaled_match_with_normalize(self, simple_df):
        series = simple_df["value_double"].values
        query = series[20:30].copy() * 2.0 + 5.0  # Scale and shift
        result = PatternRecognition.similarity_search(
            simple_df, query, top_k=1, normalize=True
        )
        assert len(result) == 1
        # With normalization, it should still find the match
        assert result.iloc[0]["dtw_distance"] < 1e-5

    def test_top_k_results(self, simple_df):
        query = np.sin(np.linspace(0, np.pi, 10))
        result = PatternRecognition.similarity_search(
            simple_df, query, top_k=3, normalize=True
        )
        assert len(result) == 3
        # Results should be sorted by distance
        assert result.iloc[0]["dtw_distance"] <= result.iloc[1]["dtw_distance"]


class TestTemplateMatch:
    def test_finds_occurrences(self, motif_df):
        pattern = np.sin(np.linspace(0, 2 * np.pi, 30))
        result = PatternRecognition.template_match(motif_df, pattern, threshold=2.0)
        # Should find at least some of the 3 inserted patterns
        assert len(result) >= 1

    def test_tight_threshold_filters(self, simple_df):
        query = simple_df["value_double"].values[:10]
        # Very tight threshold
        result = PatternRecognition.template_match(simple_df, query, threshold=0.001)
        # Should find at most 1-2 matches with such a tight threshold
        assert len(result) <= 3


class TestComputeDistanceProfile:
    def test_euclidean_profile_length(self, simple_df):
        query = np.sin(np.linspace(0, np.pi, 10))
        profile = PatternRecognition.compute_distance_profile(
            simple_df, query, metric="euclidean"
        )
        expected_len = len(simple_df) - len(query) + 1
        assert len(profile) == expected_len

    def test_dtw_profile_length(self):
        # Small dataset for DTW (expensive)
        n = 50
        df = pd.DataFrame(
            {
                "systime": pd.date_range("2024-01-01", periods=n, freq="1s"),
                "value_double": np.random.randn(n),
            }
        )
        query = np.random.randn(5)
        profile = PatternRecognition.compute_distance_profile(df, query, metric="dtw")
        expected_len = n - len(query) + 1
        assert len(profile) == expected_len

    def test_invalid_metric(self, simple_df):
        query = np.ones(5)
        with pytest.raises(ValueError):
            PatternRecognition.compute_distance_profile(
                simple_df, query, metric="invalid"
            )


class TestMass:
    def test_mass_matches_brute_force(self):
        """Compare MASS output against brute-force on small data."""
        np.random.seed(42)
        series = np.random.randn(50)
        query = series[10:15].copy()
        m = len(query)

        mass_dists = PatternRecognition._mass(series, query)

        # Brute-force z-normalized Euclidean distance
        q_norm = (query - query.mean()) / max(query.std(ddof=0), 1e-10)
        brute_dists = []
        for i in range(len(series) - m + 1):
            subseq = series[i : i + m]
            s_norm = (subseq - subseq.mean()) / max(subseq.std(ddof=0), 1e-10)
            d = np.sqrt(np.sum((q_norm - s_norm) ** 2))
            brute_dists.append(d)
        brute_dists = np.array(brute_dists)

        np.testing.assert_allclose(mass_dists, brute_dists, atol=1e-3)


class TestMatrixProfile:
    def test_profile_length(self, simple_df):
        series = simple_df["value_double"].values
        window_size = 10
        mp, mpi = PatternRecognition._compute_matrix_profile(
            series, window_size, window_size // 2
        )
        expected_len = len(series) - window_size + 1
        assert len(mp) == expected_len
        assert len(mpi) == expected_len
