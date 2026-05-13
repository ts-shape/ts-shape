import pytest
import pandas as pd
import numpy as np
from ts_shape.features.cross_signal import CrossSignalAnalytics


@pytest.fixture
def causal_df():
    """Create a DataFrame where X Granger-causes Y."""
    np.random.seed(42)
    n = 500
    x = np.cumsum(np.random.randn(n))
    y = np.zeros(n)
    y[0] = np.random.randn()
    y[1] = np.random.randn()
    for t in range(2, n):
        y[t] = 0.5 * y[t - 1] + 0.3 * x[t - 2] + np.random.randn() * 0.5

    times = pd.date_range("2024-01-01", periods=n, freq="1s")
    return pd.DataFrame({"systime": times, "X": x, "Y": y})


@pytest.fixture
def independent_df():
    """Create a DataFrame with two independent signals."""
    np.random.seed(42)
    n = 300
    times = pd.date_range("2024-01-01", periods=n, freq="1s")
    return pd.DataFrame(
        {
            "systime": times,
            "A": np.random.randn(n),
            "B": np.random.randn(n),
        }
    )


@pytest.fixture
def sine_df():
    """Create a DataFrame with sine and cosine signals."""
    n = 500
    t = np.linspace(0, 10 * np.pi, n)
    times = pd.date_range("2024-01-01", periods=n, freq="100ms")
    return pd.DataFrame(
        {
            "systime": times,
            "sine": np.sin(t),
            "cosine": np.cos(t),
            "sine_copy": np.sin(t),
        }
    )


@pytest.fixture
def lagged_df():
    """Create a DataFrame where B is a lagged copy of A."""
    np.random.seed(42)
    n = 300
    shift = 5
    a = np.sin(np.linspace(0, 20 * np.pi, n)) + np.random.randn(n) * 0.1
    b = np.zeros(n)
    b[shift:] = a[: n - shift]
    times = pd.date_range("2024-01-01", periods=n, freq="1s")
    return pd.DataFrame({"systime": times, "A": a, "B": b})


class TestGrangerCausality:
    def test_known_causal(self, causal_df):
        analytics = CrossSignalAnalytics(causal_df)
        result = analytics.granger_causality("X", "Y", max_lag=5)
        assert result["is_causal"] is True
        assert result["p_value"] < 0.05
        assert result["optimal_lag"] >= 1

    def test_independent_signals(self, independent_df):
        analytics = CrossSignalAnalytics(independent_df)
        result = analytics.granger_causality("A", "B", max_lag=5, significance=0.001)
        # With a strict significance threshold, independent signals should not be causal
        assert result["is_causal"] is False

    def test_returns_results_by_lag(self, causal_df):
        analytics = CrossSignalAnalytics(causal_df)
        result = analytics.granger_causality("X", "Y", max_lag=3)
        assert len(result["results_by_lag"]) == 3


class TestTransferEntropy:
    def test_directional(self, causal_df):
        analytics = CrossSignalAnalytics(causal_df)
        te_xy = analytics.transfer_entropy("X", "Y", lag=2, bins=8)
        te_yx = analytics.transfer_entropy("Y", "X", lag=2, bins=8)
        # X drives Y, so TE(X->Y) should be larger
        assert te_xy > te_yx

    def test_non_negative(self, independent_df):
        analytics = CrossSignalAnalytics(independent_df)
        te = analytics.transfer_entropy("A", "B")
        assert te >= 0.0

    def test_pairwise_shape(self, sine_df):
        analytics = CrossSignalAnalytics(sine_df)
        te_matrix = analytics.pairwise_transfer_entropy(bins=5)
        n_signals = len(analytics.signals)
        assert te_matrix.shape == (n_signals, n_signals)
        # Diagonal should be 0
        for sig in analytics.signals:
            assert te_matrix.loc[sig, sig] == 0.0


class TestSynchronization:
    def test_identical_signals_phase(self, sine_df):
        analytics = CrossSignalAnalytics(sine_df)
        plv = analytics.synchronization_index("sine", "sine_copy", method="phase")
        assert plv > 0.95

    def test_sine_cosine_phase_locked(self, sine_df):
        analytics = CrossSignalAnalytics(sine_df)
        plv = analytics.synchronization_index("sine", "cosine", method="phase")
        # Sine and cosine are phase-locked (constant phase difference)
        assert plv > 0.9

    def test_uncorrelated_low_sync(self, independent_df):
        analytics = CrossSignalAnalytics(independent_df)
        plv = analytics.synchronization_index("A", "B", method="phase")
        assert plv < 0.5

    def test_pairwise_shape(self, sine_df):
        analytics = CrossSignalAnalytics(sine_df)
        sync_matrix = analytics.pairwise_synchronization(method="phase")
        n_signals = len(analytics.signals)
        assert sync_matrix.shape == (n_signals, n_signals)
        # Diagonal should be 1.0
        for sig in analytics.signals:
            assert sync_matrix.loc[sig, sig] == 1.0


class TestLeadLag:
    def test_known_shift(self, lagged_df):
        analytics = CrossSignalAnalytics(lagged_df)
        result = analytics.lead_lag("A", "B", max_lag=20)
        # The absolute lag should be close to 5
        assert abs(abs(result["optimal_lag"]) - 5) <= 2

    def test_no_relationship(self, independent_df):
        analytics = CrossSignalAnalytics(independent_df)
        result = analytics.lead_lag("A", "B", max_lag=20)
        # Random signals shouldn't show significant lead-lag
        assert abs(result["correlation_at_lag"]) < 0.5

    def test_lead_lag_matrix_shape(self, sine_df):
        analytics = CrossSignalAnalytics(sine_df)
        matrix = analytics.lead_lag_matrix(max_lag=20)
        n_signals = len(analytics.signals)
        assert matrix.shape == (n_signals, n_signals)


class TestInit:
    def test_minimum_signals_required(self):
        df = pd.DataFrame(
            {"systime": pd.date_range("2024-01-01", periods=10), "A": range(10)}
        )
        with pytest.raises(ValueError):
            CrossSignalAnalytics(df)

    def test_accepts_wide_format(self, sine_df):
        analytics = CrossSignalAnalytics(sine_df)
        assert len(analytics.signals) >= 2
