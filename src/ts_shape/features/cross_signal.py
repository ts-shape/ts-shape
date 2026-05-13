import logging
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from typing import Optional, List, Dict, Any

from scipy import signal as scipy_signal  # type: ignore
from scipy import stats as scipy_stats  # type: ignore

logger = logging.getLogger(__name__)


class CrossSignalAnalytics:
    """Cross-Signal Analytics for multi-signal timeseries.

    Computes analytical metrics across pairs or groups of signals.
    Expects wide-format DataFrames (one column per signal, datetime index).

    Methods:
    - granger_causality: Test if one signal Granger-causes another.
    - transfer_entropy: Estimate information transfer between signals.
    - pairwise_transfer_entropy: Transfer entropy for all directed pairs.
    - synchronization_index: Phase or amplitude synchronization between signals.
    - pairwise_synchronization: Synchronization for all pairs.
    - lead_lag: Detect lead-lag relationships via cross-correlation.
    - lead_lag_matrix: Lead-lag for all pairs.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        time_column: str = "systime",
    ) -> None:
        if time_column in dataframe.columns:
            self.df = dataframe.set_index(time_column).copy()
        else:
            self.df = dataframe.copy()

        self.signals = [
            col
            for col in self.df.columns
            if self.df[col].dtype
            in [np.float64, np.float32, np.int64, np.int32, float, int]
        ]
        if len(self.signals) < 2:
            raise ValueError(
                f"Need at least 2 numeric signal columns, found {len(self.signals)}."
            )

    def granger_causality(
        self,
        cause: str,
        effect: str,
        max_lag: int = 10,
        significance: float = 0.05,
    ) -> dict:
        """Test whether `cause` Granger-causes `effect` using F-test.

        Args:
            cause: Column name of the potential cause signal.
            effect: Column name of the potential effect signal.
            max_lag: Maximum number of lags to test.
            significance: P-value threshold for declaring causality.

        Returns:
            Dict with optimal_lag, f_statistic, p_value, is_causal, results_by_lag.
        """
        x = self.df[cause].dropna().values
        y = self.df[effect].dropna().values
        n = min(len(x), len(y))
        x, y = x[:n], y[:n]

        results_by_lag: List[Dict[str, Any]] = []

        for p in range(1, max_lag + 1):
            if n <= 2 * p + 1:
                break

            # Build lag matrices
            y_target = y[p:]
            y_lags = np.column_stack([y[p - i - 1 : n - i - 1] for i in range(p)])
            x_lags = np.column_stack([x[p - i - 1 : n - i - 1] for i in range(p)])

            # Restricted model: y ~ y_lags
            X_r = np.column_stack([np.ones(len(y_target)), y_lags])
            beta_r, rss_r, _, _ = np.linalg.lstsq(X_r, y_target, rcond=None)
            resid_r = y_target - X_r @ beta_r
            rss_r_val = float(np.sum(resid_r**2))

            # Unrestricted model: y ~ y_lags + x_lags
            X_u = np.column_stack([np.ones(len(y_target)), y_lags, x_lags])
            beta_u, rss_u, _, _ = np.linalg.lstsq(X_u, y_target, rcond=None)
            resid_u = y_target - X_u @ beta_u
            rss_u_val = float(np.sum(resid_u**2))

            dof = len(y_target) - 2 * p - 1
            if dof <= 0 or rss_u_val <= 0:
                continue

            f_stat = ((rss_r_val - rss_u_val) / p) / (rss_u_val / dof)
            p_value = float(scipy_stats.f.sf(f_stat, p, dof))

            results_by_lag.append(
                {
                    "lag": p,
                    "f_statistic": f_stat,
                    "p_value": p_value,
                }
            )

        if not results_by_lag:
            return {
                "optimal_lag": 0,
                "f_statistic": 0.0,
                "p_value": 1.0,
                "is_causal": False,
                "results_by_lag": [],
            }

        best = min(results_by_lag, key=lambda r: r["p_value"])
        return {
            "optimal_lag": best["lag"],
            "f_statistic": best["f_statistic"],
            "p_value": best["p_value"],
            "is_causal": best["p_value"] < significance,
            "results_by_lag": results_by_lag,
        }

    def transfer_entropy(
        self,
        source: str,
        target: str,
        lag: int = 1,
        bins: int = 10,
    ) -> float:
        """Estimate transfer entropy from source to target.

        TE(X->Y) = H(Y_future | Y_past) - H(Y_future | Y_past, X_past)

        Args:
            source: Column name of source signal.
            target: Column name of target signal.
            lag: Time lag for past values.
            bins: Number of bins for discretization.

        Returns:
            Transfer entropy in bits.
        """
        x = self.df[source].dropna().values
        y = self.df[target].dropna().values
        n = min(len(x), len(y))
        x, y = x[:n], y[:n]

        if n <= lag:
            return 0.0

        y_future = y[lag:]
        y_past = y[: n - lag]
        x_past = x[: n - lag]

        # Discretize
        x_bins = np.linspace(x_past.min() - 1e-10, x_past.max() + 1e-10, bins + 1)
        y_bins = np.linspace(
            min(y_future.min(), y_past.min()) - 1e-10,
            max(y_future.max(), y_past.max()) + 1e-10,
            bins + 1,
        )

        x_d = np.digitize(x_past, x_bins) - 1
        yf_d = np.digitize(y_future, y_bins) - 1
        yp_d = np.digitize(y_past, y_bins) - 1

        # Joint distribution p(y_future, y_past, x_past)
        joint_3d, _ = np.histogramdd(
            np.column_stack([yf_d, yp_d, x_d]),
            bins=[bins, bins, bins],
            range=[[0, bins], [0, bins], [0, bins]],
        )
        joint_3d = joint_3d / joint_3d.sum()

        # p(y_future, y_past)
        joint_2d = joint_3d.sum(axis=2)

        # p(y_past, x_past)
        joint_ypxp = joint_3d.sum(axis=0)

        # p(y_past)
        p_yp = joint_2d.sum(axis=0)

        te = 0.0
        for i in range(bins):
            for j in range(bins):
                for k in range(bins):
                    p_joint = joint_3d[i, j, k]
                    if p_joint <= 0:
                        continue
                    p_yf_yp = joint_2d[i, j]
                    p_yp_xp = joint_ypxp[j, k]
                    p_yp_val = p_yp[j]

                    if p_yf_yp <= 0 or p_yp_xp <= 0 or p_yp_val <= 0:
                        continue

                    te += p_joint * np.log2((p_joint * p_yp_val) / (p_yf_yp * p_yp_xp))

        return max(0.0, float(te))

    def pairwise_transfer_entropy(
        self,
        lag: int = 1,
        bins: int = 10,
    ) -> pd.DataFrame:
        """Compute transfer entropy for all directed signal pairs.

        Returns:
            Square DataFrame where result.loc[A, B] is TE(A -> B).
        """
        n = len(self.signals)
        matrix = np.zeros((n, n))

        for i, src in enumerate(self.signals):
            for j, tgt in enumerate(self.signals):
                if i != j:
                    matrix[i, j] = self.transfer_entropy(src, tgt, lag=lag, bins=bins)

        return pd.DataFrame(matrix, index=self.signals, columns=self.signals)

    def synchronization_index(
        self,
        signal_a: str,
        signal_b: str,
        method: str = "phase",
    ) -> float:
        """Measure synchronization between two signals.

        Args:
            signal_a: First signal column name.
            signal_b: Second signal column name.
            method: 'phase' for phase-locking value, 'amplitude' for envelope correlation.

        Returns:
            Float between 0 (no sync) and 1 (perfect sync).
        """
        a = self.df[signal_a].dropna().values
        b = self.df[signal_b].dropna().values
        n = min(len(a), len(b))
        a, b = a[:n].astype(float), b[:n].astype(float)

        if method == "phase":
            analytic_a = scipy_signal.hilbert(a)
            analytic_b = scipy_signal.hilbert(b)
            phase_a = np.angle(analytic_a)
            phase_b = np.angle(analytic_b)
            plv = float(np.abs(np.mean(np.exp(1j * (phase_a - phase_b)))))
            return plv

        elif method == "amplitude":
            env_a = np.abs(scipy_signal.hilbert(a))
            env_b = np.abs(scipy_signal.hilbert(b))
            corr = float(np.corrcoef(env_a, env_b)[0, 1])
            return max(0.0, corr)

        else:
            raise ValueError(f"Unknown method: {method}. Use 'phase' or 'amplitude'.")

    def pairwise_synchronization(self, method: str = "phase") -> pd.DataFrame:
        """Compute synchronization index for all signal pairs.

        Returns:
            Symmetric DataFrame (like a correlation matrix).
        """
        n = len(self.signals)
        matrix = np.zeros((n, n))

        for i in range(n):
            matrix[i, i] = 1.0
            for j in range(i + 1, n):
                val = self.synchronization_index(
                    self.signals[i], self.signals[j], method=method
                )
                matrix[i, j] = val
                matrix[j, i] = val

        return pd.DataFrame(matrix, index=self.signals, columns=self.signals)

    def lead_lag(
        self,
        signal_a: str,
        signal_b: str,
        max_lag: int = 50,
        significance: float = 0.05,
    ) -> dict:
        """Detect lead-lag relationship via cross-correlation.

        Args:
            signal_a: First signal column name.
            signal_b: Second signal column name.
            max_lag: Maximum lag to test.
            significance: P-value threshold.

        Returns:
            Dict with optimal_lag, correlation_at_lag, leader, follower, p_value, is_significant.
        """
        a = self.df[signal_a].dropna().values.astype(float)
        b = self.df[signal_b].dropna().values.astype(float)
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]

        # Normalize
        a = (a - a.mean()) / (a.std() + 1e-12)
        b = (b - b.mean()) / (b.std() + 1e-12)

        corr_full = scipy_signal.correlate(a, b, mode="full") / n
        lags = scipy_signal.correlation_lags(n, n, mode="full")

        # Restrict to max_lag
        mask = np.abs(lags) <= max_lag
        corr_clipped = corr_full[mask]
        lags_clipped = lags[mask]

        best_idx = int(np.argmax(np.abs(corr_clipped)))
        optimal_lag = int(lags_clipped[best_idx])
        best_corr = float(corr_clipped[best_idx])

        # Permutation test for significance
        n_perms = 100
        perm_maxes = []
        rng = np.random.RandomState(42)
        for _ in range(n_perms):
            b_perm = rng.permutation(b)
            corr_perm = scipy_signal.correlate(a, b_perm, mode="full") / n
            corr_perm_clipped = corr_perm[mask]
            perm_maxes.append(np.max(np.abs(corr_perm_clipped)))

        p_value = float(np.mean(np.array(perm_maxes) >= np.abs(best_corr)))

        leader = signal_a if optimal_lag >= 0 else signal_b
        follower = signal_b if optimal_lag >= 0 else signal_a

        return {
            "optimal_lag": optimal_lag,
            "correlation_at_lag": best_corr,
            "leader": leader,
            "follower": follower,
            "p_value": p_value,
            "is_significant": p_value < significance,
        }

    def lead_lag_matrix(self, max_lag: int = 50) -> pd.DataFrame:
        """Compute lead-lag for all signal pairs.

        Returns:
            DataFrame where result.loc[A, B] is the optimal lag (positive = A leads B).
        """
        n = len(self.signals)
        matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i != j:
                    result = self.lead_lag(
                        self.signals[i], self.signals[j], max_lag=max_lag
                    )
                    matrix[i, j] = result["optimal_lag"]

        return pd.DataFrame(matrix, index=self.signals, columns=self.signals)
