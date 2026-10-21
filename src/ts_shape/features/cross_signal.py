import logging
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from ts_shape.utils.coerce import to_numeric_checked
from ts_shape.utils.frames import as_datetime, require_columns

logger = logging.getLogger(__name__)


def _is_signal(series: pd.Series) -> bool:
    """A numeric (NumPy, nullable or pyarrow), non-boolean column."""
    return ptypes.is_numeric_dtype(series.dtype) and not ptypes.is_bool_dtype(
        series.dtype
    )


def _numeric_object(series: pd.Series) -> pd.Series | None:
    """An ``object`` / text column of numbers as numbers, else ``None``.

    Numbers stored as ``object`` (a JSON or CSV load, a column of Python
    floats or numeric text) are read as a signal, as the detectors do with
    ``pd.to_numeric(errors="coerce")``; a column without a single numeric
    value (labels, ids) is not a signal. Booleans are not signals.
    """
    if not (
        ptypes.is_object_dtype(series.dtype) or ptypes.is_string_dtype(series.dtype)
    ):
        return None
    known = series.dropna()
    if known.empty or known.map(lambda v: isinstance(v, (bool, np.bool_))).any():
        return None
    numeric = pd.to_numeric(known, errors="coerce")
    if not numeric.notna().any() or ptypes.is_bool_dtype(numeric.dtype):
        return None
    return to_numeric_checked(series, what=f"column {series.name!r}")


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
        """Keep a time-ordered copy of the wide frame.

        Args:
            dataframe: Wide frame, one numeric column per signal. NumPy,
                nullable (``Float64``, ``Int64``) and pyarrow numeric columns
                are signals; they are read as ``float64``. An ``object`` /
                text column holding numbers is coerced with
                ``pd.to_numeric(errors="coerce")`` (values that are not
                numeric become NaN, with a ``DataQualityWarning``); a column
                with no numeric value is not a signal.
            time_column: Time column; when present, rows are sorted by it and
                it becomes the index.

        Raises:
            ValueError: If fewer than 2 numeric signal columns are found.
        """
        df = dataframe.copy()
        if time_column in df.columns:
            df[time_column] = as_datetime(df[time_column], where="CrossSignalAnalytics")
            df = df.sort_values(time_column, kind="stable").set_index(time_column)
        signals = []
        for col in df.columns:
            if _is_signal(df[col]):
                signals.append(col)
                continue
            coerced = _numeric_object(df[col])
            if coerced is not None:
                df[col] = coerced
                signals.append(col)
        self.signals = signals
        for col in self.signals:
            df[col] = np.asarray(
                df[col].to_numpy(dtype="float64", na_value=np.nan), dtype="float64"
            )
        self.df = df
        if len(self.signals) < 2:
            raise ValueError(
                f"Need at least 2 numeric signal columns, found {len(self.signals)}."
            )

    def _pair(
        self, first: str, second: str, where: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Both signals at the rows where both have a value, in time order.

        Raises:
            ColumnNotFoundError: If a signal column is missing.
        """
        require_columns(self.df, [first, second], where=f"CrossSignalAnalytics.{where}")
        both = self.df[[first, second]].apply(pd.to_numeric, errors="coerce")
        both = both.dropna()
        x = np.asarray(both.iloc[:, 0].to_numpy(dtype="float64"), dtype="float64")
        y = np.asarray(both.iloc[:, 1].to_numpy(dtype="float64"), dtype="float64")
        return x, y

    def granger_causality(
        self,
        cause: str,
        effect: str,
        max_lag: int = 10,
        significance: float = 0.05,
    ) -> dict:
        """Test whether `cause` Granger-causes `effect` using F-test.

        One F-test is run per lag ``1..max_lag`` and the lag with the smallest
        p-value is reported. Picking the best of several tests inflates the
        false-positive rate (about 15% instead of 5% for ``max_lag=10`` on
        independent noise), so the decision uses a Bonferroni correction for
        the number of lags tested: ``p_value_adjusted = min(1, p_value * k)``
        with ``k`` the number of lags that could be fitted, and ``is_causal``
        is ``p_value_adjusted < significance``. This is conservative (the
        lag tests are correlated) but keeps the false-positive rate at or
        below ``significance``.

        Args:
            cause: Column name of the potential cause signal.
            effect: Column name of the potential effect signal.
            max_lag: Maximum number of lags to test.
            significance: Threshold on the adjusted p-value for declaring
                causality.

        Returns:
            Dict with optimal_lag, f_statistic, p_value (the smallest,
                unadjusted p-value over the lags), p_value_adjusted (the
                Bonferroni-adjusted one), is_causal and results_by_lag (each
                entry with lag, f_statistic and unadjusted p_value). Only rows
                where both signals have a value are used. With too few of them
                to fit a lag, ``f_statistic``, ``p_value`` and
                ``p_value_adjusted`` are NaN and ``is_causal`` is False.
        """
        from scipy import stats as scipy_stats

        x, y = self._pair(cause, effect, "granger_causality")
        n = len(x)

        results_by_lag: list[dict[str, Any]] = []

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
            # Too few aligned samples to fit any lag: no test result.
            return {
                "optimal_lag": 0,
                "f_statistic": float("nan"),
                "p_value": float("nan"),
                "p_value_adjusted": float("nan"),
                "is_causal": False,
                "results_by_lag": [],
            }

        best = min(results_by_lag, key=lambda r: r["p_value"])
        # Bonferroni over the lags actually tested.
        p_adjusted = min(1.0, best["p_value"] * len(results_by_lag))
        return {
            "optimal_lag": best["lag"],
            "f_statistic": best["f_statistic"],
            "p_value": best["p_value"],
            "p_value_adjusted": p_adjusted,
            "is_causal": p_adjusted < significance,
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
            Transfer entropy in bits; NaN when there are no more aligned
                samples (rows where both signals have a value) than ``lag``.
        """
        x, y = self._pair(source, target, "transfer_entropy")
        n = len(x)

        if n <= lag:
            return float("nan")

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
            range=[(0.0, float(bins))] * 3,
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

        Args:
            lag: Number of samples between the past values and the future value of the
                target (see `transfer_entropy`). Default 1.
            bins: Number of equal-width bins used to discretize each signal. Default 10.

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
            Float between 0 (no sync) and 1 (perfect sync); NaN when no row
                has a value for both signals.
        """
        from scipy import signal as scipy_signal

        if method not in ("phase", "amplitude"):
            raise ValueError(f"Unknown method: {method}. Use 'phase' or 'amplitude'.")
        a, b = self._pair(signal_a, signal_b, "synchronization_index")
        if len(a) == 0:
            return float("nan")

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

        Args:
            method: ``'phase'`` for the phase-locking value or ``'amplitude'`` for
                envelope correlation (see `synchronization_index`). Default ``'phase'``.

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
            Dict with optimal_lag, correlation_at_lag, leader, follower, p_value,
                is_significant. ``optimal_lag`` is the number of samples by
                which ``signal_a`` leads ``signal_b``: positive when
                ``signal_b`` repeats ``signal_a`` later (``leader`` is
                ``signal_a``), negative when ``signal_b`` leads. Only rows where
                both signals have a value are used; with fewer than 2 such rows
                the numbers are NaN and ``leader`` / ``follower`` are None.
        """
        from scipy import signal as scipy_signal

        a, b = self._pair(signal_a, signal_b, "lead_lag")
        n = len(a)
        if n < 2:
            return {
                "optimal_lag": float("nan"),
                "correlation_at_lag": float("nan"),
                "leader": None,
                "follower": None,
                "p_value": float("nan"),
                "is_significant": False,
            }

        # Normalize
        a = (a - a.mean()) / (a.std() + 1e-12)
        b = (b - b.mean()) / (b.std() + 1e-12)

        # correlate(a, b)[k] = sum_n a[n + k] * b[n]: a peak at lag k < 0
        # means b repeats a k samples later, i.e. a leads.
        corr_full = scipy_signal.correlate(a, b, mode="full") / n
        lags = scipy_signal.correlation_lags(n, n, mode="full")

        # Restrict to max_lag
        mask = np.abs(lags) <= max_lag
        corr_clipped = corr_full[mask]
        lags_clipped = lags[mask]

        best_idx = int(np.argmax(np.abs(corr_clipped)))
        optimal_lag = -int(lags_clipped[best_idx])
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

        Args:
            max_lag: Largest lag, in samples, tested in either direction. Default 50.

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
