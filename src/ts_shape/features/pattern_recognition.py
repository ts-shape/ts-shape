import logging
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from typing import Optional, Tuple

from scipy.fft import fft, ifft  # type: ignore

from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class PatternRecognition(Base):
    """Pattern Recognition for univariate timeseries.

    Discover motifs, discords, and template matches using Matrix Profile
    and Dynamic Time Warping approaches.

    Methods:
    - discover_motifs: Find top-k recurring subsequence patterns.
    - discover_discords: Find top-k anomalous subsequences.
    - similarity_search: Find subsequences most similar to a query (DTW).
    - template_match: Find all occurrences of a reference template.
    - compute_distance_profile: Distance from query to every subsequence.
    """

    @classmethod
    def _mass(cls, series: np.ndarray, query: np.ndarray) -> np.ndarray:
        """MASS: compute z-normalized Euclidean distance profile via FFT.

        Args:
            series: The full time series.
            query: The query subsequence.

        Returns:
            1D array of distances for each position.
        """
        n = len(series)
        m = len(query)

        if n < m:
            return np.array([])

        # Pad query for FFT
        q_reversed = query[::-1]
        q_padded = np.zeros(n)
        q_padded[:m] = q_reversed

        # Sliding dot product via FFT
        QT = np.real(ifft(fft(series) * fft(q_padded)))
        QT = QT[m - 1 :]

        # Rolling statistics
        s = pd.Series(series)
        rolling_mean = s.rolling(window=m).mean().values[m - 1 :]
        rolling_std = s.rolling(window=m).std(ddof=0).values[m - 1 :]

        q_mean = query.mean()
        q_std = query.std(ddof=0)

        # Avoid division by zero
        rolling_std = np.where(rolling_std < 1e-10, 1e-10, rolling_std)
        q_std = max(q_std, 1e-10)

        dist_sq = (
            2 * m * (1 - (QT - m * q_mean * rolling_mean) / (m * q_std * rolling_std))
        )
        dist_sq = np.maximum(dist_sq, 0)
        return np.sqrt(dist_sq)

    @classmethod
    def _compute_matrix_profile(
        cls,
        series: np.ndarray,
        window_size: int,
        exclusion_zone: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the Matrix Profile and Matrix Profile Index.

        Args:
            series: 1D time series array.
            window_size: Subsequence length.
            exclusion_zone: Number of indices to exclude around a match.

        Returns:
            Tuple of (matrix_profile, matrix_profile_index).
        """
        n = len(series)
        profile_len = n - window_size + 1
        mp = np.full(profile_len, np.inf)
        mpi = np.full(profile_len, -1, dtype=int)

        for i in range(profile_len):
            query = series[i : i + window_size]
            dist_profile = cls._mass(series, query)

            # Apply exclusion zone
            ez_start = max(0, i - exclusion_zone)
            ez_end = min(profile_len, i + exclusion_zone + 1)
            dist_profile[ez_start:ez_end] = np.inf

            min_idx = np.argmin(dist_profile)
            min_val = dist_profile[min_idx]

            mp[i] = min_val
            mpi[i] = min_idx

        return mp, mpi

    @classmethod
    def discover_motifs(
        cls,
        dataframe: pd.DataFrame,
        value_column: str = "value_double",
        window_size: int = 50,
        top_k: int = 5,
        exclusion_zone: Optional[int] = None,
        time_column: str = "systime",
    ) -> pd.DataFrame:
        """Find the top-k recurring subsequence patterns (motifs).

        Args:
            dataframe: Input DataFrame.
            value_column: Column containing numeric values.
            window_size: Length of subsequences to compare.
            top_k: Number of motif pairs to return.
            exclusion_zone: Indices to exclude around matches. Defaults to window_size // 2.
            time_column: Column containing timestamps.

        Returns:
            DataFrame with motif_rank, index_a, index_b, distance, time_a, time_b.
        """
        series = dataframe[value_column].values.astype(float)
        if exclusion_zone is None:
            exclusion_zone = window_size // 2

        mp, mpi = cls._compute_matrix_profile(series, window_size, exclusion_zone)

        # Find top-k motifs (smallest MP values)
        sorted_indices = np.argsort(mp)
        results = []
        used = set()

        for idx in sorted_indices:
            if len(results) >= top_k:
                break
            partner = mpi[idx]
            pair = (min(idx, partner), max(idx, partner))
            if pair in used:
                continue
            if abs(idx - partner) <= exclusion_zone:
                continue
            used.add(pair)

            row = {
                "motif_rank": len(results) + 1,
                "index_a": pair[0],
                "index_b": pair[1],
                "distance": mp[idx],
            }
            if time_column in dataframe.columns:
                times = dataframe[time_column].values
                row["time_a"] = times[pair[0]]
                row["time_b"] = times[pair[1]]

            results.append(row)

        return pd.DataFrame(results)

    @classmethod
    def discover_discords(
        cls,
        dataframe: pd.DataFrame,
        value_column: str = "value_double",
        window_size: int = 50,
        top_k: int = 5,
        exclusion_zone: Optional[int] = None,
        time_column: str = "systime",
    ) -> pd.DataFrame:
        """Find the top-k anomalous subsequences (discords).

        Args:
            dataframe: Input DataFrame.
            value_column: Column containing numeric values.
            window_size: Length of subsequences to compare.
            top_k: Number of discords to return.
            exclusion_zone: Indices to exclude around matches. Defaults to window_size // 2.
            time_column: Column containing timestamps.

        Returns:
            DataFrame with discord_rank, start_index, distance, start_time.
        """
        series = dataframe[value_column].values.astype(float)
        if exclusion_zone is None:
            exclusion_zone = window_size // 2

        mp, _ = cls._compute_matrix_profile(series, window_size, exclusion_zone)

        # Find top-k discords (largest MP values)
        sorted_indices = np.argsort(mp)[::-1]
        results = []
        used_zones = []

        for idx in sorted_indices:
            if len(results) >= top_k:
                break
            # Check exclusion zone against already selected discords
            if any(abs(idx - u) <= exclusion_zone for u in used_zones):
                continue
            used_zones.append(idx)

            row = {
                "discord_rank": len(results) + 1,
                "start_index": int(idx),
                "distance": float(mp[idx]),
            }
            if time_column in dataframe.columns:
                row["start_time"] = dataframe[time_column].values[idx]

            results.append(row)

        return pd.DataFrame(results)

    @classmethod
    def _dtw_distance(cls, a: np.ndarray, b: np.ndarray) -> float:
        """Compute DTW distance between two sequences."""
        n, m = len(a), len(b)
        D = np.full((n + 1, m + 1), np.inf)
        D[0, 0] = 0.0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = (a[i - 1] - b[j - 1]) ** 2
                D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])

        return float(np.sqrt(D[n, m]))

    @classmethod
    def similarity_search(
        cls,
        dataframe: pd.DataFrame,
        query: np.ndarray,
        value_column: str = "value_double",
        top_k: int = 5,
        normalize: bool = True,
        time_column: str = "systime",
    ) -> pd.DataFrame:
        """Find the top-k most similar subsequences to a query using DTW.

        Args:
            dataframe: Input DataFrame.
            query: Query pattern as numpy array.
            value_column: Column containing numeric values.
            top_k: Number of matches to return.
            normalize: Whether to z-normalize before comparison.
            time_column: Column containing timestamps.

        Returns:
            DataFrame with rank, start_index, dtw_distance, start_time.
        """
        series = dataframe[value_column].values.astype(float)
        m = len(query)
        n = len(series)

        if n < m:
            return pd.DataFrame(
                columns=["rank", "start_index", "dtw_distance", "start_time"]
            )

        q = query.astype(float).copy()
        if normalize and q.std() > 1e-10:
            q = (q - q.mean()) / q.std()

        distances = []
        for i in range(n - m + 1):
            subseq = series[i : i + m].copy()
            if normalize and subseq.std() > 1e-10:
                subseq = (subseq - subseq.mean()) / subseq.std()
            d = cls._dtw_distance(q, subseq)
            distances.append((i, d))

        distances.sort(key=lambda x: x[1])
        results = []
        for rank, (idx, dist) in enumerate(distances[:top_k], 1):
            row = {"rank": rank, "start_index": idx, "dtw_distance": dist}
            if time_column in dataframe.columns:
                row["start_time"] = dataframe[time_column].values[idx]
            results.append(row)

        return pd.DataFrame(results)

    @classmethod
    def template_match(
        cls,
        dataframe: pd.DataFrame,
        template: np.ndarray,
        value_column: str = "value_double",
        threshold: Optional[float] = None,
        normalize: bool = True,
        time_column: str = "systime",
    ) -> pd.DataFrame:
        """Find all occurrences of a template pattern in the time series.

        Args:
            dataframe: Input DataFrame.
            template: Reference pattern as numpy array.
            value_column: Column containing numeric values.
            threshold: Maximum distance to consider a match. None = adaptive.
            normalize: Whether to z-normalize before comparison.
            time_column: Column containing timestamps.

        Returns:
            DataFrame with start_index, distance, start_time, end_time.
        """
        series = dataframe[value_column].values.astype(float)
        m = len(template)
        t = template.astype(float).copy()

        if normalize and t.std() > 1e-10:
            t = (t - t.mean()) / t.std()
            # Normalize series subsequences via MASS
            dist_profile = cls._mass(series, t)
        else:
            dist_profile = cls._mass(series, t)

        if len(dist_profile) == 0:
            return pd.DataFrame(
                columns=["start_index", "distance", "start_time", "end_time"]
            )

        if threshold is None:
            threshold = float(np.mean(dist_profile) - 2 * np.std(dist_profile))
            threshold = max(threshold, 0)

        matches = np.where(dist_profile <= threshold)[0]
        results = []
        for idx in matches:
            row = {"start_index": int(idx), "distance": float(dist_profile[idx])}
            if time_column in dataframe.columns:
                times = dataframe[time_column].values
                row["start_time"] = times[idx]
                end_idx = min(idx + m - 1, len(times) - 1)
                row["end_time"] = times[end_idx]
            results.append(row)

        return pd.DataFrame(results)

    @classmethod
    def compute_distance_profile(
        cls,
        dataframe: pd.DataFrame,
        query: np.ndarray,
        value_column: str = "value_double",
        metric: str = "euclidean",
        normalize: bool = True,
    ) -> np.ndarray:
        """Compute distance from query to every subsequence of same length.

        Args:
            dataframe: Input DataFrame.
            query: Query subsequence.
            value_column: Column containing numeric values.
            metric: 'euclidean' (FFT-based) or 'dtw'.
            normalize: Whether to z-normalize.

        Returns:
            1D numpy array of distances.
        """
        series = dataframe[value_column].values.astype(float)
        q = query.astype(float).copy()
        m = len(q)

        if normalize and q.std() > 1e-10:
            q = (q - q.mean()) / q.std()

        if metric == "euclidean":
            return cls._mass(series, q)
        elif metric == "dtw":
            n = len(series)
            distances = np.zeros(n - m + 1)
            for i in range(n - m + 1):
                subseq = series[i : i + m].copy()
                if normalize and subseq.std() > 1e-10:
                    subseq = (subseq - subseq.mean()) / subseq.std()
                distances[i] = cls._dtw_distance(q, subseq)
            return distances
        else:
            raise ValueError(f"Unknown metric: {metric}. Use 'euclidean' or 'dtw'.")
