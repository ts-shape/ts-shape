import logging

import numpy as np
import pandas as pd

from ts_shape.utils.base import Base
from ts_shape.utils.coerce import to_numeric_checked
from ts_shape.utils.frames import as_datetime, require_columns

logger = logging.getLogger(__name__)


class PatternRecognition(Base):
    """Pattern Recognition for univariate timeseries.

    Discover motifs, discords, and template matches using Matrix Profile
    and Dynamic Time Warping approaches.

    Every method reads one signal: rows are put in time order (when the frame
    has ``time_column``), and a frame with several uuids needs ``uuid=`` to
    pick one. Positions in the results (``index_a``, ``start_index``, ...)
    count samples of that time-ordered signal. A window that contains a
    missing value is skipped.

    Methods:
    - discover_motifs: Find top-k recurring subsequence patterns.
    - discover_discords: Find top-k anomalous subsequences.
    - similarity_search: Find subsequences most similar to a query (DTW).
    - template_match: Find all occurrences of a reference template.
    - compute_distance_profile: Distance from query to every subsequence.
    """

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    @classmethod
    def _signal(
        cls,
        dataframe: pd.DataFrame,
        value_column: str,
        time_column: str,
        uuid: str | None,
        where: str,
    ) -> tuple[np.ndarray, pd.Series | None]:
        """Values (float64) and times of the selected signal, in time order.

        Raises:
            ColumnNotFoundError: If ``value_column`` is missing.
            ValueError: If the frame holds several uuids and ``uuid`` is not
                given, or ``uuid`` is not in the frame.
        """
        where = f"PatternRecognition.{where}"
        require_columns(dataframe, [value_column], where=where)
        df = dataframe
        if "uuid" in df.columns and not df.empty:
            if uuid is not None:
                cls._validate_uuid(df, uuid)
                df = df[df["uuid"] == uuid]
            else:
                present = sorted(str(u) for u in df["uuid"].dropna().unique())
                if len(present) > 1:
                    raise ValueError(
                        f"{where}: the frame holds {len(present)} signals; pass "
                        f"uuid=... to pick one. Available UUIDs: {present}"
                    )
        times: pd.Series | None = None
        if time_column in df.columns:
            parsed = as_datetime(df[time_column], where=where)
            # Stable time order; NaT sorts last.
            order = np.argsort(parsed.to_numpy(dtype="datetime64[ns]"), kind="stable")
            df = df.iloc[order]
            times = parsed.iloc[order].reset_index(drop=True)
        values = to_numeric_checked(df[value_column], what=f"column {value_column!r}")
        series = np.asarray(
            values.to_numpy(dtype="float64", na_value=np.nan), dtype="float64"
        )
        return series, times

    @staticmethod
    def _query(query: np.ndarray, name: str) -> np.ndarray:
        q = np.asarray(query, dtype="float64").copy()
        if q.ndim != 1 or len(q) == 0:
            raise ValueError(f"{name} must be a non-empty 1-D array")
        if np.isnan(q).any():
            raise ValueError(f"{name} contains NaN")
        return q

    @staticmethod
    def _valid_windows(series: np.ndarray, m: int) -> np.ndarray:
        """True for each window of length ``m`` without a missing value."""
        n = len(series)
        if n < m:
            return np.zeros(0, dtype=bool)
        missing = np.r_[0, np.cumsum(np.isnan(series))]
        result: np.ndarray = (missing[m:] - missing[: n - m + 1]) == 0
        return result

    @staticmethod
    def _frame(
        rows: list[dict[str, object]],
        schema: dict[str, str],
        times: pd.Series | None,
        time_fields: list[str],
    ) -> pd.DataFrame:
        """Result frame with typed columns; time fields only with times."""
        columns: dict[str, pd.Series] = {}
        for name, dtype in schema.items():
            columns[name] = pd.Series([r[name] for r in rows], dtype=dtype)
        if times is not None:
            for name in time_fields:
                columns[name] = pd.Series([r[name] for r in rows], dtype=times.dtype)
        return pd.DataFrame(columns)

    # ------------------------------------------------------------------
    # Distances
    # ------------------------------------------------------------------

    @classmethod
    def _mass(cls, series: np.ndarray, query: np.ndarray) -> np.ndarray:
        """MASS: compute z-normalized Euclidean distance profile via FFT.

        Args:
            series: The full time series.
            query: The query subsequence.

        Returns:
            1D array of distances for each position; NaN for a window with a
            missing value.
        """
        from scipy.fft import fft, ifft

        n = len(series)
        m = len(query)

        if n < m:
            return np.array([])

        valid = cls._valid_windows(series, m)
        filled = np.where(np.isnan(series), 0.0, series)

        # Pad query for FFT
        q_reversed = query[::-1]
        q_padded = np.zeros(n)
        q_padded[:m] = q_reversed

        # Sliding dot product via FFT
        QT = np.real(ifft(fft(filled) * fft(q_padded)))
        QT = QT[m - 1 :]

        # Rolling statistics
        s = pd.Series(filled)
        rolling_mean = s.rolling(window=m).mean().to_numpy()[m - 1 :]
        rolling_std = s.rolling(window=m).std(ddof=0).to_numpy()[m - 1 :]

        q_mean = query.mean()
        q_std = query.std(ddof=0)

        # Avoid division by zero
        rolling_std = np.where(rolling_std < 1e-10, 1e-10, rolling_std)
        q_std = max(q_std, 1e-10)

        dist_sq = (
            2 * m * (1 - (QT - m * q_mean * rolling_mean) / (m * q_std * rolling_std))
        )
        dist_sq = np.maximum(dist_sq, 0)
        dist: np.ndarray = np.where(valid, np.sqrt(dist_sq), np.nan)
        return dist

    @classmethod
    def _euclidean(cls, series: np.ndarray, query: np.ndarray) -> np.ndarray:
        """Plain (not normalized) Euclidean distance profile; NaN windows NaN."""
        n = len(series)
        m = len(query)
        if n < m:
            return np.array([])
        valid = cls._valid_windows(series, m)
        filled = np.where(np.isnan(series), 0.0, series)
        windows = np.lib.stride_tricks.sliding_window_view(filled, m)
        dist: np.ndarray = np.sqrt(((windows - query) ** 2).sum(axis=1))
        return np.where(valid, dist, np.nan)

    @classmethod
    def _profile(
        cls, series: np.ndarray, query: np.ndarray, normalize: bool
    ) -> np.ndarray:
        if normalize:
            return cls._mass(series, query)
        return cls._euclidean(series, query)

    #: STOMP refreshes its running sliding dot products with one exact FFT
    #: every this many rows, so their rounding cannot accumulate.
    _STOMP_REFRESH = 64

    @classmethod
    def _compute_matrix_profile(
        cls,
        series: np.ndarray,
        window_size: int,
        exclusion_zone: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the Matrix Profile and Matrix Profile Index (STOMP).

        Row ``i`` of the distance matrix is the MASS distance profile of the
        window starting at ``i`` (see :meth:`_mass`). Its sliding dot products
        are updated from row ``i - 1`` in O(n) (STOMP, Zhu et al. 2016:
        ``QT[i, j] = QT[i-1, j-1] - T[i-1] T[j-1] + T[i+m-1] T[j+m-1]``)
        instead of one FFT per row, and recomputed exactly by FFT every
        :attr:`_STOMP_REFRESH` rows. O(n^2) instead of O(n^2 log n). The
        distances equal the per-row MASS ones to rounding: within 1e-9
        relative unless the series' level is many orders of magnitude above a
        window's spread, where MASS's own rounding error (on the raw values)
        is larger and this, working on the series centred on its mean, is the
        more accurate of the two.

        Args:
            series: 1D time series array.
            window_size: Subsequence length.
            exclusion_zone: Number of indices to exclude around a match.

        Returns:
            Tuple of (matrix_profile, matrix_profile_index). A window with a
            missing value, or without any comparable window, has profile
            ``inf`` and index ``-1``.
        """
        from scipy.fft import fft, ifft

        n = len(series)
        m = window_size
        profile_len = max(n - m + 1, 0)
        mp = np.full(profile_len, np.inf)
        mpi = np.full(profile_len, -1, dtype=int)
        valid = cls._valid_windows(series, m)
        if profile_len == 0 or not valid.any():
            return mp, mpi

        # Everything _mass derives from the series alone, computed once. The
        # series is centred on its mean first: z-normalized distances do not
        # change, and the dot products then carry no offset**2 term to cancel
        # (MASS on the raw series loses digits when the level is large
        # compared with a window's spread).
        missing = np.isnan(series)
        filled = np.where(missing, 0.0, series - float(np.mean(series[~missing])))
        fft_series = fft(filled)
        s = pd.Series(filled)
        rolling_mean = s.rolling(window=m).mean().to_numpy()[m - 1 :]
        rolling_std = s.rolling(window=m).std(ddof=0).to_numpy()[m - 1 :]
        rolling_std = np.where(rolling_std < 1e-10, 1e-10, rolling_std)
        windows = np.lib.stride_tricks.sliding_window_view(filled, m)
        q_means = windows.mean(axis=1)
        q_stds = np.maximum(windows.std(axis=1), 1e-10)

        def exact_qt(i: int) -> np.ndarray:
            q_padded = np.zeros(n)
            q_padded[:m] = filled[i : i + m][::-1]
            qt: np.ndarray = np.real(ifft(fft_series * fft(q_padded)))
            return qt[m - 1 :]

        first_row = exact_qt(0)
        qt = first_row
        head = filled[: profile_len - 1]
        tail = filled[m : m + profile_len - 1]
        for i in range(profile_len):
            if i and i % cls._STOMP_REFRESH == 0:
                qt = exact_qt(i)
            elif i:
                nxt = np.empty(profile_len)
                nxt[1:] = qt[:-1] - filled[i - 1] * head + filled[i + m - 1] * tail
                nxt[0] = first_row[i]
                qt = nxt
            if not valid[i]:
                continue
            q_mean = q_means[i]
            q_std = q_stds[i]
            dist_sq = (
                2
                * m
                * (1 - (qt - m * q_mean * rolling_mean) / (m * q_std * rolling_std))
            )
            dist_sq = np.maximum(dist_sq, 0)
            dist_profile = np.where(valid, np.sqrt(dist_sq), np.inf)

            # Apply exclusion zone
            ez_start = max(0, i - exclusion_zone)
            ez_end = min(profile_len, i + exclusion_zone + 1)
            dist_profile[ez_start:ez_end] = np.inf

            min_idx = int(np.argmin(dist_profile))
            min_val = dist_profile[min_idx]
            if np.isfinite(min_val):
                mp[i] = min_val
                mpi[i] = min_idx

        return mp, mpi

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def discover_motifs(
        cls,
        dataframe: pd.DataFrame,
        value_column: str = "value_double",
        window_size: int = 50,
        top_k: int = 5,
        exclusion_zone: int | None = None,
        time_column: str = "systime",
        *,
        uuid: str | None = None,
    ) -> pd.DataFrame:
        """Find the top-k recurring subsequence patterns (motifs).

        Args:
            dataframe: Input DataFrame.
            value_column: Column containing numeric values.
            window_size: Length of subsequences to compare.
            top_k: Number of motif pairs to return.
            exclusion_zone: Indices to exclude around matches. Defaults to
                window_size // 2. Also applied between motifs: no member of a
                returned pair starts within ``exclusion_zone`` of a member of
                a better-ranked pair, so later motifs are not the first one
                shifted by a sample or two.
            time_column: Column containing timestamps.
            uuid: Signal to analyze when the frame holds several uuids.

        Returns:
            DataFrame with motif_rank, index_a, index_b, distance, time_a, time_b
                (the time columns only when the frame has ``time_column``; they
                keep its timezone). Empty, with these columns, when the signal
                is shorter than ``window_size``.

        Raises:
            ColumnNotFoundError: If ``value_column`` is missing.
            ValueError: If the frame holds several uuids and ``uuid`` is not
                given.
        """
        series, times = cls._signal(
            dataframe, value_column, time_column, uuid, "discover_motifs"
        )
        if exclusion_zone is None:
            exclusion_zone = window_size // 2

        mp, mpi = cls._compute_matrix_profile(series, window_size, exclusion_zone)

        # Find top-k motifs (smallest MP values)
        sorted_indices = np.argsort(mp, kind="stable")
        results: list[dict[str, object]] = []
        # Window starts of the motifs already returned (both members): a
        # later pair with a member within exclusion_zone of one is a trivial
        # match (the same motif shifted by a few samples) and is skipped.
        used_starts: list[int] = []

        for idx in sorted_indices:
            if len(results) >= top_k:
                break
            partner = int(mpi[idx])
            if partner < 0 or not np.isfinite(mp[idx]):
                continue
            pair = (min(int(idx), partner), max(int(idx), partner))
            if abs(int(idx) - partner) <= exclusion_zone:
                continue
            if any(abs(p - u) <= exclusion_zone for p in pair for u in used_starts):
                continue
            used_starts.extend(pair)

            row: dict[str, object] = {
                "motif_rank": len(results) + 1,
                "index_a": pair[0],
                "index_b": pair[1],
                "distance": float(mp[idx]),
            }
            if times is not None:
                row["time_a"] = times.iloc[pair[0]]
                row["time_b"] = times.iloc[pair[1]]

            results.append(row)

        return cls._frame(
            results,
            {
                "motif_rank": "int64",
                "index_a": "int64",
                "index_b": "int64",
                "distance": "float64",
            },
            times,
            ["time_a", "time_b"],
        )

    @classmethod
    def discover_discords(
        cls,
        dataframe: pd.DataFrame,
        value_column: str = "value_double",
        window_size: int = 50,
        top_k: int = 5,
        exclusion_zone: int | None = None,
        time_column: str = "systime",
        *,
        uuid: str | None = None,
    ) -> pd.DataFrame:
        """Find the top-k anomalous subsequences (discords).

        Args:
            dataframe: Input DataFrame.
            value_column: Column containing numeric values.
            window_size: Length of subsequences to compare.
            top_k: Number of discords to return.
            exclusion_zone: Indices to exclude around matches. Defaults to window_size // 2.
            time_column: Column containing timestamps.
            uuid: Signal to analyze when the frame holds several uuids.

        Returns:
            DataFrame with discord_rank, start_index, distance, start_time (the
                time column only when the frame has ``time_column``). Windows
                with a missing value are never discords. Empty, with these
                columns, when the signal is shorter than ``window_size``.

        Raises:
            ColumnNotFoundError: If ``value_column`` is missing.
            ValueError: If the frame holds several uuids and ``uuid`` is not
                given.
        """
        series, times = cls._signal(
            dataframe, value_column, time_column, uuid, "discover_discords"
        )
        if exclusion_zone is None:
            exclusion_zone = window_size // 2

        mp, _ = cls._compute_matrix_profile(series, window_size, exclusion_zone)

        # Find top-k discords (largest MP values); skipped windows are inf.
        finite = np.where(np.isfinite(mp), mp, -np.inf)
        sorted_indices = np.argsort(-finite, kind="stable")
        results: list[dict[str, object]] = []
        used_zones: list[int] = []

        for idx in sorted_indices:
            if len(results) >= top_k:
                break
            if not np.isfinite(mp[idx]):
                continue
            # Check exclusion zone against already selected discords
            if any(abs(int(idx) - u) <= exclusion_zone for u in used_zones):
                continue
            used_zones.append(int(idx))

            row: dict[str, object] = {
                "discord_rank": len(results) + 1,
                "start_index": int(idx),
                "distance": float(mp[idx]),
            }
            if times is not None:
                row["start_time"] = times.iloc[int(idx)]

            results.append(row)

        return cls._frame(
            results,
            {"discord_rank": "int64", "start_index": "int64", "distance": "float64"},
            times,
            ["start_time"],
        )

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

    @staticmethod
    def _znorm(values: np.ndarray) -> np.ndarray:
        std = float(values.std())
        if std > 1e-10:
            scaled: np.ndarray = (values - float(values.mean())) / std
            return scaled
        return values

    #: Windows per vectorized DTW batch (bounds the (len + 1) x batch arrays).
    _DTW_BATCH = 8192

    @classmethod
    def _dtw_squared(cls, query: np.ndarray, windows: np.ndarray) -> np.ndarray:
        """Squared DTW distance of ``query`` to every row of ``windows``.

        The dynamic program of :meth:`_dtw_distance` run for all windows at
        once (one vector operation per cell instead of one per cell and
        window). Same operations in the same order, so the distances are
        bit-identical: ``np.float_power`` is the C ``pow`` that the scalar
        ``(a - b) ** 2`` calls.
        """
        n = len(query)
        count, m = windows.shape
        out = np.empty(count)
        for a in range(0, count, cls._DTW_BATCH):
            cand = np.ascontiguousarray(windows[a : a + cls._DTW_BATCH].T)
            width = cand.shape[1]
            prev = np.full((m + 1, width), np.inf)
            prev[0] = 0.0
            for i in range(1, n + 1):
                cur = np.empty((m + 1, width))
                cur[0] = np.inf
                q = query[i - 1]
                for j in range(1, m + 1):
                    cost = np.float_power(q - cand[j - 1], 2.0)
                    best = np.minimum(np.minimum(prev[j], cur[j - 1]), prev[j - 1])
                    cur[j] = cost + best
                prev = cur
            out[a : a + width] = prev[m]
        return out

    @classmethod
    def _dtw_windows(
        cls, series: np.ndarray, m: int, normalize: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        """Start indices of the windows without a missing value, and those
        windows (z-normalized as :meth:`_znorm` does when ``normalize``)."""
        valid = cls._valid_windows(series, m)
        starts = np.flatnonzero(valid)
        windows = np.lib.stride_tricks.sliding_window_view(series, m)[starts]
        if normalize and len(starts):
            std = windows.std(axis=1)
            mean = windows.mean(axis=1)
            scale = std > 1e-10
            safe = np.where(scale, std, 1.0)
            scaled = (windows - mean[:, None]) / safe[:, None]
            windows = np.where(scale[:, None], scaled, windows)
        return starts, windows

    @classmethod
    def _dtw_profile(
        cls, series: np.ndarray, query: np.ndarray, normalize: bool
    ) -> np.ndarray:
        """DTW distance of ``query`` to every window; NaN windows NaN."""
        m = len(query)
        n = len(series)
        if n < m:
            return np.array([])
        q = cls._znorm(query) if normalize else query
        starts, windows = cls._dtw_windows(series, m, normalize)
        distances = np.full(n - m + 1, np.nan)
        distances[starts] = np.sqrt(cls._dtw_squared(q, windows))
        return distances

    @staticmethod
    def _dtw_lower_bound(query: np.ndarray, windows: np.ndarray) -> np.ndarray:
        """Lower bounds of the squared DTW distance of ``query`` to each window.

        The largest of LB_Kim (the first and last cells lie on every warping
        path) and LB_Keogh in both directions (every sample is matched to at
        least one sample of the other sequence, so it costs at least its
        squared distance to the other sequence's range; without a warping
        band the envelope is that range).
        """
        lo_q, hi_q = float(query.min()), float(query.max())
        below = np.maximum(lo_q - windows, 0.0)
        above = np.maximum(windows - hi_q, 0.0)
        keogh = (below * below + above * above).sum(axis=1)
        lo_w, hi_w = windows.min(axis=1), windows.max(axis=1)
        below = np.maximum(lo_w[:, None] - query[None, :], 0.0)
        above = np.maximum(query[None, :] - hi_w[:, None], 0.0)
        keogh_rev = (below * below + above * above).sum(axis=1)
        first = (query[0] - windows[:, 0]) ** 2
        if len(query) > 1 and windows.shape[1] > 1:
            kim = first + (query[-1] - windows[:, -1]) ** 2
        else:
            kim = first
        bound: np.ndarray = np.maximum(np.maximum(keogh, keogh_rev), kim)
        return bound

    @classmethod
    def _dtw_top_k(
        cls, series: np.ndarray, query: np.ndarray, normalize: bool, top_k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """The ``top_k`` windows closest to ``query`` under DTW, as (start
        indices, distances) ranked by distance, ties by start index.

        Exactly the first ``top_k`` of the full :meth:`_dtw_profile` ranking:
        windows are visited in order of a lower bound of their distance
        (:meth:`_dtw_lower_bound`), and the search stops once the bound
        exceeds the ``top_k``-th best distance found so far (with a margin
        far above rounding), so no skipped window can rank or tie.
        """
        m = len(query)
        if len(series) < m:
            return np.zeros(0, dtype=np.int64), np.zeros(0)
        q = cls._znorm(query) if normalize else query
        starts, windows = cls._dtw_windows(series, m, normalize)
        if not 0 < top_k < len(starts):
            # Every window is needed (or none): rank the full profile.
            dist = np.sqrt(cls._dtw_squared(q, windows))
            rank = np.argsort(dist, kind="stable")[:top_k]
            return starts[rank], dist[rank]
        bound = cls._dtw_lower_bound(q, windows)
        order = np.argsort(bound, kind="stable")
        done = np.zeros(0, dtype=np.int64)
        squared = np.zeros(0)
        batch = max(4 * top_k, 256)
        pos = 0
        while pos < len(order):
            if len(squared) >= top_k:
                kth = float(np.partition(squared, top_k - 1)[top_k - 1])
                limit = kth * (1 + 1e-9) + 1e-300
                stop = int(np.searchsorted(bound[order], limit, side="right"))
                if stop <= pos:
                    break
                take = order[pos : min(stop, pos + batch)]
            else:
                take = order[pos : pos + batch]
            done = np.concatenate([done, take])
            squared = np.concatenate([squared, cls._dtw_squared(q, windows[take])])
            pos += len(take)
            batch *= 2
        # Rank as the full profile does: by distance, ties by start index.
        idx = starts[done]
        dist = np.sqrt(squared)
        rank = np.lexsort((idx, dist))[:top_k]
        return idx[rank], dist[rank]

    @classmethod
    def similarity_search(
        cls,
        dataframe: pd.DataFrame,
        query: np.ndarray,
        value_column: str = "value_double",
        top_k: int = 5,
        normalize: bool = True,
        time_column: str = "systime",
        *,
        uuid: str | None = None,
    ) -> pd.DataFrame:
        """Find the top-k most similar subsequences to a query using DTW.

        Args:
            dataframe: Input DataFrame.
            query: Query pattern as numpy array.
            value_column: Column containing numeric values.
            top_k: Number of matches to return.
            normalize: Whether to z-normalize before comparison.
            time_column: Column containing timestamps.
            uuid: Signal to analyze when the frame holds several uuids.

        Returns:
            DataFrame with rank, start_index, dtw_distance, start_time (the
                time column only when the frame has ``time_column``). Empty,
                with these columns, when the signal is shorter than the query.

        Raises:
            ColumnNotFoundError: If ``value_column`` is missing.
            ValueError: If the frame holds several uuids and ``uuid`` is not
                given, or ``query`` is empty or contains NaN.
        """
        series, times = cls._signal(
            dataframe, value_column, time_column, uuid, "similarity_search"
        )
        q = cls._query(query, "query")
        ranked, distances = cls._dtw_top_k(series, q, normalize, top_k)

        results: list[dict[str, object]] = []
        for rank, (idx, dist) in enumerate(zip(ranked, distances, strict=True), 1):
            row: dict[str, object] = {
                "rank": rank,
                "start_index": int(idx),
                "dtw_distance": float(dist),
            }
            if times is not None:
                row["start_time"] = times.iloc[int(idx)]
            results.append(row)

        return cls._frame(
            results,
            {"rank": "int64", "start_index": "int64", "dtw_distance": "float64"},
            times,
            ["start_time"],
        )

    @classmethod
    def template_match(
        cls,
        dataframe: pd.DataFrame,
        template: np.ndarray,
        value_column: str = "value_double",
        threshold: float | None = None,
        normalize: bool = True,
        time_column: str = "systime",
        *,
        uuid: str | None = None,
    ) -> pd.DataFrame:
        """Find all occurrences of a template pattern in the time series.

        Args:
            dataframe: Input DataFrame.
            template: Reference pattern as numpy array.
            value_column: Column containing numeric values.
            threshold: Maximum distance to consider a match. None = adaptive
                (mean minus two standard deviations of the distance profile).
            normalize: Whether to z-normalize before comparison. With
                ``False`` the distance is the plain Euclidean distance, so the
                level and scale of the template matter.
            time_column: Column containing timestamps.
            uuid: Signal to analyze when the frame holds several uuids.

        Returns:
            DataFrame with start_index, distance, start_time, end_time (the
                time columns only when the frame has ``time_column``). Empty,
                with these columns, when nothing matches.

        Raises:
            ColumnNotFoundError: If ``value_column`` is missing.
            ValueError: If the frame holds several uuids and ``uuid`` is not
                given, or ``template`` is empty or contains NaN.
        """
        series, times = cls._signal(
            dataframe, value_column, time_column, uuid, "template_match"
        )
        t = cls._query(template, "template")
        m = len(t)
        dist_profile = cls._profile(series, t, normalize)
        finite = dist_profile[~np.isnan(dist_profile)]

        results: list[dict[str, object]] = []
        if len(finite):
            if threshold is None:
                threshold = float(np.mean(finite) - 2 * np.std(finite))
                threshold = max(threshold, 0)

            matches = np.flatnonzero(
                ~np.isnan(dist_profile)
                & (np.nan_to_num(dist_profile, nan=np.inf) <= threshold)
            )
            for idx in matches:
                row: dict[str, object] = {
                    "start_index": int(idx),
                    "distance": float(dist_profile[idx]),
                }
                if times is not None:
                    row["start_time"] = times.iloc[int(idx)]
                    end_idx = min(int(idx) + m - 1, len(times) - 1)
                    row["end_time"] = times.iloc[end_idx]
                results.append(row)

        return cls._frame(
            results,
            {"start_index": "int64", "distance": "float64"},
            times,
            ["start_time", "end_time"],
        )

    @classmethod
    def compute_distance_profile(
        cls,
        dataframe: pd.DataFrame,
        query: np.ndarray,
        value_column: str = "value_double",
        metric: str = "euclidean",
        normalize: bool = True,
        *,
        time_column: str = "systime",
        uuid: str | None = None,
    ) -> np.ndarray:
        """Compute distance from query to every subsequence of same length.

        Args:
            dataframe: Input DataFrame.
            query: Query subsequence.
            value_column: Column containing numeric values.
            metric: 'euclidean' (FFT-based) or 'dtw'.
            normalize: Whether to z-normalize. With ``False`` the Euclidean
                distance is the plain distance of the raw values.
            time_column: Rows are put in the order of this column (when the
                frame has it).
            uuid: Signal to analyze when the frame holds several uuids.

        Returns:
            1D numpy array of distances, one per window start; NaN for a
                window with a missing value. Empty when the signal is shorter
                than the query.

        Raises:
            ColumnNotFoundError: If ``value_column`` is missing.
            ValueError: For an unknown ``metric``, several uuids without
                ``uuid``, or a query that is empty or contains NaN.
        """
        if metric not in ("euclidean", "dtw"):
            raise ValueError(f"Unknown metric: {metric}. Use 'euclidean' or 'dtw'.")
        series, _ = cls._signal(
            dataframe, value_column, time_column, uuid, "compute_distance_profile"
        )
        q = cls._query(query, "query")

        if metric == "euclidean":
            if normalize:
                q = cls._znorm(q)
            return cls._profile(series, q, normalize)
        return cls._dtw_profile(series, q, normalize)
