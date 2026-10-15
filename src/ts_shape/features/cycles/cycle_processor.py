"""Assign sample data to cycles and compute per-cycle results.

Boundary rule (used by every method): a sample belongs to a cycle when
``cycle_start <= systime < cycle_end``. The interval is half-open, so with
back-to-back cycles a sample at the shared timestamp belongs to the later
cycle only, and every sample is counted once. A cycle without an end
(``cycle_end`` NaT) gets no samples. Samples with a missing timestamp belong
to no cycle. Samples are matched on time only: pass the values of the machine
the cycles belong to.
"""

import logging
import warnings

import numpy as np
import pandas as pd

from ts_shape.errors import ColumnNotFoundError, DataQualityWarning
from ts_shape.utils.base import Base
from ts_shape.utils.deprecation import caller_stacklevel, renamed_kwargs

logger = logging.getLogger(__name__)


class CycleDataProcessor(Base):
    """
    A class to process cycle-based data and values with optimized performance.
    Samples are matched to cycles with a sorted search (no per-cycle scans).
    """

    @renamed_kwargs(cycle_uuid_col="cycle_uuid_column", systime_col="time_column")
    def __init__(
        self,
        cycles_df: pd.DataFrame,
        values_df: pd.DataFrame,
        cycle_uuid_column: str = "cycle_uuid",
        time_column: str = "systime",
    ) -> None:
        """
        Initializes the CycleDataProcessor with cycles and values DataFrames.

        Args:
            cycles_df: DataFrame containing columns 'cycle_start', 'cycle_end', and 'cycle_uuid'.
            values_df: DataFrame containing the values and timestamps in the 'systime' column.
            cycle_uuid_column: Name of the column representing cycle UUIDs.
                ``cycle_uuid_col`` is the deprecated name.
            time_column: Name of the column representing the timestamps for the
                values. ``systime_col`` is the deprecated name.

        Raises:
            ColumnNotFoundError: If a cycle column or the time column is missing.
            TypeError: If one of cycles and values is tz-aware and the other naive.
        """
        super().__init__(values_df, time_column=time_column)
        for column in ("cycle_start", "cycle_end", cycle_uuid_column):
            if column not in cycles_df.columns:
                raise ColumnNotFoundError(
                    f"cycles_df: column {column!r} not found. "
                    f"Columns: {list(cycles_df.columns)}"
                )
        if time_column not in values_df.columns:
            raise ColumnNotFoundError(
                f"values_df: column {time_column!r} not found. "
                f"Columns: {list(values_df.columns)}"
            )
        self.values_df = values_df.copy()  # Initialize self.values_df explicitly
        self.cycles_df = cycles_df.copy()
        self.cycle_uuid_column = cycle_uuid_column
        self.time_column = time_column
        # Old attribute names, kept for code that reads them.
        self.cycle_uuid_col = cycle_uuid_column
        self.systime_col = time_column

        # Ensure proper datetime format
        self.cycles_df["cycle_start"] = pd.to_datetime(self.cycles_df["cycle_start"])
        self.cycles_df["cycle_end"] = pd.to_datetime(self.cycles_df["cycle_end"])
        self.values_df[time_column] = pd.to_datetime(self.values_df[time_column])

        aware = [
            s.dt.tz is not None
            for s in (
                self.cycles_df["cycle_start"],
                self.cycles_df["cycle_end"],
                self.values_df[time_column],
            )
        ]
        if len(set(aware)) > 1:
            raise TypeError(
                "CycleDataProcessor: cycle times and value times must both be "
                "tz-aware or both naive; localize or convert one of them."
            )

        self._pairs_cache: tuple[np.ndarray, np.ndarray] | None = None
        logger.info("CycleDataProcessor initialized with cycles and values DataFrames.")

    # ------------------------------------------------------------------
    # Core assignment
    # ------------------------------------------------------------------

    @staticmethod
    def _as_ns(times: pd.Series) -> np.ndarray:
        """Datetimes as int64 nanoseconds (UTC for tz-aware input); NaT -> min."""
        if times.dt.tz is not None:
            times = times.dt.tz_convert("UTC").dt.tz_localize(None)
        return np.asarray(times.dt.as_unit("ns").to_numpy(dtype="datetime64[ns]")).view(
            "int64"
        )

    def _pairs(self) -> tuple[np.ndarray, np.ndarray]:
        """(value positions, cycle positions) of every sample-in-cycle match.

        Sorted by sample time, then by cycle start; positions index
        ``values_df`` and ``cycles_df`` (``iloc``).
        """
        if self._pairs_cache is not None:
            return self._pairs_cache
        times = self.values_df[self.time_column]
        valid_t = np.flatnonzero(times.notna().to_numpy())
        t_ns = self._as_ns(times)[valid_t]
        t_order = np.argsort(t_ns, kind="stable")
        sorted_t = t_ns[t_order]
        sorted_pos = valid_t[t_order]

        starts = self.cycles_df["cycle_start"]
        ends = self.cycles_df["cycle_end"]
        valid_c = np.flatnonzero((starts.notna() & ends.notna()).to_numpy())
        c_start = self._as_ns(starts)[valid_c]
        c_end = self._as_ns(ends)[valid_c]
        lo = np.searchsorted(sorted_t, c_start, side="left")
        hi = np.searchsorted(sorted_t, c_end, side="left")
        counts = np.clip(hi - lo, 0, None)
        total = int(counts.sum())
        if total == 0:
            empty = np.empty(0, dtype=np.int64)
            self._pairs_cache = (empty, empty)
            return self._pairs_cache
        cycle_idx = np.repeat(np.arange(len(valid_c)), counts)
        offsets = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
        sorted_idx = np.repeat(lo, counts) + offsets
        order = np.lexsort((c_start[cycle_idx], sorted_idx))
        value_pos = sorted_pos[sorted_idx[order]]
        cycle_pos = valid_c[cycle_idx[order]]
        self._pairs_cache = (value_pos, cycle_pos)
        return self._pairs_cache

    def _cycle_order(self) -> np.ndarray:
        """Cycle positions in chronological order (by start, then input order)."""
        return np.argsort(self.cycles_df["cycle_start"].to_numpy(), kind="stable")

    # ------------------------------------------------------------------
    # Split / merge / group
    # ------------------------------------------------------------------

    def split_by_cycle(self) -> dict[str, pd.DataFrame]:
        """
        Splits the values DataFrame by cycles defined in the cycles DataFrame.

        Returns:
            Dictionary where keys are cycle_uuids and values are DataFrames with the corresponding cycle data.
                Keys are in chronological cycle order; cycles without samples are
                left out. Rows keep their ``values_df`` index and are sorted by time.
        """
        merged = self.merge_dataframes_by_cycle()
        if merged.empty:
            logger.warning("No cycles or values available for splitting.")
            return {}
        result = {
            cycle_uuid: group.drop(columns=[self.cycle_uuid_column])
            for cycle_uuid, group in merged.groupby(
                self.cycle_uuid_column, sort=False, dropna=True
            )
        }
        logger.info(f"Split {len(result)} cycles.")
        return result

    def merge_dataframes_by_cycle(self) -> pd.DataFrame:
        """
        Merges the values DataFrame with the cycles DataFrame based on the cycle time intervals.

        Returns:
            The samples that fall in a cycle, with an added ``cycle_uuid`` column,
                sorted by time (original index kept). Samples in no cycle are
                dropped. A sample inside two overlapping cycles appears once per
                cycle. No match: an empty frame with the same columns.
        """
        value_pos, cycle_pos = self._pairs()
        result = self.values_df.iloc[value_pos].copy()
        result[self.cycle_uuid_column] = (
            self.cycles_df[self.cycle_uuid_column].to_numpy()[cycle_pos]
            if len(cycle_pos)
            else pd.Series(dtype=object)
        )
        dropped = int(self.values_df[self.time_column].notna().sum()) - len(
            np.unique(value_pos)
        )
        if dropped > 0:
            logger.info(f"Dropping {dropped} rows not assigned to any cycle.")
        logger.info(
            f"Merged DataFrame contains {len(result)} records across "
            f"{result[self.cycle_uuid_column].nunique()} cycles."
        )
        return result

    def group_by_cycle_uuid(
        self, data: pd.DataFrame | None = None
    ) -> list[pd.DataFrame]:
        """
        Group the DataFrame by the cycle_uuid column, resulting in a list of DataFrames, each containing data for one cycle.

        Args:
            data: DataFrame containing the data to be grouped by cycle_uuid. If None,
                uses ``merge_dataframes_by_cycle()``.

        Returns:
            List of DataFrames, one per cycle uuid, in order of first appearance.
        """
        if data is None:
            data = self.merge_dataframes_by_cycle()

        if self.cycle_uuid_column not in data.columns:
            logger.warning(
                f"Column '{self.cycle_uuid_column}' not found in data. Cannot group."
            )
            return []

        grouped_dataframes = [
            group for _, group in data.groupby(self.cycle_uuid_column, sort=False)
        ]
        logger.info(f"Grouped data into {len(grouped_dataframes)} cycle UUID groups.")
        return grouped_dataframes

    def split_dataframes_by_group(
        self, dfs: list[pd.DataFrame], column: str
    ) -> list[pd.DataFrame]:
        """
        Splits a list of DataFrames by groups based on a specified column.
        This function performs a groupby operation on each DataFrame in the list and then flattens the result.

        Args:
            dfs: List of DataFrames to be split.
            column: Column name to group by.

        Returns:
            List of DataFrames, each corresponding to a group in the original DataFrames.
                Rows whose ``column`` is missing are left out.
        """
        split_dfs = []
        for df in dfs:
            if column not in df.columns:
                logger.warning(f"Column '{column}' not found in DataFrame. Skipping.")
                continue
            groups = df.groupby(column)
            for _, group in groups:
                split_dfs.append(group)

        logger.info(
            f"Split data into {len(split_dfs)} groups based on column '{column}'."
        )
        return split_dfs

    def _filter_by_time_range(
        self, start_time: pd.Timestamp, end_time: pd.Timestamp
    ) -> pd.DataFrame:
        """Rows with ``start_time <= systime < end_time`` (the cycle rule)."""
        mask = (self.values_df[self.time_column] >= start_time) & (
            self.values_df[self.time_column] < end_time
        )
        return self.values_df[mask]

    # ------------------------------------------------------------------
    # Per-cycle results
    # ------------------------------------------------------------------

    def _per_cycle_metric(self, metric: str) -> pd.DataFrame:
        """count/mean/std of ``metric`` per cycle position (non-missing values)."""
        if metric not in self.values_df.columns:
            raise ColumnNotFoundError(
                f"values_df: column {metric!r} not found. "
                f"Columns: {list(self.values_df.columns)}"
            )
        value_pos, cycle_pos = self._pairs()
        values = pd.to_numeric(
            self.values_df[metric].iloc[value_pos], errors="coerce"
        ).astype("float64")
        frame = pd.DataFrame({"cycle": cycle_pos, "v": values.to_numpy()}).dropna()
        return frame.groupby("cycle")["v"].agg(["count", "mean", "std"])

    def compute_cycle_statistics(self) -> pd.DataFrame:
        """
        Compute statistics for each cycle.

        Returns:
            One row per cycle in ``cycles_df`` order: ``cycle_uuid``,
                ``cycle_start``, ``cycle_end``, ``duration_seconds``,
                ``value_count`` (samples in the cycle), ``unique_uuids``,
                ``mean_value_double`` / ``std_value_double`` (when ``values_df``
                has ``value_double``), and the canonical ``start``, ``end``,
                ``sample_count``. An open cycle has NaN duration and 0 samples.
        """
        has_double = "value_double" in self.values_df.columns
        n = len(self.cycles_df)
        value_pos, cycle_pos = self._pairs()
        counts = np.bincount(cycle_pos, minlength=n) if n else np.zeros(0, int)
        if "uuid" in self.values_df.columns and len(value_pos):
            uuids = (
                pd.DataFrame(
                    {
                        "cycle": cycle_pos,
                        "uuid": self.values_df["uuid"].iloc[value_pos].to_numpy(),
                    }
                )
                .groupby("cycle")["uuid"]
                .nunique()
                .reindex(range(n), fill_value=0)
                .to_numpy()
            )
        else:
            uuids = np.zeros(n, dtype=int)

        start = self.cycles_df["cycle_start"].reset_index(drop=True)
        end = self.cycles_df["cycle_end"].reset_index(drop=True)
        result = pd.DataFrame(
            {
                self.cycle_uuid_column: self.cycles_df[
                    self.cycle_uuid_column
                ].to_numpy(),
                "cycle_start": start,
                "cycle_end": end,
                "duration_seconds": (end - start).dt.total_seconds(),
                "value_count": counts.astype("int64"),
                "unique_uuids": np.asarray(uuids, dtype="int64"),
            }
        )
        if has_double:
            agg = self._per_cycle_metric("value_double").reindex(range(n))
            result["mean_value_double"] = agg["mean"].to_numpy()
            result["std_value_double"] = agg["std"].to_numpy()
        result["start"] = result["cycle_start"]
        result["end"] = result["cycle_end"]
        result["sample_count"] = result["value_count"]
        logger.info(f"Computed statistics for {len(result)} cycles.")
        return result

    def _comparison_schema(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                self.cycle_uuid_column: pd.Series(dtype=object),
                "is_reference": pd.Series(dtype=bool),
                "mean_value": pd.Series(dtype="float64"),
                "std_value": pd.Series(dtype="float64"),
                "deviation_from_ref": pd.Series(dtype="float64"),
                "deviation_pct": pd.Series(dtype="float64"),
                "variability_ratio": pd.Series(dtype="float64"),
            }
        )

    @renamed_kwargs(metric="value_column")
    def compare_cycles(
        self, reference_cycle_uuid: str, value_column: str = "value_double"
    ) -> pd.DataFrame:
        """
        Compare all cycles against a reference cycle.

        Args:
            reference_cycle_uuid: UUID of the reference cycle
            value_column: Column holding the values to compare. ``metric`` is the
                deprecated name.

        Returns:
            One row per cycle with data for ``value_column`` (chronological order):
                ``cycle_uuid``, ``is_reference``, ``mean_value``, ``std_value``,
                ``deviation_from_ref`` (mean minus reference mean),
                ``deviation_pct`` (relative to the reference mean, NaN when it
                is 0) and ``variability_ratio`` (std over reference std, NaN
                when that is 0 or undefined). Empty (full columns) with a
                ``DataQualityWarning`` when the reference cycle is unknown or
                has no data.

        Raises:
            ColumnNotFoundError: If ``value_column`` is not a column of ``values_df``.
        """
        uuids = self.cycles_df[self.cycle_uuid_column].astype(str).to_numpy()
        stats = self._per_cycle_metric(value_column)
        ref_positions = np.flatnonzero(uuids == str(reference_cycle_uuid))
        if len(ref_positions) == 0:
            warnings.warn(
                f"compare_cycles: reference cycle {reference_cycle_uuid!r} not found.",
                DataQualityWarning,
                stacklevel=caller_stacklevel(),
            )
            return self._comparison_schema()
        ref_pos = int(ref_positions[0])
        if ref_pos not in stats.index:
            warnings.warn(
                f"compare_cycles: reference cycle {reference_cycle_uuid!r} has "
                f"no {value_column!r} data.",
                DataQualityWarning,
                stacklevel=caller_stacklevel(),
            )
            return self._comparison_schema()
        ref_mean = float(stats.loc[ref_pos, "mean"])
        ref_std = float(stats.loc[ref_pos, "std"])

        order = [int(p) for p in self._cycle_order() if int(p) in stats.index]
        sub = stats.loc[order]
        mean = sub["mean"].to_numpy(dtype=float)
        std = sub["std"].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            pct = (
                (mean - ref_mean) / ref_mean * 100
                if ref_mean != 0
                else np.full(len(mean), np.nan)
            )
            ratio = (
                std / ref_std
                if ref_std != 0 and not np.isnan(ref_std)
                else np.full(len(std), np.nan)
            )
        result = pd.DataFrame(
            {
                self.cycle_uuid_column: self.cycles_df[self.cycle_uuid_column]
                .to_numpy()[order]
                .astype(object),
                "is_reference": uuids[order] == str(reference_cycle_uuid),
                "mean_value": mean,
                "std_value": std,
                "deviation_from_ref": mean - ref_mean,
                "deviation_pct": pct,
                "variability_ratio": ratio,
            }
        )
        logger.info(
            f"Compared {len(result)} cycles against reference cycle '{reference_cycle_uuid}'."
        )
        return result

    @renamed_kwargs(metric="value_column")
    def identify_golden_cycles(
        self,
        value_column: str = "value_double",
        method: str = "low_variability",
        top_n: int = 5,
        target: float | None = None,
    ) -> list[str]:
        """
        Identify the best performing cycles (golden cycles).

        Args:
            value_column: Column holding the values to evaluate. ``metric`` is the
                deprecated name.
            method: ``'low_variability'`` (smallest std / |mean|),
                ``'high_mean'`` (largest mean) or ``'target_value'`` (mean
                closest to ``target``).
            top_n: Number of golden cycles to identify
            target: Target value, required for ``'target_value'``.

        Returns:
            Up to ``top_n`` cycle UUIDs, best first; ties keep chronological
                order. Cycles without ``value_column`` data, or whose score is
                undefined (a single sample has no std), are not ranked.

        Raises:
            ValueError: For an unknown ``method``, or ``'target_value'`` without
                ``target``.
            ColumnNotFoundError: If ``value_column`` is not a column of ``values_df``.
        """
        if method not in ("low_variability", "high_mean", "target_value"):
            raise ValueError(
                f"Unknown method {method!r}; use 'low_variability', 'high_mean' "
                "or 'target_value'."
            )
        if method == "target_value" and target is None:
            raise ValueError("method='target_value' needs target=<value>.")

        stats = self._per_cycle_metric(value_column)
        if stats.empty:
            logger.warning("Could not compute scores for any cycles.")
            return []
        if method == "low_variability":
            with np.errstate(divide="ignore", invalid="ignore"):
                score = -(stats["std"] / stats["mean"].abs())
            score = score.where(stats["mean"] != 0, -np.inf)
            score = score.where(stats["std"].notna())
        elif method == "high_mean":
            score = stats["mean"]
        else:
            score = -(stats["mean"] - float(target)).abs()  # type: ignore[arg-type]

        chrono = {int(p): rank for rank, p in enumerate(self._cycle_order())}
        ranked = pd.DataFrame(
            {"score": score, "chrono": [chrono[int(p)] for p in stats.index]},
            index=stats.index,
        ).dropna(subset=["score"])
        ranked = ranked.sort_values(
            ["score", "chrono"], ascending=[False, True], kind="stable"
        )
        uuids = self.cycles_df[self.cycle_uuid_column].to_numpy()
        golden_cycles: list[str] = [uuids[int(p)] for p in ranked.index[:top_n]]

        logger.info(
            f"Identified {len(golden_cycles)} golden cycles using method '{method}'."
        )
        return golden_cycles
