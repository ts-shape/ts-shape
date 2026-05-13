import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd  # type: ignore
import numpy as np
from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class CycleDataProcessor(Base):
    """
    A class to process cycle-based data and values with optimized performance.
    Uses pandas IntervalIndex for efficient cycle assignment instead of nested loops.
    """

    def __init__(
        self,
        cycles_df: pd.DataFrame,
        values_df: pd.DataFrame,
        cycle_uuid_col: str = "cycle_uuid",
        systime_col: str = "systime",
    ):
        """
        Initializes the CycleDataProcessor with cycles and values DataFrames.

        Args:
            cycles_df: DataFrame containing columns 'cycle_start', 'cycle_end', and 'cycle_uuid'.
            values_df: DataFrame containing the values and timestamps in the 'systime' column.
            cycle_uuid_col: Name of the column representing cycle UUIDs.
            systime_col: Name of the column representing the timestamps for the values.
        """
        super().__init__(values_df)  # Call the parent constructor
        self.values_df = values_df.copy()  # Initialize self.values_df explicitly
        self.cycles_df = cycles_df.copy()
        self.cycle_uuid_col = cycle_uuid_col
        self.systime_col = systime_col

        # Ensure proper datetime format
        self.cycles_df["cycle_start"] = pd.to_datetime(self.cycles_df["cycle_start"])
        self.cycles_df["cycle_end"] = pd.to_datetime(self.cycles_df["cycle_end"])
        self.values_df[systime_col] = pd.to_datetime(self.values_df[systime_col])

        # Pre-build interval index for efficient lookups
        self._cycle_intervals = None
        self._build_interval_index()

        logger.info("CycleDataProcessor initialized with cycles and values DataFrames.")

    def _build_interval_index(self) -> None:
        """Build IntervalIndex for efficient cycle lookups."""
        if self.cycles_df.empty:
            self._cycle_intervals = None
            return

        # Create interval index from cycle start/end times
        intervals = pd.IntervalIndex.from_arrays(
            self.cycles_df["cycle_start"], self.cycles_df["cycle_end"], closed="both"
        )
        self._cycle_intervals = pd.Series(
            self.cycles_df[self.cycle_uuid_col].values, index=intervals
        )
        logger.debug(f"Built interval index with {len(intervals)} cycles.")

    def split_by_cycle(self) -> Dict[str, pd.DataFrame]:
        """
        Splits the values DataFrame by cycles defined in the cycles DataFrame.
        Uses optimized interval-based assignment.

        Return:
            Dictionary where keys are cycle_uuids and values are DataFrames with the corresponding cycle data.
        """
        if self._cycle_intervals is None or self.values_df.empty:
            logger.warning("No cycles or values available for splitting.")
            return {}

        # Use merge_dataframes_by_cycle to assign cycle_uuids efficiently
        merged = self.merge_dataframes_by_cycle()

        # Split into dictionary
        result = {
            cycle_uuid: group.drop(columns=[self.cycle_uuid_col])
            for cycle_uuid, group in merged.groupby(self.cycle_uuid_col)
        }

        logger.info(f"Split {len(result)} cycles.")
        return result

    def merge_dataframes_by_cycle(self) -> pd.DataFrame:
        """
        Merges the values DataFrame with the cycles DataFrame based on the cycle time intervals.
        Uses optimized interval-based assignment instead of nested loops.

        Return:
            DataFrame with an added 'cycle_uuid' column.
        """
        if self._cycle_intervals is None or self.values_df.empty:
            logger.warning("No cycles available for merging.")
            result = self.values_df.copy()
            result[self.cycle_uuid_col] = None
            return result

        # Create a copy to avoid modifying the original
        merged_df = self.values_df.copy()

        # Use pd.cut with interval index for vectorized assignment
        # This is much faster than iterating over cycles
        try:
            # Get time values as int64 (nanoseconds) for efficient indexing
            time_values = merged_df[self.systime_col]

            # Find which interval each timestamp belongs to
            cycle_assignment = pd.Series(index=merged_df.index, dtype="object")

            for interval, cycle_uuid in self._cycle_intervals.items():
                mask = (time_values >= interval.left) & (time_values <= interval.right)
                cycle_assignment.loc[mask] = cycle_uuid

            merged_df[self.cycle_uuid_col] = cycle_assignment

        except Exception as e:
            logger.error(
                f"Error in vectorized cycle assignment: {e}. Falling back to iterative method."
            )
            # Fallback to original method if vectorization fails
            merged_df[self.cycle_uuid_col] = None
            for _, row in self.cycles_df.iterrows():
                mask = (merged_df[self.systime_col] >= row["cycle_start"]) & (
                    merged_df[self.systime_col] <= row["cycle_end"]
                )
                merged_df.loc[mask, self.cycle_uuid_col] = row[self.cycle_uuid_col]

        # Drop rows not assigned to any cycle
        unassigned_count = merged_df[self.cycle_uuid_col].isna().sum()
        if unassigned_count > 0:
            logger.info(f"Dropping {unassigned_count} rows not assigned to any cycle.")
        result = merged_df.dropna(subset=[self.cycle_uuid_col])
        logger.info(
            f"Merged DataFrame contains {len(result)} records across {result[self.cycle_uuid_col].nunique()} cycles."
        )
        return result

    def group_by_cycle_uuid(
        self, data: Optional[pd.DataFrame] = None
    ) -> List[pd.DataFrame]:
        """
        Group the DataFrame by the cycle_uuid column, resulting in a list of DataFrames, each containing data for one cycle.

        Args:
            data: DataFrame containing the data to be grouped by cycle_uuid. If None, uses the internal values_df.

        Return:
            List of DataFrames, each containing data for a unique cycle_uuid.
        """
        if data is None:
            data = self.values_df

        if self.cycle_uuid_col not in data.columns:
            logger.warning(
                f"Column '{self.cycle_uuid_col}' not found in data. Cannot group."
            )
            return []

        grouped_dataframes = [group for _, group in data.groupby(self.cycle_uuid_col)]
        logger.info(f"Grouped data into {len(grouped_dataframes)} cycle UUID groups.")
        return grouped_dataframes

    def split_dataframes_by_group(
        self, dfs: List[pd.DataFrame], column: str
    ) -> List[pd.DataFrame]:
        """
        Splits a list of DataFrames by groups based on a specified column.
        This function performs a groupby operation on each DataFrame in the list and then flattens the result.

        Args:
            dfs: List of DataFrames to be split.
            column: Column name to group by.

        Return:
            List of DataFrames, each corresponding to a group in the original DataFrames.
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
        """
        Filters the values DataFrame by the given time range.

        Args:
            start_time: Start of the time range.
            end_time: End of the time range.

        Return:
            Filtered DataFrame containing rows within the time range.
        """
        mask = (self.values_df[self.systime_col] >= start_time) & (
            self.values_df[self.systime_col] <= end_time
        )
        return self.values_df[mask]

    def compute_cycle_statistics(self) -> pd.DataFrame:
        """
        Compute statistics for each cycle.

        Returns:
            DataFrame with cycle-level statistics including duration, value counts, etc.
        """
        if self.cycles_df.empty:
            return pd.DataFrame()

        stats = []
        for _, cycle in self.cycles_df.iterrows():
            cycle_uuid = cycle[self.cycle_uuid_col]
            cycle_start = cycle["cycle_start"]
            cycle_end = cycle["cycle_end"]

            # Get values for this cycle
            mask = (self.values_df[self.systime_col] >= cycle_start) & (
                self.values_df[self.systime_col] <= cycle_end
            )
            cycle_values = self.values_df[mask]

            # Compute stats
            stat = {
                self.cycle_uuid_col: cycle_uuid,
                "cycle_start": cycle_start,
                "cycle_end": cycle_end,
                "duration_seconds": (cycle_end - cycle_start).total_seconds(),
                "value_count": len(cycle_values),
                "unique_uuids": (
                    cycle_values["uuid"].nunique()
                    if "uuid" in cycle_values.columns
                    else 0
                ),
            }

            # Add value-type specific stats if columns exist
            if "value_double" in cycle_values.columns:
                stat["mean_value_double"] = cycle_values["value_double"].mean()
                stat["std_value_double"] = cycle_values["value_double"].std()

            stats.append(stat)

        result = pd.DataFrame(stats)
        logger.info(f"Computed statistics for {len(result)} cycles.")
        return result

    def compare_cycles(
        self, reference_cycle_uuid: str, metric: str = "value_double"
    ) -> pd.DataFrame:
        """
        Compare all cycles against a reference cycle.

        Args:
            reference_cycle_uuid: UUID of the reference cycle
            metric: Column name to use for comparison

        Returns:
            DataFrame with comparison metrics for each cycle
        """
        if reference_cycle_uuid not in self.cycles_df[self.cycle_uuid_col].values:
            logger.error(f"Reference cycle '{reference_cycle_uuid}' not found.")
            return pd.DataFrame()

        # Get reference cycle data
        ref_cycle = self.cycles_df[
            self.cycles_df[self.cycle_uuid_col] == reference_cycle_uuid
        ].iloc[0]
        ref_mask = (self.values_df[self.systime_col] >= ref_cycle["cycle_start"]) & (
            self.values_df[self.systime_col] <= ref_cycle["cycle_end"]
        )
        ref_values = self.values_df[ref_mask][metric].dropna()

        if ref_values.empty:
            logger.warning("Reference cycle has no data for the specified metric.")
            return pd.DataFrame()

        ref_mean = ref_values.mean()
        ref_std = ref_values.std()

        # Compare each cycle
        comparisons = []
        for _, cycle in self.cycles_df.iterrows():
            cycle_uuid = cycle[self.cycle_uuid_col]
            mask = (self.values_df[self.systime_col] >= cycle["cycle_start"]) & (
                self.values_df[self.systime_col] <= cycle["cycle_end"]
            )
            cycle_values = self.values_df[mask][metric].dropna()

            if cycle_values.empty:
                continue

            comparison = {
                self.cycle_uuid_col: cycle_uuid,
                "is_reference": cycle_uuid == reference_cycle_uuid,
                "mean_value": cycle_values.mean(),
                "std_value": cycle_values.std(),
                "deviation_from_ref": cycle_values.mean() - ref_mean,
                "deviation_pct": (
                    ((cycle_values.mean() - ref_mean) / ref_mean * 100)
                    if ref_mean != 0
                    else np.nan
                ),
                "variability_ratio": (
                    (cycle_values.std() / ref_std) if ref_std != 0 else np.nan
                ),
            }
            comparisons.append(comparison)

        result = pd.DataFrame(comparisons)
        logger.info(
            f"Compared {len(result)} cycles against reference cycle '{reference_cycle_uuid}'."
        )
        return result

    def identify_golden_cycles(
        self,
        metric: str = "value_double",
        method: str = "low_variability",
        top_n: int = 5,
    ) -> List[str]:
        """
        Identify the best performing cycles (golden cycles).

        Args:
            metric: Column name to evaluate
            method: Method for identification ('low_variability', 'high_mean', 'target_value')
            top_n: Number of golden cycles to identify

        Returns:
            List of cycle UUIDs identified as golden cycles
        """
        stats = self.compute_cycle_statistics()

        if stats.empty:
            logger.warning("No cycle statistics available.")
            return []

        # Calculate metric-specific scores for each cycle
        scores = []
        for _, cycle in self.cycles_df.iterrows():
            cycle_uuid = cycle[self.cycle_uuid_col]
            mask = (self.values_df[self.systime_col] >= cycle["cycle_start"]) & (
                self.values_df[self.systime_col] <= cycle["cycle_end"]
            )
            cycle_values = self.values_df[mask][metric].dropna()

            if cycle_values.empty:
                continue

            if method == "low_variability":
                # Lower coefficient of variation is better
                mean_val = cycle_values.mean()
                std_val = cycle_values.std()
                score = -(std_val / mean_val) if mean_val != 0 else -np.inf
            elif method == "high_mean":
                score = cycle_values.mean()
            else:  # target_value - would need target parameter
                score = -cycle_values.std()  # fallback to low variability

            scores.append({"cycle_uuid": cycle_uuid, "score": score})

        if not scores:
            logger.warning("Could not compute scores for any cycles.")
            return []

        scores_df = pd.DataFrame(scores).sort_values("score", ascending=False)
        golden_cycles = scores_df.head(top_n)["cycle_uuid"].tolist()

        logger.info(
            f"Identified {len(golden_cycles)} golden cycles using method '{method}'."
        )
        return golden_cycles
