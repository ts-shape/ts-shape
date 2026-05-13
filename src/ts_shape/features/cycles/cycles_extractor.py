import logging
from typing import Optional, Dict, List, Any
import pandas as pd  # type: ignore
import uuid
from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class CycleExtractor(Base):
    """Class for processing cycles based on different criteria."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        start_uuid: str,
        end_uuid: Optional[str] = None,
        value_change_threshold: float = 0.0,
    ):
        """Initializes the class with the data and the UUIDs for cycle start and end.

        Args:
            dataframe: Input DataFrame with cycle data
            start_uuid: UUID for cycle start variable
            end_uuid: UUID for cycle end variable (defaults to start_uuid)
            value_change_threshold: Minimum threshold for considering a value change significant (default: 0.0)
        """
        super().__init__(dataframe)

        # Validate input types
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("dataframe must be a pandas DataFrame")
        if not isinstance(start_uuid, str):
            raise ValueError("start_uuid must be a string")

        self.df = dataframe  # Use the provided DataFrame directly
        self.start_uuid = start_uuid
        self.end_uuid = end_uuid if end_uuid else start_uuid
        self.value_change_threshold = abs(value_change_threshold)

        # Statistics tracking
        self._stats: Dict[str, Any] = {
            "total_cycles": 0,
            "complete_cycles": 0,
            "incomplete_cycles": 0,
            "unmatched_starts": 0,
            "unmatched_ends": 0,
            "overlapping_cycles": 0,
            "warnings": [],
        }

        logger.info(
            f"CycleExtractor initialized with start_uuid: {self.start_uuid} and end_uuid: {self.end_uuid}"
        )

    def process_persistent_cycle(self) -> pd.DataFrame:
        """Processes cycles where the value of the variable stays true during the cycle."""
        # Assuming dataframe is pre-filtered
        cycle_starts = self.df[self.df["value_bool"] == True]
        cycle_ends = self.df[self.df["value_bool"] == False]

        return self._generate_cycle_dataframe(cycle_starts, cycle_ends)

    def process_trigger_cycle(self) -> pd.DataFrame:
        """Processes cycles where the value of the variable goes from true to false during the cycle."""
        # Assuming dataframe is pre-filtered
        cycle_starts = self.df[self.df["value_bool"] == True]
        cycle_ends = self.df[self.df["value_bool"] == False].shift(-1)

        return self._generate_cycle_dataframe(cycle_starts, cycle_ends)

    def process_separate_start_end_cycle(self) -> pd.DataFrame:
        """Processes cycles where different variables indicate cycle start and end.

        When the DataFrame contains a 'uuid' column and start_uuid != end_uuid,
        filters starts by start_uuid and ends by end_uuid. Otherwise falls back
        to treating all True values as both start and end candidates.
        """
        if "uuid" in self.df.columns and self.start_uuid != self.end_uuid:
            cycle_starts = self.df[
                (self.df["uuid"] == self.start_uuid) & (self.df["value_bool"] == True)
            ]
            cycle_ends = self.df[
                (self.df["uuid"] == self.end_uuid) & (self.df["value_bool"] == True)
            ]
        else:
            cycle_starts = self.df[self.df["value_bool"] == True]
            cycle_ends = self.df[self.df["value_bool"] == True]

        return self._generate_cycle_dataframe(cycle_starts, cycle_ends)

    def process_step_sequence(self, start_step: int, end_step: int) -> pd.DataFrame:
        """Processes cycles based on a step sequence, where specific integer values denote cycle start and end."""
        # Assuming dataframe is pre-filtered
        cycle_starts = self.df[self.df["value_integer"] == start_step]
        cycle_ends = self.df[self.df["value_integer"] == end_step]

        return self._generate_cycle_dataframe(cycle_starts, cycle_ends)

    def process_state_change_cycle(self) -> pd.DataFrame:
        """Processes cycles where the start of a new cycle is the end of the previous cycle."""
        # Assuming dataframe is pre-filtered
        cycle_starts = self.df.copy()
        cycle_ends = self.df.shift(-1)

        return self._generate_cycle_dataframe(cycle_starts, cycle_ends)

    def process_value_change_cycle(self) -> pd.DataFrame:
        """Processes cycles where a change in the value indicates a new cycle.

        Uses the value_change_threshold to determine if a numeric change is significant.
        """
        # Assuming dataframe is pre-filtered

        # Fill NaN or None values with appropriate defaults for diff() to work
        self.df["value_double"] = self.df["value_double"].fillna(
            0
        )  # Assuming numeric column
        self.df["value_bool"] = self.df["value_bool"].fillna(
            False
        )  # Assuming boolean column
        self.df["value_string"] = self.df["value_string"].fillna(
            ""
        )  # Assuming string column
        self.df["value_integer"] = self.df["value_integer"].fillna(
            0
        )  # Assuming integer column

        # Detect changes across the relevant columns using diff() with threshold
        self.df["value_change"] = (
            (self.df["value_double"].diff().abs() > self.value_change_threshold)
            | (self.df["value_bool"].diff().ne(0))
            | (self.df["value_string"].shift().ne(self.df["value_string"]))
            | (self.df["value_integer"].diff().abs() > self.value_change_threshold)
        )

        # Define cycle starts and ends based on changes
        cycle_starts = self.df[self.df["value_change"] == True]
        cycle_ends = self.df[self.df["value_change"] == True].shift(-1)

        return self._generate_cycle_dataframe(cycle_starts, cycle_ends)

    def _generate_cycle_dataframe(
        self, cycle_starts: pd.DataFrame, cycle_ends: pd.DataFrame
    ) -> pd.DataFrame:
        """Generates a DataFrame with cycle start and end times.

        Uses a vectorized merge_asof approach (O(n log n)) instead of a Python
        iterator loop, significantly faster on large datasets.  Tracks incomplete
        cycles and adds an 'is_complete' flag.
        """
        empty = pd.DataFrame(
            {
                "cycle_start": pd.Series(dtype="datetime64[ns]"),
                "cycle_end": pd.Series(dtype="datetime64[ns]"),
                "cycle_uuid": pd.Series(dtype="string"),
                "is_complete": pd.Series(dtype="bool"),
            }
        )

        if cycle_starts.empty or "systime" not in cycle_starts.columns:
            self._stats.update(
                {
                    "total_cycles": 0,
                    "complete_cycles": 0,
                    "incomplete_cycles": 0,
                    "unmatched_starts": 0,
                }
            )
            return empty

        starts = (
            cycle_starts[["systime"]]
            .dropna(subset=["systime"])
            .rename(columns={"systime": "cycle_start"})
            .sort_values("cycle_start")
            .reset_index(drop=True)
        )

        ends_raw = (
            cycle_ends["systime"].dropna()
            if "systime" in cycle_ends.columns
            else pd.Series(dtype="datetime64[ns]")
        )
        ends = (
            pd.DataFrame({"cycle_end": ends_raw})
            .sort_values("cycle_end")
            .reset_index(drop=True)
        )

        if ends.empty:
            starts["cycle_end"] = pd.NaT
            starts["is_complete"] = False
        else:
            # For each start, find the nearest end that is >= start (forward match).
            # merge_asof requires both keys to have the same name; use a temporary key.
            starts_keyed = starts.rename(columns={"cycle_start": "_key"})
            ends_keyed = ends.rename(columns={"cycle_end": "_key"})
            merged = pd.merge_asof(
                starts_keyed,
                ends_keyed,
                on="_key",
                direction="forward",
                suffixes=("", "_end"),
            )
            starts["cycle_start"] = merged["_key"].values
            starts["cycle_end"] = (
                merged["_key_end"].values if "_key_end" in merged.columns else pd.NaT
            )
            # merge_asof forward: unmatched rows have NaT in right key
            starts["is_complete"] = starts["cycle_end"].notna()

        starts["cycle_uuid"] = [str(uuid.uuid4()) for _ in range(len(starts))]
        cycle_df = starts[
            ["cycle_start", "cycle_end", "cycle_uuid", "is_complete"]
        ].reset_index(drop=True)

        complete_cycles = int(cycle_df["is_complete"].sum())
        incomplete_cycles = int((~cycle_df["is_complete"]).sum())

        if incomplete_cycles:
            warning_msg = f"{incomplete_cycles} cycle(s) have no matching end and are marked incomplete."
            logger.warning(warning_msg)
            self._stats["warnings"].append(warning_msg)

        self._stats.update(
            {
                "total_cycles": len(cycle_df),
                "complete_cycles": complete_cycles,
                "incomplete_cycles": incomplete_cycles,
                "unmatched_starts": incomplete_cycles,
            }
        )

        logger.info(
            f"Generated {len(cycle_df)} cycles ({complete_cycles} complete, {incomplete_cycles} incomplete)."
        )
        return cycle_df

    def _parse_duration(self, duration_str: str) -> pd.Timedelta:
        """Parse a duration string like '1s', '5m', '1h' to a pandas Timedelta.

        Args:
            duration_str: Duration string (e.g., '1s', '5m', '1h', '2d')

        Returns:
            pd.Timedelta object

        Raises:
            ValueError: If duration string format is invalid
        """
        import re

        match = re.match(r"^(\d+(?:\.\d+)?)\s*([smhd])$", duration_str.lower().strip())
        if not match:
            raise ValueError(
                f"Invalid duration format: '{duration_str}'. Expected format: number followed by s/m/h/d (e.g., '1s', '5m', '1h')"
            )

        value, unit = match.groups()
        value = float(value)

        unit_map = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days"}

        return pd.Timedelta(**{unit_map[unit]: value})

    def validate_cycles(
        self,
        cycle_df: pd.DataFrame,
        min_duration: str = "1s",
        max_duration: str = "1h",
        warn: bool = True,
    ) -> pd.DataFrame:
        """Validate cycles based on duration constraints.

        Args:
            cycle_df: DataFrame with cycle data (output from process_* methods)
            min_duration: Minimum acceptable cycle duration (default: '1s')
            max_duration: Maximum acceptable cycle duration (default: '1h')
            warn: Whether to log warnings for invalid cycles (default: True)

        Returns:
            DataFrame with additional 'is_valid' column and 'validation_issue' column
        """
        if cycle_df.empty:
            logger.warning("Empty cycle DataFrame provided for validation.")
            return cycle_df

        # Parse duration constraints
        min_td = self._parse_duration(min_duration)
        max_td = self._parse_duration(max_duration)

        # Create a copy to avoid modifying the original
        validated_df = cycle_df.copy()

        # Calculate cycle durations
        validated_df["cycle_duration"] = (
            validated_df["cycle_end"] - validated_df["cycle_start"]
        )

        # Initialize validation columns
        validated_df["is_valid"] = True
        validated_df["validation_issue"] = ""

        # Check for incomplete cycles
        incomplete_mask = ~validated_df["is_complete"]
        if incomplete_mask.any():
            validated_df.loc[incomplete_mask, "is_valid"] = False
            validated_df.loc[incomplete_mask, "validation_issue"] = "incomplete_cycle"
            if warn:
                logger.warning(f"Found {incomplete_mask.sum()} incomplete cycles.")

        # Check duration constraints (only for complete cycles)
        complete_mask = validated_df["is_complete"]
        too_short_mask = complete_mask & (validated_df["cycle_duration"] < min_td)
        too_long_mask = complete_mask & (validated_df["cycle_duration"] > max_td)

        if too_short_mask.any():
            validated_df.loc[too_short_mask, "is_valid"] = False
            validated_df.loc[too_short_mask, "validation_issue"] = (
                validated_df.loc[too_short_mask, "validation_issue"] + "too_short;"
            )
            if warn:
                logger.warning(
                    f"Found {too_short_mask.sum()} cycles shorter than {min_duration}."
                )

        if too_long_mask.any():
            validated_df.loc[too_long_mask, "is_valid"] = False
            validated_df.loc[too_long_mask, "validation_issue"] = (
                validated_df.loc[too_long_mask, "validation_issue"] + "too_long;"
            )
            if warn:
                logger.warning(
                    f"Found {too_long_mask.sum()} cycles longer than {max_duration}."
                )

        valid_count = validated_df["is_valid"].sum()
        invalid_count = (~validated_df["is_valid"]).sum()

        logger.info(
            f"Validation complete: {valid_count} valid cycles, {invalid_count} invalid cycles."
        )

        return validated_df

    def detect_overlapping_cycles(
        self, cycle_df: pd.DataFrame, resolve: str = "flag"
    ) -> pd.DataFrame:
        """Detect and optionally resolve overlapping cycles.

        Args:
            cycle_df: DataFrame with cycle data
            resolve: How to handle overlaps - 'flag' (mark only), 'keep_first', 'keep_last', 'keep_longest'

        Returns:
            DataFrame with 'has_overlap' column and potentially filtered rows
        """
        if cycle_df.empty:
            logger.warning("Empty cycle DataFrame provided for overlap detection.")
            return cycle_df

        # Create a copy to avoid modifying the original
        result_df = cycle_df.copy()

        # Sort by start time
        result_df = result_df.sort_values("cycle_start").reset_index(drop=True)

        # Initialize overlap column
        result_df["has_overlap"] = False

        # Check for overlaps (only for complete cycles)
        overlaps = []
        for i in range(len(result_df) - 1):
            if not result_df.loc[i, "is_complete"]:
                continue

            current_end = result_df.loc[i, "cycle_end"]

            for j in range(i + 1, len(result_df)):
                if not result_df.loc[j, "is_complete"]:
                    continue

                next_start = result_df.loc[j, "cycle_start"]

                if current_end > next_start:
                    # Overlap detected
                    result_df.loc[i, "has_overlap"] = True
                    result_df.loc[j, "has_overlap"] = True
                    overlaps.append((i, j))
                else:
                    break  # No more overlaps for current cycle

        overlap_count = len(overlaps)
        if overlap_count > 0:
            logger.warning(f"Detected {overlap_count} overlapping cycle pairs.")
            self._stats["overlapping_cycles"] = overlap_count

        # Resolve overlaps based on strategy
        if resolve != "flag" and overlap_count > 0:
            indices_to_drop = set()

            for i, j in overlaps:
                if resolve == "keep_first":
                    indices_to_drop.add(j)
                elif resolve == "keep_last":
                    indices_to_drop.add(i)
                elif resolve == "keep_longest":
                    duration_i = (
                        result_df.loc[i, "cycle_end"] - result_df.loc[i, "cycle_start"]
                    )
                    duration_j = (
                        result_df.loc[j, "cycle_end"] - result_df.loc[j, "cycle_start"]
                    )
                    if duration_i >= duration_j:
                        indices_to_drop.add(j)
                    else:
                        indices_to_drop.add(i)

            if indices_to_drop:
                result_df = result_df.drop(list(indices_to_drop)).reset_index(drop=True)
                logger.info(
                    f"Resolved overlaps: removed {len(indices_to_drop)} cycles using '{resolve}' strategy."
                )

        return result_df

    def suggest_method(self) -> Dict[str, Any]:
        """Suggest the best cycle extraction method based on data characteristics.

        Analyzes the input DataFrame to recommend appropriate extraction method(s).

        Returns:
            Dictionary with method suggestions and reasoning
        """
        suggestions = {
            "recommended_methods": [],
            "reasoning": [],
            "data_characteristics": {},
        }

        # Analyze data characteristics
        has_bool = (
            "value_bool" in self.df.columns and self.df["value_bool"].notna().any()
        )
        has_integer = (
            "value_integer" in self.df.columns
            and self.df["value_integer"].notna().any()
        )
        has_double = (
            "value_double" in self.df.columns and self.df["value_double"].notna().any()
        )
        has_string = (
            "value_string" in self.df.columns and self.df["value_string"].notna().any()
        )

        suggestions["data_characteristics"] = {
            "has_boolean_values": has_bool,
            "has_integer_values": has_integer,
            "has_double_values": has_double,
            "has_string_values": has_string,
            "row_count": len(self.df),
            "separate_start_end": self.start_uuid != self.end_uuid,
        }

        # Suggest methods based on characteristics
        if suggestions["data_characteristics"]["separate_start_end"]:
            suggestions["recommended_methods"].append(
                "process_separate_start_end_cycle"
            )
            suggestions["reasoning"].append("Separate start and end UUIDs detected")

        if has_bool:
            # Analyze boolean patterns
            bool_data = self.df["value_bool"].dropna()
            if len(bool_data) > 1:
                # Check for transitions
                transitions = bool_data.diff().abs().sum()
                if transitions > 0:
                    suggestions["recommended_methods"].append(
                        "process_persistent_cycle"
                    )
                    suggestions["reasoning"].append(
                        "Boolean transitions detected, suitable for persistent cycles"
                    )

                    suggestions["recommended_methods"].append("process_trigger_cycle")
                    suggestions["reasoning"].append(
                        "Boolean data present, can use trigger-based extraction"
                    )

        if has_integer:
            # Check if integers represent steps
            int_data = self.df["value_integer"].dropna()
            unique_values = int_data.nunique()
            if 2 <= unique_values <= 20:  # Reasonable step count
                suggestions["recommended_methods"].append("process_step_sequence")
                suggestions["reasoning"].append(
                    f"Integer data with {unique_values} unique values suggests step sequence"
                )

        if has_double or has_integer or has_string:
            # Value change method is versatile
            suggestions["recommended_methods"].append("process_value_change_cycle")
            suggestions["reasoning"].append(
                "Data shows changing values, suitable for value change detection"
            )

        # Check for state-like patterns
        if has_integer or has_string:
            suggestions["recommended_methods"].append("process_state_change_cycle")
            suggestions["reasoning"].append(
                "State-based data detected, consider state change cycles"
            )

        # If no specific method suggested, recommend value_change as fallback
        if not suggestions["recommended_methods"]:
            suggestions["recommended_methods"].append("process_value_change_cycle")
            suggestions["reasoning"].append(
                "Default method: works with any value changes"
            )

        logger.info(f"Method suggestion: {suggestions['recommended_methods']}")
        return suggestions

    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get statistics about the last cycle extraction.

        Returns:
            Dictionary with extraction statistics including counts, success rate, and warnings
        """
        stats = self._stats.copy()

        # Calculate success rate
        if stats["total_cycles"] > 0:
            stats["success_rate"] = stats["complete_cycles"] / stats["total_cycles"]
        else:
            stats["success_rate"] = 0.0

        # Add configuration info
        stats["configuration"] = {
            "start_uuid": self.start_uuid,
            "end_uuid": self.end_uuid,
            "value_change_threshold": self.value_change_threshold,
        }

        return stats

    def reset_stats(self):
        """Reset extraction statistics."""
        self._stats = {
            "total_cycles": 0,
            "complete_cycles": 0,
            "incomplete_cycles": 0,
            "unmatched_starts": 0,
            "unmatched_ends": 0,
            "overlapping_cycles": 0,
            "warnings": [],
        }
        logger.info("Statistics reset.")
