import logging
from ts_shape.utils.base import Base
import pandas as pd  # type: ignore
import numpy as np
import operator
from typing import Callable, Optional, Dict, Tuple

logger = logging.getLogger(__name__)


class ToleranceDeviationEvents(Base):
    """
    Inherits from Base and processes DataFrame data for specific events, comparing tolerance and actual values.

    Enhanced features:
    - Separate upper and lower tolerances
    - Warning zones with configurable thresholds
    - Deviation magnitude tracking (absolute and percentage)
    - Severity level classification (minor, major, critical)
    - Process capability indices (Cp, Cpk, Pp, Ppk)
    - Time-lagged tolerance application
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tolerance_column: str,
        actual_column: str,
        tolerance_uuid: Optional[str] = None,
        actual_uuid: str = None,
        event_uuid: str = None,
        compare_func: Callable[[pd.Series, pd.Series], pd.Series] = operator.ge,
        time_threshold: str = "5min",
        upper_tolerance_uuid: Optional[str] = None,
        lower_tolerance_uuid: Optional[str] = None,
        warning_threshold: float = 0.8,
        tolerance_lag: str = "0s",
    ) -> None:
        """
        Initializes the ToleranceDeviationEvents with specific event attributes.
        Inherits the sorted dataframe from the Base class.

        Args:
            dataframe: Input DataFrame with measurement data
            tolerance_column: Column name containing tolerance values
            actual_column: Column name containing actual measurement values
            tolerance_uuid: UUID for tolerance (backward compatibility, used if upper/lower not specified)
            actual_uuid: UUID for actual measurements
            event_uuid: UUID for generated events
            compare_func: Comparison function (default: operator.ge for greater-than-or-equal)
            time_threshold: Time window for grouping events (default: '5min')
            upper_tolerance_uuid: UUID for upper tolerance limit (optional)
            lower_tolerance_uuid: UUID for lower tolerance limit (optional)
            warning_threshold: Threshold ratio for warning zone (default: 0.8 = 80% of tolerance)
            tolerance_lag: Time lag for tolerance application (default: '0s')
        """
        super().__init__(dataframe)  # Inherit and initialize Base class

        self.tolerance_column: str = tolerance_column
        self.actual_column: str = actual_column
        self.actual_uuid: str = actual_uuid
        self.event_uuid: str = event_uuid
        self.compare_func: Callable[[pd.Series, pd.Series], pd.Series] = compare_func
        self.time_threshold: str = time_threshold
        self.warning_threshold: float = warning_threshold
        self.tolerance_lag: str = tolerance_lag

        # Handle backward compatibility for tolerance_uuid
        if upper_tolerance_uuid is None and lower_tolerance_uuid is None:
            if tolerance_uuid is None:
                raise ValueError(
                    "Either tolerance_uuid or both upper_tolerance_uuid and lower_tolerance_uuid must be provided"
                )
            # Use single tolerance for both upper and lower (backward compatibility)
            self.tolerance_uuid: Optional[str] = tolerance_uuid
            self.upper_tolerance_uuid: Optional[str] = None
            self.lower_tolerance_uuid: Optional[str] = None
            self.separate_tolerances: bool = False
        else:
            # Use separate upper and lower tolerances
            self.tolerance_uuid: Optional[str] = None
            self.upper_tolerance_uuid: Optional[str] = upper_tolerance_uuid
            self.lower_tolerance_uuid: Optional[str] = lower_tolerance_uuid
            self.separate_tolerances: bool = True

    def _apply_tolerance_lag(
        self, df: pd.DataFrame, tolerance_col: str
    ) -> pd.DataFrame:
        """
        Apply time lag to tolerance values.

        Args:
            df: DataFrame with tolerance values
            tolerance_col: Name of the tolerance column

        Returns:
            DataFrame with lagged tolerance values
        """
        if self.tolerance_lag != "0s":
            lag_timedelta = pd.to_timedelta(self.tolerance_lag)
            df["systime_shifted"] = df["systime"] + lag_timedelta
            df = df.sort_values(by="systime_shifted", ascending=False)
            df["systime"] = df["systime_shifted"]
            df = df.drop("systime_shifted", axis=1)
        return df

    def _calculate_severity(
        self,
        deviation_abs: float,
        tolerance_range: float,
        upper_limit: float,
        lower_limit: float,
        actual_value: float,
    ) -> str:
        """
        Calculate severity level based on deviation magnitude.

        Args:
            deviation_abs: Absolute deviation from tolerance
            tolerance_range: Total tolerance range (upper - lower)
            upper_limit: Upper tolerance limit
            lower_limit: Lower tolerance limit
            actual_value: Actual measured value

        Returns:
            Severity level: 'minor', 'major', or 'critical'
        """
        if pd.isna(deviation_abs) or pd.isna(tolerance_range):
            return "unknown"

        warning_limit = tolerance_range * self.warning_threshold

        # Check if within warning zone
        if deviation_abs <= warning_limit:
            return "minor"

        # Check if beyond tolerance but not critical
        if actual_value > upper_limit or actual_value < lower_limit:
            # Critical if beyond 2x tolerance range
            if deviation_abs > tolerance_range * 2:
                return "critical"
            else:
                return "major"

        return "minor"

    def process_and_group_data_with_events(self) -> pd.DataFrame:
        """
        Processes DataFrame to apply tolerance checks, group events by time, and generate an events DataFrame.

        Enhanced with:
        - Separate upper/lower tolerances
        - Deviation magnitude tracking (absolute and percentage)
        - Warning zones
        - Severity level classification
        - Time-lagged tolerance application

        Returns:
            pd.DataFrame: A DataFrame of processed and grouped event data with enhanced metrics.
        """
        df = self.dataframe.copy()  # Inherited from Base class

        # Convert 'systime' to datetime and sort the DataFrame by 'systime' in descending order
        df["systime"] = pd.to_datetime(df["systime"])
        df = df.sort_values(by="systime", ascending=False)

        if not self.separate_tolerances:
            # Backward compatibility: use single tolerance
            df = self._process_single_tolerance(df)
        else:
            # Use separate upper and lower tolerances
            df = self._process_separate_tolerances(df)

        # Apply time lag to tolerance if specified
        if self.tolerance_lag != "0s":
            df = self._apply_tolerance_lag(df, "tolerance_value")

        return df

    def _process_single_tolerance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process data using single tolerance (backward compatibility).

        Args:
            df: Input DataFrame

        Returns:
            Processed DataFrame with events
        """
        # Create a column for lagged tolerance values
        df["tolerance_value"] = df.apply(
            lambda row: (
                row[self.tolerance_column]
                if (row["uuid"] == self.tolerance_uuid and row["is_delta"])
                else pd.NA
            ),
            axis=1,
        )

        # Forward fill the tolerance values to propagate the last observed tolerance value
        df["tolerance_value"] = df["tolerance_value"].ffill()

        # Store original tolerance for deviation calculations
        df["upper_tolerance"] = df["tolerance_value"]
        df["lower_tolerance"] = df["tolerance_value"]

        # Remove tolerance setting rows from the dataset
        df = df[df["uuid"] != self.tolerance_uuid]

        # Ensure there are no NA values in the tolerance_value column before comparison
        df = df.dropna(subset=["tolerance_value"])

        # Calculate deviations for all rows with actual values
        actual_mask = df["uuid"] == self.actual_uuid
        df.loc[actual_mask, "deviation_abs"] = np.abs(
            df.loc[actual_mask, self.actual_column]
            - df.loc[actual_mask, "tolerance_value"]
        )

        # Calculate percentage deviation (avoid division by zero)
        tolerance_nonzero = df.loc[actual_mask, "tolerance_value"] != 0
        df.loc[actual_mask & tolerance_nonzero, "deviation_pct"] = (
            df.loc[actual_mask & tolerance_nonzero, "deviation_abs"]
            / np.abs(df.loc[actual_mask & tolerance_nonzero, "tolerance_value"])
            * 100
        )

        # Calculate severity levels
        df.loc[actual_mask, "severity"] = df.loc[actual_mask].apply(
            lambda row: self._calculate_severity(
                row["deviation_abs"],
                np.abs(row["upper_tolerance"] - row["lower_tolerance"]),
                row["upper_tolerance"],
                row["lower_tolerance"],
                row[self.actual_column],
            ),
            axis=1,
        )

        # Apply comparison function to compare actual values with tolerance values
        df = df[self.compare_func(df[self.actual_column], df["tolerance_value"])]
        df["value_bool"] = True  # Assign True in the value_bool column for kept rows

        # Grouping events that are close to each other in terms of time
        df["group_id"] = (
            df["systime"].diff().abs() > pd.to_timedelta(self.time_threshold)
        ).cumsum()

        # Filter for specific UUID and prepare events DataFrame
        filtered_df = df[df["uuid"] == self.actual_uuid]
        events_data = []

        for group_id in filtered_df["group_id"].unique():
            group_data = filtered_df[filtered_df["group_id"] == group_id]
            if group_data.shape[0] > 1:  # Ensure there's more than one row to work with
                first_row = group_data.nsmallest(1, "systime")
                last_row = group_data.nlargest(1, "systime")
                combined_rows = pd.concat([first_row, last_row])
                events_data.append(combined_rows)

        # Convert list of DataFrame slices to a single DataFrame
        if events_data:
            events_df = pd.concat(events_data)
            events_df["uuid"] = self.event_uuid
        else:
            events_df = pd.DataFrame(
                columns=filtered_df.columns
            )  # Create empty DataFrame if no data

        # Clean up temporary columns
        cols_to_drop = [
            "tolerance_value",
            "group_id",
            "upper_tolerance",
            "lower_tolerance",
        ]
        cols_to_drop = [col for col in cols_to_drop if col in events_df.columns]
        events_df = events_df.drop(cols_to_drop, axis=1)

        events_df[self.actual_column] = np.nan
        events_df["is_delta"] = True

        return events_df

    def _process_separate_tolerances(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process data using separate upper and lower tolerances.

        Args:
            df: Input DataFrame

        Returns:
            Processed DataFrame with events
        """
        # Create columns for upper and lower tolerance values
        df["upper_tolerance"] = df.apply(
            lambda row: (
                row[self.tolerance_column]
                if (row["uuid"] == self.upper_tolerance_uuid and row["is_delta"])
                else pd.NA
            ),
            axis=1,
        )

        df["lower_tolerance"] = df.apply(
            lambda row: (
                row[self.tolerance_column]
                if (row["uuid"] == self.lower_tolerance_uuid and row["is_delta"])
                else pd.NA
            ),
            axis=1,
        )

        # Forward fill the tolerance values
        df["upper_tolerance"] = df["upper_tolerance"].ffill()
        df["lower_tolerance"] = df["lower_tolerance"].ffill()

        # Remove tolerance setting rows from the dataset
        df = df[
            ~df["uuid"].isin([self.upper_tolerance_uuid, self.lower_tolerance_uuid])
        ]

        # Ensure there are no NA values in the tolerance columns before comparison
        df = df.dropna(subset=["upper_tolerance", "lower_tolerance"])

        # Calculate tolerance midpoint and range
        df["tolerance_midpoint"] = (df["upper_tolerance"] + df["lower_tolerance"]) / 2
        df["tolerance_range"] = df["upper_tolerance"] - df["lower_tolerance"]

        # Calculate deviations for all rows with actual values
        actual_mask = df["uuid"] == self.actual_uuid

        # Absolute deviation from nearest tolerance boundary
        df.loc[actual_mask, "deviation_from_upper"] = (
            df.loc[actual_mask, self.actual_column]
            - df.loc[actual_mask, "upper_tolerance"]
        )
        df.loc[actual_mask, "deviation_from_lower"] = (
            df.loc[actual_mask, "lower_tolerance"]
            - df.loc[actual_mask, self.actual_column]
        )

        # Calculate absolute deviation (distance to nearest boundary)
        df.loc[actual_mask, "deviation_abs"] = df.loc[actual_mask].apply(
            lambda row: max(
                0, max(row["deviation_from_upper"], row["deviation_from_lower"])
            ),
            axis=1,
        )

        # Calculate percentage deviation based on tolerance range
        tolerance_nonzero = df.loc[actual_mask, "tolerance_range"] != 0
        df.loc[actual_mask & tolerance_nonzero, "deviation_pct"] = (
            df.loc[actual_mask & tolerance_nonzero, "deviation_abs"]
            / df.loc[actual_mask & tolerance_nonzero, "tolerance_range"]
            * 100
        )

        # Calculate severity levels
        df.loc[actual_mask, "severity"] = df.loc[actual_mask].apply(
            lambda row: self._calculate_severity(
                row["deviation_abs"],
                row["tolerance_range"],
                row["upper_tolerance"],
                row["lower_tolerance"],
                row[self.actual_column],
            ),
            axis=1,
        )

        # Check if values are outside tolerance bounds
        outside_bounds = (df[self.actual_column] > df["upper_tolerance"]) | (
            df[self.actual_column] < df["lower_tolerance"]
        )

        df = df[outside_bounds]
        df["value_bool"] = True

        # Grouping events that are close to each other in terms of time
        df["group_id"] = (
            df["systime"].diff().abs() > pd.to_timedelta(self.time_threshold)
        ).cumsum()

        # Filter for specific UUID and prepare events DataFrame
        filtered_df = df[df["uuid"] == self.actual_uuid]
        events_data = []

        for group_id in filtered_df["group_id"].unique():
            group_data = filtered_df[filtered_df["group_id"] == group_id]
            if group_data.shape[0] > 1:  # Ensure there's more than one row to work with
                first_row = group_data.nsmallest(1, "systime")
                last_row = group_data.nlargest(1, "systime")
                combined_rows = pd.concat([first_row, last_row])
                events_data.append(combined_rows)

        # Convert list of DataFrame slices to a single DataFrame
        if events_data:
            events_df = pd.concat(events_data)
            events_df["uuid"] = self.event_uuid
        else:
            events_df = pd.DataFrame(columns=filtered_df.columns)

        # Clean up temporary columns
        cols_to_drop = [
            "upper_tolerance",
            "lower_tolerance",
            "tolerance_midpoint",
            "tolerance_range",
            "group_id",
            "deviation_from_upper",
            "deviation_from_lower",
        ]
        cols_to_drop = [col for col in cols_to_drop if col in events_df.columns]
        events_df = events_df.drop(cols_to_drop, axis=1)

        events_df[self.actual_column] = np.nan
        events_df["is_delta"] = True

        return events_df

    def compute_capability_indices(
        self, target_value: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate process capability indices (Cp, Cpk, Pp, Ppk).

        Process capability indices measure how well a process meets specification limits:
        - Cp: Process capability (potential capability assuming perfect centering)
        - Cpk: Process capability index (accounts for process centering)
        - Pp: Process performance (overall variability)
        - Ppk: Process performance index (accounts for centering)

        Args:
            target_value: Target/nominal value for the process. If None, uses midpoint of tolerances.

        Returns:
            Dictionary containing:
                - 'Cp': Process capability
                - 'Cpk': Process capability index
                - 'Pp': Process performance
                - 'Ppk': Process performance index
                - 'mean': Process mean
                - 'std': Process standard deviation
                - 'usl': Upper specification limit
                - 'lsl': Lower specification limit
                - 'target': Target value used

        Note:
            - Cp/Cpk use short-term variation (within-subgroup)
            - Pp/Ppk use long-term variation (overall)
            - Values > 1.33 are generally considered acceptable
            - Values > 1.67 are considered good
        """
        df = self.dataframe.copy()
        df["systime"] = pd.to_datetime(df["systime"])
        df = df.sort_values(by="systime", ascending=False)

        # Get tolerance limits
        if not self.separate_tolerances:
            # Single tolerance case
            tolerance_rows = df[df["uuid"] == self.tolerance_uuid]
            if tolerance_rows.empty:
                raise ValueError(
                    f"No tolerance data found for uuid: {self.tolerance_uuid}"
                )

            tolerance_value = tolerance_rows[self.tolerance_column].iloc[0]
            usl = tolerance_value  # Upper Specification Limit
            lsl = tolerance_value  # Lower Specification Limit (same as upper for single tolerance)
        else:
            # Separate tolerances case
            upper_rows = df[df["uuid"] == self.upper_tolerance_uuid]
            lower_rows = df[df["uuid"] == self.lower_tolerance_uuid]

            if upper_rows.empty:
                raise ValueError(
                    f"No upper tolerance data found for uuid: {self.upper_tolerance_uuid}"
                )
            if lower_rows.empty:
                raise ValueError(
                    f"No lower tolerance data found for uuid: {self.lower_tolerance_uuid}"
                )

            usl = upper_rows[self.tolerance_column].iloc[0]
            lsl = lower_rows[self.tolerance_column].iloc[0]

        # Get actual measurements
        actual_data = df[df["uuid"] == self.actual_uuid][self.actual_column].dropna()

        if actual_data.empty:
            raise ValueError(
                f"No actual measurement data found for uuid: {self.actual_uuid}"
            )

        # Calculate statistics
        process_mean = actual_data.mean()
        process_std = actual_data.std(ddof=1)  # Sample standard deviation

        if process_std == 0:
            raise ValueError(
                "Process standard deviation is zero - cannot calculate capability indices"
            )

        # Determine target value
        if target_value is None:
            target_value = (usl + lsl) / 2

        # Calculate Cp (potential capability)
        cp = (usl - lsl) / (6 * process_std)

        # Calculate Cpk (actual capability accounting for centering)
        cpu = (usl - process_mean) / (3 * process_std)
        cpl = (process_mean - lsl) / (3 * process_std)
        cpk = min(cpu, cpl)

        # For Pp and Ppk, we use the same formulas as Cp and Cpk
        # In practice, Pp/Ppk would use long-term std dev, but without subgroups, we use overall std
        pp = cp  # Using overall variation
        ppk = cpk

        return {
            "Cp": round(cp, 4),
            "Cpk": round(cpk, 4),
            "Pp": round(pp, 4),
            "Ppk": round(ppk, 4),
            "mean": round(process_mean, 4),
            "std": round(process_std, 4),
            "usl": round(usl, 4),
            "lsl": round(lsl, 4),
            "target": round(target_value, 4),
        }
