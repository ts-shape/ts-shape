import logging
import operator
import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd

from ts_shape.utils.base import Base
from ts_shape.utils.deprecation import deprecated_method, keyword_only

logger = logging.getLogger(__name__)


class ToleranceDeviationEvents(Base):
    """
    Inherits from Base and processes DataFrame data for specific events, comparing tolerance and actual values.

    Enhanced features:
    - Upper and lower tolerances (two-sided, or one-sided with only one of them)
    - Warning zones with configurable thresholds
    - Deviation magnitude tracking (absolute and percentage)
    - Severity level classification (minor, major, critical)
    - Process capability indices (Cp, Cpk, Pp, Ppk)
    - Time-lagged tolerance application
    """

    @keyword_only(4)
    def __init__(
        self,
        dataframe: pd.DataFrame,
        tolerance_column: str,
        actual_column: str,
        tolerance_uuid: str | None = None,
        actual_uuid: str | None = None,
        event_uuid: str | None = None,
        compare_func: Callable[[pd.Series, pd.Series], pd.Series] = operator.ge,
        time_threshold: str = "5min",
        upper_tolerance_uuid: str | None = None,
        lower_tolerance_uuid: str | None = None,
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
            tolerance_uuid: Deprecated (removed in the next major release; passing it without
                ``upper_tolerance_uuid``/``lower_tolerance_uuid`` emits a
                ``DeprecationWarning``). UUID of a single tolerance signal
                that is compared with the actual values through
                ``compare_func``. Use ``upper_tolerance_uuid`` and/or
                ``lower_tolerance_uuid`` instead: a single upper limit is
                ``upper_tolerance_uuid`` alone.
            actual_uuid: UUID for actual measurements. Required: a
                ``ValueError`` is raised when it is missing.
            event_uuid: UUID for generated events (default None)
            compare_func: Comparison used by the deprecated single
                ``tolerance_uuid`` path only (default: ``operator.ge``, so
                ``actual >= tolerance`` is a deviation).
            time_threshold: Time window for grouping events (default: '5min')
            upper_tolerance_uuid: UUID for the upper tolerance limit. Given
                alone, it is a one-sided upper limit.
            lower_tolerance_uuid: UUID for the lower tolerance limit. Given
                alone, it is a one-sided lower limit.
            warning_threshold: Threshold ratio for warning zone (default: 0.8 = 80% of tolerance)
            tolerance_lag: Time lag for tolerance application (default: '0s')
        """
        if actual_uuid is None:
            raise ValueError(
                "actual_uuid is required: it names the measurement signal "
                "that is checked against the tolerances"
            )
        super().__init__(dataframe)  # Inherit and initialize Base class

        self.tolerance_column: str = tolerance_column
        self.actual_column: str = actual_column
        self.actual_uuid: str = actual_uuid
        self.event_uuid: str | None = event_uuid
        self.compare_func: Callable[[pd.Series, pd.Series], pd.Series] = compare_func
        self.time_threshold: str = time_threshold
        self.warning_threshold: float = warning_threshold
        self.tolerance_lag: str = tolerance_lag

        self.tolerance_uuid: str | None = None
        self.upper_tolerance_uuid: str | None = upper_tolerance_uuid
        self.lower_tolerance_uuid: str | None = lower_tolerance_uuid
        self.separate_tolerances: bool = True
        if upper_tolerance_uuid is None and lower_tolerance_uuid is None:
            if tolerance_uuid is None:
                raise ValueError(
                    "Provide upper_tolerance_uuid and/or lower_tolerance_uuid "
                    "(or the deprecated tolerance_uuid)"
                )
            warnings.warn(
                "ToleranceDeviationEvents(): tolerance_uuid is deprecated and will "
                "be removed in the next major release; use upper_tolerance_uuid and/or "
                "lower_tolerance_uuid (a single upper limit is "
                "upper_tolerance_uuid alone)",
                DeprecationWarning,
                stacklevel=4,
            )
            self.tolerance_uuid = tolerance_uuid
            self.separate_tolerances = False

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
            df = df.sort_values(by="systime_shifted", kind="stable")
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

    process_and_group_data_with_events = deprecated_method("detect_deviations")

    def detect_deviations(self) -> pd.DataFrame:
        """
        Processes DataFrame to apply tolerance checks, group events by time, and generate an events DataFrame.

        Enhanced with:
        - Upper/lower tolerances (two-sided or one-sided)
        - Deviation magnitude tracking (absolute and percentage)
        - Warning zones
        - Severity level classification
        - Time-lagged tolerance application

        Out-of-tolerance samples of the actual signal closer than
        ``time_threshold`` form one event, reported as its first and last
        row; an event of a single sample is reported as that one row.

        Returns:
            pd.DataFrame: A DataFrame of processed and grouped event data with enhanced metrics.
            Columns: systime, uuid (the event uuid), source_uuid (the actual
            measurement signal's uuid), the value columns (``actual_column``
            holds the measured value), is_delta, deviation_abs,
            deviation_pct and severity.
        """
        df = self.dataframe.copy()  # Inherited from Base class

        # Sort chronologically so the forward-fill below hands each measurement
        # the tolerance most recently set *before* it (a descending sort made
        # every measurement pick up the next, future tolerance instead).
        df["systime"] = pd.to_datetime(df["systime"])
        df = df.sort_values(by="systime", kind="stable")

        if self.separate_tolerances:
            df = self._process_separate_tolerances(df)
        else:
            df = self._process_single_tolerance(df)

        # Apply time lag to tolerance if specified
        if self.tolerance_lag != "0s":
            df = self._apply_tolerance_lag(df, "tolerance_value")

        # Every event row is about the actual measurement signal.
        df["source_uuid"] = self.actual_uuid
        return df

    def _process_single_tolerance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process data using the deprecated single ``tolerance_uuid``.

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

        # Group the actual signal's events that are close in time. Gaps are
        # measured on the actual signal only: rows of other uuids in the frame
        # must not split or merge its events.
        filtered_df = df[df["uuid"] == self.actual_uuid].copy()
        filtered_df["group_id"] = (
            filtered_df["systime"].diff().abs() > pd.to_timedelta(self.time_threshold)
        ).cumsum()
        events_data = []

        for group_id in filtered_df["group_id"].unique():
            group_data = filtered_df[filtered_df["group_id"] == group_id]
            if group_data.shape[0] == 1:
                # A single out-of-tolerance sample is an event too: one row.
                events_data.append(group_data)
                continue
            first_row = group_data.nsmallest(1, "systime")
            last_row = group_data.nlargest(1, "systime")
            events_data.append(pd.concat([first_row, last_row]))

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
        # Upper and lower tolerance in force at each row (forward-filled from
        # the tolerance signals). A side that is not configured stays NaN.
        sides = {
            "upper_tolerance": self.upper_tolerance_uuid,
            "lower_tolerance": self.lower_tolerance_uuid,
        }
        for col, uuid in sides.items():
            if uuid is None:
                df[col] = np.nan
                continue
            setting = (df["uuid"] == uuid) & df["is_delta"].astype(bool)
            df[col] = df[self.tolerance_column].where(setting).ffill()

        # Remove tolerance setting rows from the dataset
        configured = [uuid for uuid in sides.values() if uuid is not None]
        df = df[~df["uuid"].isin(configured)]

        # Rows before the first setting of a configured side are not checked.
        required = [col for col, uuid in sides.items() if uuid is not None]
        df = df.dropna(subset=required).copy()

        upper = df["upper_tolerance"].astype(float)
        lower = df["lower_tolerance"].astype(float)
        actual = df[self.actual_column]

        # Tolerance range: the band width when two-sided; the magnitude of
        # the single limit when one-sided.
        two_sided = (
            self.upper_tolerance_uuid is not None
            and self.lower_tolerance_uuid is not None
        )
        if two_sided:
            df["tolerance_range"] = upper - lower
        else:
            df["tolerance_range"] = upper.fillna(lower).abs()

        # Calculate deviations for all rows with actual values
        actual_mask = df["uuid"] == self.actual_uuid

        # Signed distance beyond each boundary (NaN for a missing side).
        df.loc[actual_mask, "deviation_from_upper"] = (actual - upper)[actual_mask]
        df.loc[actual_mask, "deviation_from_lower"] = (lower - actual)[actual_mask]

        # Absolute deviation: distance beyond the violated boundary, 0 inside.
        beyond = np.fmax(
            df.loc[actual_mask, "deviation_from_upper"].to_numpy(dtype=float),
            df.loc[actual_mask, "deviation_from_lower"].to_numpy(dtype=float),
        )
        df.loc[actual_mask, "deviation_abs"] = np.maximum(0.0, beyond)

        # Calculate percentage deviation based on tolerance range
        tolerance_nonzero = df.loc[actual_mask, "tolerance_range"] != 0
        df.loc[actual_mask & tolerance_nonzero, "deviation_pct"] = (
            df.loc[actual_mask & tolerance_nonzero, "deviation_abs"]
            / df.loc[actual_mask & tolerance_nonzero, "tolerance_range"]
            * 100
        )

        # Calculate severity levels (a missing side never counts as violated)
        df.loc[actual_mask, "severity"] = df.loc[actual_mask].apply(
            lambda row: self._calculate_severity(
                row["deviation_abs"],
                row["tolerance_range"],
                row["upper_tolerance"] if pd.notna(row["upper_tolerance"]) else np.inf,
                row["lower_tolerance"] if pd.notna(row["lower_tolerance"]) else -np.inf,
                row[self.actual_column],
            ),
            axis=1,
        )

        # Check if values are outside tolerance bounds (NaN compares False)
        outside_bounds = (actual > upper) | (actual < lower)

        df = df[outside_bounds]
        df["value_bool"] = True

        # Group the actual signal's events that are close in time. Gaps are
        # measured on the actual signal only: rows of other uuids in the frame
        # must not split or merge its events.
        filtered_df = df[df["uuid"] == self.actual_uuid].copy()
        filtered_df["group_id"] = (
            filtered_df["systime"].diff().abs() > pd.to_timedelta(self.time_threshold)
        ).cumsum()
        events_data = []

        for group_id in filtered_df["group_id"].unique():
            group_data = filtered_df[filtered_df["group_id"] == group_id]
            if group_data.shape[0] == 1:
                # A single out-of-tolerance sample is an event too: one row.
                events_data.append(group_data)
                continue
            first_row = group_data.nsmallest(1, "systime")
            last_row = group_data.nlargest(1, "systime")
            events_data.append(pd.concat([first_row, last_row]))

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
            "tolerance_range",
            "group_id",
            "deviation_from_upper",
            "deviation_from_lower",
        ]
        cols_to_drop = [col for col in cols_to_drop if col in events_df.columns]
        events_df = events_df.drop(cols_to_drop, axis=1)

        events_df["is_delta"] = True

        return events_df

    compute_capability_indices = deprecated_method("capability_indices")

    @keyword_only(1)
    def capability_indices(self, target_value: float | None = None) -> dict[str, float]:
        """
        Calculate process capability indices (Cp, Cpk, Pp, Ppk and Cpm).

        Process capability indices measure how well a process meets specification limits:
        - Cp = (USL - LSL) / 6σ: potential capability assuming perfect centering
        - Cpk = min(USL - mean, mean - LSL) / 3σ: accounts for centering
        - Pp / Ppk: the same formulas
        - Cpm (only when ``target_value`` is given): the Taguchi index
          Cp / sqrt(1 + ((mean - T) / σ)²), which penalizes distance from
          the target T

        σ is the overall sample standard deviation (ddof=1) of all actual
        measurements. The data carries no rational subgroups, so no
        within-subgroup (short-term) σ can be estimated: Cp/Cpk are computed
        from the overall σ and are therefore equal to Pp/Ppk.

        With a one-sided spec (only an upper or only a lower limit) Cp/Pp
        are NaN and Cpk/Ppk are the one-sided index (CPU or CPL); the
        missing limit is reported as NaN.

        Args:
            target_value: Target/nominal value for the process. If None, the
                midpoint of the tolerances (or the single limit of a
                one-sided spec) is reported as ``target`` and no ``Cpm`` is
                computed.

        Returns:
            Dictionary containing:
                - 'Cp': Process capability
                - 'Cpk': Process capability index
                - 'Pp': Process performance (equal to Cp, see above)
                - 'Ppk': Process performance index (equal to Cpk)
                - 'Cpm': Taguchi capability index; only present when
                  ``target_value`` is given (NaN for a one-sided spec)
                - 'mean': Process mean
                - 'std': Process standard deviation
                - 'usl': Upper specification limit
                - 'lsl': Lower specification limit
                - 'target': Target value used

        Note:
            - Values > 1.33 are generally considered acceptable
            - Values > 1.67 are considered good
        """
        df = self.dataframe.copy()
        df["systime"] = pd.to_datetime(df["systime"])
        df = df.sort_values(by="systime", kind="stable")

        # Get tolerance limits: the latest setting is the one currently in force.
        if not self.separate_tolerances:
            # Deprecated single tolerance: a one-sided upper limit (the default
            # compare_func flags actual >= tolerance).
            tolerance_rows = df[df["uuid"] == self.tolerance_uuid]
            if tolerance_rows.empty:
                raise ValueError(
                    f"No tolerance data found for uuid: {self.tolerance_uuid}"
                )
            usl = tolerance_rows[self.tolerance_column].iloc[-1]
            lsl = np.nan
        else:
            limits = {}
            for side, uuid in (
                ("upper", self.upper_tolerance_uuid),
                ("lower", self.lower_tolerance_uuid),
            ):
                if uuid is None:
                    limits[side] = np.nan
                    continue
                rows = df[df["uuid"] == uuid]
                if rows.empty:
                    raise ValueError(f"No {side} tolerance data found for uuid: {uuid}")
                limits[side] = rows[self.tolerance_column].iloc[-1]
            usl, lsl = limits["upper"], limits["lower"]

        # Get actual measurements
        actual_data = df[df["uuid"] == self.actual_uuid][self.actual_column].dropna()

        if actual_data.empty:
            raise ValueError(
                f"No actual measurement data found for uuid: {self.actual_uuid}"
            )

        # Calculate statistics
        process_mean = actual_data.mean()
        process_std = actual_data.std(ddof=1)  # Sample standard deviation

        # NaN (a single measurement) must not slip past the zero check.
        if not process_std > 0:
            raise ValueError(
                "Process standard deviation is zero or undefined (fewer than two "
                "measurements) - cannot calculate capability indices"
            )

        # Determine target value
        target_given = target_value is not None
        if target_value is None:
            if pd.isna(lsl):
                target_value = usl
            elif pd.isna(usl):
                target_value = lsl
            else:
                target_value = (usl + lsl) / 2

        # Calculate Cp (potential capability); undefined (NaN) when one-sided
        cp = (usl - lsl) / (6 * process_std)

        # Calculate Cpk (actual capability accounting for centering); a
        # one-sided spec gives the one-sided index (CPU or CPL).
        cpu = (usl - process_mean) / (3 * process_std)
        cpl = (process_mean - lsl) / (3 * process_std)
        cpk = cpu if pd.isna(cpl) else cpl if pd.isna(cpu) else min(cpu, cpl)

        # Without subgroups there is no short-term sigma: Pp/Ppk equal Cp/Cpk.
        pp = cp
        ppk = cpk

        result = {
            "Cp": float(cp),
            "Cpk": float(cpk),
            "Pp": float(pp),
            "Ppk": float(ppk),
            "mean": float(process_mean),
            "std": float(process_std),
            "usl": float(usl),
            "lsl": float(lsl),
            "target": float(target_value),
        }
        if target_given:
            # Taguchi index (Chan, Cheng & Spiring 1988, JQT 20(3)).
            cpm = cp / np.sqrt(1 + ((process_mean - target_value) / process_std) ** 2)
            result["Cpm"] = float(cpm)
        return result
