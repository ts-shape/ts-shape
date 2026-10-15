import logging
import operator
import warnings
from collections.abc import Callable
from typing import Any, Literal, overload

import numpy as np
import pandas as pd

from ts_shape.errors import DataQualityWarning
from ts_shape.events._compat import summary_bounds, warn_dict_return
from ts_shape.events._output import drop_duplicate_events
from ts_shape.events.quality.statistical_process_control import (
    SigmaMethod,
    _check_sigma_method,
    _sigma_hat,
)
from ts_shape.utils.base import Base
from ts_shape.utils.coerce import bool_or_false
from ts_shape.utils.deprecation import (
    REMOVAL,
    caller_stacklevel,
    deprecated_method,
    keyword_only,
)

logger = logging.getLogger(__name__)


_CAPABILITY_COLUMNS = [
    "start",
    "end",
    "duration_seconds",
    "uuid",
    "source_uuid",
    "Cp",
    "Cpk",
    "Pp",
    "Ppk",
    "Cpm",
    "mean",
    "std",
    "std_within",
    "usl",
    "lsl",
    "target",
    "sample_count",
]


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

    Example:
        >>> import pandas as pd, ts_shape
        >>> act = ts_shape.make_timeseries(["act"], n_points=60, n_outliers=3)
        >>> limits = pd.DataFrame({"systime": act["systime"].min(), "uuid": ["usl", "lsl"],
        ...                        "value_double": [103.0, 97.0]})
        >>> tde = ts_shape.ToleranceDeviationEvents(
        ...     pd.concat([limits, act]), "value_double", "value_double", actual_uuid="act",
        ...     upper_tolerance_uuid="usl", lower_tolerance_uuid="lsl")
        >>> out = tde.detect_deviations()
        >>> out[["systime", "value_double", "severity"]].round({"value_double": 1})
                       systime  value_double severity
        4  2025-01-01 00:02:00         110.0    major
        33 2025-01-01 00:16:30         111.2    major
        49 2025-01-01 00:24:30         112.1    major
    """

    #: Any out-of-tolerance value is a real deviation.
    _severity_levels = {"*": {"minor": "warn"}}

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
                f"be removed in {REMOVAL}; use upper_tolerance_uuid and/or "
                "lower_tolerance_uuid (a single upper limit is "
                "upper_tolerance_uuid alone)",
                DeprecationWarning,
                stacklevel=caller_stacklevel(),
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

    def _severity_array(
        self,
        deviation_abs: np.ndarray,
        tolerance_range: np.ndarray,
        upper_limit: np.ndarray,
        lower_limit: np.ndarray,
        actual_value: np.ndarray,
    ) -> np.ndarray:
        """:meth:`_calculate_severity` for whole arrays (same rules)."""
        with np.errstate(invalid="ignore"):
            outside = (actual_value > upper_limit) | (actual_value < lower_limit)
            beyond = np.where(deviation_abs > tolerance_range * 2, "critical", "major")
            sev = np.where(
                deviation_abs <= tolerance_range * self.warning_threshold,
                "minor",
                np.where(outside, beyond, "minor"),
            )
        unknown = np.isnan(deviation_abs) | np.isnan(tolerance_range)
        return np.where(unknown, "unknown", sev).astype(object)

    @staticmethod
    def _group_event_rows(filtered_df: pd.DataFrame) -> list[pd.DataFrame]:
        """Event rows per ``group_id`` (ids are consecutive cumsum labels).

        A single sample is its own event row; otherwise the first earliest
        and the first latest sample (as ``nsmallest``/``nlargest`` pick
        them). Selected positionally for all groups at once instead of one
        full-frame filter per group.
        """
        if filtered_df.empty:
            return []
        order = pd.DataFrame(
            {
                "g": filtered_df["group_id"].to_numpy(),
                "t": filtered_df["systime"].to_numpy(),
                "p": np.arange(len(filtered_df)),
            }
        )
        size = order.groupby("g", sort=False).size()
        earliest = (
            order.sort_values(["g", "t", "p"], kind="stable")
            .groupby("g", sort=False)["p"]
            .first()
        )
        latest = (
            order.sort_values(["g", "t", "p"], ascending=[True, False, True])
            .groupby("g", sort=False)["p"]
            .first()
        )
        positions: list[int] = []
        for g, n in size.items():
            if n == 1:
                positions.append(int(earliest[g]))
            else:
                positions += [int(earliest[g]), int(latest[g])]
        return [filtered_df.iloc[positions]]

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
        # Only the actual signal and its tolerance signals take part: every
        # event row is an actual-signal row, and the tolerance in force at a
        # row is the last setting before it in time order, which rows of
        # other signals never change. Narrowing first keeps the copy, the
        # sort and the forward fill below proportional to those signals
        # instead of the whole frame. (The fill stays positional rather than
        # a merge_asof on systime: a setting stamped at the same instant but
        # listed after a measurement does not apply to it.)
        signals = [
            u
            for u in (
                self.actual_uuid,
                self.tolerance_uuid,
                self.upper_tolerance_uuid,
                self.lower_tolerance_uuid,
            )
            if u is not None
        ]
        source = self.dataframe  # Inherited from Base class
        df = source[source["uuid"].isin(signals)].copy()

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
        # Duplicated input samples would report the same row twice.
        return drop_duplicate_events(df)

    def _process_single_tolerance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process data using the deprecated single ``tolerance_uuid``.

        Args:
            df: Input DataFrame

        Returns:
            Processed DataFrame with events
        """
        # The tolerance setting rows carry the tolerance; every later row
        # uses the last one set (forward fill).
        is_setting = df["uuid"].eq(self.tolerance_uuid) & bool_or_false(df["is_delta"])
        df["tolerance_value"] = df[self.tolerance_column].where(is_setting).ffill()

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
        events_data = self._group_event_rows(filtered_df)

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
        df.loc[actual_mask, "severity"] = self._severity_array(
            df.loc[actual_mask, "deviation_abs"].to_numpy(dtype=float),
            df.loc[actual_mask, "tolerance_range"].to_numpy(dtype=float),
            df.loc[actual_mask, "upper_tolerance"].fillna(np.inf).to_numpy(float),
            df.loc[actual_mask, "lower_tolerance"].fillna(-np.inf).to_numpy(float),
            df.loc[actual_mask, self.actual_column].to_numpy(dtype=float),
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
        events_data = self._group_event_rows(filtered_df)

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

    @overload
    def capability_indices(
        self,
        target_value: float | None = ...,
        *,
        as_frame: Literal[True] = ...,
        sigma_method: SigmaMethod = ...,
    ) -> pd.DataFrame: ...

    @overload
    def capability_indices(
        self,
        target_value: float | None = ...,
        *,
        as_frame: Literal[False],
        sigma_method: SigmaMethod = ...,
    ) -> dict[str, float]: ...

    @keyword_only(1)
    def capability_indices(
        self,
        target_value: float | None = None,
        *,
        as_frame: bool = True,
        sigma_method: SigmaMethod = "moving_range",
    ) -> pd.DataFrame | dict[str, float]:
        """
        Calculate process capability indices (Cp, Cpk, Pp, Ppk and Cpm).

        Process capability indices measure how well a process meets specification limits
        (AIAG SPC manual; Montgomery, Introduction to Statistical Quality
        Control, section 8.3):
        - Cp = (USL - LSL) / 6σ_within: potential capability assuming perfect centering
        - Cpk = min(USL - mean, mean - LSL) / 3σ_within: accounts for centering
        - Pp / Ppk: the same formulas with the overall σ
        - Cpm (only when ``target_value`` is given): the Taguchi index
          (USL - LSL) / 6 sqrt(σ² + (mean - T)²) = Pp / sqrt(1 + ((mean - T) / σ)²)
          with the overall σ, which penalizes distance from the target T

        The overall σ (``std``) is the sample standard deviation (ddof=1)
        of all actual measurements. The within (short-term) σ
        (``std_within``) is, with the default ``sigma_method="moving_range"``,
        the individuals-chart estimate ``MR-bar / 1.128`` from consecutive
        measurements in time order; with ``sigma_method="std"`` it is the
        overall σ, so Cp/Cpk equal Pp/Ppk (the behavior before October 2026).

        With a one-sided spec (only an upper or only a lower limit) Cp/Pp
        are NaN and Cpk/Ppk are the one-sided index (CPU or CPL); the
        missing limit is reported as NaN.

        Measurements without spread (constant values, a single measurement or
        only missing values) give NaN indices and a
        :class:`~ts_shape.errors.DataQualityWarning`, as in
        :class:`CapabilityTrendingEvents`.

        Args:
            target_value: Target/nominal value for the process. If None, the
                midpoint of the tolerances (or the single limit of a
                one-sided spec) is reported as ``target`` and no ``Cpm`` is
                computed.
            sigma_method: Within σ for Cp/Cpk: ``"moving_range"`` (default,
                ``MR-bar / 1.128``) or ``"std"`` (the overall σ).
            as_frame: When True (default), return a one-row DataFrame (see
                below). ``as_frame=False`` returns the dictionary form; it is
                deprecated, emits a ``DeprecationWarning`` and will be removed
                in the next major release.

        Returns:
            With ``as_frame=True`` (default), a one-row summary DataFrame with
            columns start, end, duration_seconds (first to last actual
            measurement), uuid (event uuid), source_uuid (actual uuid), Cp,
            Cpk, Pp, Ppk, Cpm (always present; NaN unless ``target_value``
            is given), mean, std (overall σ), std_within (the σ behind Cp /
            Cpk), usl, lsl, target, and sample_count (the
            number of measurements the indices are computed from).

            With ``as_frame=False`` (deprecated), a dictionary containing:
                - 'Cp': Process capability
                - 'Cpk': Process capability index
                - 'Pp': Process performance
                - 'Ppk': Process performance index
                - 'Cpm': Taguchi capability index; only present when
                  ``target_value`` is given (NaN for a one-sided spec)
                - 'mean': Process mean
                - 'std': Overall process standard deviation
                - 'std_within': Within σ used for Cp / Cpk
                - 'usl': Upper specification limit
                - 'lsl': Lower specification limit
                - 'target': Target value used

        Note:
            - Values > 1.33 are generally considered acceptable
            - Values > 1.67 are considered good
        """
        _check_sigma_method(sigma_method)
        if not as_frame:
            warn_dict_return("ToleranceDeviationEvents.capability_indices")
        if as_frame and self.dataframe.empty:
            # No data at all: an empty result with the full schema, like every
            # other detector. Missing limits on real data still raise.
            return pd.DataFrame(columns=_CAPABILITY_COLUMNS)
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
        actual_rows = df[df["uuid"] == self.actual_uuid]
        actual_data = actual_rows[self.actual_column].dropna()

        if actual_rows.empty:
            raise ValueError(
                f"No actual measurement data found for uuid: {self.actual_uuid}"
            )

        # Calculate statistics
        process_mean = float(actual_data.mean()) if len(actual_data) else np.nan
        # Sample standard deviation; NaN for fewer than two measurements.
        process_std = float(actual_data.std(ddof=1)) if len(actual_data) else np.nan

        # Like CapabilityTrendingEvents: no spread (constant values, a single
        # or no measurement) gives NaN indices instead of an error.
        if not process_std > 0:
            warnings.warn(
                f"capability_indices: {len(actual_data)} measurement(s) without "
                "spread (constant, single or missing values); the capability "
                "indices are NaN",
                DataQualityWarning,
                stacklevel=caller_stacklevel(),
            )
            process_std = np.nan
        within_std = _sigma_hat(actual_data, sigma_method)
        if not within_std > 0:
            within_std = np.nan

        # Determine target value
        target_given = target_value is not None
        if target_value is None:
            if pd.isna(lsl):
                target_value = usl
            elif pd.isna(usl):
                target_value = lsl
            else:
                target_value = (usl + lsl) / 2

        def _indices(sigma: float) -> tuple[float, float]:
            # (USL - LSL) / 6 sigma is undefined (NaN) when one-sided; the
            # k-index is then the one-sided index (CPU or CPL).
            index = (usl - lsl) / (6 * sigma)
            upper = (usl - process_mean) / (3 * sigma)
            lower = (process_mean - lsl) / (3 * sigma)
            k_index = (
                upper
                if pd.isna(lower)
                else lower if pd.isna(upper) else min(upper, lower)
            )
            return index, k_index

        cp, cpk = _indices(within_std)  # within (short-term) sigma
        pp, ppk = _indices(process_std)  # overall sigma

        result: dict[str, Any] = {
            "Cp": float(cp),
            "Cpk": float(cpk),
            "Pp": float(pp),
            "Ppk": float(ppk),
            "mean": float(process_mean),
            "std": float(process_std),
            "std_within": float(within_std),
            "usl": float(usl),
            "lsl": float(lsl),
            "target": float(target_value),
        }
        if target_given:
            # Taguchi index (Chan, Cheng & Spiring 1988, JQT 20(3)).
            cpm = pp / np.sqrt(1 + ((process_mean - target_value) / process_std) ** 2)
            result["Cpm"] = float(cpm)
        if not as_frame:
            return result
        measured = (
            actual_rows.loc[actual_data.index] if len(actual_data) else actual_rows
        )
        start, end, span = summary_bounds(measured["systime"])
        row = {
            "start": start,
            "end": end,
            "duration_seconds": span,
            "uuid": self.event_uuid,
            "source_uuid": self.actual_uuid,
            **{k: result[k] for k in ("Cp", "Cpk", "Pp", "Ppk")},
            "Cpm": result.get("Cpm", float("nan")),
            **{
                k: result[k]
                for k in ("mean", "std", "std_within", "usl", "lsl", "target")
            },
            "sample_count": int(len(actual_data)),
        }
        return pd.DataFrame([row], columns=_CAPABILITY_COLUMNS)
