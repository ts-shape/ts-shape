import logging

import numpy as np
import pandas as pd  # type: ignore

from ts_shape.utils.base import Base
from ts_shape.utils.deprecation import (
    deprecated_class,
    deprecated_method,
    keyword_only,
    renamed_kwargs,
)

logger = logging.getLogger(__name__)


class StatisticalProcessControlEvents(Base):
    """
    Inherits from Base and applies SPC rules (Western Electric Rules) to a DataFrame for event detection.
    Processes data based on control limit UUIDs, actual value UUIDs, and generates events with an event UUID.

    ``rule_1`` .. ``rule_8`` are the eight tests of Nelson (1984), "The
    Shewhart Control Chart--Tests for Special Causes", Journal of Quality
    Technology 16(4); rules 1, 5 and 6 are the Western Electric zone rules
    (Montgomery, Introduction to Statistical Quality Control, Table 5.1).
    The center line and sigma come from the ``tolerance_uuid`` rows.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        value_column: str,
        tolerance_uuid: str,
        actual_uuid: str,
        event_uuid: str,
    ) -> None:
        """
        Initializes StatisticalProcessControlEvents with UUIDs for tolerance, actual, and event values.
        Inherits the sorted dataframe from the Base class.

        Args:
            dataframe (pd.DataFrame): The input DataFrame containing the data to be processed.
            value_column (str): The column containing the values to monitor.
            tolerance_uuid (str): UUID identifier for rows that set tolerance values.
            actual_uuid (str): UUID identifier for rows containing actual values.
            event_uuid (str): UUID to assign to generated events.
        """
        super().__init__(dataframe)  # Initialize the Base class
        self.value_column: str = value_column
        self.tolerance_uuid: str = tolerance_uuid
        self.actual_uuid: str = actual_uuid
        self.event_uuid: str = event_uuid
        if not self.dataframe.empty:
            self._validate_column(self.dataframe, value_column)
            self._validate_uuid(self.dataframe, tolerance_uuid)
            self._validate_uuid(self.dataframe, actual_uuid)

    calculate_control_limits = deprecated_method("control_limits")

    def control_limits(self) -> pd.DataFrame:
        """
        Calculate the control limits (mean ± 1σ, 2σ, 3σ) for the tolerance values.

        Returns:
            pd.DataFrame: DataFrame with control limits for each tolerance group.
        """
        df = self.dataframe[self.dataframe["uuid"] == self.tolerance_uuid]
        mean = df[self.value_column].mean()
        sigma = df[self.value_column].std()
        if not sigma > 0:
            # Fewer than two tolerance rows (sigma NaN) or a constant signal
            # (sigma 0): the limits are NaN or collapse onto the mean, so the
            # Western Electric rules silently never fire (or always fire).
            logger.warning(
                "Control limits for uuid %r are degenerate (sigma=%s from %d "
                "row(s)); rule checks will not be meaningful.",
                self.tolerance_uuid,
                sigma,
                len(df),
            )

        control_limits = {
            "mean": mean,
            "1sigma_upper": mean + sigma,
            "1sigma_lower": mean - sigma,
            "2sigma_upper": mean + 2 * sigma,
            "2sigma_lower": mean - 2 * sigma,
            "3sigma_upper": mean + 3 * sigma,
            "3sigma_lower": mean - 3 * sigma,
        }

        return pd.DataFrame([control_limits])

    calculate_dynamic_control_limits = deprecated_method("dynamic_control_limits")

    @keyword_only(1)
    @renamed_kwargs(window="window_size")
    def dynamic_control_limits(
        self, method: str = "moving_range", window_size: int = 20
    ) -> pd.DataFrame:
        """
        Calculate dynamic control limits that adapt over time.

        Args:
            method (str): Method for calculating dynamic limits. Options:
                - 'moving_range': Uses moving window statistics
                - 'ewma': Uses Exponentially Weighted Moving Average
            window_size (int): Number of rows in the moving window (the EWMA
                span for ``'ewma'``; default: 20)

        Returns:
            pd.DataFrame: DataFrame with dynamic control limits indexed by time.
                Columns: systime, mean, 1sigma_upper, 1sigma_lower,
                2sigma_upper, 2sigma_lower, 3sigma_upper, 3sigma_lower, uuid
                (the event uuid) and source_uuid (the actual signal's uuid).
        """
        df = self.dataframe[self.dataframe["uuid"] == self.actual_uuid].copy()
        df = df.sort_values(by="systime")

        if method == "moving_range":
            # Calculate rolling mean and std
            rolling_mean = (
                df[self.value_column].rolling(window=window_size, min_periods=1).mean()
            )
            rolling_std = (
                df[self.value_column].rolling(window=window_size, min_periods=1).std()
            )

            control_limits = pd.DataFrame(
                {
                    "systime": df["systime"],
                    "mean": rolling_mean,
                    "1sigma_upper": rolling_mean + rolling_std,
                    "1sigma_lower": rolling_mean - rolling_std,
                    "2sigma_upper": rolling_mean + 2 * rolling_std,
                    "2sigma_lower": rolling_mean - 2 * rolling_std,
                    "3sigma_upper": rolling_mean + 3 * rolling_std,
                    "3sigma_lower": rolling_mean - 3 * rolling_std,
                }
            )

        elif method == "ewma":
            # Calculate EWMA-based control limits
            # Convert window size to span for EWMA
            span = window_size
            ewma_mean = df[self.value_column].ewm(span=span, adjust=False).mean()

            # Calculate EWMA variance
            squared_diff = (df[self.value_column] - ewma_mean) ** 2
            ewma_var = squared_diff.ewm(span=span, adjust=False).mean()
            ewma_std = np.sqrt(ewma_var)

            control_limits = pd.DataFrame(
                {
                    "systime": df["systime"],
                    "mean": ewma_mean,
                    "1sigma_upper": ewma_mean + ewma_std,
                    "1sigma_lower": ewma_mean - ewma_std,
                    "2sigma_upper": ewma_mean + 2 * ewma_std,
                    "2sigma_lower": ewma_mean - 2 * ewma_std,
                    "3sigma_upper": ewma_mean + 3 * ewma_std,
                    "3sigma_lower": ewma_mean - 3 * ewma_std,
                }
            )
        else:
            raise ValueError(f"Unknown method: {method}. Use 'moving_range' or 'ewma'.")

        control_limits["uuid"] = self.event_uuid
        control_limits["source_uuid"] = self.actual_uuid
        return control_limits.reset_index(drop=True)

    # The rule definitions follow Nelson (1984), "The Shewhart Control
    # Chart--Tests for Special Causes", Journal of Quality Technology 16(4),
    # pp. 237-239, which extends the Western Electric zone rules (Western
    # Electric Statistical Quality Control Handbook, 1956; Montgomery,
    # Introduction to Statistical Quality Control, Table 5.1). Zones are
    # measured from the chart's center line (the tolerance mean): zone C is
    # within 1 sigma, zone B between 1 and 2 sigma, zone A between 2 and 3
    # sigma. Run lengths count *points*: "six points steadily increasing" is
    # five consecutive increases. A rule flags the point that completes the
    # pattern (the last point of every qualifying window).
    _RULE_NAMES = tuple(f"rule_{i}" for i in range(1, 9))

    def _rule_mask(
        self,
        rule: str,
        values: pd.Series,
        limits: pd.DataFrame | None,
        center: float | None = None,
    ) -> pd.Series:
        """Boolean mask of the points that complete a violation of ``rule``.

        Shared by the per-rule methods, :meth:`detect_violations` and
        :meth:`detect_rule_violations` so all paths flag the same points.

        Args:
            rule: One of ``"rule_1"`` .. ``"rule_8"``.
            values: The actual values, sorted by time.
            limits: Control limits as returned by :meth:`control_limits`
                (first row used). Only rules 2-4 work without limits.
            center: Center line for rule 2; defaults to ``limits["mean"]``,
                or to the mean of ``values`` when no limits are given.

        Returns:
            pd.Series: Boolean mask aligned with ``values``.
        """
        if rule in ("rule_2", "rule_3", "rule_4") and limits is None:
            lim: dict[str, float] = {}
        elif limits is None:
            raise ValueError(f"{rule} needs control limits")
        else:
            lim = {c: float(limits[c].values[0]) for c in limits.columns}
        if rule == "rule_1":
            # Nelson 1: one point more than 3 sigma from the center line.
            return (values > lim["3sigma_upper"]) | (values < lim["3sigma_lower"])
        if rule == "rule_2":
            # Nelson 2: nine points in a row on the same side of the center
            # line (a point exactly on the center line breaks the run).
            if center is None:
                center = lim["mean"] if lim else float(values.mean())
            above = (values > center).astype(int).rolling(window=9).sum() == 9
            below = (values < center).astype(int).rolling(window=9).sum() == 9
            return above | below
        if rule == "rule_3":
            # Nelson 3: six points in a row steadily increasing or decreasing
            # = five consecutive strictly positive (or negative) differences.
            diffs = values.diff()
            inc = (diffs > 0).astype(int).rolling(window=5).sum() == 5
            dec = (diffs < 0).astype(int).rolling(window=5).sum() == 5
            return inc | dec
        if rule == "rule_4":
            # Nelson 4: fourteen points in a row alternating up and down
            # = thirteen consecutive non-zero differences whose signs alternate.
            signs = np.sign(values.diff().to_numpy(dtype=float))
            flips = np.zeros(len(signs), dtype=int)
            if len(signs) > 1:
                flips[1:] = (signs[1:] * signs[:-1]) < 0  # NaN/0 compare False
            # 13 alternating differences = 12 consecutive sign flips.
            return pd.Series(flips, index=values.index).rolling(window=12).sum() == 12
        if rule == "rule_5":
            # Nelson 5 / Western Electric 2: two out of three points in a row
            # more than 2 sigma from the center line, on the same side
            # (zone A *or beyond*).
            up = (values > lim["2sigma_upper"]).astype(int).rolling(window=3).sum()
            lo = (values < lim["2sigma_lower"]).astype(int).rolling(window=3).sum()
            return (up >= 2) | (lo >= 2)
        if rule == "rule_6":
            # Nelson 6 / Western Electric 3: four out of five points in a row
            # more than 1 sigma from the center line, on the same side
            # (zone B *or beyond*).
            up = (values > lim["1sigma_upper"]).astype(int).rolling(window=5).sum()
            lo = (values < lim["1sigma_lower"]).astype(int).rolling(window=5).sum()
            return (up >= 4) | (lo >= 4)
        if rule == "rule_7":
            # Nelson 7: fifteen points in a row within 1 sigma (zone C),
            # either side of the center line.
            within = (values < lim["1sigma_upper"]) & (values > lim["1sigma_lower"])
            return within.astype(int).rolling(window=15).sum() == 15
        if rule == "rule_8":
            # Nelson 8: eight points in a row with none within 1 sigma (none
            # in zone C), on either side of the center line.
            outside = (values > lim["1sigma_upper"]) | (values < lim["1sigma_lower"])
            return outside.astype(int).rolling(window=8).sum() == 8
        raise ValueError(f"Unknown rule: {rule!r}")

    def _apply_rule(
        self, rule: str, df: pd.DataFrame, limits: pd.DataFrame | None
    ) -> pd.DataFrame:
        out = df.copy()
        out[rule] = self._rule_mask(rule, out[self.value_column], limits).astype(bool)
        return out[out[rule]]

    def rule_1(self, df: pd.DataFrame, limits: pd.DataFrame) -> pd.DataFrame:
        """
        Rule 1 (Nelson 1984, test 1): one point more than 3σ from the center line.

        Args:
            df: Rows of the actual signal, sorted by time. Not modified.
            limits: Control limits as returned by :meth:`control_limits`;
                only the first row is used.

        Returns:
            pd.DataFrame: The violating rows (a copy) with a boolean
                ``rule_1`` column.
        """
        return self._apply_rule("rule_1", df, limits)

    def rule_2(
        self, df: pd.DataFrame, *, limits: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """
        Rule 2 (Nelson 1984, test 2): nine points in a row on the same side of the center line.

        Args:
            df: Rows of the actual signal, sorted by time. Not modified.
            limits: Control limits as returned by :meth:`control_limits`;
                the center line is ``limits["mean"]``. When omitted, the
                mean of ``df``'s values is used as the center line.

        Returns:
            pd.DataFrame: The violating rows (a copy) with a boolean
                ``rule_2`` column (the ninth point of each run is flagged).
        """
        return self._apply_rule("rule_2", df, limits)

    def rule_3(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rule 3 (Nelson 1984, test 3): six points in a row steadily increasing or decreasing.

        Six points means five consecutive strictly positive (or strictly
        negative) differences; an unchanged value breaks the run.

        Args:
            df: Rows of the actual signal, sorted by time. Not modified.

        Returns:
            pd.DataFrame: The violating rows (a copy) with a boolean
                ``rule_3`` column (the sixth point of each run is flagged).
        """
        return self._apply_rule("rule_3", df, None)

    def rule_4(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rule 4 (Nelson 1984, test 4): fourteen points in a row alternating up and down.

        Fourteen points means thirteen consecutive non-zero differences
        whose signs alternate; an unchanged value breaks the run.

        Args:
            df: Rows of the actual signal, sorted by time. Not modified.

        Returns:
            pd.DataFrame: The violating rows (a copy) with a boolean
                ``rule_4`` column (the fourteenth point of each run is flagged).
        """
        return self._apply_rule("rule_4", df, None)

    def rule_5(self, df: pd.DataFrame, limits: pd.DataFrame) -> pd.DataFrame:
        """
        Rule 5 (Nelson 1984, test 5): two out of three points in a row more than 2σ from the center line, same side.

        Points beyond 3σ count as beyond 2σ (zone A or beyond).

        Args:
            df: Rows of the actual signal, sorted by time. Not modified.
            limits: Control limits as returned by :meth:`control_limits`;
                only the first row is used.

        Returns:
            pd.DataFrame: The violating rows (a copy) with a boolean
                ``rule_5`` column (the last point of each qualifying
                three-point window is flagged).
        """
        return self._apply_rule("rule_5", df, limits)

    def rule_6(self, df: pd.DataFrame, limits: pd.DataFrame) -> pd.DataFrame:
        """
        Rule 6 (Nelson 1984, test 6): four out of five points in a row more than 1σ from the center line, same side.

        Points beyond 2σ or 3σ count as beyond 1σ (zone B or beyond).

        Args:
            df: Rows of the actual signal, sorted by time. Not modified.
            limits: Control limits as returned by :meth:`control_limits`;
                only the first row is used.

        Returns:
            pd.DataFrame: The violating rows (a copy) with a boolean
                ``rule_6`` column (the last point of each qualifying
                five-point window is flagged).
        """
        return self._apply_rule("rule_6", df, limits)

    def rule_7(self, df: pd.DataFrame, limits: pd.DataFrame) -> pd.DataFrame:
        """
        Rule 7 (Nelson 1984, test 7): fifteen points in a row within 1σ of the center line (either side).

        Args:
            df: Rows of the actual signal, sorted by time. Not modified.
            limits: Control limits as returned by :meth:`control_limits`;
                only the first row is used.

        Returns:
            pd.DataFrame: The violating rows (a copy) with a boolean
                ``rule_7`` column.
        """
        return self._apply_rule("rule_7", df, limits)

    def rule_8(self, df: pd.DataFrame, limits: pd.DataFrame) -> pd.DataFrame:
        """
        Rule 8 (Nelson 1984, test 8): eight points in a row with none within 1σ of the center line.

        The points may lie on either side of the center line; what matters
        is that none falls in zone C (within 1σ).

        Args:
            df: Rows of the actual signal, sorted by time. Not modified.
            limits: Control limits as returned by :meth:`control_limits`;
                only the first row is used.

        Returns:
            pd.DataFrame: The violating rows (a copy) with a boolean
                ``rule_8`` column.
        """
        return self._apply_rule("rule_8", df, limits)

    def _calculate_rule_2_7_8_optimized(
        self, df: pd.DataFrame, limits: pd.DataFrame
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Masks for rules 2, 7 and 8 (kept for backward compatibility).

        Args:
            df (pd.DataFrame): DataFrame with values to analyze
            limits (pd.DataFrame): Control limits

        Returns:
            Tuple[pd.Series, pd.Series, pd.Series]: Boolean series for rule_2, rule_7, rule_8 violations
        """
        values = df[self.value_column]
        return (
            self._rule_mask("rule_2", values, limits),
            self._rule_mask("rule_7", values, limits),
            self._rule_mask("rule_8", values, limits),
        )

    apply_rules_vectorized = deprecated_method("detect_rule_violations")

    @keyword_only(1)
    def detect_rule_violations(
        self, selected_rules: list[str] | None = None
    ) -> pd.DataFrame:
        """
        Applies SPC rules using vectorized operations with optimized multi-rule processing.
        Processes multiple rules in fewer passes through the data for better performance.

        Args:
            selected_rules (Optional[List[str]]): List of rule names to apply.
                If None, applies all rules.

        Returns:
            pd.DataFrame: DataFrame with rule violations. Columns: systime,
                value, rule, severity, uuid (the event uuid) and source_uuid
                (the actual signal's uuid).
        """
        df = self.dataframe[self.dataframe["uuid"] == self.actual_uuid].copy()
        df["systime"] = pd.to_datetime(df["systime"])
        df = df.sort_values(by="systime").reset_index(drop=True)

        limits = self.control_limits()

        if selected_rules is None:
            selected_rules = [
                "rule_1",
                "rule_2",
                "rule_3",
                "rule_4",
                "rule_5",
                "rule_6",
                "rule_7",
                "rule_8",
            ]

        rule_severity = {
            "rule_1": "critical",
            "rule_2": "medium",
            "rule_3": "medium",
            "rule_4": "medium",
            "rule_5": "high",
            "rule_6": "medium",
            "rule_7": "low",
            "rule_8": "low",
        }
        values = df[self.value_column]
        violations = []
        for rule in self._RULE_NAMES:
            if rule not in selected_rules:
                continue
            mask = self._rule_mask(rule, values, limits)
            for idx in df[mask].index:
                violations.append(
                    {
                        "systime": df.loc[idx, "systime"],
                        "value": df.loc[idx, self.value_column],
                        "rule": rule,
                        "severity": rule_severity[rule],
                    }
                )

        # Create violations DataFrame
        if violations:
            violations_df = pd.DataFrame(violations)
            violations_df["uuid"] = self.event_uuid
            violations_df["source_uuid"] = self.actual_uuid
            return violations_df.drop_duplicates()
        else:
            return pd.DataFrame(
                columns=["systime", "value", "rule", "severity", "uuid", "source_uuid"]
            )

    @keyword_only(1)
    def detect_cusum_shifts(
        self, target: float | None = None, k: float = 0.5, h: float = 5.0
    ) -> pd.DataFrame:
        """
        Detect process shifts using CUSUM (Cumulative Sum) control chart.

        CUSUM charts are effective at detecting small shifts in the process mean
        and are more sensitive than traditional Shewhart control charts.

        Args:
            target (Optional[float]): Target mean value. If None, uses the mean of tolerance data.
            k (float): Reference value (slack parameter), typically 0.5 to 1.0 times sigma.
                Smaller k detects smaller shifts. Default: 0.5
            h (float): Decision interval (threshold). Typical values are 4-5.
                Smaller h gives faster detection but more false alarms. Default: 5.0

        Returns:
            pd.DataFrame: DataFrame with CUSUM statistics and detected shifts.
                Columns: systime, value_column, cusum_high, cusum_low,
                shift_direction, severity, uuid (the event uuid) and
                source_uuid (the actual signal's uuid).
        """
        df = self.dataframe[self.dataframe["uuid"] == self.actual_uuid].copy()
        df = df.sort_values(by="systime").reset_index(drop=True)

        # Determine target and sigma
        if target is None:
            tolerance_df = self.dataframe[self.dataframe["uuid"] == self.tolerance_uuid]
            target = tolerance_df[self.value_column].mean()

        tolerance_df = self.dataframe[self.dataframe["uuid"] == self.tolerance_uuid]
        sigma = tolerance_df[self.value_column].std()
        result_cols = [
            "systime",
            self.value_column,
            "cusum_high",
            "cusum_low",
            "shift_direction",
            "severity",
            "uuid",
            "source_uuid",
        ]
        if not sigma > 0:
            # sigma NaN (fewer than two rows) or 0 (constant): k*sigma and the
            # decision interval h*sigma are 0/NaN, which flags every deviation
            # (or none). Report nothing rather than a wall of false shifts.
            logger.warning(
                "CUSUM skipped for uuid %r: tolerance sigma is %s; at least two "
                "non-constant tolerance values are needed.",
                self.tolerance_uuid,
                sigma,
            )
            return pd.DataFrame(columns=result_cols)

        # Initialize CUSUM statistics
        cusum_high = np.zeros(len(df))
        cusum_low = np.zeros(len(df))

        # Calculate CUSUM
        for i in range(len(df)):
            value = df.loc[i, self.value_column]

            if i == 0:
                cusum_high[i] = max(0, value - target - k * sigma)
                cusum_low[i] = max(0, target - value - k * sigma)
            else:
                cusum_high[i] = max(0, cusum_high[i - 1] + value - target - k * sigma)
                cusum_low[i] = max(0, cusum_low[i - 1] + target - value - k * sigma)

        # Add CUSUM statistics to dataframe
        df["cusum_high"] = cusum_high
        df["cusum_low"] = cusum_low

        # Detect shifts (when CUSUM exceeds threshold h*sigma)
        threshold = h * sigma
        df["shift_detected"] = (cusum_high > threshold) | (cusum_low > threshold)
        df["shift_direction"] = np.where(
            cusum_high > threshold,
            "upward",
            np.where(cusum_low > threshold, "downward", "none"),
        )
        df["severity"] = np.where(
            df["shift_detected"],
            np.where(
                (cusum_high > 2 * threshold) | (cusum_low > 2 * threshold),
                "critical",
                "high",
            ),
            "none",
        )

        # Add event UUID to detected shifts
        df.loc[df["shift_detected"], "uuid"] = self.event_uuid
        df["source_uuid"] = self.actual_uuid

        # Return only rows with detected shifts
        result = df[df["shift_detected"]].copy()
        return result[result_cols]

    def interpret_violations(self, violations_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add human-readable interpretations to rule violations.

        Args:
            violations_df (pd.DataFrame): DataFrame with rule violations
                (output from detect_rule_violations or process methods)

        Returns:
            pd.DataFrame: Enhanced DataFrame with interpretation and recommendation columns.
        """
        # Rule interpretations
        rule_interpretations = {
            "rule_1": {
                "interpretation": "One or more points beyond 3-sigma control limits",
                "meaning": "Indicates a special cause - an unusual event or significant process change",
                "recommendation": "Investigate immediately for assignable causes such as equipment failure, "
                "operator error, or material defects",
                "default_severity": "critical",
            },
            "rule_2": {
                "interpretation": "Nine consecutive points on one side of the center line",
                "meaning": "Process mean has shifted - indicates a sustained change in process level",
                "recommendation": "Check for systematic changes in materials, methods, equipment settings, "
                "or environmental conditions",
                "default_severity": "medium",
            },
            "rule_3": {
                "interpretation": "Six consecutive points steadily increasing or decreasing",
                "meaning": "Indicates a trend - gradual systematic change in the process",
                "recommendation": "Look for tool wear, temperature drift, operator fatigue, "
                "or gradual equipment degradation",
                "default_severity": "medium",
            },
            "rule_4": {
                "interpretation": "Fourteen consecutive points alternating up and down",
                "meaning": "Indicates systematic oscillation - two alternating causes affecting the process",
                "recommendation": "Check for alternating operators, materials from two sources, "
                "or temperature cycling effects",
                "default_severity": "medium",
            },
            "rule_5": {
                "interpretation": "Two out of three consecutive points beyond 2-sigma "
                "limits on the same side",
                "meaning": "Process variation has increased or mean is shifting",
                "recommendation": "Monitor closely and prepare to investigate. May indicate the start "
                "of a larger problem",
                "default_severity": "high",
            },
            "rule_6": {
                "interpretation": "Four out of five consecutive points beyond 1-sigma "
                "limits on the same side",
                "meaning": "Process variation or mean has likely changed",
                "recommendation": "Check for changes in process inputs or measurement system accuracy",
                "default_severity": "medium",
            },
            "rule_7": {
                "interpretation": "Fifteen consecutive points within 1-sigma of center line",
                "meaning": "Unusually low variation - may indicate stratification or measurement issues",
                "recommendation": "Verify measurement system accuracy and check if data is being "
                "manipulated or averaged incorrectly",
                "default_severity": "low",
            },
            "rule_8": {
                "interpretation": "Eight consecutive points with none within 1-sigma "
                "of the center line (either side)",
                "meaning": "Process variation may be higher than expected",
                "recommendation": "Review process capability and consider if control limits need recalculation",
                "default_severity": "low",
            },
        }

        # Add interpretations to violations
        result = violations_df.copy()

        # Ensure severity column exists
        if "severity" not in result.columns:
            result["severity"] = result["rule"].map(
                lambda r: rule_interpretations.get(r, {}).get(
                    "default_severity", "medium"
                )
            )

        # Add interpretation columns
        result["interpretation"] = result["rule"].map(
            lambda r: rule_interpretations.get(r, {}).get(
                "interpretation", "Unknown rule"
            )
        )
        result["meaning"] = result["rule"].map(
            lambda r: rule_interpretations.get(r, {}).get(
                "meaning", "No description available"
            )
        )
        result["recommendation"] = result["rule"].map(
            lambda r: rule_interpretations.get(r, {}).get(
                "recommendation", "Review process documentation"
            )
        )

        return result

    process = deprecated_method("detect_violations")

    @keyword_only(1)
    def detect_violations(
        self, selected_rules: list[str] | None = None, include_severity: bool = False
    ) -> pd.DataFrame:
        """
        Applies the selected SPC rules and generates a DataFrame of events where any rules are violated.

        Args:
            selected_rules (Optional[List[str]]): List of rule names (e.g., ['rule_1', 'rule_3']) to apply.
            include_severity (bool): If True, includes severity and rule information in output.
                Default: False (maintains backward compatibility)

        Returns:
            pd.DataFrame: DataFrame with rule violations and detected events.
                If include_severity=False: columns are [systime, value_column, uuid, source_uuid]
                If include_severity=True: columns include [systime, value_column, uuid,
                triggered_rule, severity, source_uuid]. ``source_uuid`` is the actual
                signal's uuid.
        """
        df = self.dataframe[self.dataframe["uuid"] == self.actual_uuid].copy()
        df["systime"] = pd.to_datetime(df["systime"])
        df = df.sort_values(by="systime")

        limits = self.control_limits()

        # Dictionary of rule functions
        rules = {
            "rule_1": lambda df: self.rule_1(df, limits),
            "rule_2": lambda df: self.rule_2(df, limits=limits),
            "rule_3": lambda df: self.rule_3(df),
            "rule_4": lambda df: self.rule_4(df),
            "rule_5": lambda df: self.rule_5(df, limits),
            "rule_6": lambda df: self.rule_6(df, limits),
            "rule_7": lambda df: self.rule_7(df, limits),
            "rule_8": lambda df: self.rule_8(df, limits),
        }

        # Severity mapping for rules
        rule_severity = {
            "rule_1": "critical",
            "rule_2": "medium",
            "rule_3": "medium",
            "rule_4": "medium",
            "rule_5": "high",
            "rule_6": "medium",
            "rule_7": "low",
            "rule_8": "low",
        }

        # If no specific rules are provided, use all rules
        if selected_rules is None:
            selected_rules = list(rules.keys())

        # Apply selected rules and track which rule triggered each event
        all_events = []
        for rule_name in selected_rules:
            if rule_name in rules:
                rule_events = rules[rule_name](df)
                if not rule_events.empty:
                    rule_events["triggered_rule"] = rule_name
                    rule_events["severity"] = rule_severity.get(rule_name, "medium")
                    all_events.append(rule_events)

        if all_events:
            events = pd.concat(all_events).drop_duplicates(
                subset=["systime", self.value_column]
            )
        else:
            events = pd.DataFrame(columns=["systime", self.value_column])

        # Add the event UUID to the detected events
        events["uuid"] = self.event_uuid
        events["source_uuid"] = self.actual_uuid

        # Return appropriate columns based on include_severity flag
        if include_severity and "triggered_rule" in events.columns:
            return events[
                [
                    "systime",
                    self.value_column,
                    "uuid",
                    "triggered_rule",
                    "severity",
                    "source_uuid",
                ]
            ].drop_duplicates()
        else:
            # Backward compatible output
            return events[
                ["systime", self.value_column, "uuid", "source_uuid"]
            ].drop_duplicates()


#: Deprecated name, removed in 1.0.
StatisticalProcessControlRuleBased = deprecated_class(
    StatisticalProcessControlEvents, "StatisticalProcessControlRuleBased"
)
