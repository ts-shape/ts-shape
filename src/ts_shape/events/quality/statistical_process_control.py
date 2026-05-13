import logging
import pandas as pd  # type: ignore
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple
from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class StatisticalProcessControlRuleBased(Base):
    """
    Inherits from Base and applies SPC rules (Western Electric Rules) to a DataFrame for event detection.
    Processes data based on control limit UUIDs, actual value UUIDs, and generates events with an event UUID.
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
        Initializes the SPCMonitor with UUIDs for tolerance, actual, and event values.
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

    def calculate_control_limits(self) -> pd.DataFrame:
        """
        Calculate the control limits (mean ± 1σ, 2σ, 3σ) for the tolerance values.

        Returns:
            pd.DataFrame: DataFrame with control limits for each tolerance group.
        """
        df = self.dataframe[self.dataframe["uuid"] == self.tolerance_uuid]
        mean = df[self.value_column].mean()
        sigma = df[self.value_column].std()

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

    def calculate_dynamic_control_limits(
        self, method: str = "moving_range", window: int = 20
    ) -> pd.DataFrame:
        """
        Calculate dynamic control limits that adapt over time.

        Args:
            method (str): Method for calculating dynamic limits. Options:
                - 'moving_range': Uses moving window statistics
                - 'ewma': Uses Exponentially Weighted Moving Average
            window (int): Window size for moving calculations (default: 20)

        Returns:
            pd.DataFrame: DataFrame with dynamic control limits indexed by time.
        """
        df = self.dataframe[self.dataframe["uuid"] == self.actual_uuid].copy()
        df = df.sort_values(by="systime")

        if method == "moving_range":
            # Calculate rolling mean and std
            rolling_mean = (
                df[self.value_column].rolling(window=window, min_periods=1).mean()
            )
            rolling_std = (
                df[self.value_column].rolling(window=window, min_periods=1).std()
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
            # Convert window to span for EWMA
            span = window
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

        return control_limits.reset_index(drop=True)

    def rule_1(self, df: pd.DataFrame, limits: pd.DataFrame) -> pd.DataFrame:
        """
        Rule 1: One point beyond the 3σ control limits.

        Returns:
            pd.DataFrame: Filtered DataFrame with rule violations.
        """
        df["rule_1"] = (df[self.value_column] > limits["3sigma_upper"].values[0]) | (
            df[self.value_column] < limits["3sigma_lower"].values[0]
        )
        return df[df["rule_1"]]

    def rule_2(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rule 2: Nine consecutive points on one side of the mean.

        Returns:
            pd.DataFrame: Filtered DataFrame with rule violations.
        """
        mean = df[self.value_column].mean()
        df["above_mean"] = df[self.value_column] > mean
        df["below_mean"] = df[self.value_column] < mean
        df["rule_2"] = (df["above_mean"].rolling(window=9).sum() == 9) | (
            df["below_mean"].rolling(window=9).sum() == 9
        )
        return df[df["rule_2"]]

    def rule_3(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rule 3: Six consecutive points steadily increasing or decreasing.

        Returns:
            pd.DataFrame: Filtered DataFrame with rule violations.
        """
        df["increasing"] = df[self.value_column].diff().gt(0)
        df["decreasing"] = df[self.value_column].diff().lt(0)
        df["rule_3"] = (df["increasing"].rolling(window=6).sum() == 6) | (
            df["decreasing"].rolling(window=6).sum() == 6
        )
        return df[df["rule_3"]]

    def rule_4(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rule 4: Fourteen consecutive points alternating up and down.

        Returns:
            pd.DataFrame: Filtered DataFrame with rule violations.
        """
        df["alternating"] = np.sign(df[self.value_column].diff())
        df["rule_4"] = (
            df["alternating"]
            .rolling(window=14)
            .apply(lambda x: (x != x.shift()).sum() == 13, raw=True)
        )
        return df[df["rule_4"]]

    def rule_5(self, df: pd.DataFrame, limits: pd.DataFrame) -> pd.DataFrame:
        """
        Rule 5: Two out of three consecutive points near the control limit (beyond 2σ but within 3σ).

        Returns:
            pd.DataFrame: Filtered DataFrame with rule violations.
        """
        values = df[self.value_column]
        beyond_2sigma = (
            (values > limits["2sigma_upper"].values[0])
            & (values < limits["3sigma_upper"].values[0])
        ) | (
            (values < limits["2sigma_lower"].values[0])
            & (values > limits["3sigma_lower"].values[0])
        )
        df["rule_5"] = beyond_2sigma.astype(int).rolling(window=3).sum() >= 2
        return df[df["rule_5"]]

    def rule_6(self, df: pd.DataFrame, limits: pd.DataFrame) -> pd.DataFrame:
        """
        Rule 6: Four out of five consecutive points near the control limit (beyond 1σ but within 2σ).

        Returns:
            pd.DataFrame: Filtered DataFrame with rule violations.
        """
        values = df[self.value_column]
        beyond_1sigma = (
            (values > limits["1sigma_upper"].values[0])
            & (values < limits["2sigma_upper"].values[0])
        ) | (
            (values < limits["1sigma_lower"].values[0])
            & (values > limits["2sigma_lower"].values[0])
        )
        df["rule_6"] = beyond_1sigma.astype(int).rolling(window=5).sum() >= 4
        return df[df["rule_6"]]

    def rule_7(self, df: pd.DataFrame, limits: pd.DataFrame) -> pd.DataFrame:
        """
        Rule 7: Fifteen consecutive points within 1σ of the centerline.

        Returns:
            pd.DataFrame: Filtered DataFrame with rule violations.
        """
        values = df[self.value_column]
        within_1sigma = (values < limits["1sigma_upper"].values[0]) & (
            values > limits["1sigma_lower"].values[0]
        )
        df["rule_7"] = within_1sigma.astype(int).rolling(window=15).sum() == 15
        return df[df["rule_7"]]

    def rule_8(self, df: pd.DataFrame, limits: pd.DataFrame) -> pd.DataFrame:
        """
        Rule 8: Eight consecutive points on both sides of the mean within 1σ.

        Returns:
            pd.DataFrame: Filtered DataFrame with rule violations.
        """
        values = df[self.value_column]
        within_1sigma = (values < limits["1sigma_upper"].values[0]) & (
            values > limits["1sigma_lower"].values[0]
        )
        df["rule_8"] = within_1sigma.astype(int).rolling(window=8).sum() == 8
        return df[df["rule_8"]]

    def _calculate_rule_2_7_8_optimized(
        self, df: pd.DataFrame, limits: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Optimized calculation for rules 2, 7, and 8 that share common computations.
        Reduces multiple passes through the data by computing related patterns together.

        Args:
            df (pd.DataFrame): DataFrame with values to analyze
            limits (pd.DataFrame): Control limits

        Returns:
            Tuple[pd.Series, pd.Series, pd.Series]: Boolean series for rule_2, rule_7, rule_8 violations
        """
        values = df[self.value_column]
        mean = values.mean()
        upper_1sigma = limits["1sigma_upper"].values[0]
        lower_1sigma = limits["1sigma_lower"].values[0]

        # Shared computation: determine position relative to mean and sigma bands
        above_mean = values > mean
        below_mean = values < mean
        within_1sigma = (values < upper_1sigma) & (values > lower_1sigma)

        # Rule 2: Nine consecutive points on one side of the mean
        rule_2 = (above_mean.rolling(window=9).sum() == 9) | (
            below_mean.rolling(window=9).sum() == 9
        )

        # Rule 7: Fifteen consecutive points within 1σ
        rule_7 = within_1sigma.rolling(window=15).sum() == 15

        # Rule 8: Eight consecutive points within 1σ
        rule_8 = within_1sigma.rolling(window=8).sum() == 8

        return rule_2, rule_7, rule_8

    def apply_rules_vectorized(
        self, selected_rules: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Applies SPC rules using vectorized operations with optimized multi-rule processing.
        Processes multiple rules in fewer passes through the data for better performance.

        Args:
            selected_rules (Optional[List[str]]): List of rule names to apply.
                If None, applies all rules.

        Returns:
            pd.DataFrame: DataFrame with rule violations, including rule name and severity.
        """
        df = self.dataframe[self.dataframe["uuid"] == self.actual_uuid].copy()
        df["systime"] = pd.to_datetime(df["systime"])
        df = df.sort_values(by="systime").reset_index(drop=True)

        limits = self.calculate_control_limits()

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

        # Pre-compute values used across multiple rules
        values = df[self.value_column]
        mean = values.mean()
        upper_3sigma = limits["3sigma_upper"].values[0]
        lower_3sigma = limits["3sigma_lower"].values[0]
        upper_2sigma = limits["2sigma_upper"].values[0]
        lower_2sigma = limits["2sigma_lower"].values[0]
        upper_1sigma = limits["1sigma_upper"].values[0]
        lower_1sigma = limits["1sigma_lower"].values[0]

        # Initialize results
        violations = []

        # Rule 1: One point beyond 3σ
        if "rule_1" in selected_rules:
            rule_1_mask = (values > upper_3sigma) | (values < lower_3sigma)
            if rule_1_mask.any():
                for idx in df[rule_1_mask].index:
                    violations.append(
                        {
                            "systime": df.loc[idx, "systime"],
                            "value": df.loc[idx, self.value_column],
                            "rule": "rule_1",
                            "severity": "critical",
                        }
                    )

        # Optimized rules 2, 7, 8 (processed together)
        if any(r in selected_rules for r in ["rule_2", "rule_7", "rule_8"]):
            rule_2_mask, rule_7_mask, rule_8_mask = (
                self._calculate_rule_2_7_8_optimized(df, limits)
            )

            if "rule_2" in selected_rules and rule_2_mask.any():
                for idx in df[rule_2_mask].index:
                    violations.append(
                        {
                            "systime": df.loc[idx, "systime"],
                            "value": df.loc[idx, self.value_column],
                            "rule": "rule_2",
                            "severity": "medium",
                        }
                    )

            if "rule_7" in selected_rules and rule_7_mask.any():
                for idx in df[rule_7_mask].index:
                    violations.append(
                        {
                            "systime": df.loc[idx, "systime"],
                            "value": df.loc[idx, self.value_column],
                            "rule": "rule_7",
                            "severity": "low",
                        }
                    )

            if "rule_8" in selected_rules and rule_8_mask.any():
                for idx in df[rule_8_mask].index:
                    violations.append(
                        {
                            "systime": df.loc[idx, "systime"],
                            "value": df.loc[idx, self.value_column],
                            "rule": "rule_8",
                            "severity": "low",
                        }
                    )

        # Rule 3: Six consecutive points steadily increasing or decreasing
        if "rule_3" in selected_rules:
            diffs = values.diff()
            increasing = diffs > 0
            decreasing = diffs < 0
            rule_3_mask = (increasing.rolling(window=6).sum() == 6) | (
                decreasing.rolling(window=6).sum() == 6
            )
            if rule_3_mask.any():
                for idx in df[rule_3_mask].index:
                    violations.append(
                        {
                            "systime": df.loc[idx, "systime"],
                            "value": df.loc[idx, self.value_column],
                            "rule": "rule_3",
                            "severity": "medium",
                        }
                    )

        # Rule 4: Fourteen consecutive points alternating
        if "rule_4" in selected_rules:
            alternating = values.diff().apply(np.sign)
            rule_4_mask = (
                alternating.rolling(window=14).apply(
                    lambda x: (x != x.shift()).sum() == 13 if len(x) == 14 else False,
                    raw=True,
                )
                == 1
            )
            if rule_4_mask.any():
                for idx in df[rule_4_mask].index:
                    violations.append(
                        {
                            "systime": df.loc[idx, "systime"],
                            "value": df.loc[idx, self.value_column],
                            "rule": "rule_4",
                            "severity": "medium",
                        }
                    )

        # Rule 5: Two out of three beyond 2σ
        if "rule_5" in selected_rules:
            beyond_2sigma = ((values > upper_2sigma) & (values < upper_3sigma)) | (
                (values < lower_2sigma) & (values > lower_3sigma)
            )
            rule_5_mask = beyond_2sigma.rolling(window=3).sum() >= 2
            if rule_5_mask.any():
                for idx in df[rule_5_mask].index:
                    violations.append(
                        {
                            "systime": df.loc[idx, "systime"],
                            "value": df.loc[idx, self.value_column],
                            "rule": "rule_5",
                            "severity": "high",
                        }
                    )

        # Rule 6: Four out of five beyond 1σ
        if "rule_6" in selected_rules:
            beyond_1sigma = ((values > upper_1sigma) & (values < upper_2sigma)) | (
                (values < lower_1sigma) & (values > lower_2sigma)
            )
            rule_6_mask = beyond_1sigma.rolling(window=5).sum() >= 4
            if rule_6_mask.any():
                for idx in df[rule_6_mask].index:
                    violations.append(
                        {
                            "systime": df.loc[idx, "systime"],
                            "value": df.loc[idx, self.value_column],
                            "rule": "rule_6",
                            "severity": "medium",
                        }
                    )

        # Create violations DataFrame
        if violations:
            violations_df = pd.DataFrame(violations)
            violations_df["uuid"] = self.event_uuid
            return violations_df.drop_duplicates()
        else:
            return pd.DataFrame(
                columns=["systime", "value", "rule", "severity", "uuid"]
            )

    def detect_cusum_shifts(
        self, target: Optional[float] = None, k: float = 0.5, h: float = 5.0
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
                Columns: systime, value, cusum_high, cusum_low, shift_detected, shift_direction
        """
        df = self.dataframe[self.dataframe["uuid"] == self.actual_uuid].copy()
        df = df.sort_values(by="systime").reset_index(drop=True)

        # Determine target and sigma
        if target is None:
            tolerance_df = self.dataframe[self.dataframe["uuid"] == self.tolerance_uuid]
            target = tolerance_df[self.value_column].mean()

        tolerance_df = self.dataframe[self.dataframe["uuid"] == self.tolerance_uuid]
        sigma = tolerance_df[self.value_column].std()

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

        # Return only rows with detected shifts
        result = df[df["shift_detected"]].copy()
        result = result[
            [
                "systime",
                self.value_column,
                "cusum_high",
                "cusum_low",
                "shift_direction",
                "severity",
                "uuid",
            ]
        ]

        return result

    def interpret_violations(self, violations_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add human-readable interpretations to rule violations.

        Args:
            violations_df (pd.DataFrame): DataFrame with rule violations
                (output from apply_rules_vectorized or process methods)

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
                "interpretation": "Two out of three consecutive points beyond 2-sigma limits",
                "meaning": "Process variation has increased or mean is shifting",
                "recommendation": "Monitor closely and prepare to investigate. May indicate the start "
                "of a larger problem",
                "default_severity": "high",
            },
            "rule_6": {
                "interpretation": "Four out of five consecutive points beyond 1-sigma limits",
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
                "interpretation": "Eight consecutive points beyond 1-sigma on both sides",
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

    def process(
        self, selected_rules: Optional[List[str]] = None, include_severity: bool = False
    ) -> pd.DataFrame:
        """
        Applies the selected SPC rules and generates a DataFrame of events where any rules are violated.

        Args:
            selected_rules (Optional[List[str]]): List of rule names (e.g., ['rule_1', 'rule_3']) to apply.
            include_severity (bool): If True, includes severity and rule information in output.
                Default: False (maintains backward compatibility)

        Returns:
            pd.DataFrame: DataFrame with rule violations and detected events.
                If include_severity=False: columns are [systime, value_column, uuid]
                If include_severity=True: columns include [systime, value_column, uuid, rule, severity]
        """
        df = self.dataframe[self.dataframe["uuid"] == self.actual_uuid].copy()
        df["systime"] = pd.to_datetime(df["systime"])
        df = df.sort_values(by="systime")

        limits = self.calculate_control_limits()

        # Dictionary of rule functions
        rules = {
            "rule_1": lambda df: self.rule_1(df, limits),
            "rule_2": lambda df: self.rule_2(df),
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
                rule_events = rules[rule_name](df.copy())
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

        # Return appropriate columns based on include_severity flag
        if include_severity and "triggered_rule" in events.columns:
            return events[
                ["systime", self.value_column, "uuid", "triggered_rule", "severity"]
            ].drop_duplicates()
        else:
            # Backward compatible output
            return events[["systime", self.value_column, "uuid"]].drop_duplicates()
