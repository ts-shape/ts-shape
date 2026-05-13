"""Shift-based production reporting.

Simple module for shift summaries:
- Production by shift
- Shift performance comparison
- Shift targets and actuals
"""

import logging
import pandas as pd  # type: ignore
from typing import Optional, Dict, Tuple

from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class ShiftReporting(Base):
    """Simple shift-based production reporting.

    Each UUID represents one signal:
    - counter_uuid: production counter
    - part_id_uuid: part number (optional)

    Merge keys: [date, shift] for shift-level outputs.

    Pipeline example::

        reporter = ShiftReporting(df)
        prod = reporter.shift_production('counter')
        # → merge with QualityTracking.nok_by_shift() on [date, shift]
        # → merge with DowntimeTracking.downtime_by_shift() on [date, shift]
        # → feed combined DataFrame into ShiftHandoverReport.from_shift_data()

    Example usage:
        reporter = ShiftReporting(df, shift_definitions={
            "day": ("06:00", "14:00"),
            "afternoon": ("14:00", "22:00"),
            "night": ("22:00", "06:00"),
        })

        # Production per shift
        shift_prod = reporter.shift_production(
            counter_uuid='counter_signal',
            part_id_uuid='part_number_signal'
        )

        # Compare shifts
        comparison = reporter.shift_comparison(counter_uuid='counter_signal')
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        time_column: str = "systime",
        shift_definitions: Optional[Dict[str, Tuple[str, str]]] = None,
    ) -> None:
        """Initialize shift reporter.

        Args:
            dataframe: Input DataFrame with timeseries data
            time_column: Name of timestamp column (default: 'systime')
            shift_definitions: Dictionary mapping shift names to (start, end) times
                              Default: 3-shift operation (06:00-14:00, 14:00-22:00, 22:00-06:00)

        Example shift_definitions:
            {
                "shift_1": ("06:00", "14:00"),
                "shift_2": ("14:00", "22:00"),
                "shift_3": ("22:00", "06:00"),
            }
        """
        super().__init__(dataframe, column_name=time_column)
        self.time_column = time_column

        # Default 3-shift operation
        self.shift_definitions = shift_definitions or {
            "shift_1": ("06:00", "14:00"),
            "shift_2": ("14:00", "22:00"),
            "shift_3": ("22:00", "06:00"),
        }

    def _assign_shift(self, timestamp: pd.Timestamp) -> str:
        """Assign shift based on time of day.

        Args:
            timestamp: Timestamp to classify

        Returns:
            Shift name
        """
        time = timestamp.time()

        for shift_name, (start, end) in self.shift_definitions.items():
            start_time = pd.to_datetime(start).time()
            end_time = pd.to_datetime(end).time()

            if start_time < end_time:
                # Normal shift (e.g., 06:00-14:00)
                if start_time <= time < end_time:
                    return shift_name
            else:
                # Overnight shift (e.g., 22:00-06:00)
                if time >= start_time or time < end_time:
                    return shift_name

        return "unknown"

    def shift_production(
        self,
        counter_uuid: str,
        part_id_uuid: Optional[str] = None,
        *,
        value_column_counter: str = "value_integer",
        value_column_part: str = "value_string",
        date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Production quantity per shift.

        Args:
            counter_uuid: Production counter UUID
            part_id_uuid: Part number UUID (optional, for part-specific production)
            value_column_counter: Column containing counter values
            value_column_part: Column containing part numbers
            date: Specific date in 'YYYY-MM-DD' format (optional)

        Returns:
            DataFrame with production by shift:
            - date: Production date
            - shift: Shift name
            - part_number: Part number (if part_id_uuid provided)
            - quantity: Parts produced

        Example:
            >>> shift_production('counter', part_id_uuid='part_id')
                date        shift    part_number  quantity
            0   2024-01-01  shift_1  PART_A       450
            1   2024-01-01  shift_2  PART_A       425
            2   2024-01-01  shift_3  PART_A       380
        """
        counter_data = (
            self.dataframe[self.dataframe["uuid"] == counter_uuid]
            .copy()
            .sort_values(self.time_column)
        )

        if counter_data.empty:
            cols = ["date", "shift", "quantity"]
            if part_id_uuid:
                cols.insert(2, "part_number")
            return pd.DataFrame(columns=cols)

        counter_data[self.time_column] = pd.to_datetime(counter_data[self.time_column])

        # Filter by date if specified
        if date:
            target_date = pd.to_datetime(date).date()
            counter_data = counter_data[
                counter_data[self.time_column].dt.date == target_date
            ]

        # Assign shifts
        counter_data["shift"] = counter_data[self.time_column].apply(self._assign_shift)
        counter_data["date"] = counter_data[self.time_column].dt.date

        # Add part numbers if provided
        group_cols = ["date", "shift"]
        if part_id_uuid:
            part_data = (
                self.dataframe[self.dataframe["uuid"] == part_id_uuid]
                .copy()
                .sort_values(self.time_column)
            )

            if not part_data.empty:
                part_data[self.time_column] = pd.to_datetime(
                    part_data[self.time_column]
                )

                # Select only needed columns to avoid suffix issues in merge
                # Keep only the columns we need from counter_data
                merge_cols = [self.time_column, value_column_counter, "shift", "date"]
                counter_subset = counter_data[merge_cols].copy()
                part_subset = part_data[[self.time_column, value_column_part]].copy()

                counter_data = pd.merge_asof(
                    counter_subset,
                    part_subset,
                    on=self.time_column,
                    direction="backward",
                )

                # Rename part column
                counter_data = counter_data.rename(
                    columns={value_column_part: "part_number"}
                )
                group_cols.append("part_number")

        # Calculate quantity per shift
        results = []
        for group_key, group_data in counter_data.groupby(group_cols):
            if group_data.empty:
                continue

            first_count = group_data[value_column_counter].iloc[0]
            last_count = group_data[value_column_counter].iloc[-1]
            quantity = max(0, last_count - first_count)

            if part_id_uuid and len(group_cols) == 3:
                results.append(
                    {
                        "date": group_key[0],
                        "shift": group_key[1],
                        "part_number": group_key[2],
                        "quantity": quantity,
                    }
                )
            else:
                results.append(
                    {
                        "date": group_key[0],
                        "shift": group_key[1],
                        "quantity": quantity,
                    }
                )

        return pd.DataFrame(results)

    def shift_comparison(
        self,
        counter_uuid: str,
        *,
        value_column_counter: str = "value_integer",
        days: int = 7,
    ) -> pd.DataFrame:
        """Compare shift performance over recent days.

        Args:
            counter_uuid: Production counter UUID
            value_column_counter: Column containing counter values
            days: Number of recent days to analyze (default: 7)

        Returns:
            DataFrame with shift comparison:
            - shift: Shift name
            - avg_quantity: Average production per shift
            - min_quantity: Minimum production
            - max_quantity: Maximum production
            - std_quantity: Standard deviation
            - days_count: Number of days included

        Example:
            >>> shift_comparison('counter', days=7)
                shift    avg_quantity  min_quantity  max_quantity  std_quantity  days_count
            0   shift_1  445           420           465           15.2          7
            1   shift_2  430           405           450           12.8          7
            2   shift_3  385           360           410           18.5          7
        """
        shift_prod = self.shift_production(
            counter_uuid, value_column_counter=value_column_counter
        )

        if shift_prod.empty:
            return pd.DataFrame(
                columns=[
                    "shift",
                    "avg_quantity",
                    "min_quantity",
                    "max_quantity",
                    "std_quantity",
                    "days_count",
                ]
            )

        # Filter to recent days
        shift_prod["date"] = pd.to_datetime(shift_prod["date"])
        cutoff_date = shift_prod["date"].max() - pd.Timedelta(days=days - 1)
        shift_prod = shift_prod[shift_prod["date"] >= cutoff_date]

        # Compare shifts
        comparison = (
            shift_prod.groupby("shift")["quantity"]
            .agg(
                [
                    ("avg_quantity", "mean"),
                    ("min_quantity", "min"),
                    ("max_quantity", "max"),
                    ("std_quantity", "std"),
                    ("days_count", "count"),
                ]
            )
            .reset_index()
        )

        return comparison

    def shift_targets(
        self,
        counter_uuid: str,
        targets: Dict[str, float],
        *,
        value_column_counter: str = "value_integer",
        date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Compare actual production to shift targets.

        Args:
            counter_uuid: Production counter UUID
            targets: Dictionary mapping shift names to target quantities
            value_column_counter: Column containing counter values
            date: Specific date in 'YYYY-MM-DD' format (optional)

        Returns:
            DataFrame with target comparison:
            - date: Production date
            - shift: Shift name
            - actual: Actual production
            - target: Target production
            - variance: Difference (actual - target)
            - achievement_pct: Percentage of target achieved

        Example:
            >>> shift_targets('counter', targets={'shift_1': 450, 'shift_2': 450, 'shift_3': 400})
                date        shift    actual  target  variance  achievement_pct
            0   2024-01-01  shift_1  445     450     -5        98.9
            1   2024-01-01  shift_2  465     450     15        103.3
            2   2024-01-01  shift_3  390     400     -10       97.5
        """
        shift_prod = self.shift_production(
            counter_uuid, value_column_counter=value_column_counter, date=date
        )

        if shift_prod.empty:
            return pd.DataFrame(
                columns=[
                    "date",
                    "shift",
                    "actual",
                    "target",
                    "variance",
                    "achievement_pct",
                ]
            )

        # Add targets
        shift_prod["target"] = shift_prod["shift"].map(targets)
        shift_prod = shift_prod.dropna(subset=["target"])

        # Calculate variance and achievement
        shift_prod["actual"] = shift_prod["quantity"]
        shift_prod["variance"] = shift_prod["actual"] - shift_prod["target"]
        shift_prod["achievement_pct"] = (
            shift_prod["actual"] / shift_prod["target"] * 100
        ).round(1)

        return shift_prod[
            ["date", "shift", "actual", "target", "variance", "achievement_pct"]
        ]

    def best_and_worst_shifts(
        self,
        counter_uuid: str,
        *,
        value_column_counter: str = "value_integer",
        days: int = 30,
    ) -> Dict[str, pd.DataFrame]:
        """Identify best and worst performing shifts.

        Args:
            counter_uuid: Production counter UUID
            value_column_counter: Column containing counter values
            days: Number of recent days to analyze (default: 30)

        Returns:
            Dictionary with:
            - 'best': Top 5 best shifts
            - 'worst': Top 5 worst shifts

        Example:
            >>> results = best_and_worst_shifts('counter')
            >>> results['best']
                date        shift    quantity
            0   2024-01-15  shift_2  495
            1   2024-01-18  shift_1  490
            2   2024-01-22  shift_2  485
        """
        shift_prod = self.shift_production(
            counter_uuid, value_column_counter=value_column_counter
        )

        if shift_prod.empty:
            empty_df = pd.DataFrame(columns=["date", "shift", "quantity"])
            return {"best": empty_df, "worst": empty_df}

        # Filter to recent days
        shift_prod["date"] = pd.to_datetime(shift_prod["date"])
        cutoff_date = shift_prod["date"].max() - pd.Timedelta(days=days - 1)
        shift_prod = shift_prod[shift_prod["date"] >= cutoff_date]

        # Get best and worst
        best = shift_prod.nlargest(5, "quantity")[["date", "shift", "quantity"]]
        worst = shift_prod.nsmallest(5, "quantity")[["date", "shift", "quantity"]]

        return {"best": best, "worst": worst}
