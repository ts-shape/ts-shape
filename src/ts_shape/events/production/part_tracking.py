"""Production tracking by part number.

Simple, practical module for daily production reporting:
- How many parts were produced?
- Production by part number and time window
- Daily summaries
"""

import logging
import pandas as pd  # type: ignore
from typing import Optional

from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class PartProductionTracking(Base):
    """Track production quantities by part number.

    Each UUID represents one signal:
    - part_id_uuid: string signal with current part number
    - counter_uuid: monotonic counter for production count

    Example usage:
        tracker = PartProductionTracking(df)

        # Hourly production by part
        hourly = tracker.production_by_part(
            part_id_uuid='part_number_signal',
            counter_uuid='counter_signal',
            window='1h'
        )

        # Daily summary
        daily = tracker.daily_production_summary(
            part_id_uuid='part_number_signal',
            counter_uuid='counter_signal'
        )
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        time_column: str = "systime",
    ) -> None:
        """Initialize part production tracker.

        Args:
            dataframe: Input DataFrame with timeseries data
            time_column: Name of timestamp column (default: 'systime')
        """
        super().__init__(dataframe, column_name=time_column)
        self.time_column = time_column

    def production_by_part(
        self,
        part_id_uuid: str,
        counter_uuid: str,
        *,
        window: str = "1h",
        value_column_part: str = "value_string",
        value_column_counter: str = "value_integer",
    ) -> pd.DataFrame:
        """Calculate production quantity per part number.

        Args:
            part_id_uuid: UUID for part number signal
            counter_uuid: UUID for production counter
            window: Time window for aggregation (e.g., '1h', '8h', '1d')
            value_column_part: Column containing part numbers
            value_column_counter: Column containing counter values

        Returns:
            DataFrame with columns:
            - window_start: Start of time window
            - part_number: Part number/ID
            - quantity: Parts produced in window
            - first_count: Counter value at window start
            - last_count: Counter value at window end

        Example:
            >>> production_by_part('part_id', 'counter', window='1h')
                window_start         part_number  quantity  first_count  last_count
            0   2024-01-01 08:00:00  PART_A       150       1000        1150
            1   2024-01-01 09:00:00  PART_A       145       1150        1295
            2   2024-01-01 10:00:00  PART_B       98        1295        1393
        """
        # Get part ID changes
        part_data = (
            self.dataframe[self.dataframe["uuid"] == part_id_uuid]
            .copy()
            .sort_values(self.time_column)
        )

        if part_data.empty:
            return pd.DataFrame(
                columns=[
                    "window_start",
                    "part_number",
                    "quantity",
                    "first_count",
                    "last_count",
                ]
            )

        part_data[self.time_column] = pd.to_datetime(part_data[self.time_column])

        # Get counter data
        counter_data = (
            self.dataframe[self.dataframe["uuid"] == counter_uuid]
            .copy()
            .sort_values(self.time_column)
        )

        if counter_data.empty:
            return pd.DataFrame(
                columns=[
                    "window_start",
                    "part_number",
                    "quantity",
                    "first_count",
                    "last_count",
                ]
            )

        counter_data[self.time_column] = pd.to_datetime(counter_data[self.time_column])

        # Merge part ID with counter data (backward fill - use most recent part)
        # Select only needed columns to avoid suffix issues in merge
        counter_subset = counter_data[
            [self.time_column, value_column_counter, "uuid"]
        ].copy()
        part_subset = part_data[[self.time_column, value_column_part]].copy()

        merged = pd.merge_asof(
            counter_subset, part_subset, on=self.time_column, direction="backward"
        )

        # Rename part column
        if value_column_part in merged.columns:
            merged = merged.rename(columns={value_column_part: "part_number"})
        else:
            # Handle case where merge added suffix
            merged = merged.rename(columns={f"{value_column_part}_y": "part_number"})

        merged = merged.dropna(subset=["part_number"])

        # Group by time window and part number
        merged = merged.set_index(self.time_column)

        results = []
        for (window_start, part_num), group in merged.groupby(
            [pd.Grouper(freq=window), "part_number"]
        ):
            if group.empty:
                continue

            first_count = group[value_column_counter].iloc[0]
            last_count = group[value_column_counter].iloc[-1]
            quantity = max(0, last_count - first_count)

            results.append(
                {
                    "window_start": window_start,
                    "part_number": part_num,
                    "quantity": quantity,
                    "first_count": first_count,
                    "last_count": last_count,
                }
            )

        return pd.DataFrame(results)

    def daily_production_summary(
        self,
        part_id_uuid: str,
        counter_uuid: str,
        *,
        value_column_part: str = "value_string",
        value_column_counter: str = "value_integer",
    ) -> pd.DataFrame:
        """Daily production summary by part number.

        Args:
            part_id_uuid: UUID for part number signal
            counter_uuid: UUID for production counter
            value_column_part: Column containing part numbers
            value_column_counter: Column containing counter values

        Returns:
            DataFrame with columns:
            - date: Production date
            - part_number: Part number/ID
            - total_quantity: Total parts produced that day
            - hours_active: Number of hours with production

        Example:
            >>> daily_production_summary('part_id', 'counter')
                date        part_number  total_quantity  hours_active
            0   2024-01-01  PART_A       1200           8
            1   2024-01-01  PART_B       850            6
            2   2024-01-02  PART_A       1150           8
        """
        hourly = self.production_by_part(
            part_id_uuid,
            counter_uuid,
            window="1h",
            value_column_part=value_column_part,
            value_column_counter=value_column_counter,
        )

        if hourly.empty:
            return pd.DataFrame(
                columns=["date", "part_number", "total_quantity", "hours_active"]
            )

        hourly["date"] = hourly["window_start"].dt.date

        daily = (
            hourly.groupby(["date", "part_number"])
            .agg({"quantity": "sum", "window_start": "count"})
            .reset_index()
        )

        daily = daily.rename(
            columns={"quantity": "total_quantity", "window_start": "hours_active"}
        )

        return daily

    def production_totals(
        self,
        part_id_uuid: str,
        counter_uuid: str,
        *,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        value_column_part: str = "value_string",
        value_column_counter: str = "value_integer",
    ) -> pd.DataFrame:
        """Total production by part number for a date range.

        Args:
            part_id_uuid: UUID for part number signal
            counter_uuid: UUID for production counter
            start_date: Start date 'YYYY-MM-DD' (optional)
            end_date: End date 'YYYY-MM-DD' (optional)
            value_column_part: Column containing part numbers
            value_column_counter: Column containing counter values

        Returns:
            DataFrame with total production per part

        Example:
            >>> production_totals('part_id', 'counter',
            ...                   start_date='2024-01-01', end_date='2024-01-07')
                part_number  total_quantity  days_produced
            0   PART_A       8450           5
            1   PART_B       6200           4
        """
        daily = self.daily_production_summary(
            part_id_uuid,
            counter_uuid,
            value_column_part=value_column_part,
            value_column_counter=value_column_counter,
        )

        if daily.empty:
            return pd.DataFrame(
                columns=["part_number", "total_quantity", "days_produced"]
            )

        # Filter by date range
        daily["date"] = pd.to_datetime(daily["date"])
        if start_date:
            daily = daily[daily["date"] >= pd.to_datetime(start_date)]
        if end_date:
            daily = daily[daily["date"] <= pd.to_datetime(end_date)]

        totals = (
            daily.groupby("part_number")
            .agg({"total_quantity": "sum", "date": "count"})
            .reset_index()
        )

        totals = totals.rename(columns={"date": "days_produced"})

        return totals
