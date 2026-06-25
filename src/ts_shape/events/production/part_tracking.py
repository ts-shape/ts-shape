"""Production tracking by part number.

Simple, practical module for daily production reporting:
- How many parts were produced?
- Production by part number and time window
- Daily summaries

Counters are assumed to be monotonically increasing during normal
operation. Many real-world counters reset back to zero (or to a lower
value) - for example at a shift change, a part change, or a controller
restart. A naive ``last - first`` calculation silently loses all of the
production that happened in a window where a reset occurred. Pass
``handle_resets=True`` to account for these resets, and use
:meth:`PartProductionTracking.detect_resets` to inspect when they happened.
"""

import logging
import pandas as pd  # type: ignore

from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class PartProductionTracking(Base):
    """Track production quantities by part number.

    Each UUID represents one signal:
    - part_id_uuid: string signal with current part number
    - counter_uuid: monotonic counter for production count

    The counter is expected to increase as parts are produced. If the
    counter is reset during operation (back to zero or a lower value),
    enable ``handle_resets`` so the reset is treated as new production
    instead of being discarded.

    Example usage:
        tracker = PartProductionTracking(df)

        # Hourly production by part, accounting for counter resets
        hourly = tracker.production_by_part(
            part_id_uuid='part_number_signal',
            counter_uuid='counter_signal',
            window='1h',
            handle_resets=True,
        )

        # Daily summary
        daily = tracker.daily_production_summary(
            part_id_uuid='part_number_signal',
            counter_uuid='counter_signal',
            handle_resets=True,
        )

        # When did the counter reset?
        resets = tracker.detect_resets(
            part_id_uuid='part_number_signal',
            counter_uuid='counter_signal',
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _merge_part_counter(
        self,
        part_id_uuid: str,
        counter_uuid: str,
        value_column_part: str,
        value_column_counter: str,
    ) -> pd.DataFrame | None:
        """Merge counter readings with the active part number.

        Returns a time-sorted DataFrame with the counter value and a
        ``part_number`` column, or ``None`` if either signal is missing.
        """
        part_data = (
            self.dataframe[self.dataframe["uuid"] == part_id_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        counter_data = (
            self.dataframe[self.dataframe["uuid"] == counter_uuid]
            .copy()
            .sort_values(self.time_column)
        )

        if part_data.empty or counter_data.empty:
            return None

        part_data[self.time_column] = pd.to_datetime(part_data[self.time_column])
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
        return merged

    @staticmethod
    def _add_increments(merged: pd.DataFrame, value_column_counter: str) -> None:
        """Annotate a merged frame with per-reading production increments.

        Adds three columns in place:
        - ``_prev``: the previous counter reading
        - ``_increment``: parts produced since the previous reading. A
          decreasing counter is interpreted as a reset, so the increment
          is the new counter value (production after the reset).
        - ``_is_reset``: True where the counter decreased.

        The first reading has no predecessor, so its increment is 0 - it
        establishes the baseline rather than counting as production.
        """
        counter = merged[value_column_counter]
        prev = counter.shift(1)
        delta = counter - prev
        is_reset = delta < 0

        # On a reset, assume the counter dropped to (at most) zero and
        # climbed to its current value, so the new value is the production.
        increment = delta.where(~is_reset, counter).fillna(0)

        merged["_prev"] = prev
        merged["_increment"] = increment
        merged["_is_reset"] = is_reset.fillna(False)

    @staticmethod
    def _empty_production_frame(handle_resets: bool) -> pd.DataFrame:
        columns = ["start", "part_number", "quantity", "first_count", "last_count"]
        if handle_resets:
            columns.append("resets")
        return pd.DataFrame(columns=columns)

    def production_by_part(
        self,
        part_id_uuid: str,
        counter_uuid: str,
        *,
        window: str = "1h",
        value_column_part: str = "value_string",
        value_column_counter: str = "value_integer",
        handle_resets: bool = False,
    ) -> pd.DataFrame:
        """Calculate production quantity per part number.

        Args:
            part_id_uuid: UUID for part number signal
            counter_uuid: UUID for production counter
            window: Time window for aggregation (e.g., '1h', '8h', '1d')
            value_column_part: Column containing part numbers
            value_column_counter: Column containing counter values
            handle_resets: When True, treat a decreasing counter as a reset
                and count the production after the reset instead of losing
                it. The result gains a ``resets`` column with the number of
                resets observed in each window. When False (default), the
                quantity is simply ``max(0, last_count - first_count)`` and a
                window containing a reset reports 0.

        Returns:
            DataFrame with columns:
            - start: Start of time window
            - part_number: Part number/ID
            - quantity: Parts produced in window
            - first_count: Counter value at window start
            - last_count: Counter value at window end
            - resets: Number of counter resets in window (only when
              ``handle_resets=True``)

        Example:
            >>> production_by_part('part_id', 'counter', window='1h')
                start         part_number  quantity  first_count  last_count
            0   2024-01-01 08:00:00  PART_A       150       1000        1150
            1   2024-01-01 09:00:00  PART_A       145       1150        1295
            2   2024-01-01 10:00:00  PART_B       98        1295        1393

            With ``handle_resets=True``, a window where the counter drops
            from 432 to 61 reports a quantity of 61 (and ``resets=1``)
            instead of 0.
        """
        merged = self._merge_part_counter(
            part_id_uuid, counter_uuid, value_column_part, value_column_counter
        )

        if merged is None or merged.empty:
            return self._empty_production_frame(handle_resets)

        if handle_resets:
            self._add_increments(merged, value_column_counter)

        # Group by time window and part number
        merged = merged.set_index(self.time_column)

        results = []
        for (start, part_num), group in merged.groupby(
            [pd.Grouper(freq=window), "part_number"]
        ):
            if group.empty:
                continue

            first_count = group[value_column_counter].iloc[0]
            last_count = group[value_column_counter].iloc[-1]

            row = {
                "start": start,
                "part_number": part_num,
                "first_count": first_count,
                "last_count": last_count,
            }

            if handle_resets:
                row["quantity"] = int(group["_increment"].sum())
                row["resets"] = int(group["_is_reset"].sum())
            else:
                row["quantity"] = max(0, last_count - first_count)

            results.append(row)

        if not results:
            return self._empty_production_frame(handle_resets)

        columns = ["start", "part_number", "quantity", "first_count", "last_count"]
        if handle_resets:
            columns.append("resets")
        return pd.DataFrame(results)[columns]

    def detect_resets(
        self,
        part_id_uuid: str,
        counter_uuid: str,
        *,
        value_column_part: str = "value_string",
        value_column_counter: str = "value_integer",
    ) -> pd.DataFrame:
        """Find the points in time where the counter was reset.

        A reset is any reading where the counter value is lower than the
        previous reading.

        Args:
            part_id_uuid: UUID for part number signal
            counter_uuid: UUID for production counter
            value_column_part: Column containing part numbers
            value_column_counter: Column containing counter values

        Returns:
            DataFrame with one row per reset and columns:
            - <time_column>: Timestamp of the reading after the reset
            - part_number: Active part number at the reset
            - count_before: Counter value just before the reset
            - count_after: Counter value just after the reset
            - drop: How far the counter fell (count_before - count_after)

        Example:
            >>> detect_resets('part_id', 'counter')
                systime              part_number  count_before  count_after  drop
            0   2026-06-15 17:00:00  8842580      432           61           371
            1   2026-06-16 19:00:00  9423376      854           0            854
        """
        columns = [
            self.time_column,
            "part_number",
            "count_before",
            "count_after",
            "drop",
        ]

        merged = self._merge_part_counter(
            part_id_uuid, counter_uuid, value_column_part, value_column_counter
        )

        if merged is None or merged.empty:
            return pd.DataFrame(columns=columns)

        self._add_increments(merged, value_column_counter)

        resets = merged[merged["_is_reset"]].copy()
        if resets.empty:
            return pd.DataFrame(columns=columns)

        out = pd.DataFrame(
            {
                self.time_column: resets[self.time_column].values,
                "part_number": resets["part_number"].values,
                "count_before": resets["_prev"].astype(resets[value_column_counter].dtype).values,
                "count_after": resets[value_column_counter].values,
            }
        )
        out["drop"] = out["count_before"] - out["count_after"]
        return out.reset_index(drop=True)

    def daily_production_summary(
        self,
        part_id_uuid: str,
        counter_uuid: str,
        *,
        value_column_part: str = "value_string",
        value_column_counter: str = "value_integer",
        handle_resets: bool = False,
    ) -> pd.DataFrame:
        """Daily production summary by part number.

        Args:
            part_id_uuid: UUID for part number signal
            counter_uuid: UUID for production counter
            value_column_part: Column containing part numbers
            value_column_counter: Column containing counter values
            handle_resets: Account for counter resets (see
                :meth:`production_by_part`). Adds a ``resets`` column with
                the number of resets observed that day.

        Returns:
            DataFrame with columns:
            - date: Production date
            - part_number: Part number/ID
            - total_quantity: Total parts produced that day
            - hours_active: Number of hours with production
            - resets: Number of counter resets that day (only when
              ``handle_resets=True``)

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
            handle_resets=handle_resets,
        )

        base_columns = ["date", "part_number", "total_quantity", "hours_active"]
        if handle_resets:
            base_columns.append("resets")

        if hourly.empty:
            return pd.DataFrame(columns=base_columns)

        hourly["date"] = hourly["start"].dt.date

        agg = {"total_quantity": ("quantity", "sum"), "hours_active": ("start", "count")}
        if handle_resets:
            agg["resets"] = ("resets", "sum")

        daily = (
            hourly.groupby(["date", "part_number"]).agg(**agg).reset_index()
        )

        return daily

    def production_totals(
        self,
        part_id_uuid: str,
        counter_uuid: str,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        value_column_part: str = "value_string",
        value_column_counter: str = "value_integer",
        handle_resets: bool = False,
    ) -> pd.DataFrame:
        """Total production by part number for a date range.

        Args:
            part_id_uuid: UUID for part number signal
            counter_uuid: UUID for production counter
            start_date: Start date 'YYYY-MM-DD' (optional)
            end_date: End date 'YYYY-MM-DD' (optional)
            value_column_part: Column containing part numbers
            value_column_counter: Column containing counter values
            handle_resets: Account for counter resets (see
                :meth:`production_by_part`). Adds a ``resets`` column with
                the total number of resets in the range.

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
            handle_resets=handle_resets,
        )

        base_columns = ["part_number", "total_quantity", "days_produced"]
        if handle_resets:
            base_columns.append("resets")

        if daily.empty:
            return pd.DataFrame(columns=base_columns)

        # Filter by date range
        daily["date"] = pd.to_datetime(daily["date"])
        if start_date:
            daily = daily[daily["date"] >= pd.to_datetime(start_date)]
        if end_date:
            daily = daily[daily["date"] <= pd.to_datetime(end_date)]

        agg = {"total_quantity": ("total_quantity", "sum"), "days_produced": ("date", "count")}
        if handle_resets:
            agg["resets"] = ("resets", "sum")

        totals = daily.groupby("part_number").agg(**agg).reset_index()

        return totals
