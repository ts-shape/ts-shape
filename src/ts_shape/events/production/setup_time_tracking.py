"""Setup/changeover time tracking for SMED analysis.

Track setup durations to support Single-Minute Exchange of Die improvement:
- Setup event durations
- Setup time by product transition
- Setup statistics
- Setup time trends
"""

import logging

import pandas as pd

from ts_shape.events._windows import add_period_window
from ts_shape.utils.base import Base
from ts_shape.utils.deprecation import deprecated_class
from ts_shape.utils.shifts import production_dates

logger = logging.getLogger(__name__)


class SetupTimeEvents(Base):
    """Track and analyze setup/changeover durations.

    Each UUID represents one signal:
    - state_uuid: machine state signal containing a setup indicator value
    - part_id_uuid: part number / product type signal (optional)

    Merge keys: [date, shift] for shift-level, [period] for trend data,
    [from_product, to_product] for product transition analysis.

    Pipeline example::

        setup = SetupTimeEvents(df)
        durations = setup.setup_durations('machine_state', setup_value='Setup')
        # → merge with ShiftReportEvents.shift_production() on [date, shift]
        stats = setup.setup_statistics('machine_state')
        # → standalone KPI reporting

    Example usage:
        tracker = SetupTimeEvents(df)

        # List all setup events
        events = tracker.setup_durations(state_uuid='machine_state')

        # Setup time by product transition
        by_product = tracker.setup_by_product(
            state_uuid='machine_state',
            part_id_uuid='part_number',
        )
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        time_column: str = "systime",
        shift_definitions: dict[str, tuple[str, str]] | None = None,
        event_uuid: str = "prod:setup_time",
    ) -> None:
        """Initialize the setup-time tracker.

        Args:
            dataframe: Input DataFrame with timeseries data.
            time_column: Name of timestamp column (default: 'systime').
            shift_definitions: Mapping of shift name to (start, end) times.
                Default: 3-shift operation (06:00-14:00, 14:00-22:00,
                22:00-06:00).
            event_uuid: UUID tagged on derived interval events (the ``uuid``
                column of :meth:`setup_durations`; default: 'prod:setup_time').
        """
        super().__init__(dataframe, time_column=time_column)
        self.time_column = time_column
        self.event_uuid = event_uuid
        self.shift_definitions = shift_definitions or {
            "shift_1": ("06:00", "14:00"),
            "shift_2": ("14:00", "22:00"),
            "shift_3": ("22:00", "06:00"),
        }

    def _assign_shift(self, timestamp: pd.Timestamp) -> str:
        time = timestamp.time()
        for shift_name, (start, end) in self.shift_definitions.items():
            start_time = pd.to_datetime(start).time()
            end_time = pd.to_datetime(end).time()
            if start_time < end_time:
                if start_time <= time < end_time:
                    return shift_name
            else:
                if time >= start_time or time < end_time:
                    return shift_name
        return "unknown"

    def _extract_setup_intervals(
        self,
        state_uuid: str,
        setup_value: str,
        value_column: str,
    ) -> pd.DataFrame:
        """Extract start/end intervals where state equals the setup value."""
        data = (
            self.dataframe[self.dataframe["uuid"] == state_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        if data.empty:
            return pd.DataFrame(columns=["start", "end", "duration_minutes"])

        data[self.time_column] = pd.to_datetime(data[self.time_column])

        # Detect setup intervals: contiguous blocks where value == setup_value
        data["is_setup"] = data[value_column] == setup_value
        data["block"] = (data["is_setup"] != data["is_setup"].shift()).cumsum()

        setup_blocks = data[data["is_setup"]]
        if setup_blocks.empty:
            return pd.DataFrame(columns=["start", "end", "duration_minutes"])

        intervals = []
        for _, block in setup_blocks.groupby("block"):
            start = block[self.time_column].iloc[0]
            # End time: use next non-setup timestamp if available, else last setup timestamp
            block_end_idx = block.index[-1]
            next_rows = data.loc[data.index > block_end_idx]
            if not next_rows.empty:
                end = next_rows[self.time_column].iloc[0]
            else:
                end = block[self.time_column].iloc[-1]

            duration = (end - start).total_seconds() / 60
            if duration > 0:
                intervals.append(
                    {
                        "start": start,
                        "end": end,
                        "duration_minutes": duration,
                    }
                )

        return pd.DataFrame(intervals)

    def setup_durations(
        self,
        state_uuid: str,
        *,
        setup_value: str = "Setup",
        value_column: str = "value_string",
        production_day: bool = True,
    ) -> pd.DataFrame:
        """List every setup event with duration.

        Args:
            state_uuid: UUID for machine state signal.
            setup_value: Value that indicates a setup state.
            value_column: Column containing state values.
            production_day: Attribute an overnight shift's post-midnight
                setups to the shift's start date, so a night shift is one
                ``(date, shift)`` row (default). False groups by raw calendar
                date, which splits a night shift across two rows.

        Returns:
            DataFrame with columns:
            - start, end, duration_seconds (float seconds, end - start),
                uuid (event uuid), source_uuid (state_uuid),
                duration_minutes, date, shift
        """
        intervals = self._extract_setup_intervals(state_uuid, setup_value, value_column)
        if intervals.empty:
            return pd.DataFrame(
                columns=[
                    "start",
                    "end",
                    "duration_seconds",
                    "uuid",
                    "source_uuid",
                    "duration_minutes",
                    "date",
                    "shift",
                ]
            )

        intervals["shift"] = intervals["start"].apply(self._assign_shift)
        intervals["date"] = (
            production_dates(
                intervals["start"], intervals["shift"], self.shift_definitions
            )
            if production_day
            else intervals["start"].dt.date
        )
        intervals.insert(
            2,
            "duration_seconds",
            (intervals["end"] - intervals["start"]).dt.total_seconds(),
        )
        intervals.insert(3, "uuid", self.event_uuid)
        intervals.insert(4, "source_uuid", state_uuid)

        return intervals

    def setup_by_product(
        self,
        state_uuid: str,
        part_id_uuid: str,
        *,
        setup_value: str = "Setup",
        value_column_state: str = "value_string",
        value_column_part: str = "value_string",
    ) -> pd.DataFrame:
        """Setup time statistics by product transition (from → to).

        Args:
            state_uuid: UUID for machine state signal.
            part_id_uuid: UUID for part number / product type signal.
            setup_value: Value indicating setup state.
            value_column_state: Column containing state values.
            value_column_part: Column containing part number values.

        Returns:
            DataFrame with columns:
            - from_product, to_product, avg_minutes, min_minutes, max_minutes, count
        """
        intervals = self._extract_setup_intervals(
            state_uuid, setup_value, value_column_state
        )
        if intervals.empty:
            return pd.DataFrame(
                columns=[
                    "from_product",
                    "to_product",
                    "avg_minutes",
                    "min_minutes",
                    "max_minutes",
                    "count",
                ]
            )

        part_data = (
            self.dataframe[self.dataframe["uuid"] == part_id_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        if part_data.empty:
            return pd.DataFrame(
                columns=[
                    "from_product",
                    "to_product",
                    "avg_minutes",
                    "min_minutes",
                    "max_minutes",
                    "count",
                ]
            )

        part_data[self.time_column] = pd.to_datetime(part_data[self.time_column])

        # For each setup interval, find the product before and after
        rows = []
        for _, interval in intervals.iterrows():
            before = part_data[part_data[self.time_column] <= interval["start"]]
            after = part_data[part_data[self.time_column] >= interval["end"]]

            from_product = (
                before[value_column_part].iloc[-1] if not before.empty else "unknown"
            )
            to_product = (
                after[value_column_part].iloc[0] if not after.empty else "unknown"
            )

            rows.append(
                {
                    "from_product": from_product,
                    "to_product": to_product,
                    "duration_minutes": interval["duration_minutes"],
                }
            )

        if not rows:
            return pd.DataFrame(
                columns=[
                    "from_product",
                    "to_product",
                    "avg_minutes",
                    "min_minutes",
                    "max_minutes",
                    "count",
                ]
            )

        transitions = pd.DataFrame(rows)
        result = (
            transitions.groupby(["from_product", "to_product"])["duration_minutes"]
            .agg(
                [
                    ("avg_minutes", "mean"),
                    ("min_minutes", "min"),
                    ("max_minutes", "max"),
                    ("count", "count"),
                ]
            )
            .reset_index()
        )

        return result.sort_values("avg_minutes", ascending=False).reset_index(drop=True)

    def setup_statistics(
        self,
        state_uuid: str,
        *,
        setup_value: str = "Setup",
        value_column: str = "value_string",
    ) -> pd.DataFrame:
        """Overall setup time statistics.

        Args:
            state_uuid: UUID for machine state signal.
            setup_value: Value indicating setup state.
            value_column: Column containing state values.

        Returns:
            DataFrame (single row) with columns:
            - total_setups, total_minutes, avg_minutes, median_minutes,
                std_minutes, pct_of_available_time
        """
        intervals = self._extract_setup_intervals(state_uuid, setup_value, value_column)
        if intervals.empty:
            return pd.DataFrame(
                columns=[
                    "total_setups",
                    "total_minutes",
                    "avg_minutes",
                    "median_minutes",
                    "std_minutes",
                    "pct_of_available_time",
                ]
            )

        durations = intervals["duration_minutes"]

        # Calculate total available time from the data
        state_data = (
            self.dataframe[self.dataframe["uuid"] == state_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        state_data[self.time_column] = pd.to_datetime(state_data[self.time_column])
        total_span_minutes = (
            state_data[self.time_column].max() - state_data[self.time_column].min()
        ).total_seconds() / 60

        total_setup = durations.sum()
        pct = (total_setup / total_span_minutes * 100) if total_span_minutes > 0 else 0

        return pd.DataFrame(
            [
                {
                    "total_setups": len(durations),
                    "total_minutes": total_setup,
                    "avg_minutes": durations.mean(),
                    "median_minutes": durations.median(),
                    "std_minutes": durations.std() if len(durations) > 1 else 0.0,
                    "pct_of_available_time": pct,
                }
            ]
        )

    def setup_trend(
        self,
        state_uuid: str,
        *,
        setup_value: str = "Setup",
        value_column: str = "value_string",
        window: str = "1W",
    ) -> pd.DataFrame:
        """Track setup time trend over time.

        Args:
            state_uuid: UUID for machine state signal.
            setup_value: Value indicating setup state.
            value_column: Column containing state values.
            window: Time window for aggregation (default '1W').

        Returns:
            DataFrame with columns:
            - start, end, duration_seconds: the period's bin bounds (for the
                default ``'1W'`` the Monday-to-Monday week; ``period`` is the
                pandas bin label, i.e. the week's Sunday)
            - period, avg_setup_minutes, setup_count, total_setup_minutes
        """
        trend_cols = [
            "period",
            "avg_setup_minutes",
            "setup_count",
            "total_setup_minutes",
        ]
        intervals = self._extract_setup_intervals(state_uuid, setup_value, value_column)
        if intervals.empty:
            return add_period_window(pd.DataFrame(columns=trend_cols), window)

        intervals = intervals.set_index("start")

        results = []
        for period, grp in intervals.groupby(pd.Grouper(freq=window)):
            if grp.empty:
                continue
            results.append(
                {
                    "period": period,
                    "avg_setup_minutes": grp["duration_minutes"].mean(),
                    "setup_count": len(grp),
                    "total_setup_minutes": grp["duration_minutes"].sum(),
                }
            )

        return add_period_window(pd.DataFrame(results, columns=trend_cols), window)


#: Deprecated name, removed in 2.0.
SetupTimeTracking = deprecated_class(SetupTimeEvents, "SetupTimeTracking")
