"""Performance/speed loss tracking.

Track when machine runs slower than ideal/target speed:
- Identify slow periods vs target cycle time
- Calculate performance loss in minutes and parts
- Performance trend over time
"""

import logging

import pandas as pd  # type: ignore

from ts_shape.events._windows import (
    add_period_window,
    add_shift_window,
    data_tz,
)
from ts_shape.utils.base import Base
from ts_shape.utils.deprecation import deprecated_class
from ts_shape.utils.shifts import production_dates

logger = logging.getLogger(__name__)

_SHIFT_COLS = [
    "date",
    "shift",
    "actual_parts",
    "avg_cycle_time_s",
    "target_cycle_time_s",
    "performance_pct",
    "loss_minutes",
]
_SLOW_COLS = [
    "start",
    "end",
    "duration_seconds",
    "uuid",
    "source_uuid",
    "actual_parts",
    "avg_cycle_time_s",
    "performance_pct",
    "loss_minutes",
]
_TREND_COLS = [
    "period",
    "actual_parts",
    "avg_cycle_time_s",
    "performance_pct",
    "loss_minutes",
]


class PerformanceLossEvents(Base):
    """Track performance and speed losses against target cycle times.

    Identifies hidden losses where the machine is running but slower than
    it should be.  Typically 10-20% of production time is lost to speed losses.

    Each UUID represents one signal:
    - cycle_uuid: cycle trigger or production counter
    - part_id_uuid: part number signal (optional, for per-part targets)

    Merge keys: [date, shift] for shift-level, [period] for trend data.

    Pipeline example::

        perf = PerformanceLossEvents(df)
        shift_perf = perf.performance_by_shift('cycle', target_cycle_time=45)
        # → merge with DowntimeEvents.downtime_by_shift() on [date, shift]
        # → merge with QualityEvents.nok_by_shift() on [date, shift]
        # → feed into ShiftHandoverEvents.from_shift_data()

    Example usage:
        tracker = PerformanceLossEvents(df)

        # Performance by shift
        perf = tracker.performance_by_shift(
            cycle_uuid='cycle_trigger',
            target_cycle_time=45.0,
        )

        # Identify slow periods
        slow = tracker.slow_periods(
            cycle_uuid='cycle_trigger',
            target_cycle_time=45.0,
            threshold_pct=90,
        )
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        time_column: str = "systime",
        shift_definitions: dict[str, tuple[str, str]] | None = None,
        event_uuid: str = "prod:performance_loss",
    ) -> None:
        """Initialize the performance-loss tracker.

        Args:
            dataframe: Input DataFrame with timeseries data.
            time_column: Name of timestamp column (default: 'systime').
            shift_definitions: Mapping of shift name to (start, end) times.
                Default: 3-shift operation (06:00-14:00, 14:00-22:00,
                22:00-06:00).
            event_uuid: UUID tagged on derived interval events (the ``uuid``
                column of :meth:`slow_periods`; default:
                'prod:performance_loss').
        """
        super().__init__(dataframe, column_name=time_column)
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

    def _compute_cycle_times(
        self,
        cycle_uuid: str,
        value_column: str = "value_integer",
        *,
        max_cycle_gap: str | None = None,
    ) -> pd.DataFrame:
        """Compute per-cycle times from a counter or trigger signal.

        When ``max_cycle_gap`` is set, intervals longer than it are dropped as
        stoppages (breaks, downtime) rather than counted as slow cycles.
        """
        data = (
            self.dataframe[self.dataframe["uuid"] == cycle_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        if data.empty:
            return pd.DataFrame(columns=[self.time_column, "cycle_time_s"])

        data[self.time_column] = pd.to_datetime(data[self.time_column])

        # If counter: detect increments.  If trigger (bool): each row is a cycle.
        if value_column in data.columns and pd.api.types.is_numeric_dtype(
            data[value_column]
        ):
            data["delta"] = data[value_column].diff()
            data = data[data["delta"] > 0]

        data["cycle_time_s"] = data[self.time_column].diff().dt.total_seconds()
        data = data[data["cycle_time_s"].notna() & (data["cycle_time_s"] > 0)]
        if max_cycle_gap is not None:
            cap_seconds = pd.to_timedelta(max_cycle_gap).total_seconds()
            data = data[data["cycle_time_s"] <= cap_seconds]
        return data

    def performance_by_shift(
        self,
        cycle_uuid: str,
        target_cycle_time: float,
        *,
        value_column: str = "value_integer",
        max_cycle_gap: str | None = None,
        production_day: bool = True,
    ) -> pd.DataFrame:
        """Calculate performance percentage per shift.

        Performance = (target_cycle_time * actual_parts) / elapsed_run_time.

        Args:
            cycle_uuid: UUID of cycle trigger or counter signal.
            target_cycle_time: Target (ideal) cycle time in seconds.
            value_column: Column with counter values.
            max_cycle_gap: Optional cap (e.g. "5min"). Intervals longer than
                this are treated as stoppages and excluded, so a break or
                downtime is not counted as slow-running time (which would
                otherwise report a large, spurious speed loss).
            production_day: Attribute an overnight shift's post-midnight
                readings to the shift's start date, so a night shift is one
                ``(date, shift)`` row (default). False groups by raw calendar
                date, which splits a night shift across two rows.

        Returns:
            DataFrame with columns:
            - start, end, duration_seconds: the shift window from
                ``shift_definitions`` (tz-aware in the data's tz; an overnight
                shift ends on the next day)
            - date, shift, actual_parts, avg_cycle_time_s,
                target_cycle_time_s, performance_pct, loss_minutes
        """
        data = self._compute_cycle_times(
            cycle_uuid, value_column, max_cycle_gap=max_cycle_gap
        )
        if data.empty:
            return add_shift_window(
                pd.DataFrame(columns=_SHIFT_COLS), self.shift_definitions, None
            )

        data["shift"] = data[self.time_column].apply(self._assign_shift)
        data["date"] = (
            production_dates(
                data[self.time_column], data["shift"], self.shift_definitions
            )
            if production_day
            else data[self.time_column].dt.date
        )

        results = []
        for (date, shift), grp in data.groupby(["date", "shift"]):
            actual_parts = len(grp)
            avg_ct = grp["cycle_time_s"].mean()
            elapsed = grp["cycle_time_s"].sum()

            if elapsed > 0 and target_cycle_time > 0:
                ideal_time = actual_parts * target_cycle_time
                performance = (ideal_time / elapsed) * 100.0
            else:
                performance = 0.0

            loss_seconds = max(0, elapsed - actual_parts * target_cycle_time)

            results.append(
                {
                    "date": date,
                    "shift": shift,
                    "actual_parts": actual_parts,
                    "avg_cycle_time_s": round(avg_ct, 2),
                    "target_cycle_time_s": target_cycle_time,
                    "performance_pct": round(min(performance, 100.0), 1),
                    "loss_minutes": round(loss_seconds / 60, 1),
                }
            )

        return add_shift_window(
            pd.DataFrame(results, columns=_SHIFT_COLS),
            self.shift_definitions,
            data_tz(data[self.time_column]),
        )

    def slow_periods(
        self,
        cycle_uuid: str,
        target_cycle_time: float,
        *,
        threshold_pct: float = 90.0,
        window: str = "1h",
        value_column: str = "value_integer",
        max_cycle_gap: str | None = None,
    ) -> pd.DataFrame:
        """Identify time windows where performance is below threshold.

        Args:
            cycle_uuid: UUID of cycle trigger or counter signal.
            target_cycle_time: Target cycle time in seconds.
            threshold_pct: Performance must be below this to flag (default 90%).
            window: Rolling window size (default '1h').
            value_column: Column with counter values.

        Returns:
            DataFrame with columns:
            - start, end, duration_seconds (float seconds, end - start),
                uuid (event uuid), source_uuid (cycle_uuid), actual_parts,
                avg_cycle_time_s, performance_pct, loss_minutes
        """
        data = self._compute_cycle_times(
            cycle_uuid, value_column, max_cycle_gap=max_cycle_gap
        )
        if data.empty:
            return pd.DataFrame(columns=_SLOW_COLS)

        data = data.set_index(self.time_column)
        results = []

        for start, grp in data.groupby(pd.Grouper(freq=window)):
            if grp.empty or len(grp) < 2:
                continue

            actual_parts = len(grp)
            avg_ct = grp["cycle_time_s"].mean()
            elapsed = grp["cycle_time_s"].sum()

            if elapsed > 0 and target_cycle_time > 0:
                ideal_time = actual_parts * target_cycle_time
                performance = (ideal_time / elapsed) * 100.0
            else:
                performance = 0.0

            if performance < threshold_pct:
                loss_seconds = max(0, elapsed - actual_parts * target_cycle_time)
                results.append(
                    {
                        "start": start,
                        "end": start + pd.Timedelta(window),
                        "duration_seconds": pd.Timedelta(window).total_seconds(),
                        "uuid": self.event_uuid,
                        "source_uuid": cycle_uuid,
                        "actual_parts": actual_parts,
                        "avg_cycle_time_s": round(avg_ct, 2),
                        "performance_pct": round(performance, 1),
                        "loss_minutes": round(loss_seconds / 60, 1),
                    }
                )

        return pd.DataFrame(results, columns=_SLOW_COLS)

    def performance_trend(
        self,
        cycle_uuid: str,
        target_cycle_time: float,
        *,
        window: str = "1D",
        value_column: str = "value_integer",
        max_cycle_gap: str | None = None,
    ) -> pd.DataFrame:
        """Track performance trend over time.

        Args:
            cycle_uuid: UUID of cycle trigger or counter signal.
            target_cycle_time: Target cycle time in seconds.
            window: Time window for aggregation (default '1D').
            value_column: Column with counter values.

        Returns:
            DataFrame with columns:
            - start, end (start + window), duration_seconds
            - period, actual_parts, avg_cycle_time_s, performance_pct, loss_minutes
        """
        data = self._compute_cycle_times(
            cycle_uuid, value_column, max_cycle_gap=max_cycle_gap
        )
        if data.empty:
            return add_period_window(pd.DataFrame(columns=_TREND_COLS), window)

        data = data.set_index(self.time_column)
        results = []

        for period, grp in data.groupby(pd.Grouper(freq=window)):
            if grp.empty:
                continue

            actual_parts = len(grp)
            avg_ct = grp["cycle_time_s"].mean()
            elapsed = grp["cycle_time_s"].sum()

            if elapsed > 0 and target_cycle_time > 0:
                ideal_time = actual_parts * target_cycle_time
                performance = (ideal_time / elapsed) * 100.0
            else:
                performance = 0.0

            loss_seconds = max(0, elapsed - actual_parts * target_cycle_time)

            results.append(
                {
                    "period": period,
                    "actual_parts": actual_parts,
                    "avg_cycle_time_s": round(avg_ct, 2),
                    "performance_pct": round(min(performance, 100.0), 1),
                    "loss_minutes": round(loss_seconds / 60, 1),
                }
            )

        return add_period_window(pd.DataFrame(results, columns=_TREND_COLS), window)


#: Deprecated name, removed in 1.0.
PerformanceLossTracking = deprecated_class(
    PerformanceLossEvents, "PerformanceLossTracking"
)
