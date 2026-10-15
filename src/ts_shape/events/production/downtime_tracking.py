"""Downtime tracking by shift and reason.

Essential module for daily downtime analysis:
- Downtime by shift
- Downtime by reason code
- Availability calculations
- Downtime trends
"""

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING

import pandas as pd

from ts_shape.events._compat import state_mask
from ts_shape.events._windows import add_period_window, add_shift_window, data_tz
from ts_shape.signals.spec import SignalSpec
from ts_shape.utils.base import Base
from ts_shape.utils.deprecation import deprecated_class, renamed_kwargs
from ts_shape.utils.time_accounting import (
    EXCLUDED,
    grouper_periods,
    reading_intervals,
    shift_periods,
    time_by_period,
)

if TYPE_CHECKING:
    from typing_extensions import deprecated

logger = logging.getLogger(__name__)

_SHIFT_COLS = [
    "date",
    "shift",
    "total_minutes",
    "total_seconds",
    "downtime_minutes",
    "downtime_seconds",
    "uptime_minutes",
    "uptime_seconds",
    "availability_pct",
    "excluded_seconds",
]
_TREND_COLS = [
    "period",
    "availability_pct",
    "uptime_minutes",
    "uptime_seconds",
    "downtime_minutes",
    "downtime_seconds",
    "excluded_seconds",
]
_REASON_COLS = [
    "reason",
    "occurrences",
    "total_minutes",
    "total_seconds",
    "avg_minutes",
    "avg_seconds",
    "pct_of_total",
]
_TOP_COLS = [
    "reason",
    "total_minutes",
    "total_seconds",
    "pct_of_total",
    "cumulative_pct",
]


class DowntimeEvents(Base):
    """Track machine downtimes by shift and reason.

    Each UUID represents one signal:
    - run_state_uuid: machine state (running/stopped/idle)
    - reason_uuid: downtime reason code (optional)

    Merge keys: [date, shift] for shift-level, [period] for trend data.

    Run state comes in one of two forms, and both give the same result:

    - a string state signal in ``value_string`` (``running_value="Running"``,
      ``stopped_value="Stopped"``), or
    - a boolean run signal in ``value_bool`` (True = running). It is used when
      the state rows carry ``value_bool`` but no ``value_string``, or when
      ``running_value`` / ``stopped_value`` is a bool (then matched as given).

    Durations are reported in seconds (``*_seconds``); the ``*_minutes``
    columns are kept as a convenience.

    Pipeline example::

        downtime = DowntimeEvents(df)
        shift_dt = downtime.downtime_by_shift('machine_state')
        # → merge with QualityEvents.nok_by_shift() on [date, shift]
        # → merge with ShiftReportEvents.shift_production() on [date, shift]
        # → feed into ShiftHandoverEvents.from_shift_data()

    Example:
        >>> import ts_shape
        >>> df = ts_shape.make_id_signal("state", ["Running", "Stopped", "Running"],
        ...                              hold=20, freq="1min", start="2025-01-01 08:00")
        >>> out = ts_shape.DowntimeEvents(df).downtime_by_shift("state")
        >>> out[["date", "shift", "downtime_minutes", "availability_pct"]].round(1)
                 date    shift  downtime_minutes  availability_pct
        0  2025-01-01  shift_1              20.0              66.1
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        time_column: str = "systime",
        shift_definitions: dict[str, tuple[str, str]] | None = None,
        specs: Mapping[str, SignalSpec] | None = None,
    ) -> None:
        """Initialize downtime tracker.

        Args:
            dataframe: Input DataFrame with timeseries data
            time_column: Name of timestamp column (default: 'systime')
            shift_definitions: Dictionary mapping shift names to (start, end) times
                              Default: 3-shift operation (06:00-14:00, 14:00-22:00, 22:00-06:00)
            specs: Optional signal semantics (see :mod:`ts_shape.signals`). A
                state signal's ``max_gap`` then caps durations by default.
        """
        super().__init__(dataframe, time_column=time_column)
        self.time_column = time_column
        self.specs: Mapping[str, SignalSpec] = specs or {}

        # Default 3-shift operation
        self.shift_definitions = shift_definitions or {
            "shift_1": ("06:00", "14:00"),
            "shift_2": ("14:00", "22:00"),
            "shift_3": ("22:00", "06:00"),
        }

    @renamed_kwargs(state_uuid="run_state_uuid")
    def downtime_by_shift(
        self,
        run_state_uuid: str,
        *,
        running_value: str | bool = "Running",
        value_column: str = "value_string",
        max_gap: str | None = None,
        production_day: bool = True,
    ) -> pd.DataFrame:
        """Calculate downtime duration per shift.

        Each reading's state holds until the next reading, and that interval
        is split at the shift boundaries (the ISO 22400-2 time model): a stop
        from 07:00 to 15:00 is 7 h of downtime in the 06:00-14:00 shift and
        1 h in the 14:00-22:00 shift. A shift is reported when any observed
        time falls in it.

        Args:
            run_state_uuid: UUID for machine state signal
            running_value: Value that indicates machine is running. A bool
                matches a boolean run signal (``value_bool`` by default); a
                boolean run signal is also used, with True as running, when
                the state rows carry no ``value_column`` value.
            value_column: Column containing state values
            max_gap: Optional cap (e.g. "15min"). A duration between two
                consecutive readings longer than this is treated as a logging
                gap ("no data") and excluded, instead of being attributed to
                the preceding state. Without it, a logger outage between a
                ``Running`` sample and the next reading is booked as uptime,
                inflating availability toward 100 % over a period with no data.
                ``None`` (default) uses the state signal's spec ``max_gap``
                when ``specs`` were given and applies no cap otherwise, so
                every interval between readings, however long, counts toward
                the preceding state.
            production_day: Attribute an overnight shift's post-midnight
                readings to the shift's start date, so a night shift is one
                ``(date, shift)`` row (default). False groups by raw calendar
                date, which splits a night shift across two rows.

        Returns:
            DataFrame with downtime by shift:
            - start: Start of the shift window (tz-aware in the data's tz)
            - end: End of the shift window (next day for overnight shifts)
            - duration_seconds: Shift window length in seconds
            - date: Production date
            - shift: Shift name
            - total_minutes / total_seconds: Observed shift time (uptime +
                downtime)
            - downtime_minutes / downtime_seconds: Downtime duration
            - uptime_minutes / uptime_seconds: Running time
            - availability_pct: Uptime percentage
            - excluded_seconds: Time removed by ``max_gap`` as "no data"
                (0 without a cap), split at the shift boundaries like the
                rest; it is in neither uptime nor downtime
            The ``*_minutes`` columns are a convenience (``*_seconds`` / 60).

        Example:
            >>> import ts_shape
            >>> df = ts_shape.make_id_signal(
            ...     "state", ["Running", "Stopped", "Running"], hold=20, freq="1min",
            ...     start="2025-01-01 08:00")
            >>> out = ts_shape.DowntimeEvents(df).downtime_by_shift("state")
            >>> out[["shift", "downtime_minutes", "uptime_minutes", "availability_pct"]]
                 shift  downtime_minutes  uptime_minutes  availability_pct
            0  shift_1              20.0            39.0         66.101695
        """
        state_data = (
            self.dataframe[self.dataframe["uuid"] == run_state_uuid]
            .copy()
            .sort_values(self.time_column)
        )

        if state_data.empty:
            return add_shift_window(
                pd.DataFrame(columns=_SHIFT_COLS), self.shift_definitions, None
            )

        state_data[self.time_column] = pd.to_datetime(state_data[self.time_column])
        times = state_data[self.time_column]

        if max_gap is None and run_state_uuid in self.specs:
            spec_gap = self.specs[run_state_uuid].max_gap
            max_gap = str(spec_gap) if spec_gap is not None else None

        # One interval per reading (held until the next reading), split at
        # the shift boundaries: a stop from 07:00 to 15:00 is 7 h of the
        # 06-14 shift and 1 h of the 14-22 shift (ISO 22400-2 time model).
        is_running = state_mask(
            state_data, value_column, running_value, bool_value=True
        )
        intervals = reading_intervals(
            times,
            is_running.map({True: "up", False: "down"}),
            max_gap=max_gap,
        )
        periods = shift_periods(
            times.iloc[0],
            times.iloc[-1],
            self.shift_definitions,
            production_day=production_day,
        )
        seconds = time_by_period(
            intervals,
            periods,
            keys=["date", "shift"],
            states=["up", "down", EXCLUDED],
        )
        seconds = seconds[(seconds["up"] + seconds["down"]) > 0]

        results = []
        for date, shift, up, down, excluded in seconds[
            ["date", "shift", "up", "down", EXCLUDED]
        ].itertuples(index=False, name=None):
            total = up + down
            results.append(
                {
                    "date": date,
                    "shift": shift,
                    "total_minutes": total / 60,
                    "total_seconds": float(total),
                    "downtime_minutes": down / 60,
                    "downtime_seconds": float(down),
                    "uptime_minutes": up / 60,
                    "uptime_seconds": float(up),
                    "availability_pct": up / total * 100,
                    "excluded_seconds": float(excluded),
                }
            )

        return add_shift_window(
            pd.DataFrame(results, columns=_SHIFT_COLS),
            self.shift_definitions,
            data_tz(times),
        )

    @renamed_kwargs(state_uuid="run_state_uuid")
    def downtime_by_reason(
        self,
        run_state_uuid: str,
        reason_uuid: str,
        *,
        stopped_value: str | bool = "Stopped",
        value_column_state: str = "value_string",
        value_column_reason: str = "value_string",
    ) -> pd.DataFrame:
        """Analyze downtime by reason code.

        Args:
            run_state_uuid: UUID for machine state signal
            reason_uuid: UUID for downtime reason signal
            stopped_value: Value indicating machine is stopped. A bool
                matches a boolean run signal (``value_bool`` by default); a
                boolean run signal is also used, with False as stopped, when
                the state rows carry no ``value_column_state`` value.
            value_column_state: Column containing state values
            value_column_reason: Column containing reason codes

        Returns:
            DataFrame with downtime by reason:
            - reason: Reason code
            - occurrences: Number of downtime events
            - total_minutes / total_seconds: Total downtime for this reason
            - avg_minutes / avg_seconds: Average downtime per occurrence
            - pct_of_total: Percentage of total downtime
            The ``*_minutes`` columns are a convenience (``*_seconds`` / 60).

        Example:
            >>> import pandas as pd, ts_shape
            >>> states = ["Running", "Stopped", "Running", "Stopped", "Running"]
            >>> df = pd.concat([ts_shape.make_id_signal("state", states, hold=10, freq="1min"),
            ...     ts_shape.make_id_signal("reason", ["-", "Jam", "-", "Tool", "-"], hold=10,
            ...                             freq="1min")])
            >>> out = ts_shape.DowntimeEvents(df).downtime_by_reason("state", "reason")
            >>> out[["reason", "occurrences", "total_minutes", "pct_of_total"]]
              reason  occurrences  total_minutes  pct_of_total
            0    Jam           10           10.0          50.0
            1   Tool           10           10.0          50.0
        """
        state_data = (
            self.dataframe[self.dataframe["uuid"] == run_state_uuid]
            .copy()
            .sort_values(self.time_column)
        )

        reason_data = (
            self.dataframe[self.dataframe["uuid"] == reason_uuid]
            .copy()
            .sort_values(self.time_column)
        )

        if state_data.empty or reason_data.empty:
            return pd.DataFrame(columns=_REASON_COLS)

        state_data[self.time_column] = pd.to_datetime(state_data[self.time_column])
        reason_data[self.time_column] = pd.to_datetime(reason_data[self.time_column])

        # Merge state with reason - select only needed columns to avoid suffix issues
        # Note: both dataframes likely have value_string, so we need to be careful
        state_data["is_stopped"] = state_mask(
            state_data, value_column_state, stopped_value, bool_value=False
        )
        state_clean = state_data[[self.time_column, "is_stopped", "uuid"]].copy()

        reason_clean = reason_data[[self.time_column, value_column_reason]].copy()
        reason_clean = reason_clean.rename(columns={value_column_reason: "reason"})

        merged = pd.merge_asof(
            state_clean, reason_clean, on=self.time_column, direction="backward"
        )

        # Calculate each interval's duration on the FULL state timeline, before
        # filtering to stopped rows. A stop lasts until the next state change
        # (the next reading), not until the next *stop* - computing the diff
        # after filtering would book the intervening running time as downtime.
        merged = merged.sort_values(self.time_column)
        merged["next_time"] = merged[self.time_column].shift(-1)
        merged["duration_s"] = (
            merged["next_time"] - merged[self.time_column]
        ).dt.total_seconds()
        merged["duration_minutes"] = merged["duration_s"] / 60

        # Filter for stopped states, then drop the trailing open interval (NaN).
        stopped = merged[merged["is_stopped"]].copy()
        stopped = stopped[stopped["duration_minutes"].notna()]

        if stopped.empty:
            return pd.DataFrame(columns=_REASON_COLS)

        # Group by reason
        grouped = stopped.groupby("reason")
        reason_stats = (
            grouped["duration_minutes"]
            .agg(
                [
                    ("occurrences", "count"),
                    ("total_minutes", "sum"),
                    ("avg_minutes", "mean"),
                ]
            )
            .reset_index()
        )
        seconds = grouped["duration_s"].agg(["sum", "mean"]).reset_index(drop=True)
        reason_stats.insert(3, "total_seconds", seconds["sum"].to_numpy())
        reason_stats.insert(5, "avg_seconds", seconds["mean"].to_numpy())

        # Calculate percentage of total
        total_downtime = reason_stats["total_minutes"].sum()
        reason_stats["pct_of_total"] = (
            (reason_stats["total_minutes"] / total_downtime * 100)
            if total_downtime > 0
            else 0
        )

        # Sort by total downtime descending
        reason_stats = reason_stats.sort_values("total_minutes", ascending=False)

        return reason_stats[_REASON_COLS].reset_index(drop=True)

    @renamed_kwargs(state_uuid="run_state_uuid")
    def top_downtime_reasons(
        self,
        run_state_uuid: str,
        reason_uuid: str,
        *,
        top_n: int = 5,
        stopped_value: str | bool = "Stopped",
        value_column_state: str = "value_string",
        value_column_reason: str = "value_string",
    ) -> pd.DataFrame:
        """Get top N downtime reasons (Pareto analysis).

        Args:
            run_state_uuid: UUID for machine state signal
            reason_uuid: UUID for downtime reason signal
            top_n: Number of top reasons to return
            stopped_value: Value indicating machine is stopped (a string, or
                a bool for a boolean run signal; see :meth:`downtime_by_reason`)
            value_column_state: Column containing state values
            value_column_reason: Column containing reason codes

        Returns:
            DataFrame with top N reasons and cumulative percentage: reason,
            total_minutes, total_seconds, pct_of_total, cumulative_pct

        Example:
            >>> import pandas as pd, ts_shape
            >>> states = ["Running", "Stopped", "Running", "Stopped", "Running"]
            >>> df = pd.concat([ts_shape.make_id_signal("state", states, hold=10, freq="1min"),
            ...     ts_shape.make_id_signal("reason", ["-", "Jam", "-", "Tool", "-"], hold=10,
            ...                             freq="1min")])
            >>> top = ts_shape.DowntimeEvents(df).top_downtime_reasons("state", "reason", top_n=1)
            >>> top[["reason", "total_minutes", "cumulative_pct"]]
              reason  total_minutes  cumulative_pct
            0    Jam           10.0            50.0
        """
        reason_stats = self.downtime_by_reason(
            run_state_uuid,
            reason_uuid,
            stopped_value=stopped_value,
            value_column_state=value_column_state,
            value_column_reason=value_column_reason,
        )

        if reason_stats.empty:
            return pd.DataFrame(columns=_TOP_COLS)

        # Get top N
        top_reasons = reason_stats.head(top_n).copy()

        # Calculate cumulative percentage
        top_reasons["cumulative_pct"] = top_reasons["pct_of_total"].cumsum()

        return top_reasons[_TOP_COLS]

    @renamed_kwargs(state_uuid="run_state_uuid")
    def availability_trend(
        self,
        run_state_uuid: str,
        *,
        running_value: str | bool = "Running",
        value_column: str = "value_string",
        window: str = "1D",
        max_gap: str | None = None,
    ) -> pd.DataFrame:
        """Calculate availability trend over time.

        Each reading's state holds until the next reading, and that interval
        is split at the window boundaries (the ISO 22400-2 time model), so
        every window holds exactly the up / down time that fell inside it. A
        window is reported when any observed time falls in it.

        Args:
            run_state_uuid: UUID for machine state signal
            running_value: Value that indicates machine is running (a string,
                or a bool for a boolean run signal; see
                :meth:`downtime_by_shift`)
            value_column: Column containing state values
            window: Time window for aggregation (e.g., '1D', '1W')
            max_gap: Optional cap (e.g. "15min"). A gap between consecutive
                readings longer than this is treated as "no data" and excluded,
                instead of being attributed to the preceding state (which would
                inflate availability across a logging outage). ``None``
                (default) applies no cap: every interval between readings,
                however long, counts toward the preceding state.

        Returns:
            DataFrame with availability trend:
            - start: Period start
            - end: Period end (start + window)
            - duration_seconds: Period length in seconds
            - period: Time period
            - availability_pct: Availability percentage
            - uptime_minutes / uptime_seconds: Total uptime
            - downtime_minutes / downtime_seconds: Total downtime
            - excluded_seconds: Time removed by ``max_gap`` as "no data"
                (0 without a cap), split at the window boundaries
            The ``*_minutes`` columns are a convenience (``*_seconds`` / 60).

        Example:
            >>> import ts_shape
            >>> df = ts_shape.make_id_signal(
            ...     "state", ["Running", "Stopped", "Running"], hold=20, freq="1min")
            >>> out = ts_shape.DowntimeEvents(df).availability_trend("state", window="30min")
            >>> out[["period", "availability_pct", "downtime_minutes"]]
                           period  availability_pct  downtime_minutes
            0 2025-01-01 00:00:00         66.666667              10.0
            1 2025-01-01 00:30:00         65.517241              10.0
        """
        state_data = (
            self.dataframe[self.dataframe["uuid"] == run_state_uuid]
            .copy()
            .sort_values(self.time_column)
        )

        if state_data.empty:
            return add_period_window(pd.DataFrame(columns=_TREND_COLS), window)

        state_data[self.time_column] = pd.to_datetime(state_data[self.time_column])
        times = state_data[self.time_column]

        # One interval per reading, split at the window boundaries, so each
        # window holds exactly the up / down time that fell inside it.
        is_running = state_mask(
            state_data, value_column, running_value, bool_value=True
        )
        intervals = reading_intervals(
            times,
            is_running.map({True: "up", False: "down"}),
            max_gap=max_gap,
        )
        periods = grouper_periods(times.iloc[0], times.iloc[-1], window)
        seconds = time_by_period(
            intervals, periods, keys=["period"], states=["up", "down", EXCLUDED]
        )
        seconds = seconds[(seconds["up"] + seconds["down"]) > 0]

        if seconds.empty:
            return add_period_window(pd.DataFrame(columns=_TREND_COLS), window)

        results = []
        for period, up, down, excluded in seconds[
            ["period", "up", "down", EXCLUDED]
        ].itertuples(index=False, name=None):
            results.append(
                {
                    "period": period,
                    "availability_pct": up / (up + down) * 100,
                    "uptime_minutes": up / 60,
                    "uptime_seconds": float(up),
                    "downtime_minutes": down / 60,
                    "downtime_seconds": float(down),
                    "excluded_seconds": float(excluded),
                }
            )

        return add_period_window(pd.DataFrame(results, columns=_TREND_COLS), window)


#: Deprecated name, removed in the next major release.
if TYPE_CHECKING:

    @deprecated("DowntimeTracking is deprecated, use DowntimeEvents")
    class DowntimeTracking(DowntimeEvents): ...

else:
    DowntimeTracking = deprecated_class(DowntimeEvents, "DowntimeTracking")
