"""Downtime tracking by shift and reason.

Essential module for daily downtime analysis:
- Downtime by shift
- Downtime by reason code
- Availability calculations
- Downtime trends
"""

import logging
from collections.abc import Mapping

import pandas as pd

from ts_shape.events._compat import state_mask
from ts_shape.events._windows import (
    add_period_window,
    add_shift_window,
    data_tz,
    window_bounds,
)
from ts_shape.signals.spec import SignalSpec
from ts_shape.utils.base import Base
from ts_shape.utils.deprecation import deprecated_class
from ts_shape.utils.shifts import assign_shift, production_dates

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
    - state_uuid: machine state (running/stopped/idle)
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

    def downtime_by_shift(
        self,
        state_uuid: str,
        *,
        running_value: str | bool = "Running",
        value_column: str = "value_string",
        max_gap: str | None = None,
        production_day: bool = True,
    ) -> pd.DataFrame:
        """Calculate downtime duration per shift.

        Args:
            state_uuid: UUID for machine state signal
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
                (0 without a cap); it is in neither uptime nor downtime
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
            self.dataframe[self.dataframe["uuid"] == state_uuid]
            .copy()
            .sort_values(self.time_column)
        )

        if state_data.empty:
            return add_shift_window(
                pd.DataFrame(columns=_SHIFT_COLS), self.shift_definitions, None
            )

        state_data[self.time_column] = pd.to_datetime(state_data[self.time_column])

        # Assign shifts and dates
        state_data["shift"] = assign_shift(
            state_data[self.time_column], self.shift_definitions
        )
        state_data["date"] = (
            production_dates(
                state_data[self.time_column],
                state_data["shift"],
                self.shift_definitions,
            )
            if production_day
            else state_data[self.time_column].dt.date
        )

        # Calculate duration for each state
        state_data["next_time"] = state_data[self.time_column].shift(-1)
        state_data["duration_s"] = (
            state_data["next_time"] - state_data[self.time_column]
        ).dt.total_seconds()
        state_data["duration_minutes"] = state_data["duration_s"] / 60
        state_data["excluded_s"] = 0.0

        if max_gap is None and state_uuid in self.specs:
            spec_gap = self.specs[state_uuid].max_gap
            max_gap = str(spec_gap) if spec_gap is not None else None

        # Treat over-long gaps as "no data" so they are excluded from the sums.
        if max_gap is not None:
            cap_minutes = pd.to_timedelta(max_gap).total_seconds() / 60
            over_gap = state_data["duration_minutes"] > cap_minutes
            state_data.loc[over_gap, "excluded_s"] = state_data.loc[
                over_gap, "duration_s"
            ]
            state_data.loc[over_gap, ["duration_minutes", "duration_s"]] = float("nan")

        # Mark running vs downtime
        state_data["is_running"] = state_mask(
            state_data, value_column, running_value, bool_value=True
        )

        # Seconds per (date, shift), aggregated in one pass.
        seconds = (
            state_data.assign(
                _up=state_data["duration_s"].where(state_data["is_running"], 0.0),
                _down=state_data["duration_s"].where(~state_data["is_running"], 0.0),
            )
            .groupby(["date", "shift"])[["_up", "_down", "excluded_s"]]
            .sum()
        )
        seconds_by_group = dict(
            zip(seconds.index, seconds.itertuples(index=False, name=None), strict=True)
        )

        # Group by date and shift
        results = []
        for (date, shift), group in state_data.groupby(["date", "shift"]):
            if group.empty:
                continue

            # Calculate times (exclude last row with NaN duration)
            valid_data = group[group["duration_minutes"].notna()]

            if valid_data.empty:
                continue

            uptime_minutes = valid_data[valid_data["is_running"]][
                "duration_minutes"
            ].sum()
            downtime_minutes = valid_data[~valid_data["is_running"]][
                "duration_minutes"
            ].sum()
            total_minutes = uptime_minutes + downtime_minutes
            uptime_seconds, downtime_seconds, excluded_seconds = seconds_by_group[
                (date, shift)
            ]

            availability_pct = (
                (uptime_minutes / total_minutes * 100) if total_minutes > 0 else 0
            )

            results.append(
                {
                    "date": date,
                    "shift": shift,
                    "total_minutes": total_minutes,
                    "total_seconds": float(uptime_seconds + downtime_seconds),
                    "downtime_minutes": downtime_minutes,
                    "downtime_seconds": float(downtime_seconds),
                    "uptime_minutes": uptime_minutes,
                    "uptime_seconds": float(uptime_seconds),
                    "availability_pct": availability_pct,
                    "excluded_seconds": float(excluded_seconds),
                }
            )

        return add_shift_window(
            pd.DataFrame(results, columns=_SHIFT_COLS),
            self.shift_definitions,
            data_tz(state_data[self.time_column]),
        )

    def downtime_by_reason(
        self,
        state_uuid: str,
        reason_uuid: str,
        *,
        stopped_value: str | bool = "Stopped",
        value_column_state: str = "value_string",
        value_column_reason: str = "value_string",
    ) -> pd.DataFrame:
        """Analyze downtime by reason code.

        Args:
            state_uuid: UUID for machine state signal
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
            self.dataframe[self.dataframe["uuid"] == state_uuid]
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

    def top_downtime_reasons(
        self,
        state_uuid: str,
        reason_uuid: str,
        *,
        top_n: int = 5,
        stopped_value: str | bool = "Stopped",
        value_column_state: str = "value_string",
        value_column_reason: str = "value_string",
    ) -> pd.DataFrame:
        """Get top N downtime reasons (Pareto analysis).

        Args:
            state_uuid: UUID for machine state signal
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
            state_uuid,
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

    def availability_trend(
        self,
        state_uuid: str,
        *,
        running_value: str | bool = "Running",
        value_column: str = "value_string",
        window: str = "1D",
        max_gap: str | None = None,
    ) -> pd.DataFrame:
        """Calculate availability trend over time.

        Args:
            state_uuid: UUID for machine state signal
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
                (0 without a cap)
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
            self.dataframe[self.dataframe["uuid"] == state_uuid]
            .copy()
            .sort_values(self.time_column)
        )

        if state_data.empty:
            return add_period_window(pd.DataFrame(columns=_TREND_COLS), window)

        state_data[self.time_column] = pd.to_datetime(state_data[self.time_column])

        # Calculate duration for each state
        state_data["next_time"] = state_data[self.time_column].shift(-1)
        state_data["duration_s"] = (
            state_data["next_time"] - state_data[self.time_column]
        ).dt.total_seconds()
        state_data["duration_minutes"] = state_data["duration_s"] / 60

        # Treat over-long gaps as "no data" so they are excluded below.
        excluded: pd.Series | None = None
        if max_gap is not None:
            cap_minutes = pd.to_timedelta(max_gap).total_seconds() / 60
            over_gap = state_data["duration_minutes"] > cap_minutes
            excluded = state_data.loc[over_gap].set_index(self.time_column)[
                "duration_s"
            ]
            state_data.loc[over_gap, ["duration_minutes", "duration_s"]] = float("nan")

        # Mark running vs downtime
        state_data["is_running"] = state_mask(
            state_data, value_column, running_value, bool_value=True
        )

        # Remove NaN durations
        state_data = state_data[state_data["duration_minutes"].notna()]

        if state_data.empty:
            return add_period_window(pd.DataFrame(columns=_TREND_COLS), window)

        # Group by time window
        state_data = state_data.set_index(self.time_column)

        grouper = pd.Grouper(freq=window)
        seconds = (
            state_data.assign(
                _up=state_data["duration_s"].where(state_data["is_running"], 0.0),
                _down=state_data["duration_s"].where(~state_data["is_running"], 0.0),
            )
            .groupby(grouper)[["_up", "_down"]]
            .sum()
        )
        seconds_by_period = dict(
            zip(seconds.index, seconds.itertuples(index=False, name=None), strict=True)
        )
        # Excluded time is booked to the window (as reported in start/end)
        # holding the reading before the gap.
        excluded_by_period: dict = {}
        if excluded is not None and len(excluded):
            labels = pd.Series(seconds.index)
            lo, hi = window_bounds(labels, window)
            bins = pd.IntervalIndex.from_arrays(lo, hi, closed="left")
            pos = bins.get_indexer(excluded.index)
            hit = pos >= 0
            sums = excluded[hit].groupby(pos[hit]).sum()
            excluded_by_period = {labels.iloc[i]: v for i, v in sums.items()}

        results = []
        for period, group in state_data.groupby(grouper):
            if group.empty:
                continue
            excluded_seconds = float(excluded_by_period.get(period, 0.0))

            uptime = group[group["is_running"]]["duration_minutes"].sum()
            downtime = group[~group["is_running"]]["duration_minutes"].sum()
            total = uptime + downtime

            availability = (uptime / total * 100) if total > 0 else 0

            results.append(
                {
                    "period": period,
                    "availability_pct": availability,
                    "uptime_minutes": uptime,
                    "uptime_seconds": float(seconds_by_period[period][0]),
                    "downtime_minutes": downtime,
                    "downtime_seconds": float(seconds_by_period[period][1]),
                    "excluded_seconds": excluded_seconds,
                }
            )

        return add_period_window(pd.DataFrame(results, columns=_TREND_COLS), window)


#: Deprecated name, removed in the next major release.
DowntimeTracking = deprecated_class(DowntimeEvents, "DowntimeTracking")
