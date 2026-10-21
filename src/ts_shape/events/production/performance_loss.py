"""Performance/speed loss tracking.

Track when machine runs slower than ideal/target speed:
- Identify slow periods vs target cycle time
- Calculate performance loss in minutes and parts
- Performance trend over time
"""

import logging
from typing import TYPE_CHECKING

import pandas as pd

from ts_shape.errors import ColumnNotFoundError
from ts_shape.events._windows import (
    add_period_window,
    add_shift_window,
    data_tz,
)
from ts_shape.utils.base import Base
from ts_shape.utils.counters import counter_increments
from ts_shape.utils.deprecation import deprecated_class
from ts_shape.utils.time_accounting import (
    grouper_periods,
    shift_periods,
    split_intervals,
)

if TYPE_CHECKING:
    from typing_extensions import deprecated

logger = logging.getLogger(__name__)

_SHIFT_COLS = [
    "date",
    "shift",
    "actual_parts",
    "avg_cycle_time_s",
    "avg_cycle_time_seconds",
    "target_cycle_time_s",
    "target_cycle_time_seconds",
    "performance_pct",
    "loss_minutes",
    "loss_seconds",
    "run_seconds",
]
_SLOW_COLS = [
    "start",
    "end",
    "duration_seconds",
    "uuid",
    "source_uuid",
    "actual_parts",
    "avg_cycle_time_s",
    "avg_cycle_time_seconds",
    "performance_pct",
    "loss_minutes",
    "loss_seconds",
    "run_seconds",
]
_TREND_COLS = [
    "period",
    "actual_parts",
    "avg_cycle_time_s",
    "avg_cycle_time_seconds",
    "performance_pct",
    "loss_minutes",
    "loss_seconds",
    "run_seconds",
]


def _loss_row(
    actual_parts: float, elapsed: float, target_cycle_time: float
) -> tuple[int | float, float, float, float, float]:
    """``(parts, avg cycle s, performance %, loss s, run s)`` of a period.

    ISO 22400-2: performance = ideal cycle time x produced quantity / run
    time, and the actual (average) cycle time is run time / produced
    quantity. ``actual_parts`` is the produced quantity (counter increments,
    not the number of readings) and ``elapsed`` the run time in the period.
    """
    actual_parts = round(float(actual_parts), 9)
    elapsed = float(elapsed)
    avg_ct = elapsed / actual_parts if actual_parts > 0 else float("nan")
    if elapsed > 0 and target_cycle_time > 0:
        performance = (actual_parts * target_cycle_time / elapsed) * 100.0
    else:
        performance = 0.0
    loss_seconds = max(0.0, elapsed - actual_parts * target_cycle_time)
    parts_out = int(actual_parts) if actual_parts.is_integer() else actual_parts
    return parts_out, avg_ct, performance, loss_seconds, elapsed


def _by_period(
    data: pd.DataFrame, time_column: str, periods: pd.DataFrame, keys: list[str]
) -> pd.DataFrame:
    """Run seconds and parts per period, each cycle split at the boundaries.

    A cycle completed by the reading at ``t`` ran during ``(t - interval_s,
    t]``; its time is cut at the period boundaries and its parts are shared
    in proportion to the time in each period (constant rate within a cycle).
    Returns the period rows (``keys``, ``start``, ``end``) that received run
    time, in time order, with ``run_seconds`` and ``parts``.
    """
    end = data[time_column].reset_index(drop=True)
    intervals = pd.DataFrame(
        {
            "start": end - pd.to_timedelta(data["interval_s"].to_numpy(), unit="s"),
            "end": end,
            "parts": data["parts"].to_numpy(dtype="float64"),
        }
    )
    per = periods.sort_values("start", kind="stable").reset_index(drop=True)
    per["_p"] = range(len(per))
    pieces = split_intervals(intervals, per, keys=["_p"])
    pieces["_parts"] = pieces["parts"] * pieces["fraction"]
    sums = pieces.groupby("_p")[["seconds", "_parts"]].sum()
    sums = sums[sums["seconds"] > 0]
    out = per.loc[sums.index.astype(int), [*keys, "start", "end"]]
    out = out.reset_index(drop=True)
    out["run_seconds"] = sums["seconds"].to_numpy(dtype="float64")
    out["parts"] = sums["_parts"].to_numpy(dtype="float64")
    return out


class PerformanceLossEvents(Base):
    """Track performance and speed losses against target cycle times.

    Identifies hidden losses where the machine is running but slower than
    it should be.  Typically 10-20% of production time is lost to speed losses.

    Each method takes one signal:
    - cycle_uuid: cycle trigger or production counter

    All methods compare against a single ``target_cycle_time``; there are no
    per-part targets.

    Definitions follow ISO 22400-2: ``actual_parts`` is the produced
    quantity (the sum of reset-aware counter increments, so a counter that
    steps by 2 counts 2 parts; for a trigger signal one part per row),
    ``run_seconds`` is the running time the parts were made in (the summed
    intervals between producing readings, minus any dropped by
    ``max_cycle_gap``), ``avg_cycle_time_seconds = run_seconds /
    actual_parts`` and ``performance_pct = target_cycle_time x actual_parts
    / run_seconds x 100``. The speed loss is ``run_seconds - actual_parts x
    target_cycle_time``. Durations are in seconds (``*_seconds``);
    ``loss_minutes`` and the ``*_s`` cycle time columns are kept as a
    convenience. For a line that runs several part numbers, filter the data
    to one part (or one product run) per call.

    Each cycle ``(t_prev, t]`` is booked to the windows / shifts it ran in:
    its time is cut at the boundaries and its parts are shared in
    proportion to the time in each (constant rate within a cycle), so
    ``run_seconds`` never exceeds the window and ``actual_parts`` can be
    fractional for a window that holds part of a cycle.

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
        super().__init__(dataframe, time_column=time_column)
        self.time_column = time_column
        self.event_uuid = event_uuid
        self.shift_definitions = shift_definitions or {
            "shift_1": ("06:00", "14:00"),
            "shift_2": ("14:00", "22:00"),
            "shift_3": ("22:00", "06:00"),
        }

    def _compute_cycle_times(
        self,
        cycle_uuid: str,
        value_column: str = "value_integer",
        *,
        max_cycle_gap: str | None = None,
    ) -> pd.DataFrame:
        """Production intervals from a counter or trigger signal.

        One row per reading that reports production, with ``parts`` (the
        parts produced since the previous producing reading), ``interval_s``
        (the time since it) and ``cycle_time_s`` (``interval_s / parts``).

        For a numeric counter the parts are the reset-aware counter
        increments (:func:`~ts_shape.utils.counters.counter_increments`), so
        a counter that steps by 2 reports 2 parts, and the first increment
        after the baseline reading is counted (its interval runs from the
        baseline). Readings with no increment do not end an interval. For a
        trigger signal (non-numeric ``value_column``) each row is one part
        and the first row is the baseline. A ``value_column`` the frame does
        not have raises :class:`~ts_shape.errors.ColumnNotFoundError` (it is
        not read as a trigger signal).

        When ``max_cycle_gap`` is set, intervals longer than it are dropped
        (with their parts) as stoppages (breaks, downtime) rather than
        counted as slow cycles.
        """
        cols = [self.time_column, "parts", "interval_s", "cycle_time_s"]
        if value_column not in self.dataframe.columns and not self.dataframe.empty:
            raise ColumnNotFoundError(
                f"PerformanceLossEvents: the data has no {value_column!r} column "
                f"for signal {cycle_uuid!r}. Columns: {list(self.dataframe.columns)}"
            )
        data = (
            self.dataframe[self.dataframe["uuid"] == cycle_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        if data.empty:
            return pd.DataFrame(columns=cols)

        data[self.time_column] = pd.to_datetime(data[self.time_column])

        if pd.api.types.is_numeric_dtype(data[value_column]):
            # Counter: the reset-aware increments are the parts produced. The
            # first reading is the baseline the first increment is measured
            # from, so it stays in to anchor the first interval.
            data = data[data[value_column].notna()]
            if data.empty:
                return pd.DataFrame(columns=cols)
            data["parts"] = counter_increments(data[value_column]).to_numpy()
            anchor = pd.Series(False, index=data.index)
            anchor.iloc[0] = True
            data = data[(data["parts"] > 0) | anchor]
        else:
            # Trigger: each row is one cycle.
            data["parts"] = 1.0

        data["interval_s"] = data[self.time_column].diff().dt.total_seconds()
        data = data[data["interval_s"].notna() & (data["interval_s"] > 0)]
        data = data[data["parts"] > 0]
        if max_cycle_gap is not None:
            cap_seconds = pd.to_timedelta(max_cycle_gap).total_seconds()
            data = data[data["interval_s"] <= cap_seconds]
        data["cycle_time_s"] = data["interval_s"] / data["parts"]
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

        Performance = (target_cycle_time * actual_parts) / run_seconds, with
        ``actual_parts`` the produced quantity (counter increments).

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
                avg_cycle_time_seconds, target_cycle_time_s,
                target_cycle_time_seconds, performance_pct, loss_minutes,
                loss_seconds
            - run_seconds: running time of the shift (the basis of
                ``performance_pct``)
            ``avg_cycle_time_s`` / ``target_cycle_time_s`` / ``loss_minutes``
            are convenience duplicates of the ``*_seconds`` columns.
        """
        data = self._compute_cycle_times(
            cycle_uuid, value_column, max_cycle_gap=max_cycle_gap
        )
        if data.empty:
            return add_shift_window(
                pd.DataFrame(columns=_SHIFT_COLS), self.shift_definitions, None
            )

        times = data[self.time_column]
        starts = times - pd.to_timedelta(data["interval_s"].to_numpy(), unit="s")
        periods = shift_periods(
            starts.min(),
            times.max(),
            self.shift_definitions,
            production_day=production_day,
        )
        booked = _by_period(data, self.time_column, periods, ["date", "shift"])

        results = []
        for date, shift, run, parts_in in booked[
            ["date", "shift", "run_seconds", "parts"]
        ].itertuples(index=False, name=None):
            parts, avg_ct, performance, loss_seconds, run_seconds = _loss_row(
                parts_in, run, target_cycle_time
            )
            results.append(
                {
                    "date": date,
                    "shift": shift,
                    "actual_parts": parts,
                    "avg_cycle_time_s": avg_ct,
                    "avg_cycle_time_seconds": avg_ct,
                    "target_cycle_time_s": target_cycle_time,
                    "target_cycle_time_seconds": target_cycle_time,
                    "performance_pct": min(performance, 100.0),
                    "loss_minutes": loss_seconds / 60,
                    "loss_seconds": float(loss_seconds),
                    "run_seconds": run_seconds,
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
            window: Fixed time window per row (default '1h').
            value_column: Column with counter values.
            max_cycle_gap: Optional cap (e.g. "5min"). Cycle intervals longer
                than it are dropped as stops instead of counted as slow
                cycles. None keeps every interval.

        Returns:
            DataFrame with columns:
            - start, end, duration_seconds (float seconds, end - start),
                uuid (event uuid), source_uuid (cycle_uuid), actual_parts,
                avg_cycle_time_s, avg_cycle_time_seconds, performance_pct,
                loss_minutes, loss_seconds
            - run_seconds: running time in the window (the basis of
                ``performance_pct``)
        """
        data = self._compute_cycle_times(
            cycle_uuid, value_column, max_cycle_gap=max_cycle_gap
        )
        if data.empty:
            return pd.DataFrame(columns=_SLOW_COLS)

        booked = self._window_losses(data, window)
        results = []
        for start, end, run, parts_in in booked[
            ["start", "end", "run_seconds", "parts"]
        ].itertuples(index=False, name=None):
            parts, avg_ct, performance, loss_seconds, run_seconds = _loss_row(
                parts_in, run, target_cycle_time
            )
            if performance < threshold_pct:
                results.append(
                    {
                        "start": start,
                        "end": end,
                        "duration_seconds": (end - start).total_seconds(),
                        "uuid": self.event_uuid,
                        "source_uuid": cycle_uuid,
                        "actual_parts": parts,
                        "avg_cycle_time_s": avg_ct,
                        "avg_cycle_time_seconds": avg_ct,
                        "performance_pct": performance,
                        "loss_minutes": loss_seconds / 60,
                        "loss_seconds": float(loss_seconds),
                        "run_seconds": run_seconds,
                    }
                )

        return pd.DataFrame(results, columns=_SLOW_COLS)

    def _window_losses(self, data: pd.DataFrame, window: str) -> pd.DataFrame:
        """:func:`_by_period` over the ``pd.Grouper(freq=window)`` bins."""
        times = data[self.time_column]
        starts = times - pd.to_timedelta(data["interval_s"].to_numpy(), unit="s")
        periods = grouper_periods(starts.min(), times.max(), window)
        return _by_period(data, self.time_column, periods, ["period"])

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
            max_cycle_gap: Optional cap (e.g. "5min"). Cycle intervals longer
                than it are dropped as stops instead of counted as slow
                cycles. None keeps every interval.

        Returns:
            DataFrame with columns:
            - start, end (start + window), duration_seconds
            - period, actual_parts, avg_cycle_time_s, avg_cycle_time_seconds,
                performance_pct, loss_minutes, loss_seconds
            - run_seconds: running time in the period (the basis of
                ``performance_pct``)
        """
        data = self._compute_cycle_times(
            cycle_uuid, value_column, max_cycle_gap=max_cycle_gap
        )
        if data.empty:
            return add_period_window(pd.DataFrame(columns=_TREND_COLS), window)

        booked = self._window_losses(data, window)
        results = []
        for period, run, parts_in in booked[
            ["period", "run_seconds", "parts"]
        ].itertuples(index=False, name=None):
            parts, avg_ct, performance, loss_seconds, run_seconds = _loss_row(
                parts_in, run, target_cycle_time
            )
            results.append(
                {
                    "period": period,
                    "actual_parts": parts,
                    "avg_cycle_time_s": avg_ct,
                    "avg_cycle_time_seconds": avg_ct,
                    "performance_pct": min(performance, 100.0),
                    "loss_minutes": loss_seconds / 60,
                    "loss_seconds": float(loss_seconds),
                    "run_seconds": run_seconds,
                }
            )

        return add_period_window(pd.DataFrame(results, columns=_TREND_COLS), window)


#: Deprecated name, removed in the next major release.
if TYPE_CHECKING:

    @deprecated("PerformanceLossTracking is deprecated, use PerformanceLossEvents")
    class PerformanceLossTracking(PerformanceLossEvents): ...

else:
    PerformanceLossTracking = deprecated_class(
        PerformanceLossEvents, "PerformanceLossTracking"
    )
