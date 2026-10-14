"""Target vs actual comparison module.

Generic module for comparing any metric to targets:
- Shift/daily target comparison
- Variance analysis
- Target achievement tracking over time
"""

import logging
from typing import Literal, overload

import pandas as pd

from ts_shape.events._compat import warn_dict_return
from ts_shape.events._windows import data_tz
from ts_shape.utils.base import Base
from ts_shape.utils.counters import count_resets, counter_quantity
from ts_shape.utils.deprecation import deprecated_class
from ts_shape.utils.shifts import assign_shift, production_dates

logger = logging.getLogger(__name__)


class TargetEvents(Base):
    """Compare any production metric to targets.

    Every plant has daily/shift targets.  This module provides a generic way to
    compare actual performance against those targets.

    Merge keys: [date, shift] for shift-level, [date] for daily.

    Pipeline example::

        target = TargetEvents(df)
        result = target.compare_to_target('counter', {'shift_1': 450})
        # → merge with PerformanceLossEvents.performance_by_shift() on [date, shift]
        # → merge with QualityEvents.nok_by_shift() on [date, shift]
        # → result['status'] column enables filtering/alerting

    Example usage:
        tracker = TargetEvents(df)

        # Compare counter to fixed target
        result = tracker.compare_to_target(
            metric_uuid='production_counter',
            targets={'shift_1': 450, 'shift_2': 450, 'shift_3': 400},
        )

        # Achievement summary over time
        summary = tracker.target_achievement_summary(
            metric_uuid='production_counter',
            daily_target=1300,
        )
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        time_column: str = "systime",
        shift_definitions: dict[str, tuple[str, str]] | None = None,
    ) -> None:
        super().__init__(dataframe, time_column=time_column)
        self.time_column = time_column
        self.shift_definitions = shift_definitions or {
            "shift_1": ("06:00", "14:00"),
            "shift_2": ("14:00", "22:00"),
            "shift_3": ("22:00", "06:00"),
        }

    def _assign_shift(self, timestamp: pd.Timestamp) -> str:
        """Assign shift based on time of day."""
        return assign_shift(timestamp, self.shift_definitions)

    def compare_to_target(
        self,
        metric_uuid: str,
        targets: dict[str, float],
        *,
        value_column: str = "value_integer",
        handle_resets: bool = True,
        production_day: bool = True,
    ) -> pd.DataFrame:
        """Compare actual metric to per-shift targets.

        Args:
            metric_uuid: UUID of the metric signal (counter).
            targets: Dict mapping shift names to target values.
            value_column: Column containing metric values.
            handle_resets: When True (default), treat a decreasing counter as a reset and
                count the production after the reset instead of losing it, so
                ``actual`` is the reset-aware total. The result gains a
                ``resets`` column with the number of resets observed in each
                shift. When False, ``actual`` is simply
                ``max(0, last - first)`` and a shift containing a reset reports
                too little.
            production_day: When True, the post-midnight portion of an overnight
                shift (e.g. ``22:00 -> 06:00``) is attributed to the shift's
                start date, so each ``(date, shift)`` group is one contiguous
                interval (the default). When False, readings are grouped by their
                raw calendar date; a night shift is then split across two dates
                and, within a date, its group spans the whole day so
                ``last - first`` sweeps in other shifts' production. Only
                affects schedules with an overnight shift.

        Returns:
            DataFrame with columns:
            - date, shift, actual, target, variance, achievement_pct, status
            - resets (only when ``handle_resets=True``)
        """
        base_columns = [
            "date",
            "shift",
            "actual",
            "target",
            "variance",
            "achievement_pct",
            "status",
        ]
        if handle_resets:
            base_columns.append("resets")
        data = (
            self.dataframe[self.dataframe["uuid"] == metric_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        if data.empty:
            return pd.DataFrame(columns=base_columns)

        data[self.time_column] = pd.to_datetime(data[self.time_column])
        data["shift"] = assign_shift(data[self.time_column], self.shift_definitions)
        if production_day:
            data["date"] = production_dates(
                data[self.time_column], data["shift"], self.shift_definitions
            )
        else:
            data["date"] = data[self.time_column].dt.date

        results = []
        for (date, shift), grp in data.groupby(["date", "shift"]):
            grp = grp.sort_values(self.time_column)
            if handle_resets:
                actual = counter_quantity(grp[value_column], handle_resets=True)
            else:
                actual = max(0, grp[value_column].iloc[-1] - grp[value_column].iloc[0])
            target = targets.get(shift, 0.0)

            variance = actual - target
            achievement = (actual / target * 100) if target > 0 else 0.0

            if achievement >= 100:
                status = "on_target"
            elif achievement >= 90:
                status = "warning"
            else:
                status = "below_target"

            row = {
                "date": date,
                "shift": shift,
                "actual": int(actual),
                "target": target,
                "variance": variance,
                "achievement_pct": achievement,
                "status": status,
            }
            if handle_resets:
                row["resets"] = count_resets(grp[value_column])
            results.append(row)

        return pd.DataFrame(results, columns=base_columns)

    def target_achievement_summary(
        self,
        metric_uuid: str,
        daily_target: float,
        *,
        value_column: str = "value_integer",
        handle_resets: bool = True,
    ) -> pd.DataFrame:
        """Summarize target achievement over time.

        Args:
            metric_uuid: UUID of the metric signal (counter).
            daily_target: Daily production target.
            value_column: Column containing metric values.
            handle_resets: When True (default), treat a decreasing counter as a reset and
                count the production after the reset instead of losing it, so
                ``actual`` is the reset-aware daily total. The result gains a
                ``resets`` column with the number of resets observed each day.
                When False, ``actual`` is simply
                ``max(0, last - first)`` and a day containing a reset reports
                too little.

        Returns:
            DataFrame with columns:
            - date, actual, target, variance, achievement_pct, status
            - resets (only when ``handle_resets=True``)
        """
        base_columns = [
            "date",
            "actual",
            "target",
            "variance",
            "achievement_pct",
            "status",
        ]
        if handle_resets:
            base_columns.append("resets")
        data = (
            self.dataframe[self.dataframe["uuid"] == metric_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        if data.empty:
            return pd.DataFrame(columns=base_columns)

        data[self.time_column] = pd.to_datetime(data[self.time_column])
        data["date"] = data[self.time_column].dt.date

        results = []
        for date, grp in data.groupby("date"):
            grp = grp.sort_values(self.time_column)
            if handle_resets:
                actual = counter_quantity(grp[value_column], handle_resets=True)
            else:
                actual = max(0, grp[value_column].iloc[-1] - grp[value_column].iloc[0])

            variance = actual - daily_target
            achievement = (actual / daily_target * 100) if daily_target > 0 else 0.0

            if achievement >= 100:
                status = "on_target"
            elif achievement >= 90:
                status = "warning"
            else:
                status = "below_target"

            row = {
                "date": date,
                "actual": int(actual),
                "target": daily_target,
                "variance": variance,
                "achievement_pct": achievement,
                "status": status,
            }
            if handle_resets:
                row["resets"] = count_resets(grp[value_column])
            results.append(row)

        return pd.DataFrame(results, columns=base_columns)

    @overload
    def target_hit_rate(
        self,
        metric_uuid: str,
        daily_target: float,
        *,
        value_column: str = ...,
        as_frame: Literal[True] = ...,
    ) -> pd.DataFrame: ...

    @overload
    def target_hit_rate(
        self,
        metric_uuid: str,
        daily_target: float,
        *,
        value_column: str = ...,
        as_frame: Literal[False],
    ) -> dict[str, float | int]: ...

    def target_hit_rate(
        self,
        metric_uuid: str,
        daily_target: float,
        *,
        value_column: str = "value_integer",
        as_frame: bool = True,
    ) -> pd.DataFrame | dict[str, float | int]:
        """How often are targets met?

        Args:
            metric_uuid: UUID of the metric signal (counter).
            daily_target: Daily production target.
            value_column: Column containing metric values.
            as_frame: When True (default), return a one-row DataFrame (see
                below). ``as_frame=False`` returns the dictionary form; it is
                deprecated, emits a ``DeprecationWarning`` and will be removed
                in the next major release.

        Returns:
            With ``as_frame=True`` (default), a one-row summary DataFrame with
            columns:
            - start, end, duration_seconds: first analysed day's midnight to
                the midnight after the last day (tz-aware in the data's tz;
                NaT / 0 without data)
            - total_days: Number of days analyzed.
            - days_on_target: Days where target was met.
            - hit_rate_pct: Percentage of days meeting target.
            - avg_achievement_pct: Average achievement across days.

            With ``as_frame=False`` (deprecated), a dict with the keys
            total_days, days_on_target, hit_rate_pct, avg_achievement_pct.
        """
        if not as_frame:
            warn_dict_return("TargetEvents.target_hit_rate")
        summary = self.target_achievement_summary(
            metric_uuid, daily_target, value_column=value_column
        )
        if summary.empty:
            rate: dict[str, float | int] = {
                "total_days": 0,
                "days_on_target": 0,
                "hit_rate_pct": 0.0,
                "avg_achievement_pct": 0.0,
            }
        else:
            total_days = len(summary)
            days_on_target = int((summary["achievement_pct"] >= 100).sum())
            hit_rate = days_on_target / total_days * 100 if total_days > 0 else 0.0
            avg_achievement = summary["achievement_pct"].mean()
            rate = {
                "total_days": total_days,
                "days_on_target": days_on_target,
                "hit_rate_pct": hit_rate,
                "avg_achievement_pct": avg_achievement,
            }
        if not as_frame:
            return rate

        if summary.empty:
            start, end, span = pd.NaT, pd.NaT, 0.0
        else:
            tz = data_tz(
                self.dataframe.loc[
                    self.dataframe["uuid"] == metric_uuid, self.time_column
                ]
            )
            first = pd.Timestamp(min(summary["date"]))
            last = pd.Timestamp(max(summary["date"])) + pd.Timedelta(days=1)
            if tz is not None:
                first = first.tz_localize(
                    tz, nonexistent="shift_forward", ambiguous=True
                )
                last = last.tz_localize(tz, nonexistent="shift_forward", ambiguous=True)
            start, end = first, last
            span = (end - start).total_seconds()
        return pd.DataFrame(
            [{"start": start, "end": end, "duration_seconds": span, **rate}]
        )


#: Deprecated name, removed in the next major release.
TargetTracking = deprecated_class(TargetEvents, "TargetTracking")
