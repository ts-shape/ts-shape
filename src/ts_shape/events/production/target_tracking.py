"""Target vs actual comparison module.

Generic module for comparing any metric to targets:
- Shift/daily target comparison
- Variance analysis
- Target achievement tracking over time
"""

import logging
import pandas as pd  # type: ignore
import numpy as np
from typing import Optional, Dict, Union

from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class TargetTracking(Base):
    """Compare any production metric to targets.

    Every plant has daily/shift targets.  This module provides a generic way to
    compare actual performance against those targets.

    Merge keys: [date, shift] for shift-level, [date] for daily.

    Pipeline example::

        target = TargetTracking(df)
        result = target.compare_to_target('counter', {'shift_1': 450})
        # → merge with PerformanceLossTracking.performance_by_shift() on [date, shift]
        # → merge with QualityTracking.nok_by_shift() on [date, shift]
        # → result['status'] column enables filtering/alerting

    Example usage:
        tracker = TargetTracking(df)

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
        shift_definitions: Optional[Dict[str, tuple[str, str]]] = None,
    ) -> None:
        super().__init__(dataframe, column_name=time_column)
        self.time_column = time_column
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

    def compare_to_target(
        self,
        metric_uuid: str,
        targets: Dict[str, float],
        *,
        value_column: str = "value_integer",
    ) -> pd.DataFrame:
        """Compare actual metric to per-shift targets.

        Args:
            metric_uuid: UUID of the metric signal (counter).
            targets: Dict mapping shift names to target values.
            value_column: Column containing metric values.

        Returns:
            DataFrame with columns:
            - date, shift, actual, target, variance, achievement_pct, status
        """
        data = (
            self.dataframe[self.dataframe["uuid"] == metric_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        if data.empty:
            return pd.DataFrame(
                columns=[
                    "date",
                    "shift",
                    "actual",
                    "target",
                    "variance",
                    "achievement_pct",
                    "status",
                ]
            )

        data[self.time_column] = pd.to_datetime(data[self.time_column])
        data["shift"] = data[self.time_column].apply(self._assign_shift)
        data["date"] = data[self.time_column].dt.date

        results = []
        for (date, shift), grp in data.groupby(["date", "shift"]):
            grp = grp.sort_values(self.time_column)
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

            results.append(
                {
                    "date": date,
                    "shift": shift,
                    "actual": int(actual),
                    "target": target,
                    "variance": round(variance, 1),
                    "achievement_pct": round(achievement, 1),
                    "status": status,
                }
            )

        return pd.DataFrame(results)

    def target_achievement_summary(
        self,
        metric_uuid: str,
        daily_target: float,
        *,
        value_column: str = "value_integer",
    ) -> pd.DataFrame:
        """Summarize target achievement over time.

        Args:
            metric_uuid: UUID of the metric signal (counter).
            daily_target: Daily production target.
            value_column: Column containing metric values.

        Returns:
            DataFrame with columns:
            - date, actual, target, variance, achievement_pct, status
        """
        data = (
            self.dataframe[self.dataframe["uuid"] == metric_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        if data.empty:
            return pd.DataFrame(
                columns=[
                    "date",
                    "actual",
                    "target",
                    "variance",
                    "achievement_pct",
                    "status",
                ]
            )

        data[self.time_column] = pd.to_datetime(data[self.time_column])
        data["date"] = data[self.time_column].dt.date

        results = []
        for date, grp in data.groupby("date"):
            grp = grp.sort_values(self.time_column)
            actual = max(0, grp[value_column].iloc[-1] - grp[value_column].iloc[0])

            variance = actual - daily_target
            achievement = (actual / daily_target * 100) if daily_target > 0 else 0.0

            if achievement >= 100:
                status = "on_target"
            elif achievement >= 90:
                status = "warning"
            else:
                status = "below_target"

            results.append(
                {
                    "date": date,
                    "actual": int(actual),
                    "target": daily_target,
                    "variance": round(variance, 1),
                    "achievement_pct": round(achievement, 1),
                    "status": status,
                }
            )

        return pd.DataFrame(results)

    def target_hit_rate(
        self,
        metric_uuid: str,
        daily_target: float,
        *,
        value_column: str = "value_integer",
    ) -> Dict[str, Union[float, int]]:
        """How often are targets met?

        Args:
            metric_uuid: UUID of the metric signal (counter).
            daily_target: Daily production target.
            value_column: Column containing metric values.

        Returns:
            Dict with:
            - total_days: Number of days analyzed.
            - days_on_target: Days where target was met.
            - hit_rate_pct: Percentage of days meeting target.
            - avg_achievement_pct: Average achievement across days.
        """
        summary = self.target_achievement_summary(
            metric_uuid, daily_target, value_column=value_column
        )
        if summary.empty:
            return {
                "total_days": 0,
                "days_on_target": 0,
                "hit_rate_pct": 0.0,
                "avg_achievement_pct": 0.0,
            }

        total_days = len(summary)
        days_on_target = int((summary["achievement_pct"] >= 100).sum())
        hit_rate = days_on_target / total_days * 100 if total_days > 0 else 0.0
        avg_achievement = summary["achievement_pct"].mean()

        return {
            "total_days": total_days,
            "days_on_target": days_on_target,
            "hit_rate_pct": round(hit_rate, 1),
            "avg_achievement_pct": round(avg_achievement, 1),
        }
