"""Rework tracking for parts requiring re-processing.

Track rework events and their impact:
- Rework count by shift
- Rework by reason code
- Rework rate as percentage of production
- Rework cost estimation
- Rework trends over time
"""

import logging
import pandas as pd  # type: ignore
import numpy as np
from typing import Optional, Dict

from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class ReworkTracking(Base):
    """Track parts that require rework (re-processing through a station).

    Each UUID represents one signal:
    - rework_uuid: rework event counter (monotonic integer)
    - reason_uuid: rework reason code signal (optional)
    - total_production_uuid: total production counter (optional)
    - part_id_uuid: part number / product type signal (optional)

    Merge keys: [date, shift] for shift-level, [period] for trend,
    [reason] for reason-level, [part_number] for part-level.

    Pipeline example::

        rework = ReworkTracking(df)
        shift_rework = rework.rework_by_shift('rework_counter')
        # → merge with ShiftReporting.shift_production() on [date, shift]
        # → merge with QualityTracking.nok_by_shift() on [date, shift]
        rate = rework.rework_rate('rework_counter', 'total_counter')
        # → merge with ShiftReporting.shift_production() on [date, shift]

    Example usage:
        tracker = ReworkTracking(df)

        # Rework per shift
        shift_rework = tracker.rework_by_shift(rework_uuid='rework_counter')

        # Rework rate
        rate = tracker.rework_rate(
            rework_uuid='rework_counter',
            total_production_uuid='total_counter',
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

    def _get_counter_quantity(
        self,
        data: pd.DataFrame,
        value_column: str,
    ) -> float:
        """Get quantity from a monotonic counter (last - first)."""
        if data.empty:
            return 0.0
        first_val = data[value_column].iloc[0]
        last_val = data[value_column].iloc[-1]
        return max(0.0, last_val - first_val)

    def rework_by_shift(
        self,
        rework_uuid: str,
        *,
        value_column: str = "value_integer",
    ) -> pd.DataFrame:
        """Rework count per shift.

        Args:
            rework_uuid: UUID for rework event counter signal.
            value_column: Column containing counter values.

        Returns:
            DataFrame with columns:
            - date, shift, rework_count
        """
        data = (
            self.dataframe[self.dataframe["uuid"] == rework_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        if data.empty:
            return pd.DataFrame(columns=["date", "shift", "rework_count"])

        data[self.time_column] = pd.to_datetime(data[self.time_column])
        data["shift"] = data[self.time_column].apply(self._assign_shift)
        data["date"] = data[self.time_column].dt.date

        results = []
        for (date, shift), grp in data.groupby(["date", "shift"]):
            qty = self._get_counter_quantity(grp, value_column)
            results.append(
                {
                    "date": date,
                    "shift": shift,
                    "rework_count": round(qty, 0),
                }
            )

        return pd.DataFrame(results)

    def rework_by_reason(
        self,
        rework_uuid: str,
        reason_uuid: str,
        *,
        value_column_rework: str = "value_integer",
        value_column_reason: str = "value_string",
    ) -> pd.DataFrame:
        """Rework quantity by reason code.

        Args:
            rework_uuid: UUID for rework counter signal.
            reason_uuid: UUID for rework reason code signal.
            value_column_rework: Column containing rework counter values.
            value_column_reason: Column containing reason codes.

        Returns:
            DataFrame with columns:
            - reason, rework_count, pct_of_total
        """
        rework_data = (
            self.dataframe[self.dataframe["uuid"] == rework_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        reason_data = (
            self.dataframe[self.dataframe["uuid"] == reason_uuid]
            .copy()
            .sort_values(self.time_column)
        )

        if rework_data.empty or reason_data.empty:
            return pd.DataFrame(columns=["reason", "rework_count", "pct_of_total"])

        rework_data[self.time_column] = pd.to_datetime(rework_data[self.time_column])
        reason_data[self.time_column] = pd.to_datetime(reason_data[self.time_column])

        rework_clean = rework_data[[self.time_column, value_column_rework]].copy()
        rework_clean = rework_clean.rename(columns={value_column_rework: "rework_val"})

        reason_clean = reason_data[[self.time_column, value_column_reason]].copy()
        reason_clean = reason_clean.rename(columns={value_column_reason: "reason"})

        merged = pd.merge_asof(
            rework_clean,
            reason_clean,
            on=self.time_column,
            direction="backward",
        )
        merged = merged.dropna(subset=["reason"])

        if merged.empty:
            return pd.DataFrame(columns=["reason", "rework_count", "pct_of_total"])

        results = []
        for reason, grp in merged.groupby("reason"):
            qty = self._get_counter_quantity(grp, "rework_val")
            results.append({"reason": reason, "rework_count": round(qty, 0)})

        if not results:
            return pd.DataFrame(columns=["reason", "rework_count", "pct_of_total"])

        result_df = pd.DataFrame(results)
        total = result_df["rework_count"].sum()
        result_df["pct_of_total"] = (
            (result_df["rework_count"] / total * 100).round(1) if total > 0 else 0.0
        )

        return result_df.sort_values("rework_count", ascending=False).reset_index(
            drop=True
        )

    def rework_rate(
        self,
        rework_uuid: str,
        total_production_uuid: str,
        *,
        value_column_rework: str = "value_integer",
        value_column_production: str = "value_integer",
    ) -> pd.DataFrame:
        """Rework rate as percentage of total production per shift.

        Args:
            rework_uuid: UUID for rework counter signal.
            total_production_uuid: UUID for total production counter signal.
            value_column_rework: Column containing rework counter values.
            value_column_production: Column containing production counter values.

        Returns:
            DataFrame with columns:
            - date, shift, total_produced, rework_count, rework_rate_pct
        """
        rework_data = (
            self.dataframe[self.dataframe["uuid"] == rework_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        prod_data = (
            self.dataframe[self.dataframe["uuid"] == total_production_uuid]
            .copy()
            .sort_values(self.time_column)
        )

        if rework_data.empty or prod_data.empty:
            return pd.DataFrame(
                columns=[
                    "date",
                    "shift",
                    "total_produced",
                    "rework_count",
                    "rework_rate_pct",
                ]
            )

        rework_data[self.time_column] = pd.to_datetime(rework_data[self.time_column])
        prod_data[self.time_column] = pd.to_datetime(prod_data[self.time_column])

        rework_data["shift"] = rework_data[self.time_column].apply(self._assign_shift)
        rework_data["date"] = rework_data[self.time_column].dt.date
        prod_data["shift"] = prod_data[self.time_column].apply(self._assign_shift)
        prod_data["date"] = prod_data[self.time_column].dt.date

        # Get all date/shift combos
        date_shifts = set(zip(rework_data["date"], rework_data["shift"])) | set(
            zip(prod_data["date"], prod_data["shift"])
        )

        results = []
        for date, shift in sorted(date_shifts):
            rw_grp = rework_data[
                (rework_data["date"] == date) & (rework_data["shift"] == shift)
            ]
            pr_grp = prod_data[
                (prod_data["date"] == date) & (prod_data["shift"] == shift)
            ]

            rework_count = (
                self._get_counter_quantity(rw_grp, value_column_rework)
                if not rw_grp.empty
                else 0
            )
            total_produced = (
                self._get_counter_quantity(pr_grp, value_column_production)
                if not pr_grp.empty
                else 0
            )

            rate = (rework_count / total_produced * 100) if total_produced > 0 else 0

            results.append(
                {
                    "date": date,
                    "shift": shift,
                    "total_produced": round(total_produced, 0),
                    "rework_count": round(rework_count, 0),
                    "rework_rate_pct": round(rate, 1),
                }
            )

        return pd.DataFrame(results)

    def rework_cost(
        self,
        rework_uuid: str,
        part_id_uuid: str,
        rework_costs: Dict[str, float],
        *,
        value_column_rework: str = "value_integer",
        value_column_part: str = "value_string",
    ) -> pd.DataFrame:
        """Convert rework counts to monetary cost by part number.

        Args:
            rework_uuid: UUID for rework counter signal.
            part_id_uuid: UUID for part number / product type signal.
            rework_costs: Dict mapping part numbers to cost per rework.
            value_column_rework: Column containing rework counter values.
            value_column_part: Column containing part number values.

        Returns:
            DataFrame with columns:
            - part_number, rework_count, cost_per_rework, total_cost
        """
        rework_data = (
            self.dataframe[self.dataframe["uuid"] == rework_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        part_data = (
            self.dataframe[self.dataframe["uuid"] == part_id_uuid]
            .copy()
            .sort_values(self.time_column)
        )

        if rework_data.empty or part_data.empty:
            return pd.DataFrame(
                columns=["part_number", "rework_count", "cost_per_rework", "total_cost"]
            )

        rework_data[self.time_column] = pd.to_datetime(rework_data[self.time_column])
        part_data[self.time_column] = pd.to_datetime(part_data[self.time_column])

        rework_clean = rework_data[[self.time_column, value_column_rework]].copy()
        rework_clean = rework_clean.rename(columns={value_column_rework: "rework_val"})

        part_clean = part_data[[self.time_column, value_column_part]].copy()
        part_clean = part_clean.rename(columns={value_column_part: "part_number"})

        merged = pd.merge_asof(
            rework_clean,
            part_clean,
            on=self.time_column,
            direction="backward",
        )
        merged = merged.dropna(subset=["part_number"])

        results = []
        for part_num, grp in merged.groupby("part_number"):
            qty = self._get_counter_quantity(grp, "rework_val")
            cost = rework_costs.get(part_num, 0.0)
            results.append(
                {
                    "part_number": part_num,
                    "rework_count": round(qty, 0),
                    "cost_per_rework": cost,
                    "total_cost": round(qty * cost, 2),
                }
            )

        return (
            pd.DataFrame(results)
            .sort_values("total_cost", ascending=False)
            .reset_index(drop=True)
        )

    def rework_trend(
        self,
        rework_uuid: str,
        *,
        value_column: str = "value_integer",
        window: str = "1D",
    ) -> pd.DataFrame:
        """Track rework count trend over time.

        Args:
            rework_uuid: UUID for rework counter signal.
            value_column: Column containing counter values.
            window: Time window for aggregation (default '1D').

        Returns:
            DataFrame with columns:
            - period, rework_count
        """
        data = (
            self.dataframe[self.dataframe["uuid"] == rework_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        if data.empty:
            return pd.DataFrame(columns=["period", "rework_count"])

        data[self.time_column] = pd.to_datetime(data[self.time_column])
        data = data.set_index(self.time_column)

        results = []
        for period, grp in data.groupby(pd.Grouper(freq=window)):
            if grp.empty:
                continue
            qty = self._get_counter_quantity(grp.reset_index(), value_column)
            results.append(
                {
                    "period": period,
                    "rework_count": round(qty, 0),
                }
            )

        return pd.DataFrame(results)
