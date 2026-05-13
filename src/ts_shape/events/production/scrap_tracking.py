"""Scrap and material waste tracking.

Track material waste and scrap (different from NOK parts):
- Scrap by shift and reason
- Scrap cost calculation
- Scrap trends over time
"""

import logging
import pandas as pd  # type: ignore
import numpy as np
from typing import Optional, Dict

from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class ScrapTracking(Base):
    """Track material scrap and waste.

    Each UUID represents one signal:
    - scrap_uuid: scrap weight or count signal
    - reason_uuid: scrap reason code (optional)
    - part_id_uuid: part number / material type (optional)

    Merge keys: [date, shift] for shift-level, [period] for trend,
    [reason] for reason-level, [part_number] for part-level.

    Pipeline example::

        scrap = ScrapTracking(df)
        shift_scrap = scrap.scrap_by_shift('scrap_weight')
        # → merge with ShiftReporting.shift_production() on [date, shift]
        # → merge with QualityTracking.nok_by_shift() on [date, shift]
        cost = scrap.scrap_cost('scrap_weight', 'part_id', {'A': 12.5})
        # → merge with QualityTracking.quality_by_part() on [part_number]

    Example usage:
        tracker = ScrapTracking(df)

        # Scrap per shift
        shift_scrap = tracker.scrap_by_shift(scrap_uuid='scrap_weight')

        # Scrap by reason
        reasons = tracker.scrap_by_reason(
            scrap_uuid='scrap_weight',
            reason_uuid='scrap_reason'
        )

        # Convert to cost
        cost = tracker.scrap_cost(
            scrap_uuid='scrap_weight',
            part_id_uuid='part_number',
            material_costs={'PART_A': 12.50, 'PART_B': 8.75}
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

    def scrap_by_shift(
        self,
        scrap_uuid: str,
        *,
        value_column: str = "value_double",
    ) -> pd.DataFrame:
        """Scrap quantity per shift.

        Args:
            scrap_uuid: UUID for scrap weight/count signal.
            value_column: Column containing scrap values.

        Returns:
            DataFrame with columns:
            - date, shift, scrap_quantity
        """
        data = (
            self.dataframe[self.dataframe["uuid"] == scrap_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        if data.empty:
            return pd.DataFrame(columns=["date", "shift", "scrap_quantity"])

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
                    "scrap_quantity": round(qty, 2),
                }
            )

        return pd.DataFrame(results)

    def scrap_by_reason(
        self,
        scrap_uuid: str,
        reason_uuid: str,
        *,
        value_column_scrap: str = "value_double",
        value_column_reason: str = "value_string",
    ) -> pd.DataFrame:
        """Scrap quantity by reason code.

        Args:
            scrap_uuid: UUID for scrap weight/count signal.
            reason_uuid: UUID for scrap reason code signal.
            value_column_scrap: Column containing scrap values.
            value_column_reason: Column containing reason codes.

        Returns:
            DataFrame with columns:
            - reason, scrap_quantity, pct_of_total
        """
        scrap_data = (
            self.dataframe[self.dataframe["uuid"] == scrap_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        reason_data = (
            self.dataframe[self.dataframe["uuid"] == reason_uuid]
            .copy()
            .sort_values(self.time_column)
        )

        if scrap_data.empty or reason_data.empty:
            return pd.DataFrame(columns=["reason", "scrap_quantity", "pct_of_total"])

        scrap_data[self.time_column] = pd.to_datetime(scrap_data[self.time_column])
        reason_data[self.time_column] = pd.to_datetime(reason_data[self.time_column])

        scrap_clean = scrap_data[[self.time_column, value_column_scrap]].copy()
        scrap_clean = scrap_clean.rename(columns={value_column_scrap: "scrap_val"})

        reason_clean = reason_data[[self.time_column, value_column_reason]].copy()
        reason_clean = reason_clean.rename(columns={value_column_reason: "reason"})

        merged = pd.merge_asof(
            scrap_clean,
            reason_clean,
            on=self.time_column,
            direction="backward",
        )
        merged = merged.dropna(subset=["reason"])

        if merged.empty:
            return pd.DataFrame(columns=["reason", "scrap_quantity", "pct_of_total"])

        # For each reason group, compute counter delta
        results = []
        for reason, grp in merged.groupby("reason"):
            qty = self._get_counter_quantity(grp, "scrap_val")
            results.append({"reason": reason, "scrap_quantity": round(qty, 2)})

        if not results:
            return pd.DataFrame(columns=["reason", "scrap_quantity", "pct_of_total"])

        result_df = pd.DataFrame(results)
        total = result_df["scrap_quantity"].sum()
        result_df["pct_of_total"] = (
            (result_df["scrap_quantity"] / total * 100).round(1) if total > 0 else 0.0
        )
        return result_df.sort_values("scrap_quantity", ascending=False).reset_index(
            drop=True
        )

    def scrap_cost(
        self,
        scrap_uuid: str,
        part_id_uuid: str,
        material_costs: Dict[str, float],
        *,
        value_column_scrap: str = "value_double",
        value_column_part: str = "value_string",
    ) -> pd.DataFrame:
        """Convert scrap quantities to monetary cost.

        Args:
            scrap_uuid: UUID for scrap weight/count signal.
            part_id_uuid: UUID for part number / material type signal.
            material_costs: Dict mapping part numbers to cost per unit scrap.
            value_column_scrap: Column containing scrap values.
            value_column_part: Column containing part numbers.

        Returns:
            DataFrame with columns:
            - part_number, scrap_quantity, cost_per_unit, total_cost
        """
        scrap_data = (
            self.dataframe[self.dataframe["uuid"] == scrap_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        part_data = (
            self.dataframe[self.dataframe["uuid"] == part_id_uuid]
            .copy()
            .sort_values(self.time_column)
        )

        if scrap_data.empty or part_data.empty:
            return pd.DataFrame(
                columns=["part_number", "scrap_quantity", "cost_per_unit", "total_cost"]
            )

        scrap_data[self.time_column] = pd.to_datetime(scrap_data[self.time_column])
        part_data[self.time_column] = pd.to_datetime(part_data[self.time_column])

        scrap_clean = scrap_data[[self.time_column, value_column_scrap]].copy()
        scrap_clean = scrap_clean.rename(columns={value_column_scrap: "scrap_val"})

        part_clean = part_data[[self.time_column, value_column_part]].copy()
        part_clean = part_clean.rename(columns={value_column_part: "part_number"})

        merged = pd.merge_asof(
            scrap_clean,
            part_clean,
            on=self.time_column,
            direction="backward",
        )
        merged = merged.dropna(subset=["part_number"])

        results = []
        for part_num, grp in merged.groupby("part_number"):
            qty = self._get_counter_quantity(grp, "scrap_val")
            cost_per_unit = material_costs.get(part_num, 0.0)
            results.append(
                {
                    "part_number": part_num,
                    "scrap_quantity": round(qty, 2),
                    "cost_per_unit": cost_per_unit,
                    "total_cost": round(qty * cost_per_unit, 2),
                }
            )

        return (
            pd.DataFrame(results)
            .sort_values("total_cost", ascending=False)
            .reset_index(drop=True)
        )

    def scrap_trend(
        self,
        scrap_uuid: str,
        *,
        value_column: str = "value_double",
        window: str = "1D",
    ) -> pd.DataFrame:
        """Track scrap quantity trend over time.

        Args:
            scrap_uuid: UUID for scrap weight/count signal.
            value_column: Column containing scrap values.
            window: Time window for aggregation (default '1D').

        Returns:
            DataFrame with columns:
            - period, scrap_quantity
        """
        data = (
            self.dataframe[self.dataframe["uuid"] == scrap_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        if data.empty:
            return pd.DataFrame(columns=["period", "scrap_quantity"])

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
                    "scrap_quantity": round(qty, 2),
                }
            )

        return pd.DataFrame(results)
