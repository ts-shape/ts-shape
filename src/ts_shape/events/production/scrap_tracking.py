"""Scrap and material waste tracking.

Track material waste and scrap (different from NOK parts):
- Scrap by shift and reason
- Scrap cost calculation
- Scrap trends over time
"""

import logging

import pandas as pd

from ts_shape.events._windows import (
    add_period_window,
    add_shift_window,
    data_tz,
)
from ts_shape.utils.base import Base
from ts_shape.utils.counters import count_resets, counter_increments, counter_quantity
from ts_shape.utils.deprecation import deprecated_class
from ts_shape.utils.shifts import production_dates

logger = logging.getLogger(__name__)


class ScrapEvents(Base):
    """Track material scrap and waste.

    Each UUID represents one signal:
    - scrap_uuid: scrap weight or count signal
    - reason_uuid: scrap reason code (optional)
    - part_id_uuid: part number / material type (optional)

    Merge keys: [date, shift] for shift-level, [period] for trend,
    [reason] for reason-level, [part_number] for part-level.

    Pipeline example::

        scrap = ScrapEvents(df)
        shift_scrap = scrap.scrap_by_shift('scrap_weight')
        # → merge with ShiftReportEvents.shift_production() on [date, shift]
        # → merge with QualityEvents.nok_by_shift() on [date, shift]
        cost = scrap.scrap_cost('scrap_weight', 'part_id', {'A': 12.5})
        # → merge with QualityEvents.quality_by_part() on [part_number]

    Example usage:
        tracker = ScrapEvents(df)

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
        *,
        handle_resets: bool = False,
    ) -> float:
        """Get quantity from a monotonic counter.

        When ``handle_resets`` is False the quantity is the
        backward-compatible ``max(0, last - first)``. When True a decreasing
        counter is treated as a reset and the post-reset production is
        counted (see :func:`ts_shape.utils.counters.counter_quantity`).
        """
        if data.empty:
            return 0.0
        return counter_quantity(data[value_column], handle_resets=handle_resets)

    def scrap_by_shift(
        self,
        scrap_uuid: str,
        *,
        value_column: str = "value_double",
        handle_resets: bool = False,
        production_day: bool = True,
    ) -> pd.DataFrame:
        """Scrap quantity per shift.

        Args:
            scrap_uuid: UUID for scrap weight/count signal.
            value_column: Column containing scrap values.
            handle_resets: When True, treat a decreasing counter as a reset
                and count the scrap accumulated after the reset instead of
                losing it. The result gains a ``resets`` column with the
                number of resets observed in each shift. When False (default),
                the quantity is simply ``max(0, last - first)`` and a shift
                containing a reset reports too little (often 0).
            production_day: When True, an overnight shift's post-midnight
                readings are attributed to the shift's start date, so a night
                shift is one contiguous ``(date, shift)`` group. Default True;
                False groups by raw calendar date, which splits a night shift
                across two rows. Only affects overnight schedules.

        Returns:
            DataFrame with columns:
            - start, end, duration_seconds: the shift window from
                ``shift_definitions`` (``date`` at the shift start time,
                tz-aware in the data's tz; an overnight shift ends on the
                next day)
            - date, shift, scrap_quantity
            - resets: Number of counter resets in the shift (only when
                ``handle_resets=True``)
        """
        columns = ["date", "shift", "scrap_quantity"]
        if handle_resets:
            columns.append("resets")

        data = (
            self.dataframe[self.dataframe["uuid"] == scrap_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        if data.empty:
            return add_shift_window(
                pd.DataFrame(columns=columns), self.shift_definitions, None
            )

        data[self.time_column] = pd.to_datetime(data[self.time_column])
        data["shift"] = data[self.time_column].apply(self._assign_shift)
        if production_day:
            data["date"] = production_dates(
                data[self.time_column], data["shift"], self.shift_definitions
            )
        else:
            data["date"] = data[self.time_column].dt.date

        results = []
        for (date, shift), grp in data.groupby(["date", "shift"]):
            qty = self._get_counter_quantity(
                grp, value_column, handle_resets=handle_resets
            )
            row = {
                "date": date,
                "shift": shift,
                "scrap_quantity": round(qty, 2),
            }
            if handle_resets:
                row["resets"] = count_resets(grp[value_column])
            results.append(row)

        return add_shift_window(
            pd.DataFrame(results, columns=columns),
            self.shift_definitions,
            data_tz(data[self.time_column]),
        )

    def scrap_by_reason(
        self,
        scrap_uuid: str,
        reason_uuid: str,
        *,
        value_column_scrap: str = "value_double",
        value_column_reason: str = "value_string",
        handle_resets: bool = False,
    ) -> pd.DataFrame:
        """Scrap quantity by reason code.

        Args:
            scrap_uuid: UUID for scrap weight/count signal.
            reason_uuid: UUID for scrap reason code signal.
            value_column_scrap: Column containing scrap values.
            value_column_reason: Column containing reason codes.
            handle_resets: When True, treat a decreasing counter as a reset
                and count the scrap accumulated after the reset. The reason
                is attached to a single global counter, so the reset-aware
                quantity is the sum of per-reading increments over the full
                time-sorted series, grouped by reason -- this also corrects
                the over-count that ``last - first`` produces when reasons
                alternate over a non-contiguous group. The result gains a
                ``resets`` column with the number of resets attributed to
                each reason. When False (default), the quantity is
                ``max(0, last - first)`` per reason group and a reset is lost.

        Returns:
            DataFrame with columns:
            - reason, scrap_quantity, pct_of_total
            - resets: Number of counter resets for the reason (only when
                ``handle_resets=True``)
        """
        columns = ["reason", "scrap_quantity", "pct_of_total"]
        if handle_resets:
            columns.append("resets")

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
            return pd.DataFrame(columns=columns)

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
            return pd.DataFrame(columns=columns)

        if handle_resets:
            # Compute increments/resets on the FULL time-sorted counter before
            # grouping, so alternating reasons neither over-count (interleaving)
            # nor lose post-reset production.
            merged["_increment"] = counter_increments(merged["scrap_val"])
            merged["_is_reset"] = (merged["scrap_val"].diff() < 0).fillna(False)

        # For each reason group, compute counter delta
        results = []
        for reason, grp in merged.groupby("reason"):
            if handle_resets:
                qty = float(grp["_increment"].sum())
                row = {"reason": reason, "scrap_quantity": round(qty, 2)}
                row["resets"] = int(grp["_is_reset"].sum())
            else:
                qty = self._get_counter_quantity(grp, "scrap_val")
                row = {"reason": reason, "scrap_quantity": round(qty, 2)}
            results.append(row)

        if not results:
            return pd.DataFrame(columns=columns)

        result_df = pd.DataFrame(results)
        total = result_df["scrap_quantity"].sum()
        result_df["pct_of_total"] = (
            (result_df["scrap_quantity"] / total * 100).round(1) if total > 0 else 0.0
        )
        return result_df.sort_values("scrap_quantity", ascending=False).reset_index(
            drop=True
        )[columns]

    def scrap_cost(
        self,
        scrap_uuid: str,
        part_id_uuid: str,
        material_costs: dict[str, float],
        *,
        value_column_scrap: str = "value_double",
        value_column_part: str = "value_string",
        handle_resets: bool = False,
    ) -> pd.DataFrame:
        """Convert scrap quantities to monetary cost.

        Args:
            scrap_uuid: UUID for scrap weight/count signal.
            part_id_uuid: UUID for part number / material type signal.
            material_costs: Dict mapping part numbers to cost per unit scrap.
            value_column_scrap: Column containing scrap values.
            value_column_part: Column containing part numbers.
            handle_resets: When True, treat a decreasing counter as a reset
                and count the scrap accumulated after the reset. The result
                gains a ``resets`` column with the number of resets observed
                for each part. When False (default), the quantity is
                ``max(0, last - first)`` per part and a reset is lost.

        Returns:
            DataFrame with columns:
            - part_number, scrap_quantity, cost_per_unit, total_cost
            - resets: Number of counter resets for the part (only when
                ``handle_resets=True``)
        """
        columns = ["part_number", "scrap_quantity", "cost_per_unit", "total_cost"]
        if handle_resets:
            columns.append("resets")

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
            return pd.DataFrame(columns=columns)

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
            qty = self._get_counter_quantity(
                grp, "scrap_val", handle_resets=handle_resets
            )
            cost_per_unit = material_costs.get(part_num, 0.0)
            row = {
                "part_number": part_num,
                "scrap_quantity": round(qty, 2),
                "cost_per_unit": cost_per_unit,
                "total_cost": round(qty * cost_per_unit, 2),
            }
            if handle_resets:
                row["resets"] = count_resets(grp["scrap_val"])
            results.append(row)

        return (
            pd.DataFrame(results, columns=columns)
            .sort_values("total_cost", ascending=False)
            .reset_index(drop=True)
        )

    def scrap_trend(
        self,
        scrap_uuid: str,
        *,
        value_column: str = "value_double",
        window: str = "1D",
        handle_resets: bool = False,
    ) -> pd.DataFrame:
        """Track scrap quantity trend over time.

        Args:
            scrap_uuid: UUID for scrap weight/count signal.
            value_column: Column containing scrap values.
            window: Time window for aggregation (default '1D').
            handle_resets: When True, treat a decreasing counter as a reset
                and count the scrap accumulated after the reset. The result
                gains a ``resets`` column with the number of resets observed
                in each period. When False (default), the quantity is
                ``max(0, last - first)`` per period and a reset is lost.

        Returns:
            DataFrame with columns:
            - start, end (start + window), duration_seconds
            - period, scrap_quantity
            - resets: Number of counter resets in the period (only when
                ``handle_resets=True``)
        """
        columns = ["period", "scrap_quantity"]
        if handle_resets:
            columns.append("resets")

        data = (
            self.dataframe[self.dataframe["uuid"] == scrap_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        if data.empty:
            return add_period_window(pd.DataFrame(columns=columns), window)

        data[self.time_column] = pd.to_datetime(data[self.time_column])
        data = data.set_index(self.time_column)

        results = []
        for period, grp in data.groupby(pd.Grouper(freq=window)):
            if grp.empty:
                continue
            grp = grp.reset_index()
            qty = self._get_counter_quantity(
                grp, value_column, handle_resets=handle_resets
            )
            row = {
                "period": period,
                "scrap_quantity": round(qty, 2),
            }
            if handle_resets:
                row["resets"] = count_resets(grp[value_column])
            results.append(row)

        return add_period_window(pd.DataFrame(results, columns=columns), window)


#: Deprecated name, removed in 1.0.
ScrapTracking = deprecated_class(ScrapEvents, "ScrapTracking")
