"""Rework tracking for parts requiring re-processing.

Track rework events and their impact:
- Rework count by shift
- Rework by reason code
- Rework rate as percentage of production
- Rework cost estimation
- Rework trends over time
"""

import logging
from typing import TYPE_CHECKING

import pandas as pd

from ts_shape.events._output import twin_columns
from ts_shape.events._windows import (
    add_period_window,
    add_shift_window,
    data_tz,
)
from ts_shape.events.production._counter_groups import (
    counter_totals,
    counter_totals_per_period,
)
from ts_shape.utils.base import Base
from ts_shape.utils.counters import counter_increments, counter_quantity
from ts_shape.utils.deprecation import deprecated_class
from ts_shape.utils.shifts import assign_shift, production_dates

if TYPE_CHECKING:
    from typing_extensions import deprecated

logger = logging.getLogger(__name__)


class ReworkEvents(Base):
    """Track parts that require rework (re-processing through a station).

    Each UUID represents one signal:
    - rework_uuid: rework event counter (monotonic integer)
    - reason_uuid: rework reason code signal (optional)
    - total_production_uuid: total production counter (optional)
    - part_id_uuid: part number / product type signal (optional)

    Merge keys: [date, shift] for shift-level, [period] for trend,
    [reason] for reason-level, [part_number] for part-level.

    Pipeline example::

        rework = ReworkEvents(df)
        shift_rework = rework.rework_by_shift('rework_counter')
        # → merge with ShiftReportEvents.shift_production() on [date, shift]
        # → merge with QualityEvents.nok_by_shift() on [date, shift]
        rate = rework.rework_rate('rework_counter', 'total_counter')
        # → merge with ShiftReportEvents.shift_production() on [date, shift]

    Example usage:
        tracker = ReworkEvents(df)

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
        shift_definitions: dict[str, tuple[str, str]] | None = None,
    ) -> None:
        super().__init__(dataframe, time_column=time_column)
        self.time_column = time_column
        self.shift_definitions = shift_definitions or {
            "shift_1": ("06:00", "14:00"),
            "shift_2": ("14:00", "22:00"),
            "shift_3": ("22:00", "06:00"),
        }

    def rework_by_shift(
        self,
        rework_uuid: str,
        *,
        value_column: str = "value_integer",
        handle_resets: bool = True,
        production_day: bool = True,
    ) -> pd.DataFrame:
        """Rework count per shift.

        Args:
            rework_uuid: UUID for rework event counter signal.
            value_column: Column containing counter values.
            handle_resets: When True (default), treat a decreasing counter as a reset
                and count the rework accumulated after the reset instead of
                losing it. The increments are computed once over the whole
                signal and booked to the shift of the reading that reports
                them, so the shifts add up to the signal total. The result
                gains a ``resets`` column with the number of resets observed
                in each shift. When False,
                the quantity is ``max(0, last - first)`` of the shift's own
                readings: a shift containing a reset reports too little (often
                0) and the production between two shifts' readings is lost.
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
            - date, shift, rework_count
            - resets: Number of counter resets in the shift (only when
                ``handle_resets=True``)
        """
        columns = ["date", "shift", "rework_count"]
        if handle_resets:
            columns.append("resets")

        data = (
            self.dataframe[self.dataframe["uuid"] == rework_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        if data.empty:
            return add_shift_window(
                pd.DataFrame(columns=columns), self.shift_definitions, None
            )

        data[self.time_column] = pd.to_datetime(data[self.time_column])
        data["shift"] = assign_shift(data[self.time_column], self.shift_definitions)
        if production_day:
            data["date"] = production_dates(
                data[self.time_column], data["shift"], self.shift_definitions
            )
        else:
            data["date"] = data[self.time_column].dt.date

        totals = counter_totals(
            data, value_column, ["date", "shift"], handle_resets=handle_resets
        ).rename(columns={"quantity": "rework_count"})

        return add_shift_window(
            totals[columns],
            self.shift_definitions,
            data_tz(data[self.time_column]),
        )

    @twin_columns({"pct_of_total": "share_of_total_pct"})
    def rework_by_reason(
        self,
        rework_uuid: str,
        reason_uuid: str,
        *,
        value_column_rework: str = "value_integer",
        value_column_reason: str = "value_string",
        handle_resets: bool = True,
    ) -> pd.DataFrame:
        """Rework quantity by reason code.

        Args:
            rework_uuid: UUID for rework counter signal.
            reason_uuid: UUID for rework reason code signal.
            value_column_rework: Column containing rework counter values.
            value_column_reason: Column containing reason codes.
            handle_resets: When True (default), treat a decreasing counter as a reset
                and count the rework accumulated after the reset. The reason
                is attached to a single global counter, so the reset-aware
                quantity is the sum of per-reading increments over the full
                time-sorted series, grouped by reason -- this also corrects
                the over-count that ``last - first`` produces when reasons
                alternate over a non-contiguous group. The result gains a
                ``resets`` column with the number of resets attributed to
                each reason. When False, the quantity is
                ``max(0, last - first)`` per reason group and a reset is lost.

        Returns:
            DataFrame with columns:
            - reason, rework_count, pct_of_total
            - resets: Number of counter resets for the reason (only when
                ``handle_resets=True``)

            Canonical twin columns: ``share_of_total_pct`` (after ``pct_of_total``); each
            holds the same values as the deprecated column right before it, which is
            removed in the next major release.
        """
        columns = ["reason", "rework_count", "pct_of_total"]
        if handle_resets:
            columns.append("resets")

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
            return pd.DataFrame(columns=columns)

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
            return pd.DataFrame(columns=columns)

        if handle_resets:
            # Compute increments/resets on the FULL time-sorted counter before
            # grouping, so alternating reasons neither over-count (interleaving)
            # nor lose post-reset production.
            merged["_increment"] = counter_increments(merged["rework_val"])
            merged["_is_reset"] = (merged["rework_val"].diff() < 0).fillna(False)

        results = []
        for reason, grp in merged.groupby("reason"):
            if handle_resets:
                qty = float(grp["_increment"].sum())
                row = {"reason": reason, "rework_count": qty}
                row["resets"] = int(grp["_is_reset"].sum())
            else:
                qty = counter_quantity(grp["rework_val"], handle_resets=False)
                row = {"reason": reason, "rework_count": qty}
            results.append(row)

        if not results:
            return pd.DataFrame(columns=columns)

        result_df = pd.DataFrame(results)
        total = result_df["rework_count"].sum()
        result_df["pct_of_total"] = (
            (result_df["rework_count"] / total * 100) if total > 0 else 0.0
        )

        return result_df.sort_values("rework_count", ascending=False).reset_index(
            drop=True
        )[columns]

    def rework_rate(
        self,
        rework_uuid: str,
        total_production_uuid: str,
        *,
        value_column_rework: str = "value_integer",
        value_column_production: str = "value_integer",
        handle_resets: bool = True,
        production_day: bool = True,
    ) -> pd.DataFrame:
        """Rework rate as percentage of total production per shift.

        Args:
            rework_uuid: UUID for rework counter signal.
            total_production_uuid: UUID for total production counter signal.
            value_column_rework: Column containing rework counter values.
            value_column_production: Column containing production counter values.
            handle_resets: When True (default), treat a decreasing counter as a reset
                and count the production/rework accumulated after the reset,
                for both the numerator (rework) and denominator (total
                produced). Both are computed once over the whole signal and
                booked to the shift of the reading that reports them, so the
                shifts add up to the signal totals. The result gains ``rework_resets``,
                ``production_resets`` and ``resets`` (their sum) columns with
                the number of resets observed in each shift. When False, both use ``max(0, last - first)`` per shift and a
                reset is lost.
            production_day: When True, an overnight shift's post-midnight
                readings are attributed to the shift's start date. Default True;
                False groups by raw calendar date, which splits a night shift
                across two rows. Only affects overnight schedules.

        Returns:
            DataFrame with columns:
            - start, end, duration_seconds: the shift window from
                ``shift_definitions`` (``date`` at the shift start time,
                tz-aware in the data's tz; an overnight shift ends on the
                next day)
            - date, shift, total_produced, rework_count, rework_rate_pct
            - rework_resets, production_resets, resets: Number of counter
                resets of the rework counter, the production counter, and
                both together in the shift (only when ``handle_resets=True``)
        """
        rate_cols = [
            "date",
            "shift",
            "total_produced",
            "rework_count",
            "rework_rate_pct",
        ]
        if handle_resets:
            rate_cols += ["rework_resets", "production_resets", "resets"]
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
            return add_shift_window(
                pd.DataFrame(columns=rate_cols), self.shift_definitions, None
            )

        rework_data[self.time_column] = pd.to_datetime(rework_data[self.time_column])
        prod_data[self.time_column] = pd.to_datetime(prod_data[self.time_column])

        rework_data["shift"] = assign_shift(
            rework_data[self.time_column], self.shift_definitions
        )
        prod_data["shift"] = assign_shift(
            prod_data[self.time_column], self.shift_definitions
        )
        if production_day:
            rework_data["date"] = production_dates(
                rework_data[self.time_column],
                rework_data["shift"],
                self.shift_definitions,
            )
            prod_data["date"] = production_dates(
                prod_data[self.time_column],
                prod_data["shift"],
                self.shift_definitions,
            )
        else:
            rework_data["date"] = rework_data[self.time_column].dt.date
            prod_data["date"] = prod_data[self.time_column].dt.date

        # Increments over each whole signal, booked to the shift of the reading
        # that reports them, so the shifts add up to the signal totals.
        keys = ["date", "shift"]
        rw_totals = counter_totals(
            rework_data, value_column_rework, keys, handle_resets=handle_resets
        ).rename(columns={"quantity": "rework_count", "resets": "rework_resets"})
        pr_totals = counter_totals(
            prod_data, value_column_production, keys, handle_resets=handle_resets
        ).rename(columns={"quantity": "total_produced", "resets": "production_resets"})
        merged = (
            pr_totals.merge(rw_totals, on=keys, how="outer")
            .sort_values(keys, kind="stable")
            .reset_index(drop=True)
        )
        for col in ("rework_count", "total_produced"):
            merged[col] = merged[col].fillna(0.0)
        for col in ("rework_resets", "production_resets"):
            merged[col] = merged[col].fillna(0).astype(int)
        merged["rework_rate_pct"] = (
            merged["rework_count"] / merged["total_produced"] * 100
        ).where(merged["total_produced"] > 0, 0.0)
        merged["resets"] = merged["rework_resets"] + merged["production_resets"]

        return add_shift_window(
            merged[rate_cols],
            self.shift_definitions,
            data_tz(prod_data[self.time_column]),
        )

    def rework_cost(
        self,
        rework_uuid: str,
        part_id_uuid: str,
        rework_costs: dict[str, float],
        *,
        value_column_rework: str = "value_integer",
        value_column_part: str = "value_string",
        handle_resets: bool = True,
    ) -> pd.DataFrame:
        """Convert rework counts to monetary cost by part number.

        Args:
            rework_uuid: UUID for rework counter signal.
            part_id_uuid: UUID for part number / product type signal.
            rework_costs: Dict mapping part numbers to cost per rework.
            value_column_rework: Column containing rework counter values.
            value_column_part: Column containing part number values.
            handle_resets: When True (default), treat a decreasing counter as a reset
                and count the rework accumulated after the reset. The
                increments are computed once over the whole signal and booked
                to the part of the reading that reports them, so the parts add
                up to the signal total (also when parts alternate). The result
                gains a ``resets`` column with the number of resets observed
                for each part. When False, the quantity is
                ``max(0, last - first)`` per part and a reset is lost.

        Returns:
            DataFrame with columns:
            - part_number, rework_count, cost_per_rework, total_cost
            - resets: Number of counter resets for the part (only when
                ``handle_resets=True``)
        """
        columns = ["part_number", "rework_count", "cost_per_rework", "total_cost"]
        if handle_resets:
            columns.append("resets")

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
            return pd.DataFrame(columns=columns)

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

        # Increments over the whole signal, booked to the part of the reading
        # that reports them (no boundary loss, no interleaving over-count).
        totals = counter_totals(
            merged, "rework_val", ["part_number"], handle_resets=handle_resets
        ).rename(columns={"quantity": "rework_count"})
        totals["cost_per_rework"] = [
            rework_costs.get(p, 0.0) for p in totals["part_number"]
        ]
        totals["total_cost"] = totals["rework_count"] * totals["cost_per_rework"]

        return (
            totals[columns]
            .sort_values("total_cost", ascending=False)
            .reset_index(drop=True)
        )

    def rework_trend(
        self,
        rework_uuid: str,
        *,
        value_column: str = "value_integer",
        window: str = "1D",
        handle_resets: bool = True,
    ) -> pd.DataFrame:
        """Track rework count trend over time.

        Args:
            rework_uuid: UUID for rework counter signal.
            value_column: Column containing counter values.
            window: Time window for aggregation (default '1D').
            handle_resets: When True (default), treat a decreasing counter as a reset
                and count the rework accumulated after the reset. The
                increments are computed once over the whole signal and booked
                to the period of the reading that reports them, so the periods
                add up to the signal total. The result gains a ``resets``
                column with the number of resets observed in each period.
                When False, the quantity is ``max(0, last - first)`` of the
                period's own readings: a reset and the production between two
                periods' readings are lost.

        Returns:
            DataFrame with columns:
            - start, end (start + window), duration_seconds
            - period, rework_count
            - resets: Number of counter resets in the period (only when
                ``handle_resets=True``)
        """
        columns = ["period", "rework_count"]
        if handle_resets:
            columns.append("resets")

        data = (
            self.dataframe[self.dataframe["uuid"] == rework_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        if data.empty:
            return add_period_window(pd.DataFrame(columns=columns), window)

        data[self.time_column] = pd.to_datetime(data[self.time_column])
        totals = counter_totals_per_period(
            data.set_index(self.time_column)[value_column],
            window,
            handle_resets=handle_resets,
        ).rename(columns={"quantity": "rework_count"})

        return add_period_window(totals[columns], window)


#: Deprecated name, removed in the next major release.
if TYPE_CHECKING:

    @deprecated("ReworkTracking is deprecated, use ReworkEvents")
    class ReworkTracking(ReworkEvents): ...

else:
    ReworkTracking = deprecated_class(ReworkEvents, "ReworkTracking")
