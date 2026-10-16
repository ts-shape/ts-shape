"""Scrap and material waste tracking.

Track material waste and scrap (different from NOK parts):
- Scrap by shift and reason
- Scrap cost calculation
- Scrap trends over time
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

    Example:
        >>> import pandas as pd, ts_shape
        >>> scrap = ts_shape.make_timeseries(["scrap_kg"], n_points=20, freq="1min",
        ...                                  baseline=0.0, noise=0, drift=9.5)
        >>> reason = ts_shape.make_id_signal("reason", ["Burr", "Crack"], freq="1min")
        >>> out = ts_shape.ScrapEvents(pd.concat([scrap, reason])).scrap_by_reason(
        ...     "scrap_kg", "reason")
        >>> out[["reason", "scrap_quantity", "share_of_total_pct"]].round(1)
          reason  scrap_quantity  share_of_total_pct
        0  Crack             5.0          52.6
        1   Burr             4.5          47.4
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

    def scrap_by_shift(
        self,
        scrap_uuid: str,
        *,
        value_column: str = "value_double",
        handle_resets: bool = True,
        production_day: bool = True,
    ) -> pd.DataFrame:
        """Scrap quantity per shift.

        Args:
            scrap_uuid: UUID for scrap weight/count signal.
            value_column: Column containing scrap values.
            handle_resets: When True (default), treat a decreasing counter as a reset
                and count the scrap accumulated after the reset instead of
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
        data["shift"] = assign_shift(data[self.time_column], self.shift_definitions)
        if production_day:
            data["date"] = production_dates(
                data[self.time_column], data["shift"], self.shift_definitions
            )
        else:
            data["date"] = data[self.time_column].dt.date

        # Increments over the whole signal, booked to the shift of the reading
        # that reports them, so the shifts add up to the signal total.
        totals = counter_totals(
            data, value_column, ["date", "shift"], handle_resets=handle_resets
        ).rename(columns={"quantity": "scrap_quantity"})

        return add_shift_window(
            totals[columns],
            self.shift_definitions,
            data_tz(data[self.time_column]),
        )

    @twin_columns({"pct_of_total": "share_of_total_pct"})
    def scrap_by_reason(
        self,
        scrap_uuid: str,
        reason_uuid: str,
        *,
        value_column_scrap: str = "value_double",
        value_column_reason: str = "value_string",
        handle_resets: bool = True,
    ) -> pd.DataFrame:
        """Scrap quantity by reason code.

        Args:
            scrap_uuid: UUID for scrap weight/count signal.
            reason_uuid: UUID for scrap reason code signal.
            value_column_scrap: Column containing scrap values.
            value_column_reason: Column containing reason codes.
            handle_resets: When True (default), treat a decreasing counter as a reset
                and count the scrap accumulated after the reset. The reason
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
            - reason, scrap_quantity, pct_of_total
            - resets: Number of counter resets for the reason (only when
                ``handle_resets=True``)

            Canonical twin columns: ``share_of_total_pct`` (after ``pct_of_total``); each
            holds the same values as the deprecated column right before it, which is
            removed in the next major release.
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
                row = {"reason": reason, "scrap_quantity": qty}
                row["resets"] = int(grp["_is_reset"].sum())
            else:
                qty = counter_quantity(grp["scrap_val"], handle_resets=False)
                row = {"reason": reason, "scrap_quantity": qty}
            results.append(row)

        if not results:
            return pd.DataFrame(columns=columns)

        result_df = pd.DataFrame(results)
        total = result_df["scrap_quantity"].sum()
        result_df["pct_of_total"] = (
            (result_df["scrap_quantity"] / total * 100) if total > 0 else 0.0
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
        handle_resets: bool = True,
    ) -> pd.DataFrame:
        """Convert scrap quantities to monetary cost.

        Args:
            scrap_uuid: UUID for scrap weight/count signal.
            part_id_uuid: UUID for part number / material type signal.
            material_costs: Dict mapping part numbers to cost per unit scrap.
            value_column_scrap: Column containing scrap values.
            value_column_part: Column containing part numbers.
            handle_resets: When True (default), treat a decreasing counter as a reset
                and count the scrap accumulated after the reset. The
                increments are computed once over the whole signal and booked
                to the part of the reading that reports them, so the parts add
                up to the signal total (also when parts alternate). The result
                gains a ``resets`` column with the number of resets observed
                for each part. When False, the quantity is
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

        # Increments over the whole signal, booked to the part of the reading
        # that reports them (no boundary loss, no interleaving over-count).
        totals = counter_totals(
            merged, "scrap_val", ["part_number"], handle_resets=handle_resets
        ).rename(columns={"quantity": "scrap_quantity"})
        totals["cost_per_unit"] = [
            material_costs.get(p, 0.0) for p in totals["part_number"]
        ]
        totals["total_cost"] = totals["scrap_quantity"] * totals["cost_per_unit"]

        return (
            totals[columns]
            .sort_values("total_cost", ascending=False)
            .reset_index(drop=True)
        )

    def scrap_trend(
        self,
        scrap_uuid: str,
        *,
        value_column: str = "value_double",
        window: str = "1D",
        handle_resets: bool = True,
    ) -> pd.DataFrame:
        """Track scrap quantity trend over time.

        Args:
            scrap_uuid: UUID for scrap weight/count signal.
            value_column: Column containing scrap values.
            window: Time window for aggregation (default '1D').
            handle_resets: When True (default), treat a decreasing counter as a reset
                and count the scrap accumulated after the reset. The
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
        totals = counter_totals_per_period(
            data.set_index(self.time_column)[value_column],
            window,
            handle_resets=handle_resets,
        ).rename(columns={"quantity": "scrap_quantity"})

        return add_period_window(totals[columns], window)


#: Deprecated name, removed in the next major release.
if TYPE_CHECKING:

    @deprecated("ScrapTracking is deprecated, use ScrapEvents")
    class ScrapTracking(ScrapEvents): ...

else:
    ScrapTracking = deprecated_class(ScrapEvents, "ScrapTracking")
