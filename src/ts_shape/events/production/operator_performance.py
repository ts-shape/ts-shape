"""Operator/team performance tracking.

Compare production output and quality across operators:
- Production by operator
- Operator efficiency vs targets
- Quality by operator
- Operator ranking/comparison
"""

import logging
from typing import TYPE_CHECKING

import pandas as pd

from ts_shape.events._output import twin_columns
from ts_shape.events.production._counter_groups import counter_totals
from ts_shape.utils.base import Base
from ts_shape.utils.deprecation import deprecated_class
from ts_shape.utils.shifts import assign_shift, production_dates

if TYPE_CHECKING:
    from typing_extensions import deprecated

logger = logging.getLogger(__name__)


class OperatorPerformanceEvents(Base):
    """Track and compare operator performance.

    Each UUID represents one signal:
    - operator_uuid: operator or team identifier signal
    - counter_uuid: production counter signal
    - ok_uuid / nok_uuid: OK/NOK part counters (optional, for quality)

    Merge keys: [operator] for operator-level, [date, shift] for shift-level.

    Pipeline example::

        ops = OperatorPerformanceEvents(df)
        by_op = ops.production_by_operator('operator_id', 'part_counter')
        # → standalone KPI report
        quality = ops.quality_by_operator('operator_id', 'good_parts', 'bad_parts')
        # → merge with QualityEvents outputs on [date, shift]

    Example usage:
        tracker = OperatorPerformanceEvents(df)

        # Parts per operator
        prod = tracker.production_by_operator(
            operator_uuid='operator_id',
            counter_uuid='part_counter',
        )

        # Efficiency vs target
        eff = tracker.operator_efficiency(
            operator_uuid='operator_id',
            counter_uuid='part_counter',
            target_per_shift=500,
        )
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        time_column: str = "systime",
        shift_definitions: dict[str, tuple[str, str]] | None = None,
    ) -> None:
        """Initialize the operator performance tracker.

        Args:
            dataframe: Long-format timeseries frame (one row per sample, ``uuid``
                identifies the signal).
            time_column: Timestamp column.
            shift_definitions: Mapping of shift name to ``(start, end)`` ``"HH:MM"``
                times. ``None`` (default) uses three shifts: 06:00-14:00, 14:00-22:00,
                22:00-06:00.
        """
        super().__init__(dataframe, time_column=time_column)
        self.time_column = time_column
        self.shift_definitions = shift_definitions or {
            "shift_1": ("06:00", "14:00"),
            "shift_2": ("14:00", "22:00"),
            "shift_3": ("22:00", "06:00"),
        }

    def _correlate_operator_counter(
        self,
        operator_uuid: str,
        counter_uuid: str,
        value_column_operator: str,
        value_column_counter: str,
        *,
        production_day: bool = True,
    ) -> pd.DataFrame:
        """Merge operator signal with counter signal by timestamp."""
        operator_data = (
            self.dataframe[self.dataframe["uuid"] == operator_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        counter_data = (
            self.dataframe[self.dataframe["uuid"] == counter_uuid]
            .copy()
            .sort_values(self.time_column)
        )

        if operator_data.empty or counter_data.empty:
            return pd.DataFrame()

        operator_data[self.time_column] = pd.to_datetime(
            operator_data[self.time_column]
        )
        counter_data[self.time_column] = pd.to_datetime(counter_data[self.time_column])

        counter_clean = counter_data[[self.time_column, value_column_counter]].copy()
        counter_clean = counter_clean.rename(
            columns={value_column_counter: "counter_val"}
        )

        operator_clean = operator_data[[self.time_column, value_column_operator]].copy()
        operator_clean = operator_clean.rename(
            columns={value_column_operator: "operator"}
        )

        merged = pd.merge_asof(
            counter_clean,
            operator_clean,
            on=self.time_column,
            direction="backward",
        )
        merged = merged.dropna(subset=["operator"])
        merged["shift"] = assign_shift(merged[self.time_column], self.shift_definitions)
        if production_day:
            merged["date"] = production_dates(
                merged[self.time_column], merged["shift"], self.shift_definitions
            )
        else:
            merged["date"] = merged[self.time_column].dt.date

        return merged

    def production_by_operator(
        self,
        operator_uuid: str,
        counter_uuid: str,
        *,
        value_column_operator: str = "value_string",
        value_column_counter: str = "value_integer",
        handle_resets: bool = True,
        production_day: bool = True,
    ) -> pd.DataFrame:
        """Parts produced per operator.

        Args:
            operator_uuid: UUID for operator/team identifier signal.
            counter_uuid: UUID for production counter signal.
            value_column_operator: Column containing operator names.
            value_column_counter: Column containing counter values.
            handle_resets: When True (default), treat a decreasing counter as a reset
                and count the production accumulated after the reset instead
                of losing it. The increments are computed once over the whole
                signal and booked to the (operator, date, shift) of the reading
                that reports them, so the operators add up to the signal total
                and nothing is lost at a shift or operator change.
                The result gains a ``resets`` column with the number of resets
                observed across the operator's shifts. When False,
                each shift uses ``max(0, last - first)`` and a shift
                containing a reset reports too little (often 0).
            production_day: When True, an overnight shift's post-midnight
                readings are attributed to the shift's start date, so a night
                shift counts as one worked shift. Default True; False groups by
                raw calendar date. Only affects overnight schedules.

        Returns:
            DataFrame with columns:
            - operator, total_produced, shifts_worked, avg_per_shift
            - resets: Number of counter resets across the operator's shifts
                (only when ``handle_resets=True``)
        """
        columns = ["operator", "total_produced", "shifts_worked", "avg_per_shift"]
        if handle_resets:
            columns.append("resets")

        merged = self._correlate_operator_counter(
            operator_uuid,
            counter_uuid,
            value_column_operator,
            value_column_counter,
            production_day=production_day,
        )
        if merged.empty:
            return pd.DataFrame(columns=columns)

        # Increments over the whole signal, booked to the (operator, date,
        # shift) of the reading that reports them, so no production is lost
        # at shift or operator changes.
        per_shift = counter_totals(
            merged,
            "counter_val",
            ["operator", "date", "shift"],
            handle_resets=handle_resets,
        )
        results = []
        for operator, op_shifts in per_shift.groupby("operator", sort=True):
            # A shift counts as worked when it produced something.
            worked = op_shifts["quantity"][op_shifts["quantity"] > 0]
            resets = int(op_shifts["resets"].sum())
            total = float(worked.sum())
            shifts_worked = len(worked)

            row = {
                "operator": operator,
                "total_produced": total,
                "shifts_worked": shifts_worked,
                "avg_per_shift": (total / shifts_worked) if shifts_worked > 0 else 0,
            }
            if handle_resets:
                row["resets"] = resets
            results.append(row)

        return (
            pd.DataFrame(results, columns=columns)
            .sort_values("total_produced", ascending=False)
            .reset_index(drop=True)
        )

    def operator_efficiency(
        self,
        operator_uuid: str,
        counter_uuid: str,
        target_per_shift: int,
        *,
        value_column_operator: str = "value_string",
        value_column_counter: str = "value_integer",
    ) -> pd.DataFrame:
        """Operator efficiency vs a per-shift target.

        Args:
            operator_uuid: UUID for operator/team identifier signal.
            counter_uuid: UUID for production counter signal.
            target_per_shift: Target production quantity per shift.
            value_column_operator: Column containing operator names.
            value_column_counter: Column containing counter values.

        Returns:
            DataFrame with columns:
            - operator, total_produced, target, efficiency_pct
        """
        prod = self.production_by_operator(
            operator_uuid,
            counter_uuid,
            value_column_operator=value_column_operator,
            value_column_counter=value_column_counter,
        )
        if prod.empty:
            return pd.DataFrame(
                columns=["operator", "total_produced", "target", "efficiency_pct"]
            )

        prod["target"] = prod["shifts_worked"] * target_per_shift
        prod["efficiency_pct"] = (prod["total_produced"] / prod["target"]) * 100
        prod.loc[prod["target"] == 0, "efficiency_pct"] = 0.0

        return prod[
            ["operator", "total_produced", "target", "efficiency_pct"]
        ].reset_index(drop=True)

    def quality_by_operator(
        self,
        operator_uuid: str,
        ok_uuid: str,
        nok_uuid: str,
        *,
        value_column_operator: str = "value_string",
        value_column_counter: str = "value_integer",
        handle_resets: bool = True,
    ) -> pd.DataFrame:
        """Quality metrics (First Pass Yield) per operator.

        Args:
            operator_uuid: UUID for operator/team identifier signal.
            ok_uuid: UUID for good parts counter.
            nok_uuid: UUID of the NOK part counter.
            value_column_operator: Column containing operator names.
            value_column_counter: Column containing counter values.
            handle_resets: When True (default), treat a decreasing counter as a reset
                and count the OK/NOK parts accumulated after the reset, for
                both the OK and NOK counters. Because the counts are taken
                over the whole range, a single reset otherwise wipes out an
                operator's entire tally. The result gains a ``resets`` column
                with the number of resets observed across the operator's OK
                and NOK counters. The increments are computed once over each
                whole signal and booked to the operator of the reading that
                reports them, so the operators add up to the signal totals
                (also when operators alternate). When False, both counters use
                ``max(0, last - first)`` per operator and a reset is lost.

        Returns:
            DataFrame with columns:
            - operator, ok_count, nok_count, first_pass_yield_pct
            - resets: Number of counter resets across the operator's OK and
                NOK counters (only when ``handle_resets=True``)
        """
        columns = ["operator", "ok_count", "nok_count", "first_pass_yield_pct"]
        if handle_resets:
            columns.append("resets")

        operator_data = (
            self.dataframe[self.dataframe["uuid"] == operator_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        ok_data = (
            self.dataframe[self.dataframe["uuid"] == ok_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        nok_data = (
            self.dataframe[self.dataframe["uuid"] == nok_uuid]
            .copy()
            .sort_values(self.time_column)
        )

        if operator_data.empty or (ok_data.empty and nok_data.empty):
            return pd.DataFrame(columns=columns)

        operator_data[self.time_column] = pd.to_datetime(
            operator_data[self.time_column]
        )
        operator_clean = operator_data[[self.time_column, value_column_operator]].copy()
        operator_clean = operator_clean.rename(
            columns={value_column_operator: "operator"}
        )

        ok_by_operator = {}
        nok_by_operator = {}
        resets_by_operator: dict[str, int] = {}

        # Process OK parts
        if not ok_data.empty:
            ok_data[self.time_column] = pd.to_datetime(ok_data[self.time_column])
            ok_clean = ok_data[[self.time_column, value_column_counter]].copy()
            ok_clean = ok_clean.rename(columns={value_column_counter: "counter_val"})

            merged_ok = pd.merge_asof(
                ok_clean,
                operator_clean,
                on=self.time_column,
                direction="backward",
            )
            merged_ok = merged_ok.dropna(subset=["operator"])

            ok_totals = counter_totals(
                merged_ok, "counter_val", ["operator"], handle_resets=handle_resets
            )
            for op, qty, n_resets in ok_totals.itertuples(index=False):
                ok_by_operator[op] = float(qty)
                if handle_resets:
                    resets_by_operator[op] = resets_by_operator.get(op, 0) + n_resets

        # Process NOK parts
        if not nok_data.empty:
            nok_data[self.time_column] = pd.to_datetime(nok_data[self.time_column])
            nok_clean = nok_data[[self.time_column, value_column_counter]].copy()
            nok_clean = nok_clean.rename(columns={value_column_counter: "counter_val"})

            merged_nok = pd.merge_asof(
                nok_clean,
                operator_clean,
                on=self.time_column,
                direction="backward",
            )
            merged_nok = merged_nok.dropna(subset=["operator"])

            nok_totals = counter_totals(
                merged_nok, "counter_val", ["operator"], handle_resets=handle_resets
            )
            for op, qty, n_resets in nok_totals.itertuples(index=False):
                nok_by_operator[op] = float(qty)
                if handle_resets:
                    resets_by_operator[op] = resets_by_operator.get(op, 0) + n_resets

        # Combine
        all_operators = set(ok_by_operator.keys()) | set(nok_by_operator.keys())
        results = []
        for op in sorted(all_operators):
            ok = ok_by_operator.get(op, 0)
            nok = nok_by_operator.get(op, 0)
            total = ok + nok
            fpy = (ok / total * 100) if total > 0 else 0

            row = {
                "operator": op,
                "ok_count": ok,
                "nok_count": nok,
                "first_pass_yield_pct": fpy,
            }
            if handle_resets:
                row["resets"] = resets_by_operator.get(op, 0)
            results.append(row)

        return pd.DataFrame(results, columns=columns)

    @twin_columns({"pct_of_best": "share_of_best_pct"})
    def operator_comparison(
        self,
        operator_uuid: str,
        counter_uuid: str,
        *,
        value_column_operator: str = "value_string",
        value_column_counter: str = "value_integer",
    ) -> pd.DataFrame:
        """Ranked operator performance comparison.

        Args:
            operator_uuid: UUID for operator/team identifier signal.
            counter_uuid: UUID for production counter signal.
            value_column_operator: Column containing operator names.
            value_column_counter: Column containing counter values.

        Returns:
            DataFrame with columns:
            - operator, total_produced, rank, pct_of_best

            Canonical twin columns: ``share_of_best_pct`` (after ``pct_of_best``); each holds
            the same values as the deprecated column right before it, which is removed
            in the next major release.
        """
        prod = self.production_by_operator(
            operator_uuid,
            counter_uuid,
            value_column_operator=value_column_operator,
            value_column_counter=value_column_counter,
        )
        if prod.empty:
            return pd.DataFrame(
                columns=["operator", "total_produced", "rank", "pct_of_best"]
            )

        prod = prod.sort_values("total_produced", ascending=False).reset_index(
            drop=True
        )
        prod["rank"] = range(1, len(prod) + 1)

        best = prod["total_produced"].max()
        prod["pct_of_best"] = (prod["total_produced"] / best * 100) if best > 0 else 0

        return prod[["operator", "total_produced", "rank", "pct_of_best"]]


#: Deprecated name, removed in the next major release.
if TYPE_CHECKING:

    @deprecated(
        "OperatorPerformanceTracking is deprecated, use OperatorPerformanceEvents"
    )
    class OperatorPerformanceTracking(OperatorPerformanceEvents): ...

else:
    OperatorPerformanceTracking = deprecated_class(
        OperatorPerformanceEvents, "OperatorPerformanceTracking"
    )
