"""Operator/team performance tracking.

Compare production output and quality across operators:
- Production by operator
- Operator efficiency vs targets
- Quality by operator
- Operator ranking/comparison
"""

import logging
import pandas as pd  # type: ignore
from typing import Optional, Dict

from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class OperatorPerformanceTracking(Base):
    """Track and compare operator performance.

    Each UUID represents one signal:
    - operator_uuid: operator or team identifier signal
    - counter_uuid: production counter signal
    - ok_uuid / nok_uuid: good/bad parts counters (optional, for quality)

    Merge keys: [operator] for operator-level, [date, shift] for shift-level.

    Pipeline example::

        ops = OperatorPerformanceTracking(df)
        by_op = ops.production_by_operator('operator_id', 'part_counter')
        # → standalone KPI report
        quality = ops.quality_by_operator('operator_id', 'good_parts', 'bad_parts')
        # → merge with QualityTracking outputs on [date, shift]

    Example usage:
        tracker = OperatorPerformanceTracking(df)

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

    def _correlate_operator_counter(
        self,
        operator_uuid: str,
        counter_uuid: str,
        value_column_operator: str,
        value_column_counter: str,
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
        merged["shift"] = merged[self.time_column].apply(self._assign_shift)
        merged["date"] = merged[self.time_column].dt.date

        return merged

    def production_by_operator(
        self,
        operator_uuid: str,
        counter_uuid: str,
        *,
        value_column_operator: str = "value_string",
        value_column_counter: str = "value_integer",
    ) -> pd.DataFrame:
        """Parts produced per operator.

        Args:
            operator_uuid: UUID for operator/team identifier signal.
            counter_uuid: UUID for production counter signal.
            value_column_operator: Column containing operator names.
            value_column_counter: Column containing counter values.

        Returns:
            DataFrame with columns:
            - operator, total_produced, shifts_worked, avg_per_shift
        """
        merged = self._correlate_operator_counter(
            operator_uuid,
            counter_uuid,
            value_column_operator,
            value_column_counter,
        )
        if merged.empty:
            return pd.DataFrame(
                columns=["operator", "total_produced", "shifts_worked", "avg_per_shift"]
            )

        results = []
        for operator, op_data in merged.groupby("operator"):
            # Calculate production per shift, then sum
            shift_production = []
            for (date, shift), grp in op_data.groupby(["date", "shift"]):
                qty = self._get_counter_quantity(grp, "counter_val")
                if qty > 0:
                    shift_production.append(qty)

            total = sum(shift_production)
            shifts_worked = len(shift_production)

            results.append(
                {
                    "operator": operator,
                    "total_produced": round(total, 0),
                    "shifts_worked": shifts_worked,
                    "avg_per_shift": (
                        round(total / shifts_worked, 1) if shifts_worked > 0 else 0
                    ),
                }
            )

        return (
            pd.DataFrame(results)
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
        prod["efficiency_pct"] = (
            (prod["total_produced"] / prod["target"]) * 100
        ).round(1)
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
    ) -> pd.DataFrame:
        """Quality metrics (First Pass Yield) per operator.

        Args:
            operator_uuid: UUID for operator/team identifier signal.
            ok_uuid: UUID for good parts counter.
            nok_uuid: UUID for bad parts counter.
            value_column_operator: Column containing operator names.
            value_column_counter: Column containing counter values.

        Returns:
            DataFrame with columns:
            - operator, ok_count, nok_count, first_pass_yield_pct
        """
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
            return pd.DataFrame(
                columns=["operator", "ok_count", "nok_count", "first_pass_yield_pct"]
            )

        operator_data[self.time_column] = pd.to_datetime(
            operator_data[self.time_column]
        )
        operator_clean = operator_data[[self.time_column, value_column_operator]].copy()
        operator_clean = operator_clean.rename(
            columns={value_column_operator: "operator"}
        )

        ok_by_operator = {}
        nok_by_operator = {}

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

            for op, grp in merged_ok.groupby("operator"):
                ok_by_operator[op] = self._get_counter_quantity(grp, "counter_val")

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

            for op, grp in merged_nok.groupby("operator"):
                nok_by_operator[op] = self._get_counter_quantity(grp, "counter_val")

        # Combine
        all_operators = set(ok_by_operator.keys()) | set(nok_by_operator.keys())
        results = []
        for op in sorted(all_operators):
            ok = ok_by_operator.get(op, 0)
            nok = nok_by_operator.get(op, 0)
            total = ok + nok
            fpy = (ok / total * 100) if total > 0 else 0

            results.append(
                {
                    "operator": op,
                    "ok_count": round(ok, 0),
                    "nok_count": round(nok, 0),
                    "first_pass_yield_pct": round(fpy, 1),
                }
            )

        return pd.DataFrame(results)

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
        prod["pct_of_best"] = (
            (prod["total_produced"] / best * 100).round(1) if best > 0 else 0
        )

        return prod[["operator", "total_produced", "rank", "pct_of_best"]]
