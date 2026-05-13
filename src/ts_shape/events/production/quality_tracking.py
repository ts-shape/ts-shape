"""Quality and NOK (defective parts) tracking.

Essential module for daily quality analysis:
- NOK parts by shift and part number
- Scrap tracking
- First pass yield
- Quality rates
"""

import logging
import pandas as pd  # type: ignore
import numpy as np
from typing import Optional, Dict

from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class QualityTracking(Base):
    """Track NOK (defective) parts and quality metrics.

    Each UUID represents one signal:
    - ok_counter_uuid: counter for good parts
    - nok_counter_uuid: counter for defective parts
    - part_id_uuid: part number signal (optional)
    - defect_reason_uuid: defect reason code (optional)

    Merge keys: [date, shift] for shift-level, [date] for daily, [part_number] for part-level.

    Pipeline outputs include ``quality_pct`` (alias for ``first_pass_yield_pct``)
    so downstream modules (ShiftHandoverReport, PeriodSummary, OEECalculator)
    can join on a consistent column name.

    Example usage:
        tracker = QualityTracking(df)

        # NOK parts per shift
        shift_nok = tracker.nok_by_shift(
            ok_counter_uuid='good_parts',
            nok_counter_uuid='bad_parts'
        )

        # Quality by part number
        part_quality = tracker.quality_by_part(
            ok_counter_uuid='good_parts',
            nok_counter_uuid='bad_parts',
            part_id_uuid='part_number'
        )

        # Pipeline: feed into PeriodSummary
        daily = tracker.daily_quality_summary('good', 'bad')
        # daily has [date, ok_parts, nok_parts, total_parts, quality_pct, ...]
    """

    MERGE_KEYS_SHIFT = ["date", "shift"]
    MERGE_KEYS_DAILY = ["date"]
    MERGE_KEYS_PART = ["part_number"]

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        time_column: str = "systime",
        shift_definitions: Optional[Dict[str, tuple[str, str]]] = None,
    ) -> None:
        """Initialize quality tracker.

        Args:
            dataframe: Input DataFrame with timeseries data
            time_column: Name of timestamp column (default: 'systime')
            shift_definitions: Dictionary mapping shift names to (start, end) times
                              Default: 3-shift operation (06:00-14:00, 14:00-22:00, 22:00-06:00)
        """
        super().__init__(dataframe, column_name=time_column)
        self.time_column = time_column

        # Default 3-shift operation
        self.shift_definitions = shift_definitions or {
            "shift_1": ("06:00", "14:00"),
            "shift_2": ("14:00", "22:00"),
            "shift_3": ("22:00", "06:00"),
        }

    def _assign_shift(self, timestamp: pd.Timestamp) -> str:
        """Assign shift based on time of day."""
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

    def nok_by_shift(
        self,
        ok_counter_uuid: str,
        nok_counter_uuid: str,
        *,
        value_column: str = "value_integer",
    ) -> pd.DataFrame:
        """Calculate NOK (defective) parts per shift.

        Args:
            ok_counter_uuid: UUID for good parts counter
            nok_counter_uuid: UUID for defective parts counter
            value_column: Column containing counter values

        Returns:
            DataFrame with quality metrics by shift:
            - date: Production date
            - shift: Shift name
            - ok_parts: Good parts produced
            - nok_parts: Defective parts
            - total_parts: Total parts produced
            - nok_rate_pct: Percentage of defective parts
            - first_pass_yield_pct: Percentage of good parts

        Example:
            >>> nok_by_shift('good_counter', 'bad_counter')
                date        shift    ok_parts  nok_parts  total_parts  nok_rate_pct  first_pass_yield_pct
            0   2024-01-01  shift_1  450       12         462          2.6           97.4
            1   2024-01-01  shift_2  425       18         443          4.1           95.9
            2   2024-01-01  shift_3  380       25         405          6.2           93.8
        """
        # Get OK counter data
        ok_data = (
            self.dataframe[self.dataframe["uuid"] == ok_counter_uuid]
            .copy()
            .sort_values(self.time_column)
        )

        # Get NOK counter data
        nok_data = (
            self.dataframe[self.dataframe["uuid"] == nok_counter_uuid]
            .copy()
            .sort_values(self.time_column)
        )

        if ok_data.empty and nok_data.empty:
            return pd.DataFrame(
                columns=[
                    "date",
                    "shift",
                    "ok_parts",
                    "nok_parts",
                    "total_parts",
                    "nok_rate_pct",
                    "first_pass_yield_pct",
                ]
            )

        # Process OK data
        ok_processed = None
        if not ok_data.empty:
            ok_data[self.time_column] = pd.to_datetime(ok_data[self.time_column])
            ok_data["shift"] = ok_data[self.time_column].apply(self._assign_shift)
            ok_data["date"] = ok_data[self.time_column].dt.date
            ok_processed = ok_data

        # Process NOK data
        nok_processed = None
        if not nok_data.empty:
            nok_data[self.time_column] = pd.to_datetime(nok_data[self.time_column])
            nok_data["shift"] = nok_data[self.time_column].apply(self._assign_shift)
            nok_data["date"] = nok_data[self.time_column].dt.date
            nok_processed = nok_data

        # Calculate quantities per shift
        results = []

        # Get all unique date/shift combinations
        date_shifts = set()
        if ok_processed is not None:
            date_shifts.update(zip(ok_processed["date"], ok_processed["shift"]))
        if nok_processed is not None:
            date_shifts.update(zip(nok_processed["date"], nok_processed["shift"]))

        for date, shift in sorted(date_shifts):
            ok_parts = 0
            nok_parts = 0

            # Calculate OK parts for this shift
            if ok_processed is not None:
                ok_shift = ok_processed[
                    (ok_processed["date"] == date) & (ok_processed["shift"] == shift)
                ]
                if not ok_shift.empty:
                    first_count = ok_shift[value_column].iloc[0]
                    last_count = ok_shift[value_column].iloc[-1]
                    ok_parts = max(0, last_count - first_count)

            # Calculate NOK parts for this shift
            if nok_processed is not None:
                nok_shift = nok_processed[
                    (nok_processed["date"] == date) & (nok_processed["shift"] == shift)
                ]
                if not nok_shift.empty:
                    first_count = nok_shift[value_column].iloc[0]
                    last_count = nok_shift[value_column].iloc[-1]
                    nok_parts = max(0, last_count - first_count)

            total_parts = ok_parts + nok_parts

            if total_parts > 0:
                nok_rate = (nok_parts / total_parts) * 100
                fpy = (ok_parts / total_parts) * 100
            else:
                nok_rate = 0
                fpy = 0

            results.append(
                {
                    "date": date,
                    "shift": shift,
                    "ok_parts": ok_parts,
                    "nok_parts": nok_parts,
                    "total_parts": total_parts,
                    "nok_rate_pct": round(nok_rate, 1),
                    "first_pass_yield_pct": round(fpy, 1),
                    "quality_pct": round(fpy, 1),
                }
            )

        return pd.DataFrame(results)

    def quality_by_part(
        self,
        ok_counter_uuid: str,
        nok_counter_uuid: str,
        part_id_uuid: str,
        *,
        value_column_counter: str = "value_integer",
        value_column_part: str = "value_string",
    ) -> pd.DataFrame:
        """Calculate quality metrics by part number.

        Args:
            ok_counter_uuid: UUID for good parts counter
            nok_counter_uuid: UUID for defective parts counter
            part_id_uuid: UUID for part number signal
            value_column_counter: Column containing counter values
            value_column_part: Column containing part numbers

        Returns:
            DataFrame with quality by part:
            - part_number: Part number/ID
            - ok_parts: Good parts produced
            - nok_parts: Defective parts
            - total_parts: Total parts produced
            - nok_rate_pct: Percentage of defective parts
            - first_pass_yield_pct: Percentage of good parts

        Example:
            >>> quality_by_part('good', 'bad', 'part_id')
                part_number  ok_parts  nok_parts  total_parts  nok_rate_pct  first_pass_yield_pct
            0   PART_A       1255      55         1310         4.2           95.8
            1   PART_B       890       38         928          4.1           95.9
        """
        # Get part ID data
        part_data = (
            self.dataframe[self.dataframe["uuid"] == part_id_uuid]
            .copy()
            .sort_values(self.time_column)
        )

        if part_data.empty:
            return pd.DataFrame(
                columns=[
                    "part_number",
                    "ok_parts",
                    "nok_parts",
                    "total_parts",
                    "nok_rate_pct",
                    "first_pass_yield_pct",
                ]
            )

        part_data[self.time_column] = pd.to_datetime(part_data[self.time_column])

        # Get OK counter data
        ok_data = (
            self.dataframe[self.dataframe["uuid"] == ok_counter_uuid]
            .copy()
            .sort_values(self.time_column)
        )

        # Get NOK counter data
        nok_data = (
            self.dataframe[self.dataframe["uuid"] == nok_counter_uuid]
            .copy()
            .sort_values(self.time_column)
        )

        results = []

        # Process OK parts by part number
        ok_by_part = {}
        if not ok_data.empty:
            ok_data[self.time_column] = pd.to_datetime(ok_data[self.time_column])

            # Merge with part data
            ok_subset = ok_data[[self.time_column, value_column_counter]].copy()
            part_subset = part_data[[self.time_column, value_column_part]].copy()

            merged_ok = pd.merge_asof(
                ok_subset, part_subset, on=self.time_column, direction="backward"
            )

            merged_ok = merged_ok.rename(columns={value_column_part: "part_number"})
            merged_ok = merged_ok.dropna(subset=["part_number"])

            # Calculate OK parts per part number
            for part_num, group in merged_ok.groupby("part_number"):
                if not group.empty:
                    first_count = group[value_column_counter].iloc[0]
                    last_count = group[value_column_counter].iloc[-1]
                    ok_by_part[part_num] = max(0, last_count - first_count)

        # Process NOK parts by part number
        nok_by_part = {}
        if not nok_data.empty:
            nok_data[self.time_column] = pd.to_datetime(nok_data[self.time_column])

            # Merge with part data
            nok_subset = nok_data[[self.time_column, value_column_counter]].copy()
            part_subset = part_data[[self.time_column, value_column_part]].copy()

            merged_nok = pd.merge_asof(
                nok_subset, part_subset, on=self.time_column, direction="backward"
            )

            merged_nok = merged_nok.rename(columns={value_column_part: "part_number"})
            merged_nok = merged_nok.dropna(subset=["part_number"])

            # Calculate NOK parts per part number
            for part_num, group in merged_nok.groupby("part_number"):
                if not group.empty:
                    first_count = group[value_column_counter].iloc[0]
                    last_count = group[value_column_counter].iloc[-1]
                    nok_by_part[part_num] = max(0, last_count - first_count)

        # Combine results
        all_parts = set(ok_by_part.keys()) | set(nok_by_part.keys())

        for part_num in sorted(all_parts):
            ok_parts = ok_by_part.get(part_num, 0)
            nok_parts = nok_by_part.get(part_num, 0)
            total_parts = ok_parts + nok_parts

            if total_parts > 0:
                nok_rate = (nok_parts / total_parts) * 100
                fpy = (ok_parts / total_parts) * 100
            else:
                nok_rate = 0
                fpy = 0

            results.append(
                {
                    "part_number": part_num,
                    "ok_parts": ok_parts,
                    "nok_parts": nok_parts,
                    "total_parts": total_parts,
                    "nok_rate_pct": round(nok_rate, 1),
                    "first_pass_yield_pct": round(fpy, 1),
                    "quality_pct": round(fpy, 1),
                }
            )

        return pd.DataFrame(results)

    def nok_by_reason(
        self,
        nok_counter_uuid: str,
        defect_reason_uuid: str,
        *,
        value_column_counter: str = "value_integer",
        value_column_reason: str = "value_string",
    ) -> pd.DataFrame:
        """Analyze NOK parts by defect reason.

        Args:
            nok_counter_uuid: UUID for defective parts counter
            defect_reason_uuid: UUID for defect reason signal
            value_column_counter: Column containing counter values
            value_column_reason: Column containing reason codes

        Returns:
            DataFrame with NOK by reason:
            - reason: Defect reason code
            - nok_parts: Number of defective parts
            - pct_of_total: Percentage of total NOK

        Example:
            >>> nok_by_reason('bad_parts', 'defect_reason')
                reason              nok_parts  pct_of_total
            0   Dimension_Error     45         40.5
            1   Surface_Defect      28         25.2
            2   Wrong_Color         22         19.8
        """
        nok_data = (
            self.dataframe[self.dataframe["uuid"] == nok_counter_uuid]
            .copy()
            .sort_values(self.time_column)
        )

        reason_data = (
            self.dataframe[self.dataframe["uuid"] == defect_reason_uuid]
            .copy()
            .sort_values(self.time_column)
        )

        if nok_data.empty or reason_data.empty:
            return pd.DataFrame(columns=["reason", "nok_parts", "pct_of_total"])

        nok_data[self.time_column] = pd.to_datetime(nok_data[self.time_column])
        reason_data[self.time_column] = pd.to_datetime(reason_data[self.time_column])

        # Merge NOK counter with reason - rename to avoid suffix issues
        nok_subset = nok_data[[self.time_column, value_column_counter]].copy()
        nok_subset = nok_subset.rename(columns={value_column_counter: "nok_count"})

        reason_subset = reason_data[[self.time_column, value_column_reason]].copy()
        reason_subset = reason_subset.rename(columns={value_column_reason: "reason"})

        merged = pd.merge_asof(
            nok_subset, reason_subset, on=self.time_column, direction="backward"
        )

        # Filter out empty or NaN reasons
        merged = merged.dropna(subset=["reason"])
        merged = merged[merged["reason"] != ""]

        if merged.empty:
            return pd.DataFrame(columns=["reason", "nok_parts", "pct_of_total"])

        # Calculate NOK parts per reason
        results = []
        for reason, group in merged.groupby("reason"):
            if not group.empty:
                first_count = group["nok_count"].iloc[0]
                last_count = group["nok_count"].iloc[-1]
                nok_parts = max(0, last_count - first_count)

                results.append(
                    {
                        "reason": reason,
                        "nok_parts": nok_parts,
                    }
                )

        if not results:
            return pd.DataFrame(columns=["reason", "nok_parts", "pct_of_total"])

        result_df = pd.DataFrame(results)

        # Calculate percentage
        total_nok = result_df["nok_parts"].sum()
        result_df["pct_of_total"] = (
            (result_df["nok_parts"] / total_nok * 100) if total_nok > 0 else 0
        )

        result_df["pct_of_total"] = result_df["pct_of_total"].round(1)

        # Sort by NOK parts descending
        result_df = result_df.sort_values("nok_parts", ascending=False)

        return result_df.reset_index(drop=True)

    def daily_quality_summary(
        self,
        ok_counter_uuid: str,
        nok_counter_uuid: str,
        *,
        value_column: str = "value_integer",
    ) -> pd.DataFrame:
        """Daily quality summary.

        Args:
            ok_counter_uuid: UUID for good parts counter
            nok_counter_uuid: UUID for defective parts counter
            value_column: Column containing counter values

        Returns:
            DataFrame with daily quality:
            - date: Production date
            - ok_parts: Good parts produced
            - nok_parts: Defective parts
            - total_parts: Total parts produced
            - nok_rate_pct: Percentage of defective parts
            - first_pass_yield_pct: Percentage of good parts

        Example:
            >>> daily_quality_summary('good', 'bad')
                date        ok_parts  nok_parts  total_parts  nok_rate_pct  first_pass_yield_pct
            0   2024-01-01  1255      55         1310         4.2           95.8
            1   2024-01-02  1308      42         1350         3.1           96.9
            2   2024-01-03  1290      60         1350         4.4           95.6
        """
        shift_quality = self.nok_by_shift(
            ok_counter_uuid, nok_counter_uuid, value_column=value_column
        )

        if shift_quality.empty:
            return pd.DataFrame(
                columns=[
                    "date",
                    "ok_parts",
                    "nok_parts",
                    "total_parts",
                    "nok_rate_pct",
                    "first_pass_yield_pct",
                    "quality_pct",
                ]
            )

        # Aggregate by date
        daily = (
            shift_quality.groupby("date")
            .agg(
                {
                    "ok_parts": "sum",
                    "nok_parts": "sum",
                    "total_parts": "sum",
                }
            )
            .reset_index()
        )

        # Recalculate percentages
        daily["nok_rate_pct"] = (daily["nok_parts"] / daily["total_parts"] * 100).round(
            1
        )
        daily["first_pass_yield_pct"] = (
            daily["ok_parts"] / daily["total_parts"] * 100
        ).round(1)
        daily["quality_pct"] = daily["first_pass_yield_pct"]

        return daily
