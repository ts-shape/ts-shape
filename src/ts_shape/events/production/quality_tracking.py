"""Quality and NOK (defective parts) tracking.

Essential module for daily quality analysis:
- NOK parts by shift and part number
- Scrap tracking
- First pass yield
- Quality rates
"""

import logging
from typing import Any

import pandas as pd

from ts_shape.events._windows import (
    add_day_window,
    add_shift_window,
    data_tz,
)
from ts_shape.utils.base import Base
from ts_shape.utils.counters import count_resets, counter_increments, counter_quantity
from ts_shape.utils.deprecation import deprecated_class
from ts_shape.utils.shifts import assign_shift, production_dates

logger = logging.getLogger(__name__)

_QUALITY_COLS = [
    "ok_parts",
    "nok_parts",
    "total_parts",
    "nok_rate_pct",
    "first_pass_yield_pct",
    "quality_pct",
]


class QualityEvents(Base):
    """Track NOK (defective) parts and quality metrics.

    Each UUID represents one signal:
    - ok_counter_uuid: counter for good parts
    - nok_counter_uuid: counter for NOK parts
    - part_id_uuid: part number signal (optional)
    - defect_reason_uuid: defect reason code (optional)

    Merge keys: [date, shift] for shift-level, [date] for daily, [part_number] for part-level.

    Pipeline outputs include ``quality_pct`` (alias for ``first_pass_yield_pct``)
    so downstream modules (ShiftHandoverEvents, PeriodSummaryEvents, OEEEvents)
    can join on a consistent column name.

    Example usage:
        tracker = QualityEvents(df)

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

        # Pipeline: feed into PeriodSummaryEvents
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
        shift_definitions: dict[str, tuple[str, str]] | None = None,
    ) -> None:
        """Initialize quality tracker.

        Args:
            dataframe: Input DataFrame with timeseries data
            time_column: Name of timestamp column (default: 'systime')
            shift_definitions: Dictionary mapping shift names to (start, end) times
                              Default: 3-shift operation (06:00-14:00, 14:00-22:00, 22:00-06:00)
        """
        super().__init__(dataframe, time_column=time_column)
        self.time_column = time_column

        # Default 3-shift operation
        self.shift_definitions = shift_definitions or {
            "shift_1": ("06:00", "14:00"),
            "shift_2": ("14:00", "22:00"),
            "shift_3": ("22:00", "06:00"),
        }

    def _assign_shift(self, timestamp: pd.Timestamp) -> str:
        """Assign shift based on time of day."""
        return assign_shift(timestamp, self.shift_definitions)

    def nok_by_shift(
        self,
        ok_counter_uuid: str,
        nok_counter_uuid: str,
        *,
        value_column: str = "value_integer",
        handle_resets: bool = True,
        production_day: bool = True,
    ) -> pd.DataFrame:
        """Calculate NOK (defective) parts per shift.

        Args:
            ok_counter_uuid: UUID for good parts counter
            nok_counter_uuid: UUID of the NOK part counter
            value_column: Column containing counter values
            handle_resets: When True (default), treat a decreasing counter as a reset
                and count the parts accumulated after the reset, for both the
                OK and NOK counters. The result gains a ``resets`` column with
                the number of resets observed in each shift (across both
                counters). When False, each counter uses
                ``max(0, last - first)`` and a shift containing a reset
                reports too little (often 0).
            production_day: Attribute an overnight shift's post-midnight
                readings to the shift's start date, so a night shift is one
                ``(date, shift)`` row (default). False groups by raw calendar
                date, which splits a night shift across two rows.

        Returns:
            DataFrame with quality metrics by shift:
            - start: Start of the shift window (``date`` at the shift start
                time from ``shift_definitions``, tz-aware in the data's tz)
            - end: End of the shift window (next day for overnight shifts)
            - duration_seconds: Shift window length in seconds
            - date: Production date
            - shift: Shift name
            - ok_parts: Good parts produced
            - nok_parts: NOK parts
            - total_parts: Total parts produced
            - nok_rate_pct: Percentage of NOK parts
            - first_pass_yield_pct: Percentage of good parts
            - quality_pct: Same as first_pass_yield_pct
            - resets: Number of counter resets in the shift (only when
                ``handle_resets=True``)

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

        shift_cols = ["date", "shift", *_QUALITY_COLS]
        if handle_resets:
            shift_cols.append("resets")
        if ok_data.empty and nok_data.empty:
            return add_shift_window(
                pd.DataFrame(columns=shift_cols), self.shift_definitions, None
            )

        # Process OK data
        ok_processed = None
        if not ok_data.empty:
            ok_data[self.time_column] = pd.to_datetime(ok_data[self.time_column])
            ok_data["shift"] = assign_shift(
                ok_data[self.time_column], self.shift_definitions
            )
            if production_day:
                ok_data["date"] = production_dates(
                    ok_data[self.time_column], ok_data["shift"], self.shift_definitions
                )
            else:
                ok_data["date"] = ok_data[self.time_column].dt.date
            ok_processed = ok_data

        # Process NOK data
        nok_processed = None
        if not nok_data.empty:
            nok_data[self.time_column] = pd.to_datetime(nok_data[self.time_column])
            nok_data["shift"] = assign_shift(
                nok_data[self.time_column], self.shift_definitions
            )
            if production_day:
                nok_data["date"] = production_dates(
                    nok_data[self.time_column],
                    nok_data["shift"],
                    self.shift_definitions,
                )
            else:
                nok_data["date"] = nok_data[self.time_column].dt.date
            nok_processed = nok_data

        # Calculate quantities per shift
        results = []

        # Get all unique date/shift combinations
        date_shifts: set[tuple[Any, Any]] = set()
        if ok_processed is not None:
            date_shifts.update(zip(ok_processed["date"], ok_processed["shift"]))
        if nok_processed is not None:
            date_shifts.update(zip(nok_processed["date"], nok_processed["shift"]))

        for date, shift in sorted(date_shifts):
            ok_parts: float = 0
            nok_parts: float = 0
            resets = 0

            # Calculate OK parts for this shift
            if ok_processed is not None:
                ok_shift = ok_processed[
                    (ok_processed["date"] == date) & (ok_processed["shift"] == shift)
                ]
                if not ok_shift.empty:
                    ok_parts = counter_quantity(
                        ok_shift[value_column], handle_resets=handle_resets
                    )
                    if handle_resets:
                        resets += count_resets(ok_shift[value_column])

            # Calculate NOK parts for this shift
            if nok_processed is not None:
                nok_shift = nok_processed[
                    (nok_processed["date"] == date) & (nok_processed["shift"] == shift)
                ]
                if not nok_shift.empty:
                    nok_parts = counter_quantity(
                        nok_shift[value_column], handle_resets=handle_resets
                    )
                    if handle_resets:
                        resets += count_resets(nok_shift[value_column])

            total_parts = ok_parts + nok_parts

            if total_parts > 0:
                nok_rate = (nok_parts / total_parts) * 100
                fpy = (ok_parts / total_parts) * 100
            else:
                nok_rate = 0
                fpy = 0

            row = {
                "date": date,
                "shift": shift,
                "ok_parts": ok_parts,
                "nok_parts": nok_parts,
                "total_parts": total_parts,
                "nok_rate_pct": nok_rate,
                "first_pass_yield_pct": fpy,
                "quality_pct": fpy,
            }
            if handle_resets:
                row["resets"] = int(resets)
            results.append(row)

        return add_shift_window(
            pd.DataFrame(results, columns=shift_cols),
            self.shift_definitions,
            data_tz(self.dataframe[self.time_column]),
        )

    def quality_by_part(
        self,
        ok_counter_uuid: str,
        nok_counter_uuid: str,
        part_id_uuid: str,
        *,
        value_column_counter: str = "value_integer",
        value_column_part: str = "value_string",
        handle_resets: bool = True,
    ) -> pd.DataFrame:
        """Calculate quality metrics by part number.

        Args:
            ok_counter_uuid: UUID for good parts counter
            nok_counter_uuid: UUID of the NOK part counter
            part_id_uuid: UUID for part number signal
            value_column_counter: Column containing counter values
            value_column_part: Column containing part numbers
            handle_resets: When True (default), treat a decreasing counter as a reset
                and count the parts accumulated after the reset. The part
                number is attached to a single global counter, so the
                reset-aware quantity is the sum of per-reading increments over
                the full time-sorted series, grouped by part -- this also
                corrects the over-count that ``last - first`` produces when
                parts alternate over a non-contiguous group. The result gains
                a ``resets`` column with the number of resets attributed to
                each part (across the OK and NOK counters). When False, the quantity is ``max(0, last - first)`` per part
                group and a reset is lost.

        Returns:
            DataFrame with quality by part:
            - part_number: Part number/ID
            - ok_parts: Good parts produced
            - nok_parts: NOK parts
            - total_parts: Total parts produced
            - nok_rate_pct: Percentage of NOK parts
            - first_pass_yield_pct: Percentage of good parts
            - quality_pct: Same as first_pass_yield_pct
            - resets: Number of counter resets for the part (only when
                ``handle_resets=True``)

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

        part_cols = ["part_number", *_QUALITY_COLS]
        if handle_resets:
            part_cols.append("resets")
        if part_data.empty:
            return pd.DataFrame(columns=part_cols)

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
        resets_by_part: dict[str, int] = {}
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

            if handle_resets:
                # Compute increments on the FULL time-sorted counter before
                # grouping (fixes both reset under-count and interleaving).
                merged_ok["_increment"] = counter_increments(
                    merged_ok[value_column_counter]
                )
                merged_ok["_is_reset"] = (
                    merged_ok[value_column_counter].diff() < 0
                ).fillna(False)

            # Calculate OK parts per part number
            for part_num, group in merged_ok.groupby("part_number"):
                if not group.empty:
                    if handle_resets:
                        ok_by_part[part_num] = float(group["_increment"].sum())
                        resets_by_part[part_num] = resets_by_part.get(
                            part_num, 0
                        ) + int(group["_is_reset"].sum())
                    else:
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

            if handle_resets:
                merged_nok["_increment"] = counter_increments(
                    merged_nok[value_column_counter]
                )
                merged_nok["_is_reset"] = (
                    merged_nok[value_column_counter].diff() < 0
                ).fillna(False)

            # Calculate NOK parts per part number
            for part_num, group in merged_nok.groupby("part_number"):
                if not group.empty:
                    if handle_resets:
                        nok_by_part[part_num] = float(group["_increment"].sum())
                        resets_by_part[part_num] = resets_by_part.get(
                            part_num, 0
                        ) + int(group["_is_reset"].sum())
                    else:
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

            row = {
                "part_number": part_num,
                "ok_parts": ok_parts,
                "nok_parts": nok_parts,
                "total_parts": total_parts,
                "nok_rate_pct": nok_rate,
                "first_pass_yield_pct": fpy,
                "quality_pct": fpy,
            }
            if handle_resets:
                row["resets"] = resets_by_part.get(part_num, 0)
            results.append(row)

        return pd.DataFrame(results, columns=part_cols)

    def nok_by_reason(
        self,
        nok_counter_uuid: str,
        defect_reason_uuid: str,
        *,
        value_column_counter: str = "value_integer",
        value_column_reason: str = "value_string",
        handle_resets: bool = True,
    ) -> pd.DataFrame:
        """Analyze NOK parts by defect reason.

        Args:
            nok_counter_uuid: UUID of the NOK part counter
            defect_reason_uuid: UUID for defect reason signal
            value_column_counter: Column containing counter values
            value_column_reason: Column containing reason codes
            handle_resets: When True (default), treat a decreasing counter as a reset
                and count the parts accumulated after the reset. The reason is
                attached to a single global counter, so the reset-aware
                quantity is the sum of per-reading increments over the full
                time-sorted series, grouped by reason -- this also corrects
                the over-count that ``last - first`` produces when reasons
                alternate over a non-contiguous group. The result gains a
                ``resets`` column with the number of resets attributed to each
                reason. When False, the quantity is
                ``max(0, last - first)`` per reason group and a reset is lost.

        Returns:
            DataFrame with NOK by reason:
            - reason: Defect reason code
            - nok_parts: Number of NOK parts
            - pct_of_total: Percentage of total NOK
            - resets: Number of counter resets for the reason (only when
                ``handle_resets=True``)

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

        columns = ["reason", "nok_parts", "pct_of_total"]
        if handle_resets:
            columns.append("resets")

        if nok_data.empty or reason_data.empty:
            return pd.DataFrame(columns=columns)

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
            return pd.DataFrame(columns=columns)

        if handle_resets:
            # Compute increments/resets on the FULL time-sorted counter before
            # grouping (fixes both reset under-count and interleaving).
            merged["_increment"] = counter_increments(merged["nok_count"])
            merged["_is_reset"] = (merged["nok_count"].diff() < 0).fillna(False)

        # Calculate NOK parts per reason
        results = []
        for reason, group in merged.groupby("reason"):
            if not group.empty:
                if handle_resets:
                    nok_parts = float(group["_increment"].sum())
                    row = {"reason": reason, "nok_parts": nok_parts}
                    row["resets"] = int(group["_is_reset"].sum())
                else:
                    first_count = group["nok_count"].iloc[0]
                    last_count = group["nok_count"].iloc[-1]
                    nok_parts = max(0, last_count - first_count)
                    row = {"reason": reason, "nok_parts": nok_parts}
                results.append(row)

        if not results:
            return pd.DataFrame(columns=columns)

        result_df = pd.DataFrame(results)

        # Calculate percentage
        total_nok = result_df["nok_parts"].sum()
        result_df["pct_of_total"] = (
            (result_df["nok_parts"] / total_nok * 100) if total_nok > 0 else 0
        )

        # Sort by NOK parts descending
        result_df = result_df.sort_values("nok_parts", ascending=False)

        return result_df.reset_index(drop=True)[columns]

    def daily_quality_summary(
        self,
        ok_counter_uuid: str,
        nok_counter_uuid: str,
        *,
        value_column: str = "value_integer",
        handle_resets: bool = True,
        production_day: bool = True,
    ) -> pd.DataFrame:
        """Daily quality summary.

        Args:
            ok_counter_uuid: UUID for good parts counter
            nok_counter_uuid: UUID of the NOK part counter
            value_column: Column containing counter values
            handle_resets: When True (default), treat a decreasing counter as a reset
                and count the parts accumulated after the reset (see
                :meth:`nok_by_shift`). The result gains a ``resets`` column
                with the number of resets observed that day. When False, a shift containing a reset reports too little.
            production_day: Forwarded to :meth:`nok_by_shift`; attributes an
                overnight shift's post-midnight readings to the shift's start
                date. Default True; False groups by raw calendar date, which splits a night shift across two rows.

        Returns:
            DataFrame with daily quality:
            - start: Midnight of ``date`` (tz-aware in the data's tz)
            - end: Midnight of the following day
            - duration_seconds: Day length in seconds
            - date: Production date
            - ok_parts: Good parts produced
            - nok_parts: NOK parts
            - total_parts: Total parts produced
            - nok_rate_pct: Percentage of NOK parts
            - first_pass_yield_pct: Percentage of good parts
            - quality_pct: Same as first_pass_yield_pct
            - resets: Number of counter resets that day (only when
                ``handle_resets=True``)

        Example:
            >>> daily_quality_summary('good', 'bad')
                date        ok_parts  nok_parts  total_parts  nok_rate_pct  first_pass_yield_pct
            0   2024-01-01  1255      55         1310         4.2           95.8
            1   2024-01-02  1308      42         1350         3.1           96.9
            2   2024-01-03  1290      60         1350         4.4           95.6
        """
        shift_quality = self.nok_by_shift(
            ok_counter_uuid,
            nok_counter_uuid,
            value_column=value_column,
            handle_resets=handle_resets,
            production_day=production_day,
        )

        base_columns = [
            "date",
            "ok_parts",
            "nok_parts",
            "total_parts",
            "nok_rate_pct",
            "first_pass_yield_pct",
            "quality_pct",
        ]
        if handle_resets:
            base_columns.append("resets")

        if shift_quality.empty:
            return add_day_window(pd.DataFrame(columns=base_columns), None)

        # Aggregate by date
        agg = {
            "ok_parts": "sum",
            "nok_parts": "sum",
            "total_parts": "sum",
        }
        if handle_resets:
            agg["resets"] = "sum"

        daily = shift_quality.groupby("date").agg(agg).reset_index()

        # Recalculate percentages. Guard against zero-total days (0/0 = NaN).
        daily["nok_rate_pct"] = (daily["nok_parts"] / daily["total_parts"] * 100).where(
            daily["total_parts"] > 0, 0.0
        )
        daily["first_pass_yield_pct"] = (
            daily["ok_parts"] / daily["total_parts"] * 100
        ).where(daily["total_parts"] > 0, 0.0)
        daily["quality_pct"] = daily["first_pass_yield_pct"]

        return add_day_window(daily, data_tz(self.dataframe[self.time_column]))


#: Deprecated name, removed in the next major release.
QualityTracking = deprecated_class(QualityEvents, "QualityTracking")
