"""Quality and NOK (defective parts) tracking.

Essential module for daily quality analysis:
- NOK parts by shift and part number
- Scrap tracking
- First pass yield
- Quality rates
"""

import logging
from typing import TYPE_CHECKING, Any

import pandas as pd

from ts_shape.events._windows import (
    add_day_window,
    add_shift_window,
    data_tz,
)
from ts_shape.events.production._counter_groups import counter_totals
from ts_shape.utils.base import Base
from ts_shape.utils.counters import counter_increments
from ts_shape.utils.deprecation import deprecated_class
from ts_shape.utils.shifts import assign_shift, production_dates
from ts_shape.utils.time_accounting import counter_by_period, shift_periods

if TYPE_CHECKING:
    from typing_extensions import deprecated

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

    Example:
        >>> import pandas as pd, ts_shape
        >>> kw = dict(n_points=20, freq="1min", noise=0, value_column="value_integer")
        >>> df = pd.concat([ts_shape.make_timeseries(["ok"], drift=190, **kw),
        ...                 ts_shape.make_timeseries(["nok"], drift=19, **kw),
        ...                 ts_shape.make_id_signal("part", ["A", "B"], freq="1min")])
        >>> out = ts_shape.QualityEvents(df).quality_by_part("ok", "nok", "part")
        >>> out[["part_number", "ok_parts", "nok_parts", "first_pass_yield_pct"]].round(1)
          part_number  ok_parts  nok_parts  first_pass_yield_pct
        0           A      90.0        9.0                  90.9
        1           B     100.0       10.0                  90.9
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
                OK and NOK counters. The increments are computed once over
                each whole counter signal; the increment between two readings
                is spread over the time between them and split at the shift
                boundaries in proportion to time (ISO 22400-2: a part belongs
                to the period it was produced in), so the shifts add up to
                the signal totals and the parts made in the last hour of a
                night shift stay in that shift even though the reading that
                reports them falls in the morning shift. Quantities can
                therefore be fractional. A reset counts in the shift of the
                reading that shows it. The result gains a ``resets`` column with the number of resets
                observed in each shift (across both counters). When False,
                each counter uses ``max(0, last - first)`` of the shift's own
                readings; a shift containing a reset reports too little (often
                0) and the production between two shifts' readings is lost.
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
            >>> import pandas as pd, ts_shape
            >>> kw = dict(n_points=16, freq="30min", start="2025-01-01 06:00", noise=0,
            ...           value_column="value_integer")
            >>> df = pd.concat([ts_shape.make_timeseries(["ok"], drift=150, **kw),
            ...                 ts_shape.make_timeseries(["nok"], drift=15, **kw)])
            >>> out = ts_shape.QualityEvents(df).nok_by_shift("ok", "nok")
            >>> out[["shift", "ok_parts", "nok_parts", "first_pass_yield_pct"]].round(1)
                 shift  ok_parts  nok_parts  first_pass_yield_pct
            0  shift_1     150.0       15.0                  90.9
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

        # Quantities per shift. With handle_resets the reset-aware increment
        # between two readings is spread over the time between them and split
        # at the shift boundaries in proportion to time (ISO 22400-2: a part
        # belongs to the period it was produced in), so a night shift keeps
        # the parts its last hour produced even though the reading reporting
        # them falls in the morning shift.
        ok_totals = (
            self._counter_by_shift(
                ok_processed, value_column, handle_resets, production_day
            )
            if ok_processed is not None
            else None
        )
        nok_totals = (
            self._counter_by_shift(
                nok_processed, value_column, handle_resets, production_day
            )
            if nok_processed is not None
            else None
        )
        ok_by_shift: dict[tuple[Any, Any], tuple[float, int]] = {}
        nok_by_shift: dict[tuple[Any, Any], tuple[float, int]] = {}
        for totals, target in ((ok_totals, ok_by_shift), (nok_totals, nok_by_shift)):
            if totals is None:
                continue
            for d, sh, qty, n_resets in totals[
                ["date", "shift", "quantity", "resets"]
            ].itertuples(index=False):
                target[(d, sh)] = (float(qty), int(n_resets))

        results = []
        for date, shift in sorted(set(ok_by_shift) | set(nok_by_shift)):
            ok_parts, ok_resets = ok_by_shift.get((date, shift), (0.0, 0))
            nok_parts, nok_resets = nok_by_shift.get((date, shift), (0.0, 0))
            resets = ok_resets + nok_resets

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

    def _counter_by_shift(
        self,
        data: pd.DataFrame,
        value_column: str,
        handle_resets: bool,
        production_day: bool,
    ) -> pd.DataFrame:
        """Counter quantity and resets per ``(date, shift)`` of one counter.

        ``data`` is time-sorted with ``date`` / ``shift`` assigned per
        reading. With ``handle_resets`` increments are spread over the time
        they were produced in (see :func:`~ts_shape.utils.time_accounting.
        counter_by_period`); without it each shift reports
        ``max(0, last - first)`` of its own readings.
        """
        keys = ["date", "shift"]
        if not handle_resets:
            return counter_totals(data, value_column, keys, handle_resets=False)
        times = data[self.time_column]
        periods = shift_periods(
            times.iloc[0],
            times.iloc[-1],
            self.shift_definitions,
            production_day=production_day,
        )
        return counter_by_period(
            times, data[value_column], periods, data[keys], keys=keys
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
            >>> import pandas as pd, ts_shape
            >>> kw = dict(n_points=20, freq="1min", noise=0, value_column="value_integer")
            >>> df = pd.concat([ts_shape.make_timeseries(["ok"], drift=190, **kw),
            ...                 ts_shape.make_timeseries(["nok"], drift=19, **kw),
            ...                 ts_shape.make_id_signal("part", ["A", "B"], freq="1min")])
            >>> out = ts_shape.QualityEvents(df).quality_by_part("ok", "nok", "part")
            >>> out[["part_number", "ok_parts", "nok_parts", "nok_rate_pct"]]
              part_number  ok_parts  nok_parts  nok_rate_pct
            0           A      90.0        9.0      9.090909
            1           B     100.0       10.0      9.090909
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
            >>> import pandas as pd, ts_shape
            >>> kw = dict(n_points=20, freq="1min", noise=0, value_column="value_integer")
            >>> df = pd.concat([ts_shape.make_timeseries(["nok"], drift=19, **kw),
            ...                 ts_shape.make_id_signal("reason", ["Scratch", "Dent"], freq="1min")])
            >>> ts_shape.QualityEvents(df).nok_by_reason("nok", "reason")
                reason  nok_parts  pct_of_total  resets
            0     Dent       10.0     52.631579       0
            1  Scratch        9.0     47.368421       0
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
                :meth:`nok_by_shift`; the day totals add up to the signal
                totals). The result gains a ``resets`` column
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
            >>> import pandas as pd, ts_shape
            >>> kw = dict(n_points=16, freq="30min", start="2025-01-01 06:00", noise=0,
            ...           value_column="value_integer")
            >>> df = pd.concat([ts_shape.make_timeseries(["ok"], drift=150, **kw),
            ...                 ts_shape.make_timeseries(["nok"], drift=15, **kw)])
            >>> out = ts_shape.QualityEvents(df).daily_quality_summary("ok", "nok")
            >>> out[["date", "ok_parts", "nok_parts", "nok_rate_pct"]].round(1)
                     date  ok_parts  nok_parts  nok_rate_pct
            0  2025-01-01     150.0       15.0           9.1
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
if TYPE_CHECKING:

    @deprecated("QualityTracking is deprecated, use QualityEvents")
    class QualityTracking(QualityEvents): ...

else:
    QualityTracking = deprecated_class(QualityEvents, "QualityTracking")
