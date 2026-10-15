"""Shift-based production reporting.

Simple module for shift summaries:
- Production by shift
- Shift performance comparison
- Shift targets and actuals
"""

import logging
from typing import Literal, overload

import pandas as pd

from ts_shape.events._compat import warn_dict_return
from ts_shape.events._windows import add_shift_window, data_tz
from ts_shape.events.production._counter_groups import counter_totals
from ts_shape.utils.base import Base
from ts_shape.utils.deprecation import deprecated_class, keyword_only
from ts_shape.utils.shifts import assign_shift, production_dates

logger = logging.getLogger(__name__)


class ShiftReportEvents(Base):
    """Simple shift-based production reporting.

    Each UUID represents one signal:
    - counter_uuid: production counter
    - part_id_uuid: part number (optional)

    Merge keys: [date, shift] for shift-level outputs.

    Pipeline example::

        reporter = ShiftReportEvents(df)
        prod = reporter.shift_production('counter')
        # → merge with QualityEvents.nok_by_shift() on [date, shift]
        # → merge with DowntimeEvents.downtime_by_shift() on [date, shift]
        # → feed combined DataFrame into ShiftHandoverEvents.from_shift_data()

    Example usage:
        reporter = ShiftReportEvents(df, shift_definitions={
            "day": ("06:00", "14:00"),
            "afternoon": ("14:00", "22:00"),
            "night": ("22:00", "06:00"),
        })

        # Production per shift
        shift_prod = reporter.shift_production(
            counter_uuid='counter_signal',
            part_id_uuid='part_number_signal'
        )

        # Compare shifts
        comparison = reporter.shift_comparison(counter_uuid='counter_signal')
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        time_column: str = "systime",
        shift_definitions: dict[str, tuple[str, str]] | None = None,
    ) -> None:
        """Initialize shift reporter.

        Args:
            dataframe: Input DataFrame with timeseries data
            time_column: Name of timestamp column (default: 'systime')
            shift_definitions: Dictionary mapping shift names to (start, end) times
                              Default: 3-shift operation (06:00-14:00, 14:00-22:00, 22:00-06:00)

        Example shift_definitions:
            {
                "shift_1": ("06:00", "14:00"),
                "shift_2": ("14:00", "22:00"),
                "shift_3": ("22:00", "06:00"),
            }
        """
        super().__init__(dataframe, time_column=time_column)
        self.time_column = time_column

        # Default 3-shift operation
        self.shift_definitions = shift_definitions or {
            "shift_1": ("06:00", "14:00"),
            "shift_2": ("14:00", "22:00"),
            "shift_3": ("22:00", "06:00"),
        }

    @keyword_only(2)
    def shift_production(
        self,
        counter_uuid: str,
        part_id_uuid: str | None = None,
        *,
        value_column_counter: str = "value_integer",
        value_column_part: str = "value_string",
        date: str | None = None,
        handle_resets: bool = True,
        production_day: bool = True,
    ) -> pd.DataFrame:
        """Production quantity per shift.

        The reset-aware increments are computed once over the whole counter
        signal and each one is booked to the shift of the reading that
        reports it, so the shift quantities add up to the signal total: the
        parts made between the last reading of one shift and the first
        reading of the next are counted in the next shift (in the example,
        310 parts over 06:00-21:30 are 150 + 160, not 150 + 150). A shift
        boundary is also where counters commonly reset (shift change, part
        change, PLC restart); a decreasing counter is treated as a reset and
        the post-reset production is counted.

        Args:
            counter_uuid: Production counter UUID
            part_id_uuid: Part number UUID (optional, for part-specific production)
            value_column_counter: Column containing counter values
            value_column_part: Column containing part numbers
            date: Specific date in 'YYYY-MM-DD' format (optional). Selects the
                readings of that calendar date; their increments are still
                computed on the full signal, so the first reading of the day
                carries the production since the reading before it.
            handle_resets: When True (default), treat a decreasing counter as a reset and
                count the production after the reset instead of losing it. The
                result gains a ``resets`` column with the number of resets
                observed in each shift (a drop at the shift's first reading
                counts in that shift). When False, the quantity is
                ``max(0, last_count - first_count)`` of the shift's own
                readings: a shift containing a reset reports too little and
                the production between two shifts' readings is lost.
            production_day: When True, the post-midnight portion of an overnight
                shift (e.g. ``22:00 -> 06:00``) is attributed to the shift's
                start date, so a night shift is one contiguous ``(date, shift)``
                group instead of being split across two calendar dates (the
                default). When False, grouping is by raw calendar date. Only affects
                schedules with an overnight shift.

        Returns:
            DataFrame with production by shift:
            - start: Start of the shift window (``date`` at the shift start
                time from ``shift_definitions``, tz-aware in the data's tz)
            - end: End of the shift window (next day for overnight shifts)
            - duration_seconds: Shift window length in seconds
            - date: Production date
            - shift: Shift name
            - part_number: Part number (if part_id_uuid provided)
            - quantity: Parts produced: the reset-aware increments reported
                by the shift's readings (each since the previous reading)
            - resets: Number of counter resets in the shift (only when
                ``handle_resets=True``)

        Example:
            >>> import ts_shape
            >>> df = ts_shape.make_timeseries(["counter"], n_points=32, freq="30min",
            ...     start="2025-01-01 06:00", noise=0, drift=310, value_column="value_integer")
            >>> out = ts_shape.ShiftReportEvents(df).shift_production("counter")
            >>> out[["date", "shift", "quantity"]]
                     date    shift  quantity
            0  2025-01-01  shift_1       150
            1  2025-01-01  shift_2       160
        """
        counter_data = (
            self.dataframe[self.dataframe["uuid"] == counter_uuid]
            .copy()
            .sort_values(self.time_column)
        )

        cols = ["date", "shift", "quantity"]
        if part_id_uuid:
            cols.insert(2, "part_number")
        if handle_resets:
            cols.append("resets")
        if counter_data.empty:
            return add_shift_window(
                pd.DataFrame(columns=cols), self.shift_definitions, None
            )

        counter_data[self.time_column] = pd.to_datetime(counter_data[self.time_column])

        # Shifts and dates on the FULL signal: a date filter only selects which
        # groups are reported, so the first selected reading still carries the
        # production since the reading before it.
        select = None
        if date:
            target_date = pd.to_datetime(date).date()
            select = counter_data[self.time_column].dt.date == target_date

        # Assign shifts
        counter_data["shift"] = assign_shift(
            counter_data[self.time_column], self.shift_definitions
        )
        if production_day:
            counter_data["date"] = production_dates(
                counter_data[self.time_column],
                counter_data["shift"],
                self.shift_definitions,
            )
        else:
            counter_data["date"] = counter_data[self.time_column].dt.date

        # Add part numbers if provided
        group_cols = ["date", "shift"]
        if part_id_uuid:
            part_data = (
                self.dataframe[self.dataframe["uuid"] == part_id_uuid]
                .copy()
                .sort_values(self.time_column)
            )

            if not part_data.empty:
                part_data[self.time_column] = pd.to_datetime(
                    part_data[self.time_column]
                )

                # Select only needed columns to avoid suffix issues in merge
                # Keep only the columns we need from counter_data
                merge_cols = [self.time_column, value_column_counter, "shift", "date"]
                counter_subset = counter_data[merge_cols].copy()
                part_subset = part_data[[self.time_column, value_column_part]].copy()

                counter_data = pd.merge_asof(
                    counter_subset,
                    part_subset,
                    on=self.time_column,
                    direction="backward",
                )
                if select is not None:
                    select = select.reset_index(drop=True)

                # Rename part column
                counter_data = counter_data.rename(
                    columns={value_column_part: "part_number"}
                )
                group_cols.append("part_number")

        # Quantity per shift: increments over the whole signal, each booked to
        # the shift of the reading that reports it, so shifts add up.
        totals = counter_totals(
            counter_data,
            value_column_counter,
            group_cols,
            handle_resets=handle_resets,
            select=select,
        )
        out = totals[[*group_cols, "quantity"]].copy()
        out["quantity"] = out["quantity"].astype(int)
        if handle_resets:
            out["resets"] = totals["resets"]

        if out.empty:
            out = pd.DataFrame(columns=cols)
        return add_shift_window(
            out,
            self.shift_definitions,
            data_tz(counter_data[self.time_column]),
        )

    def shift_comparison(
        self,
        counter_uuid: str,
        *,
        value_column_counter: str = "value_integer",
        days: int = 7,
    ) -> pd.DataFrame:
        """Compare shift performance over recent days.

        Args:
            counter_uuid: Production counter UUID
            value_column_counter: Column containing counter values
            days: Number of recent days to analyze (default: 7)

        Returns:
            DataFrame with shift comparison:
            - shift: Shift name
            - avg_quantity: Average production per shift
            - min_quantity: Minimum production
            - max_quantity: Maximum production
            - std_quantity: Standard deviation
            - days_count: Number of days included

        Example:
            >>> import ts_shape
            >>> df = ts_shape.make_timeseries(["counter"], n_points=96, freq="30min",
            ...     start="2025-01-01 06:00", baseline=10, noise=3, value_column="value_integer")
            >>> df["value_integer"] = df["value_integer"].cumsum()  # running part counter
            >>> out = ts_shape.ShiftReportEvents(df).shift_comparison("counter")
            >>> out[["shift", "avg_quantity", "min_quantity", "max_quantity"]]
                 shift  avg_quantity  min_quantity  max_quantity
            0  shift_1         150.0           144           156
            1  shift_2         163.5           157           170
            2  shift_3         156.5           147           166
        """
        shift_prod = self.shift_production(
            counter_uuid, value_column_counter=value_column_counter
        )

        if shift_prod.empty:
            return pd.DataFrame(
                columns=[
                    "shift",
                    "avg_quantity",
                    "min_quantity",
                    "max_quantity",
                    "std_quantity",
                    "days_count",
                ]
            )

        # Filter to recent days
        shift_prod["date"] = pd.to_datetime(shift_prod["date"])
        cutoff_date = shift_prod["date"].max() - pd.Timedelta(days=days - 1)
        shift_prod = shift_prod[shift_prod["date"] >= cutoff_date]

        # Compare shifts
        comparison = (
            shift_prod.groupby("shift")["quantity"]
            .agg(
                [
                    ("avg_quantity", "mean"),
                    ("min_quantity", "min"),
                    ("max_quantity", "max"),
                    ("std_quantity", "std"),
                    ("days_count", "count"),
                ]
            )
            .reset_index()
        )

        return comparison

    def shift_targets(
        self,
        counter_uuid: str,
        targets: dict[str, float],
        *,
        value_column_counter: str = "value_integer",
        date: str | None = None,
    ) -> pd.DataFrame:
        """Compare actual production to shift targets.

        Args:
            counter_uuid: Production counter UUID
            targets: Dictionary mapping shift names to target quantities
            value_column_counter: Column containing counter values
            date: Specific date in 'YYYY-MM-DD' format (optional)

        Returns:
            DataFrame with target comparison:
            - date: Production date
            - shift: Shift name
            - actual: Actual production
            - target: Target production
            - variance: Difference (actual - target)
            - achievement_pct: Percentage of target achieved

        Example:
            >>> import ts_shape
            >>> df = ts_shape.make_timeseries(["counter"], n_points=32, freq="30min",
            ...     start="2025-01-01 06:00", noise=0, drift=310, value_column="value_integer")
            >>> rep = ts_shape.ShiftReportEvents(df)
            >>> out = rep.shift_targets("counter", targets={"shift_1": 150, "shift_2": 200})
            >>> out[["shift", "actual", "target", "variance", "achievement_pct"]]
                 shift  actual  target  variance  achievement_pct
            0  shift_1     150     150         0            100.0
            1  shift_2     160     200       -40             80.0
        """
        shift_prod = self.shift_production(
            counter_uuid, value_column_counter=value_column_counter, date=date
        )

        if shift_prod.empty:
            return pd.DataFrame(
                columns=[
                    "date",
                    "shift",
                    "actual",
                    "target",
                    "variance",
                    "achievement_pct",
                ]
            )

        # Add targets
        shift_prod["target"] = shift_prod["shift"].map(targets)
        shift_prod = shift_prod.dropna(subset=["target"])

        # Calculate variance and achievement
        shift_prod["actual"] = shift_prod["quantity"]
        shift_prod["variance"] = shift_prod["actual"] - shift_prod["target"]
        # Guard a zero target (unscheduled shift / placeholder) which would
        # otherwise yield inf, matching the guarded sibling in ShiftHandover.
        shift_prod["achievement_pct"] = (
            shift_prod["actual"] / shift_prod["target"] * 100
        ).where(shift_prod["target"] > 0, 0.0)

        return shift_prod[
            ["date", "shift", "actual", "target", "variance", "achievement_pct"]
        ]

    @overload
    def best_and_worst_shifts(
        self,
        counter_uuid: str,
        *,
        value_column_counter: str = ...,
        days: int = ...,
        as_frame: Literal[True] = ...,
    ) -> pd.DataFrame: ...

    @overload
    def best_and_worst_shifts(
        self,
        counter_uuid: str,
        *,
        value_column_counter: str = ...,
        days: int = ...,
        as_frame: Literal[False],
    ) -> dict[str, pd.DataFrame]: ...

    def best_and_worst_shifts(
        self,
        counter_uuid: str,
        *,
        value_column_counter: str = "value_integer",
        days: int = 30,
        as_frame: bool = True,
    ) -> dict[str, pd.DataFrame] | pd.DataFrame:
        """Identify best and worst performing shifts.

        Args:
            counter_uuid: Production counter UUID
            value_column_counter: Column containing counter values
            days: Number of recent days to analyze (default: 30)
            as_frame: When True (default), return a single DataFrame (see
                below). ``as_frame=False`` returns the dictionary form; it is
                deprecated, emits a ``DeprecationWarning`` and will be removed
                in the next major release.

        Returns:
            With ``as_frame=True`` (default), one DataFrame with columns:
            - start, end, duration_seconds: the shift window (``date`` at the
                shift start time, tz-aware in the data's tz; an overnight
                shift ends on the next day)
            - rank_type: ``'best'`` or ``'worst'``
            - date, shift, quantity
            The best rows come first (highest quantity first), then the worst
            rows (lowest quantity first). A shift can appear in both groups
            when there are fewer than ten shifts.

            With ``as_frame=False`` (deprecated), a dictionary with:
            - 'best': Top 5 best shifts (columns date, shift, quantity)
            - 'worst': Top 5 worst shifts (columns date, shift, quantity)

        Example:
            >>> import ts_shape
            >>> df = ts_shape.make_timeseries(["counter"], n_points=96, freq="30min",
            ...     start="2025-01-01 06:00", baseline=10, noise=3, value_column="value_integer")
            >>> df["value_integer"] = df["value_integer"].cumsum()  # running part counter
            >>> ranked = ts_shape.ShiftReportEvents(df).best_and_worst_shifts("counter")
            >>> best = ranked[ranked["rank_type"] == "best"]
            >>> best[["date", "shift", "quantity"]].head(3)
                    date    shift  quantity
            0 2025-01-01  shift_2       170
            1 2025-01-01  shift_3       166
            2 2025-01-02  shift_2       157
        """
        if not as_frame:
            warn_dict_return("ShiftReportEvents.best_and_worst_shifts")

        shift_prod = self.shift_production(
            counter_uuid, value_column_counter=value_column_counter
        )
        key_cols = ["date", "shift", "quantity"]
        frame_cols = ["start", "end", "duration_seconds", "rank_type", *key_cols]

        if shift_prod.empty:
            if as_frame:
                return pd.DataFrame(columns=frame_cols)
            empty_df = pd.DataFrame(columns=key_cols)
            return {"best": empty_df, "worst": empty_df}

        # Filter to recent days
        shift_prod["date"] = pd.to_datetime(shift_prod["date"])
        cutoff_date = shift_prod["date"].max() - pd.Timedelta(days=days - 1)
        shift_prod = shift_prod[shift_prod["date"] >= cutoff_date]

        # Get best and worst
        best_rows = shift_prod.nlargest(5, "quantity")
        worst_rows = shift_prod.nsmallest(5, "quantity")

        if as_frame:
            out = pd.concat(
                [
                    best_rows.assign(rank_type="best"),
                    worst_rows.assign(rank_type="worst"),
                ],
                ignore_index=True,
            )
            return out[frame_cols]

        best = best_rows[key_cols]
        worst = worst_rows[key_cols]

        return {"best": best, "worst": worst}


#: Deprecated name, removed in the next major release.
ShiftReporting = deprecated_class(ShiftReportEvents, "ShiftReporting")
