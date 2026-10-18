"""Automated shift handover report generation.

Standardize shift-to-shift communication:
- Production summary
- Quality summary
- Downtime summary
- Issues to watch

Two modes of operation:
1. From raw signals: ``generate_report(counter_uuid, ok_counter_uuid, ...)``
2. From pre-computed DataFrames: ``from_shift_data(production_df, quality_df, downtime_df)``
"""

import logging
import warnings
from typing import TYPE_CHECKING, Any, Literal, overload

import pandas as pd

from ts_shape.events._compat import state_mask, warn_dict_return
from ts_shape.events._windows import add_shift_window, data_tz
from ts_shape.events.production._counter_groups import counter_totals
from ts_shape.utils.base import Base
from ts_shape.utils.deprecation import (
    REMOVAL,
    caller_stacklevel,
    deprecated_class,
    keyword_only,
    renamed_kwargs,
)
from ts_shape.utils.shifts import assign_shift, production_dates
from ts_shape.utils.time_accounting import (
    counter_by_period,
    reading_intervals,
    shift_periods,
    time_by_period,
)

if TYPE_CHECKING:
    from typing_extensions import deprecated

logger = logging.getLogger(__name__)


class ShiftHandoverEvents(Base):
    """Generate automated shift handover reports.

    Combines production, quality, and downtime data into a single summary
    suitable for shift handover meetings.

    Merge keys: [date, shift] — the report output is keyed on these columns.

    Two usage patterns:

    **Pattern A — from raw signals (one-step):**

        report = ShiftHandoverEvents(df)
        result = report.generate_report(
            counter_uuid='prod', ok_counter_uuid='ok',
            nok_counter_uuid='nok', run_state_uuid='state',
            targets={'shift_1': 450, 'shift_2': 450},
        )

    **Pattern B — pipeline (compose from upstream modules):**

        # Step 1: compute upstream DataFrames
        prod   = ShiftReportEvents(df).shift_production('counter')
        qual   = QualityEvents(df).nok_by_shift('ok', 'nok')
        downtime = DowntimeEvents(df).downtime_by_shift('state')

        # Step 2: assemble into handover report
        result = ShiftHandoverEvents.from_shift_data(
            production_df=prod,
            quality_df=qual,
            downtime_df=downtime,
            targets={'shift_1': 450, 'shift_2': 450},
        )

    Both patterns return the same output schema.

    Run state may be a string state signal (``running_value="Running"`` in
    ``value_string``) or a boolean run signal (``value_bool``, True =
    running); see :meth:`generate_report`. ``downtime_seconds`` is the
    downtime in seconds; ``downtime_minutes`` is kept as a convenience.
    """

    MERGE_KEYS = ["date", "shift"]
    OUTPUT_COLUMNS = [
        "date",
        "shift",
        "production",
        "production_target",
        "production_achievement_pct",
        "ok_parts",
        "nok_parts",
        "quality_pct",
        "availability_pct",
        "downtime_minutes",
        "downtime_seconds",
    ]
    ISSUE_COLUMNS = [
        "start",
        "end",
        "duration_seconds",
        "date",
        "shift",
        "metric",
        "value",
        "threshold",
        "severity",
        "severity_level",
    ]

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        time_column: str = "systime",
        shift_definitions: dict[str, tuple[str, str]] | None = None,
    ) -> None:
        """Initialize the shift handover report.

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

    def _counter_by_shift(
        self,
        uuid: str,
        value_column: str,
        *,
        handle_resets: bool = True,
        production_day: bool = True,
    ) -> pd.DataFrame:
        """Get counter deltas grouped by date/shift.

        When ``handle_resets`` is True a decreasing counter is treated as a
        reset and the post-reset production is counted, and a ``resets`` column
        with the per-shift reset count is added. Increments are computed once
        over the whole signal; the increment between two readings is spread
        over the time between them and split at the shift boundaries in
        proportion to time (ISO 22400-2: a part belongs to the shift it was
        produced in), so the shifts add up to the signal total and the
        quantities can be fractional. A reset counts in the shift of the
        reading that shows it. When False each shift reports
        ``max(0, last - first)`` of its own readings. When ``production_day``
        is True an overnight shift's post-midnight time is attributed to the
        shift's start date.
        """
        data = (
            self.dataframe[self.dataframe["uuid"] == uuid]
            .copy()
            .sort_values(self.time_column)
        )
        if data.empty:
            cols = ["date", "shift", "quantity"]
            if handle_resets:
                cols.append("resets")
            return pd.DataFrame(columns=cols)

        data[self.time_column] = pd.to_datetime(data[self.time_column])
        data["shift"] = assign_shift(data[self.time_column], self.shift_definitions)
        data["date"] = self._group_dates(data, production_day)

        keys = ["date", "shift"]
        if handle_resets:
            # Increments spread over the time they were produced in, split at
            # the shift boundaries (same accounting as OEE / quality).
            times = data[self.time_column]
            periods = shift_periods(
                times.iloc[0],
                times.iloc[-1],
                self.shift_definitions,
                production_day=production_day,
            )
            totals = counter_by_period(
                times, data[value_column], periods, data[keys], keys=keys
            )
        else:
            totals = counter_totals(data, value_column, keys, handle_resets=False)
        totals["quantity"] = totals["quantity"].astype(float)
        cols = ["date", "shift", "quantity"]
        if handle_resets:
            cols.append("resets")
        return totals[cols]

    def _group_dates(self, data: pd.DataFrame, production_day: bool) -> pd.Series:
        """Grouping date per row: calendar date, or production day when set."""
        if production_day:
            return production_dates(
                data[self.time_column], data["shift"], self.shift_definitions
            )
        return data[self.time_column].dt.date

    def _availability_by_shift(
        self,
        run_state_uuid: str,
        running_value: str | bool,
        value_column: str,
        *,
        production_day: bool = True,
    ) -> pd.DataFrame:
        """Compute availability grouped by date/shift.

        Each reading's state holds until the next reading; the intervals are
        split at the shift boundaries, so each shift gets the time that fell
        inside it.
        """
        data = (
            self.dataframe[self.dataframe["uuid"] == run_state_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        if data.empty:
            return pd.DataFrame(
                columns=[
                    "date",
                    "shift",
                    "availability_pct",
                    "downtime_minutes",
                    "downtime_seconds",
                ]
            )

        data[self.time_column] = pd.to_datetime(data[self.time_column])
        times = data[self.time_column]
        is_running = state_mask(data, value_column, running_value, bool_value=True)
        # Each reading's state holds until the next reading; the intervals
        # are split at the shift boundaries (ISO 22400-2 time model).
        intervals = reading_intervals(
            times, is_running.map({True: "up", False: "down"})
        )
        periods = shift_periods(
            times.iloc[0],
            times.iloc[-1],
            self.shift_definitions,
            production_day=production_day,
        )
        seconds = time_by_period(
            intervals, periods, keys=["date", "shift"], states=["up", "down"]
        )

        results = []
        for dt, shift, up, down in seconds[["date", "shift", "up", "down"]].itertuples(
            index=False, name=None
        ):
            total = up + down
            avail = (up / total * 100) if total > 0 else 0.0
            results.append(
                {
                    "date": dt,
                    "shift": shift,
                    "availability_pct": avail,
                    "downtime_minutes": down / 60,
                    "downtime_seconds": float(down),
                }
            )

        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Pipeline entry-point: build report from pre-computed DataFrames
    # ------------------------------------------------------------------

    @staticmethod
    @keyword_only(1)
    def from_shift_data(
        production_df: pd.DataFrame,
        quality_df: pd.DataFrame | None = None,
        downtime_df: pd.DataFrame | None = None,
        *,
        targets: dict[str, float] | None = None,
        report_date: str | None = None,
    ) -> pd.DataFrame:
        """Build a handover report from pre-computed shift-level DataFrames.

        This is the **pipeline-friendly** entry-point.  Instead of reading raw
        signals, it accepts DataFrames that were already computed by upstream
        modules (ShiftReportEvents, QualityEvents, DowntimeEvents, etc.).

        Args:
            production_df: DataFrame with [date, shift, quantity] — e.g. from
                ``ShiftReportEvents.shift_production()``.
            quality_df: DataFrame with [date, shift, ok_parts, nok_parts, quality_pct]
                — e.g. from ``QualityEvents.nok_by_shift()``.  Optional.
            downtime_df: DataFrame with [date, shift, availability_pct,
                downtime_seconds] — e.g. from
                ``DowntimeEvents.downtime_by_shift()``.  Optional. A frame
                with only the legacy ``downtime_minutes`` still works, with a
                ``DeprecationWarning``, until the next major release.
            targets: Per-shift production targets (dict).
            report_date: Specific date (YYYY-MM-DD).  If None, uses latest.

        Returns:
            DataFrame with columns:
            date, shift, production, production_target, production_achievement_pct,
            ok_parts, nok_parts, quality_pct, availability_pct, downtime_minutes,
            downtime_seconds. When ``downtime_df`` carries only one of
            ``downtime_seconds`` / ``downtime_minutes``, the other is derived
            from it.
        """
        output_cols = ShiftHandoverEvents.OUTPUT_COLUMNS

        if production_df.empty:
            return pd.DataFrame(columns=output_cols)

        prod = production_df.copy()

        # Normalize column name
        if "quantity" in prod.columns and "production" not in prod.columns:
            prod = prod.rename(columns={"quantity": "production"})

        # Dates may come as strings, datetimes or dates: compare them as dates
        # in every frame, or the merges below silently match nothing.
        def as_dates(frame: pd.DataFrame) -> pd.DataFrame:
            frame = frame.copy()
            frame["date"] = pd.to_datetime(frame["date"]).dt.date
            return frame

        prod = as_dates(prod)
        if quality_df is not None and not quality_df.empty:
            quality_df = as_dates(quality_df)
        if downtime_df is not None and not downtime_df.empty:
            downtime_df = as_dates(downtime_df)

        # Filter to report date
        if report_date:
            target_date = pd.to_datetime(report_date).date()
        else:
            target_date = prod["date"].max()

        result = prod[prod["date"] == target_date].copy()

        if result.empty:
            return pd.DataFrame(columns=output_cols)

        # Merge quality
        if quality_df is not None and not quality_df.empty:
            q = quality_df[quality_df["date"] == target_date].copy()
            q_cols = ["date", "shift"]
            if "ok_parts" in q.columns:
                q_cols.append("ok_parts")
            if "nok_parts" in q.columns:
                q_cols.append("nok_parts")
            if "quality_pct" in q.columns:
                q_cols.append("quality_pct")
            elif "first_pass_yield_pct" in q.columns:
                q = q.rename(columns={"first_pass_yield_pct": "quality_pct"})
                q_cols.append("quality_pct")
            result = result.merge(q[q_cols], on=["date", "shift"], how="left")

        for col in ["ok_parts", "nok_parts", "quality_pct"]:
            if col not in result.columns:
                result[col] = 0 if col != "quality_pct" else 0.0

        # Merge downtime
        if downtime_df is not None and not downtime_df.empty:
            if (
                "downtime_minutes" in downtime_df.columns
                and "downtime_seconds" not in downtime_df.columns
            ):
                warnings.warn(
                    "ShiftHandoverEvents.from_shift_data(): a downtime_df with "
                    "'downtime_minutes' but no 'downtime_seconds' is deprecated; "
                    "pass 'downtime_seconds' (DowntimeEvents.downtime_by_shift() "
                    f"returns it). The minutes input will be removed in {REMOVAL}",
                    DeprecationWarning,
                    stacklevel=caller_stacklevel(),
                )
            d = downtime_df[downtime_df["date"] == target_date].copy()
            d_cols = ["date", "shift"]
            if "availability_pct" in d.columns:
                d_cols.append("availability_pct")
            if "downtime_minutes" in d.columns:
                d_cols.append("downtime_minutes")
            if "downtime_seconds" in d.columns:
                d_cols.append("downtime_seconds")
            result = result.merge(d[d_cols], on=["date", "shift"], how="left")

        has_minutes = "downtime_minutes" in result.columns
        has_seconds = "downtime_seconds" in result.columns
        if has_seconds and not has_minutes:
            result["downtime_minutes"] = result["downtime_seconds"] / 60
        elif has_minutes and not has_seconds:
            result["downtime_seconds"] = result["downtime_minutes"] * 60

        for col in ["availability_pct", "downtime_minutes", "downtime_seconds"]:
            if col not in result.columns:
                result[col] = 0.0

        result = result.fillna(0)

        # Targets
        if targets:
            result["production_target"] = result["shift"].map(targets).fillna(0)
            result["production_achievement_pct"] = (
                result["production"] / result["production_target"] * 100
            ).where(result["production_target"] > 0, 0)
        else:
            result["production_target"] = 0.0
            result["production_achievement_pct"] = 0.0

        # Ensure quality_pct is computed if we have ok/nok but no quality_pct from upstream
        if (result["quality_pct"] == 0).all() and (result["ok_parts"] > 0).any():
            total = result["ok_parts"] + result["nok_parts"]
            result["quality_pct"] = (result["ok_parts"] / total * 100).where(
                total > 0, 0
            )

        # Return with consistent column order
        for col in output_cols:
            if col not in result.columns:
                result[col] = 0.0

        return result[output_cols].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Raw-signal entry-point
    # ------------------------------------------------------------------

    @renamed_kwargs(state_uuid="run_state_uuid")
    def generate_report(
        self,
        counter_uuid: str,
        ok_counter_uuid: str,
        nok_counter_uuid: str,
        run_state_uuid: str,
        *,
        targets: dict[str, float] | None = None,
        quality_target_pct: float = 98.0,
        availability_target_pct: float = 90.0,
        running_value: str | bool = "Running",
        value_column_counter: str = "value_integer",
        value_column_state: str = "value_string",
        report_date: str | None = None,
        handle_resets: bool = True,
        production_day: bool = True,
    ) -> pd.DataFrame:
        """Generate a shift handover report from raw timeseries signals.

        For pipeline usage with pre-computed DataFrames, use
        :meth:`from_shift_data` instead.

        Args:
            counter_uuid: UUID of production counter.
            ok_counter_uuid: UUID of good parts counter.
            nok_counter_uuid: UUID of the NOK part counter.
            run_state_uuid: UUID of machine state signal.
            targets: Per-shift production targets.
            quality_target_pct: Quality target percentage.
            availability_target_pct: Availability target percentage.
            running_value: Value indicating machine is running. A bool
                matches a boolean run signal (``value_bool`` by default); a
                boolean run signal is also used, with True as running, when
                the state rows carry no ``value_column_state`` value.
            value_column_counter: Column for counter values.
            value_column_state: Column for state values.
            report_date: Specific date (YYYY-MM-DD). If None, uses latest date.
            handle_resets: When True (default), treat a decreasing counter as a reset and
                count the production after the reset for production, ok_parts and
                nok_parts. A shift boundary is exactly where counters commonly
                reset, so ``max(0, last - first)`` (``handle_resets=False``) under-reports any
                shift containing a reset. The increments are computed once over
                each whole signal (before ``report_date`` is selected); the
                increment between two readings is spread over the time
                between them and split at the shift boundaries in proportion
                to time, so the shifts add up to the signal totals and
                ``production`` / ``ok_parts`` / ``nok_parts`` can be
                fractional; with
                ``handle_resets=False`` the production between two shifts'
                readings is lost. The result gains a ``resets`` column
                with the number of production-counter resets in each shift.
            production_day: When True, an overnight shift's post-midnight
                readings are attributed to the shift's start date, so a night
                shift is one contiguous ``(date, shift)`` group instead of being
                split across two calendar dates. Default True. Only affects schedules with an
                overnight shift.

        Returns:
            DataFrame with columns:
            start, end, duration_seconds (the shift window from
            ``shift_definitions``, tz-aware in the data's tz; an overnight
            shift ends on the next day), date, shift, production,
            production_target, production_achievement_pct, ok_parts,
            nok_parts, quality_pct, availability_pct, downtime_minutes
            (convenience), downtime_seconds, and ``resets`` (only when
            ``handle_resets=True``).
        """
        output_cols = list(self.OUTPUT_COLUMNS)
        if handle_resets:
            output_cols.append("resets")

        # Production
        prod = self._counter_by_shift(
            counter_uuid,
            value_column_counter,
            handle_resets=handle_resets,
            production_day=production_day,
        )
        ok = self._counter_by_shift(
            ok_counter_uuid,
            value_column_counter,
            handle_resets=handle_resets,
            production_day=production_day,
        )
        nok = self._counter_by_shift(
            nok_counter_uuid,
            value_column_counter,
            handle_resets=handle_resets,
            production_day=production_day,
        )
        avail = self._availability_by_shift(
            run_state_uuid,
            running_value,
            value_column_state,
            production_day=production_day,
        )

        if prod.empty:
            return add_shift_window(
                pd.DataFrame(columns=output_cols), self.shift_definitions, None
            )

        # Filter to report date
        if report_date:
            target_date = pd.to_datetime(report_date).date()
        else:
            target_date = prod["date"].max()

        prod = prod[prod["date"] == target_date]

        # Merge all data
        result = prod.rename(columns={"quantity": "production"})

        if not ok.empty:
            ok_filt = ok[ok["date"] == target_date].rename(
                columns={"quantity": "ok_parts"}
            )
            result = result.merge(
                ok_filt[["date", "shift", "ok_parts"]], on=["date", "shift"], how="left"
            )
        else:
            result["ok_parts"] = 0

        if not nok.empty:
            nok_filt = nok[nok["date"] == target_date].rename(
                columns={"quantity": "nok_parts"}
            )
            result = result.merge(
                nok_filt[["date", "shift", "nok_parts"]],
                on=["date", "shift"],
                how="left",
            )
        else:
            result["nok_parts"] = 0

        if not avail.empty:
            avail_filt = avail[avail["date"] == target_date]
            result = result.merge(
                avail_filt[
                    [
                        "date",
                        "shift",
                        "availability_pct",
                        "downtime_minutes",
                        "downtime_seconds",
                    ]
                ],
                on=["date", "shift"],
                how="left",
            )
        else:
            result["availability_pct"] = 0.0
            result["downtime_minutes"] = 0.0
            result["downtime_seconds"] = 0.0

        result = result.fillna(0)

        # Targets
        if targets:
            result["production_target"] = result["shift"].map(targets).fillna(0)
            result["production_achievement_pct"] = (
                result["production"] / result["production_target"] * 100
            ).where(result["production_target"] > 0, 0)
        else:
            result["production_target"] = 0.0
            result["production_achievement_pct"] = 0.0

        # Quality
        total = result["ok_parts"] + result["nok_parts"]
        result["quality_pct"] = (result["ok_parts"] / total * 100).where(total > 0, 0)

        return add_shift_window(
            result[output_cols].reset_index(drop=True),
            self.shift_definitions,
            data_tz(self.dataframe[self.time_column]),
        )

    @overload
    def highlight_issues(
        self,
        counter_uuid: str | None = ...,
        ok_counter_uuid: str | None = ...,
        nok_counter_uuid: str | None = ...,
        run_state_uuid: str | None = ...,
        *,
        report_df: pd.DataFrame | None = ...,
        thresholds: dict[str, float] | None = ...,
        targets: dict[str, float] | None = ...,
        running_value: str | bool = ...,
        value_column_counter: str = ...,
        value_column_state: str = ...,
        report_date: str | None = ...,
        as_frame: Literal[True] = ...,
    ) -> pd.DataFrame: ...

    @overload
    def highlight_issues(
        self,
        counter_uuid: str | None = ...,
        ok_counter_uuid: str | None = ...,
        nok_counter_uuid: str | None = ...,
        run_state_uuid: str | None = ...,
        *,
        report_df: pd.DataFrame | None = ...,
        thresholds: dict[str, float] | None = ...,
        targets: dict[str, float] | None = ...,
        running_value: str | bool = ...,
        value_column_counter: str = ...,
        value_column_state: str = ...,
        report_date: str | None = ...,
        as_frame: Literal[False],
    ) -> list[dict[str, Any]]: ...

    @keyword_only(1)
    @renamed_kwargs(state_uuid="run_state_uuid")
    def highlight_issues(
        self,
        counter_uuid: str | None = None,
        ok_counter_uuid: str | None = None,
        nok_counter_uuid: str | None = None,
        run_state_uuid: str | None = None,
        *,
        report_df: pd.DataFrame | None = None,
        thresholds: dict[str, float] | None = None,
        targets: dict[str, float] | None = None,
        running_value: str | bool = "Running",
        value_column_counter: str = "value_integer",
        value_column_state: str = "value_string",
        report_date: str | None = None,
        as_frame: bool = True,
    ) -> pd.DataFrame | list[dict[str, Any]]:
        """Identify issues that need attention.

        Can be called in two ways:

        1. From raw signals (provide UUIDs):
           ``highlight_issues('prod', 'ok', 'nok', 'state', thresholds={...})``

        2. From a pre-built report DataFrame:
           ``highlight_issues(report_df=my_report, thresholds={...})``

        Args:
            counter_uuid: UUID of production counter (raw-signal mode).
            ok_counter_uuid: UUID of good parts counter (raw-signal mode).
            nok_counter_uuid: UUID of the NOK part counter (raw-signal mode).
            run_state_uuid: UUID of machine state signal (raw-signal mode).
            report_df: Pre-computed report DataFrame (pipeline mode).
                If provided, UUID arguments are ignored.
            thresholds: Minimum acceptable values for each metric.
                Defaults to production_achievement_pct=95, quality_pct=98, availability_pct=90.
            targets: Per-shift production targets.
            running_value: Value indicating machine is running.
            value_column_counter: Column for counter values.
            value_column_state: Column for state values.
            report_date: Specific date (YYYY-MM-DD).
            as_frame: When True (default), return a DataFrame (see below).
                ``as_frame=False`` returns the list of dicts; it is
                deprecated, emits a ``DeprecationWarning`` and will be removed
                in the next major release.

        Returns:
            With ``as_frame=True`` (default), one row per issue with columns
            (:attr:`ISSUE_COLUMNS`): start, end, duration_seconds (the shift
            window of the report row; NaT / NaN when the report has none,
            e.g. one built by :meth:`from_shift_data`), date, shift, metric,
            value, threshold, severity, severity_level (warn / critical).

            With ``as_frame=False`` (deprecated), a list of dicts with keys
            shift, metric, value, threshold, severity.

            severity is 'warning' (within 5% of threshold) or 'critical'.

        Raises:
            ValueError: If ``report_df`` is None and any of the four signal
                UUIDs is missing, so no report can be built.
        """
        if not as_frame:
            warn_dict_return("ShiftHandoverEvents.highlight_issues", kind="list")
        thresholds = thresholds or {
            "production_achievement_pct": 95.0,
            "quality_pct": 98.0,
            "availability_pct": 90.0,
        }

        if report_df is not None:
            report = report_df
        elif counter_uuid and ok_counter_uuid and nok_counter_uuid and run_state_uuid:
            report = self.generate_report(
                counter_uuid,
                ok_counter_uuid,
                nok_counter_uuid,
                run_state_uuid,
                targets=targets,
                running_value=running_value,
                value_column_counter=value_column_counter,
                value_column_state=value_column_state,
                report_date=report_date,
            )
        else:
            missing = [
                name
                for name, val in (
                    ("counter_uuid", counter_uuid),
                    ("ok_counter_uuid", ok_counter_uuid),
                    ("nok_counter_uuid", nok_counter_uuid),
                    ("run_state_uuid", run_state_uuid),
                )
                if not val
            ]
            raise ValueError(
                "highlight_issues() needs either report_df or all four signal "
                f"UUIDs to build the report; missing: {', '.join(missing)}"
            )

        issues: list[dict[str, Any]] = []
        rows: list[dict[str, Any]] = []
        for _, row in report.iterrows():
            for metric, threshold in thresholds.items():
                if metric not in row:
                    continue
                value = row[metric]
                if value < threshold:
                    severity = "warning" if value >= threshold * 0.95 else "critical"
                    issue = {
                        "shift": row["shift"],
                        "metric": metric,
                        "value": float(value),
                        "threshold": threshold,
                        "severity": severity,
                    }
                    issues.append(issue)
                    rows.append(
                        {
                            "start": row.get("start", pd.NaT),
                            "end": row.get("end", pd.NaT),
                            "duration_seconds": row.get(
                                "duration_seconds", float("nan")
                            ),
                            "date": row.get("date"),
                            **issue,
                        }
                    )

        if not as_frame:
            return issues
        return pd.DataFrame(rows, columns=self.ISSUE_COLUMNS)


#: Deprecated name, removed in the next major release.
if TYPE_CHECKING:

    @deprecated("ShiftHandoverReport is deprecated, use ShiftHandoverEvents")
    class ShiftHandoverReport(ShiftHandoverEvents): ...

else:
    ShiftHandoverReport = deprecated_class(ShiftHandoverEvents, "ShiftHandoverReport")
