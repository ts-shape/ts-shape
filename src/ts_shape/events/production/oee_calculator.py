"""Overall Equipment Effectiveness (OEE) calculator.

OEE = Availability x Performance x Quality

Industry-standard metric for manufacturing productivity:
- Availability: actual run time / planned production time
- Performance: actual throughput / ideal throughput
- Quality: good parts / total parts
"""

import logging
import warnings
from math import nan
from typing import TYPE_CHECKING

import pandas as pd

from ts_shape.errors import DataQualityWarning
from ts_shape.events._output import (
    COL_DURATION_S,
    COL_END,
    COL_START,
    empty_event_df,
)
from ts_shape.events.production._counter_groups import counter_totals
from ts_shape.utils.base import Base
from ts_shape.utils.coerce import bool_or_false
from ts_shape.utils.deprecation import (
    caller_stacklevel,
    deprecated_class,
    deprecated_method,
    keyword_only,
    renamed_kwargs,
)
from ts_shape.utils.time_accounting import (
    EXCLUDED,
    counter_by_period,
    grouper_periods,
    reading_intervals,
    time_by_period,
)

if TYPE_CHECKING:
    from typing_extensions import deprecated

logger = logging.getLogger(__name__)

# Rounding slack before a percentage counts as above 100 %.
_PCT_TOLERANCE = 1e-6


def _daily_summary(
    rows: list[dict], extra_cols: list[str], tz: object | None = None
) -> pd.DataFrame:
    """Convert a list of {date, ...} rows into the canonical summary schema.

    Each ``date`` becomes ``start`` (local midnight) + ``end`` (next local
    midnight) + ``duration_seconds`` (the real day length: 23 h / 25 h on DST
    days). With ``tz`` the bounds are tz-aware in the data's timezone. The
    ``date`` key is dropped.
    """
    if not rows:
        return empty_event_df("summary", extra_cols=extra_cols)
    df = pd.DataFrame(rows)
    start = pd.to_datetime(df["date"])
    end = start + pd.Timedelta(days=1)
    if tz is not None:
        start = start.dt.tz_localize(tz, nonexistent="shift_forward", ambiguous=True)
        end = end.dt.tz_localize(tz, nonexistent="shift_forward", ambiguous=True)
    df[COL_START] = start
    df[COL_END] = end
    df[COL_DURATION_S] = (end - start).dt.total_seconds()
    df = df.drop(columns=["date"])
    leading = [COL_START, COL_END, COL_DURATION_S]
    return df[leading + [c for c in df.columns if c not in leading]]


def _day_periods(times: pd.Series) -> pd.DataFrame:
    """Calendar days (local midnights) covering ``times``, keyed by ``date``."""
    days = grouper_periods(times.iloc[0], times.iloc[-1], "1D")
    days["date"] = [ts.date() for ts in days["period"]]
    return days


def _warn_above_100(metric: str, rows: list[dict]) -> None:
    """Warn about days whose ``<metric>_pct`` exceeds 100 %.

    A ratio above 100 % means the inputs disagree (an ideal cycle time that is
    too long, run time missing from the state signal, a planned time shorter
    than the run time). The value is reported as computed, not clipped.
    """
    key = f"{metric}_pct"
    over = [r["date"] for r in rows if r[key] > 100.0 + _PCT_TOLERANCE]
    if over:
        days = ", ".join(str(d) for d in over)
        warnings.warn(
            f"OEEEvents: {key} above 100 % in the day window(s) starting {days}; "
            "the inputs disagree (check the ideal cycle time, the planned time "
            "and that the state and counter signals cover the same time). The "
            "value is reported unclipped.",
            DataQualityWarning,
            stacklevel=caller_stacklevel(),
        )


class OEEEvents(Base):
    """Calculate Overall Equipment Effectiveness from timeseries signals.

    Combines availability (from run/idle state), performance (from part counters
    and ideal cycle time), and quality (from total/reject counters) into a single
    OEE metric per day. Each daily row uses the canonical summary schema:
    ``start`` (midnight) / ``end`` (next midnight) / ``duration_seconds`` (the
    real day length: 86400, or 23 h / 25 h on DST days with tz-aware data).

    Example:
        >>> import pandas as pd, ts_shape
        >>> kw = dict(n_points=60, freq="1min")
        >>> run = ts_shape.make_timeseries(["run"], value_column="value_bool", **kw)
        >>> parts = ts_shape.make_timeseries(["parts"], noise=0, drift=59,
        ...                                  value_column="value_integer", **kw)
        >>> daily = ts_shape.OEEEvents(pd.concat([run, parts])).oee(
        ...     "run", "parts", ideal_cycle_time=30.0)
        >>> daily[["availability_pct", "performance_pct", "quality_pct", "oee_pct"]].round(1)
           availability_pct  performance_pct  quality_pct  oee_pct
        0              54.2             92.2        100.0     50.0
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        time_column: str = "systime",
    ) -> None:
        """Initialize OEE calculator.

        Args:
            dataframe: Input DataFrame with timeseries data.
            time_column: Name of timestamp column (default: 'systime').
        """
        super().__init__(dataframe, time_column=time_column)
        self.time_column = time_column
        times = self.dataframe.get(time_column)
        self._tz = (
            getattr(times.dt, "tz", None)
            if times is not None and pd.api.types.is_datetime64_any_dtype(times)
            else None
        )

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    if TYPE_CHECKING:

        @deprecated(
            "OEEEvents.calculate_availability() is deprecated, use .availability()"
        )
        def calculate_availability(
            self,
            run_state_uuid: str,
            planned_time_hours: float | None = None,
            *,
            value_column: str = "value_bool",
            max_gap: str | None = None,
        ) -> pd.DataFrame: ...

    else:
        calculate_availability = deprecated_method("availability")

    @keyword_only(2)
    def availability(
        self,
        run_state_uuid: str,
        planned_time_hours: float | None = None,
        *,
        value_column: str = "value_bool",
        max_gap: str | None = None,
    ) -> pd.DataFrame:
        """Calculate availability percentage from run/idle intervals.

        Availability = run_time / planned_time (ISO 22400-2: APT / PBT).
        Each reading's state holds until the next reading; the interval is
        split at midnight, so a run across midnight counts in both days. When
        *planned_time_hours* is ``None`` the planned time is the time the
        state signal covers that day. A day the state signal does not cover
        is not reported. A value above 100 % (run time longer than a fixed
        planned time) is kept and reported with a
        :class:`~ts_shape.errors.DataQualityWarning` naming the day.

        Args:
            run_state_uuid: UUID of the boolean run-state signal (True = running).
            planned_time_hours: Fixed planned production hours per day.  If None,
                the time the state signal covers each day is used.
            value_column: Column holding the boolean state.
            max_gap: Optional cap (e.g. "15min"). A sample's duration is the gap
                to the next reading; a gap longer than this is treated as a
                logging outage ("no data") and excluded from both run time and
                planned time. Without it, an overnight or logger gap while the
                last state was ``running`` is booked as run time, inflating
                availability. Only meaningful when ``planned_time_hours`` is
                None (the planned time is derived from the data). ``None``
                (default) applies no cap: every interval between readings,
                however long, counts toward the preceding state.

        Returns:
            Summary-shape DataFrame with columns:
            - start, end, duration_seconds (canonical day boundaries; real day length)
            - run_seconds
            - planned_seconds
            - availability_pct
            - excluded_seconds: time removed by ``max_gap`` as "no data"
                (split at midnight like the rest; 0 without ``max_gap``)
        """
        avail_extra = [
            "run_seconds",
            "planned_seconds",
            "availability_pct",
            "excluded_seconds",
        ]
        state_data = (
            self.dataframe[self.dataframe["uuid"] == run_state_uuid]
            .copy()
            .sort_values(self.time_column)
        )

        if state_data.empty:
            return _daily_summary([], avail_extra, tz=self._tz)

        state_data[self.time_column] = pd.to_datetime(state_data[self.time_column])
        times = state_data[self.time_column]
        state = bool_or_false(state_data[value_column])

        # Each reading's state holds until the next reading (gaps longer than
        # max_gap are "no data"); the intervals are split at midnight, so a
        # run across midnight counts in both days (ISO 22400-2 time model).
        intervals = reading_intervals(
            times, state.map({True: "run", False: "idle"}), max_gap=max_gap
        )
        seconds = time_by_period(
            intervals,
            _day_periods(times),
            keys=["date"],
            states=["run", "idle", EXCLUDED],
        )
        seconds = seconds[(seconds["run"] + seconds["idle"]) > 0]

        results = []
        for date, run_seconds, idle_seconds, excluded in seconds[
            ["date", "run", "idle", EXCLUDED]
        ].itertuples(index=False, name=None):
            if planned_time_hours is not None:
                planned_seconds = planned_time_hours * 3600.0
            else:
                planned_seconds = run_seconds + idle_seconds

            availability = (
                (run_seconds / planned_seconds * 100.0) if planned_seconds > 0 else 0.0
            )

            results.append(
                {
                    "date": date,
                    "run_seconds": float(run_seconds),
                    "planned_seconds": float(planned_seconds),
                    "availability_pct": availability,
                    "excluded_seconds": float(excluded),
                }
            )

        _warn_above_100("availability", results)
        return _daily_summary(results, avail_extra, tz=self._tz)

    # ------------------------------------------------------------------
    # Performance
    # ------------------------------------------------------------------

    if TYPE_CHECKING:

        @deprecated(
            "OEEEvents.calculate_performance() is deprecated, use .performance()"
        )
        def calculate_performance(
            self,
            counter_uuid: str,
            ideal_cycle_time: float,
            run_state_uuid: str | None = None,
            *,
            value_column: str = "value_integer",
            value_column_state: str = "value_bool",
            handle_resets: bool = True,
            max_gap: str | None = None,
        ) -> pd.DataFrame: ...

    else:
        calculate_performance = deprecated_method("performance")

    @keyword_only(3)
    @renamed_kwargs(run_value_column="value_column_state")
    def performance(
        self,
        counter_uuid: str,
        ideal_cycle_time: float,
        run_state_uuid: str | None = None,
        *,
        value_column: str = "value_integer",
        value_column_state: str = "value_bool",
        handle_resets: bool = True,
        max_gap: str | None = None,
    ) -> pd.DataFrame:
        """Calculate performance percentage (actual vs ideal throughput).

        Performance = (actual_parts * ideal_cycle_time) / run_time
        (ISO 22400-2: PQ x pICT / APT). If *run_state_uuid* is None, the time
        the counter readings cover each day is used as run time. With a
        state signal, a day it does not cover has unknown run time:
        ``run_seconds``, ``ideal_parts`` and ``performance_pct`` are NaN.
        A value above 100 % means the inputs disagree; it is kept (not
        clipped) and reported with a
        :class:`~ts_shape.errors.DataQualityWarning` naming the day.

        Args:
            counter_uuid: UUID of the monotonic part counter.
            ideal_cycle_time: Ideal cycle time in seconds per part.
            run_state_uuid: Optional UUID for run-state to compute actual run time.
            value_column: Column holding counter values.
            value_column_state: Column holding boolean run state.
            handle_resets: When True (default), treat a decreasing counter as a reset and
                count the production after the reset instead of losing it, so
                ``actual_parts`` is the reset-aware total. The increments are
                computed once over the whole signal; the increment between
                two readings is spread over the time between them and split
                at midnight in proportion to time (ISO 22400-2: a part
                belongs to the period it was produced in), so the days add
                up to the signal total and ``actual_parts`` can be
                fractional. The result gains a
                ``resets`` column with the number of resets observed each day.
                When False, ``actual_parts`` is simply
                ``max(0, last - first)`` and a day containing a reset reports
                too little.
            max_gap: Optional cap forwarded to :meth:`availability`
                when ``run_state_uuid`` is given, so a logging gap does not
                inflate the run time used as the performance denominator.
                ``None`` (default) applies no cap.

        Returns:
            Summary-shape DataFrame with columns:
            - start, end, duration_seconds
            - actual_parts
            - ideal_parts
            - run_seconds
            - performance_pct
            - resets (only when ``handle_resets=True``)
        """
        perf_extra = ["actual_parts", "ideal_parts", "run_seconds", "performance_pct"]
        if handle_resets:
            perf_extra.append("resets")
        counter_data = (
            self.dataframe[self.dataframe["uuid"] == counter_uuid]
            .copy()
            .sort_values(self.time_column)
        )

        if counter_data.empty:
            return _daily_summary([], perf_extra, tz=self._tz)

        counter_data[self.time_column] = pd.to_datetime(counter_data[self.time_column])
        counter_data["date"] = counter_data[self.time_column].dt.date

        # Run time per day from the state signal. A day the state signal does
        # not cover has unknown run time (NaN), not the counter's time span.
        run_per_day: dict = {}
        if run_state_uuid is not None:
            avail_df = self.availability(
                run_state_uuid, value_column=value_column_state, max_gap=max_gap
            )
            for start, run_s in zip(
                avail_df[COL_START], avail_df["run_seconds"], strict=True
            ):
                run_per_day[start.date()] = float(run_s)

        parts = self._counter_by_day(counter_data, value_column, handle_resets)
        if run_state_uuid is None:
            # No state signal: the time the counter readings cover each day.
            covered = time_by_period(
                reading_intervals(
                    counter_data[self.time_column],
                    pd.Series("covered", index=counter_data.index),
                ),
                _day_periods(counter_data[self.time_column]),
                keys=["date"],
                states=["covered"],
            )
            run_per_day = dict(zip(covered["date"], covered["covered"], strict=True))

        results = []
        for date, actual_parts, n_resets in parts[
            ["date", "quantity", "resets"]
        ].itertuples(index=False, name=None):
            run_seconds = run_per_day.get(date, 0.0 if run_state_uuid is None else nan)

            if pd.isna(run_seconds):
                ideal_parts = nan
                performance = nan
            elif run_seconds > 0 and ideal_cycle_time > 0:
                ideal_parts = run_seconds / ideal_cycle_time
                performance = (actual_parts / ideal_parts) * 100.0
            else:
                ideal_parts = 0.0
                performance = 0.0

            row = {
                "date": date,
                "actual_parts": float(actual_parts),
                "ideal_parts": ideal_parts,
                "run_seconds": run_seconds,
                "performance_pct": performance,
            }
            if handle_resets:
                row["resets"] = int(n_resets)
            results.append(row)

        _warn_above_100("performance", results)
        return _daily_summary(results, perf_extra, tz=self._tz)

    def _counter_by_day(
        self, data: pd.DataFrame, value_column: str, handle_resets: bool
    ) -> pd.DataFrame:
        """Counter quantity and resets per calendar day.

        With ``handle_resets`` the reset-aware increment between two readings
        is spread over the time between them, split at midnight in proportion
        to time (ISO 22400-2: the quantity belongs to the period it was
        produced in). Without it each day reports ``max(0, last - first)`` of
        its own readings. Columns: ``date``, ``quantity``, ``resets``.
        """
        if not handle_resets:
            return counter_totals(data, value_column, ["date"], handle_resets=False)
        times = data[self.time_column]
        return counter_by_period(
            times,
            data[value_column],
            _day_periods(times),
            data[["date"]],
            keys=["date"],
        )

    # ------------------------------------------------------------------
    # Quality
    # ------------------------------------------------------------------

    if TYPE_CHECKING:

        @deprecated("OEEEvents.calculate_quality() is deprecated, use .quality()")
        def calculate_quality(
            self,
            total_uuid: str,
            reject_uuid: str,
            *,
            value_column: str = "value_integer",
            handle_resets: bool = True,
        ) -> pd.DataFrame: ...

    else:
        calculate_quality = deprecated_method("quality")

    def quality(
        self,
        total_uuid: str,
        reject_uuid: str,
        *,
        value_column: str = "value_integer",
        handle_resets: bool = True,
    ) -> pd.DataFrame:
        """Calculate quality percentage (good parts / total parts).

        Args:
            total_uuid: UUID of the total-parts counter.
            reject_uuid: UUID of the reject-parts counter.
            value_column: Column holding counter values.
            handle_resets: When True (default), treat a decreasing counter as a reset and
                count the production after the reset for both the total and the
                reject counters, so a day containing a reset is not
                under-reported. Increments are spread over the time between
                two readings and split at midnight in proportion to time (see
                :meth:`performance`), so the days add up to the signal
                totals and the part counts can be fractional. The result gains a ``resets`` column with the
                combined number of resets (total + reject) observed each day.
                When False, both quantities are ``max(0, last - first)``.

        Returns:
            Summary-shape DataFrame with columns:
            - start, end, duration_seconds
            - total_parts
            - reject_parts
            - good_parts
            - quality_pct
            - resets (only when ``handle_resets=True``)
        """
        qual_extra = ["total_parts", "reject_parts", "good_parts", "quality_pct"]
        if handle_resets:
            qual_extra.append("resets")
        total_data = (
            self.dataframe[self.dataframe["uuid"] == total_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        reject_data = (
            self.dataframe[self.dataframe["uuid"] == reject_uuid]
            .copy()
            .sort_values(self.time_column)
        )

        if total_data.empty:
            return _daily_summary([], qual_extra, tz=self._tz)

        total_data[self.time_column] = pd.to_datetime(total_data[self.time_column])
        total_data["date"] = total_data[self.time_column].dt.date

        reject_by_day = {}
        reject_resets_by_day = {}
        if not reject_data.empty:
            reject_data[self.time_column] = pd.to_datetime(
                reject_data[self.time_column]
            )
            reject_data["date"] = reject_data[self.time_column].dt.date
            rejects = self._counter_by_day(reject_data, value_column, handle_resets)
            for date, qty, n_resets in rejects[
                ["date", "quantity", "resets"]
            ].itertuples(index=False):
                reject_by_day[date] = float(qty)
                if handle_resets:
                    reject_resets_by_day[date] = int(n_resets)

        totals = self._counter_by_day(total_data, value_column, handle_resets)
        results = []
        for date, total_parts, n_resets in totals[
            ["date", "quantity", "resets"]
        ].itertuples(index=False):
            reject_parts = reject_by_day.get(date, 0.0)
            good_parts = max(0.0, total_parts - reject_parts)
            quality = (good_parts / total_parts * 100.0) if total_parts > 0 else 0.0

            row = {
                "date": date,
                "total_parts": float(total_parts),
                "reject_parts": float(reject_parts),
                "good_parts": float(good_parts),
                "quality_pct": min(quality, 100.0),
            }
            if handle_resets:
                row["resets"] = int(n_resets) + int(reject_resets_by_day.get(date, 0))
            results.append(row)

        return _daily_summary(results, qual_extra, tz=self._tz)

    # ------------------------------------------------------------------
    # Combined OEE
    # ------------------------------------------------------------------

    if TYPE_CHECKING:

        @deprecated("OEEEvents.calculate_oee() is deprecated, use .oee()")
        def calculate_oee(
            self,
            run_state_uuid: str,
            counter_uuid: str,
            ideal_cycle_time: float,
            total_uuid: str | None = None,
            reject_uuid: str | None = None,
            *,
            planned_time_hours: float | None = None,
            handle_resets: bool = True,
            max_gap: str | None = None,
        ) -> pd.DataFrame: ...

    else:
        calculate_oee = deprecated_method("oee")

    @keyword_only(4)
    def oee(
        self,
        run_state_uuid: str,
        counter_uuid: str,
        ideal_cycle_time: float,
        total_uuid: str | None = None,
        reject_uuid: str | None = None,
        *,
        planned_time_hours: float | None = None,
        handle_resets: bool = True,
        max_gap: str | None = None,
    ) -> pd.DataFrame:
        """Calculate daily OEE = Availability * Performance * Quality.

        When *total_uuid* / *reject_uuid* are not provided, quality is
        assumed to be 100 %.

        Args:
            run_state_uuid: UUID of the boolean run-state signal.
            counter_uuid: UUID of the monotonic part counter.
            ideal_cycle_time: Ideal cycle time in seconds per part.
            total_uuid: Optional UUID of total-parts counter (for quality).
            reject_uuid: Optional UUID of reject-parts counter (for quality).
            planned_time_hours: Fixed planned production hours per day.
            handle_resets: When True (default), the part / total / reject counters are
                treated as reset-aware (see :meth:`performance` and
                :meth:`quality`), so a day where a counter resets does
                not silently drag OEE toward zero. Forwarded to the performance
                and quality components. Default True.
            max_gap: Optional cap (e.g. "15min") forwarded to the availability
                and performance components so a logging gap is not booked as run
                time (which would inflate availability and understate
                performance). ``None`` (default) applies no cap: every
                interval between readings, however long, counts toward the
                preceding state.

        Returns:
            Summary-shape DataFrame with columns:
            - start, end, duration_seconds
            - availability_pct
            - performance_pct
            - quality_pct
            - oee_pct

            followed by the basis each percentage was computed from, taken
            from the components (a day missing from a component reports 0, except
            that a day the state signal does not cover has NaN
            ``availability_pct``, ``performance_pct``, ``oee_pct``,
            ``run_seconds``, ``planned_seconds`` and ``ideal_parts``: its run
            time is unknown):

            - run_seconds, planned_seconds, excluded_seconds: from
                :meth:`availability` (``excluded_seconds`` is the time
                ``max_gap`` removed as "no data")
            - actual_parts, ideal_parts: from :meth:`performance`
            - total_parts, reject_parts, good_parts: from :meth:`quality`;
                NaN when ``total_uuid`` / ``reject_uuid`` are not given
                (quality is then assumed to be 100 %)
            - resets (only when ``handle_resets=True``): counter resets of
                the part counter plus the total / reject counters
        """
        oee_extra = [
            "availability_pct",
            "performance_pct",
            "quality_pct",
            "oee_pct",
        ]
        avail_basis = ["run_seconds", "planned_seconds", "excluded_seconds"]
        perf_basis = ["actual_parts", "ideal_parts"]
        qual_basis = ["total_parts", "reject_parts", "good_parts"]
        basis = avail_basis + perf_basis + qual_basis
        if handle_resets:
            basis.append("resets")
        avail_df = self.availability(
            run_state_uuid, planned_time_hours=planned_time_hours, max_gap=max_gap
        )
        perf_df = self.performance(
            counter_uuid,
            ideal_cycle_time,
            run_state_uuid=run_state_uuid,
            handle_resets=handle_resets,
            max_gap=max_gap,
        )

        if total_uuid is not None and reject_uuid is not None:
            qual_df = self.quality(total_uuid, reject_uuid, handle_resets=handle_resets)
        else:
            qual_df = None

        if avail_df.empty and perf_df.empty:
            return empty_event_df("summary", extra_cols=oee_extra + basis)

        # Merge on canonical day boundaries (start/end/duration_seconds)
        key_cols = [COL_START, COL_END, COL_DURATION_S]
        perf_resets = ["resets"] if handle_resets else []
        merged = avail_df[key_cols + ["availability_pct"] + avail_basis].copy()
        if not perf_df.empty:
            perf_part = perf_df[key_cols + ["performance_pct"] + perf_basis]
            perf_part = perf_part.assign(
                _perf_resets=perf_df["resets"] if perf_resets else 0
            )
            merged = merged.merge(perf_part, on=key_cols, how="outer")
        else:
            merged["performance_pct"] = 0.0
            for col in perf_basis:
                merged[col] = 0.0
            merged["_perf_resets"] = 0

        has_quality = qual_df is not None
        if qual_df is not None and not qual_df.empty:
            qual_part = qual_df[key_cols + ["quality_pct"] + qual_basis]
            qual_part = qual_part.assign(
                _qual_resets=qual_df["resets"] if handle_resets else 0
            )
            merged = merged.merge(qual_part, on=key_cols, how="outer")
        else:
            merged["quality_pct"] = 100.0
            for col in qual_basis:
                merged[col] = 0.0
            merged["_qual_resets"] = 0

        # A day the state signal does not cover has unknown availability and
        # run time (so unknown performance and OEE), not 0 %.
        no_state = merged["availability_pct"].isna()
        merged = merged.fillna(0.0)
        unknown = ["availability_pct", "run_seconds", "planned_seconds"]
        unknown += ["performance_pct", "ideal_parts"]
        merged.loc[no_state, unknown] = float("nan")
        if not has_quality:
            # Quality was not measured: the 100 % is an assumption, not a count.
            for col in qual_basis:
                merged[col] = float("nan")
        if handle_resets:
            merged["resets"] = (merged["_perf_resets"] + merged["_qual_resets"]).astype(
                int
            )

        merged["oee_pct"] = (
            merged["availability_pct"]
            * merged["performance_pct"]
            * merged["quality_pct"]
            / 10000.0  # three percentages multiplied: /100 /100
        )

        return (
            merged[key_cols + oee_extra + basis]
            .sort_values(COL_START)
            .reset_index(drop=True)
        )


#: Deprecated name, removed in the next major release.
if TYPE_CHECKING:

    @deprecated("OEECalculator is deprecated, use OEEEvents")
    class OEECalculator(OEEEvents): ...

else:
    OEECalculator = deprecated_class(OEEEvents, "OEECalculator")
