"""Weekly/monthly summary module.

Roll up daily metrics to weekly/monthly summaries:
- Weekly summary of KPIs
- Monthly totals and averages
- Period-over-period comparison

Two modes of operation:
1. From raw signals: ``weekly_summary(counter_uuid, ...)``
2. From pre-computed daily DataFrames: ``from_daily_data(daily_df)``
"""

import logging
import warnings
from typing import TYPE_CHECKING, Any

import pandas as pd

from ts_shape.errors import DataQualityWarning
from ts_shape.events._windows import add_day_window, data_tz
from ts_shape.events.production._counter_groups import counter_totals
from ts_shape.utils.base import Base
from ts_shape.utils.deprecation import (
    caller_stacklevel,
    deprecated_class,
    keyword_only,
    renamed_kwargs,
)

if TYPE_CHECKING:
    from typing_extensions import deprecated

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Roll-up rules for from_daily_data
# ----------------------------------------------------------------------
#
# A ratio KPI of a period is the ratio of the period's summed bases, never
# the sum of the daily ratios and, when the bases are known, not their plain
# mean either (ISO 22400-2: a KPI over a longer period is computed from the
# summed elements). Each rule lists the basis columns that make the result
# exact; the first rule whose columns are all present wins.

Sums = dict[str, float]


def _ratio(num: float, den: float, *, scale: float = 1.0, zero: float) -> float:
    return scale * num / den if den > 0 else zero


def _nan() -> float:
    return float("nan")


def _good(s: Sums) -> float:
    return s["good_parts"] if "good_parts" in s else s["ok_parts"]


def _total(s: Sums) -> float:
    if "total_parts" in s:
        return s["total_parts"]
    if "ok_parts" in s and "nok_parts" in s:
        return s["ok_parts"] + s["nok_parts"]
    return s["good_parts"] + s["reject_parts"]


def _bad(s: Sums) -> float:
    for c in ("nok_parts", "reject_parts", "scrap_parts"):
        if c in s:
            return s[c]
    return _total(s) - _good(s)


_GOOD_TOTAL = [
    ("good_parts", "total_parts"),
    ("ok_parts", "total_parts"),
    ("ok_parts", "nok_parts"),
    ("good_parts", "reject_parts"),
]
_BAD_TOTAL = [
    ("nok_parts", "total_parts"),
    ("reject_parts", "total_parts"),
    ("ok_parts", "nok_parts"),
    ("good_parts", "reject_parts"),
]


def _quality(s: Sums) -> float:
    return min(_ratio(_good(s), _total(s), scale=100.0, zero=0.0), 100.0)


def _availability(s: Sums) -> float:
    if "run_seconds" in s and "planned_seconds" in s:
        run, planned = s["run_seconds"], s["planned_seconds"]
    elif "uptime_seconds" in s and "downtime_seconds" in s:
        run, planned = s["uptime_seconds"], s["uptime_seconds"] + s["downtime_seconds"]
    elif "uptime_seconds" in s and "total_seconds" in s:
        run, planned = s["uptime_seconds"], s["total_seconds"]
    elif "uptime_minutes" in s and "downtime_minutes" in s:
        run, planned = s["uptime_minutes"], s["uptime_minutes"] + s["downtime_minutes"]
    else:  # ReliabilityEvents: operating / (operating + repair)
        run = s["operating_hours"]
        planned = s["operating_hours"] + s["repair_hours"]
    return min(_ratio(run, planned, scale=100.0, zero=0.0), 100.0)


_AVAILABILITY_BASES = [
    ("run_seconds", "planned_seconds"),
    ("uptime_seconds", "downtime_seconds"),
    ("uptime_seconds", "total_seconds"),
    ("uptime_minutes", "downtime_minutes"),
    ("operating_hours", "repair_hours"),
]


def _performance(s: Sums) -> float:
    return min(
        _ratio(s["actual_parts"], s["ideal_parts"], scale=100.0, zero=0.0), 100.0
    )


# column -> (exact rules [(basis columns, function of the summed bases)],
#            fallback weight columns in order of preference,
#            whether a calendar-time weight (duration_seconds) is acceptable)
_RULES: dict[str, tuple[list[tuple[tuple[str, ...], Any]], list[str], bool]] = {
    "quality_pct": ([(b, _quality) for b in _GOOD_TOTAL], ["total_parts"], False),
    "first_pass_yield_pct": (
        [(b, _quality) for b in _GOOD_TOTAL],
        ["total_parts"],
        False,
    ),
    "nok_rate_pct": (
        [
            (b, lambda s: _ratio(_bad(s), _total(s), scale=100.0, zero=0.0))
            for b in _BAD_TOTAL
        ],
        ["total_parts"],
        False,
    ),
    "availability_pct": (
        [(b, _availability) for b in _AVAILABILITY_BASES],
        ["planned_seconds", "total_seconds"],
        True,
    ),
    "performance_pct": (
        [(("actual_parts", "ideal_parts"), _performance)],
        ["ideal_parts", "run_seconds"],
        False,
    ),
    "mttr_hours": (
        [
            (
                ("repair_hours", "failures"),
                lambda s: _ratio(s["repair_hours"], s["failures"], zero=_nan()),
            )
        ],
        ["failures"],
        False,
    ),
    "mtbf_hours": (
        [
            (
                ("operating_hours", "failures"),
                lambda s: _ratio(s["operating_hours"], s["failures"], zero=_nan()),
            )
        ],
        ["failures"],
        False,
    ),
    "mtbf_iso22400_hours": (
        [
            (
                ("operating_hours", "failures"),
                lambda s: s["operating_hours"] / (s["failures"] + 1),
            )
        ],
        [],
        False,
    ),
    "failure_rate_per_hour": (
        [
            (
                ("operating_hours", "failures"),
                lambda s: _ratio(s["failures"], s["operating_hours"], zero=_nan()),
            )
        ],
        ["operating_hours"],
        False,
    ),
    "enpi": (
        [
            (
                ("energy_kwh", "units_produced"),
                lambda s: _ratio(s["energy_kwh"], s["units_produced"], zero=_nan()),
            )
        ],
        ["units_produced"],
        False,
    ),
    "sec": (
        [
            (
                ("total_energy", "total_output"),
                lambda s: _ratio(s["total_energy"], s["total_output"], zero=_nan()),
            )
        ],
        ["total_output"],
        False,
    ),
    "efficiency": (
        [
            (
                ("units", "energy"),
                lambda s: _ratio(s["units"], s["energy"], zero=0.0),
            )
        ],
        ["energy"],
        False,
    ),
}
_OEE_PARTS = ("availability_pct", "performance_pct", "quality_pct")

#: Name fragments of additive columns (counts, durations, energy).
_SUM_TOKENS = (
    "parts",
    "quantity",
    "production",
    "count",
    "occurrences",
    "failures",
    "events",
    "resets",
    "stops",
    "seconds",
    "minutes",
    "hours",
    "kwh",
    "energy",
    "units",
    "output",
    "waste",
)
#: Words (``_``-separated) or fragments that mark an intensive column -- a
#: ratio or average -- which is never summed.
_RATIO_WORDS = frozenset(
    {
        "pct",
        "percent",
        "ratio",
        "rate",
        "avg",
        "mean",
        "median",
        "per",
        "mttr",
        "mtbf",
        "mttf",
        "enpi",
        "sec",
        "efficiency",
        "yield",
        "oee",
        "factor",
        "score",
        "index",
    }
)
_RATIO_FRAGMENTS = ("cycle_time",)
#: Name fragments of KPI ratios that do not carry a unit suffix.
_WEAK_RATIO_FRAGMENTS = ("availability", "performance", "quality")
_AGGREGATIONS = ("sum", "mean", "min", "max", "first", "last", "drop")


def _classify(col: str) -> str:
    """``"ratio"``, ``"sum"`` or ``"unknown"`` from the column name."""
    low = col.lower()
    if set(low.split("_")) & _RATIO_WORDS or any(f in low for f in _RATIO_FRAGMENTS):
        return "ratio"
    if any(tok in low for tok in _SUM_TOKENS):
        return "sum"
    if any(f in low for f in _WEAK_RATIO_FRAGMENTS):
        return "ratio"
    return "unknown"


def _plan_rollup(
    columns: list[str],
    weights: list[str],
    aggregations: dict[str, str],
) -> tuple[dict[str, tuple[str, Any]], list[str], list[str]]:
    """Aggregation per column: ``{col: (kind, arg)}``, mean-without-basis, dropped.

    ``kind`` is ``"sum"``, ``"exact"`` (``arg`` = function of the summed
    bases), ``"oee"``, ``"wmean"`` (``arg`` = weight column), ``"mean"``,
    ``"min"``, ``"max"``, ``"first"``, ``"last"`` or ``"drop"``.
    """
    present = set(columns) | set(weights)
    plan: dict[str, tuple[str, Any]] = {}
    unbased: list[str] = []
    dropped: list[str] = []

    def weight_for(col: str, prefs: list[str], calendar_ok: bool) -> str | None:
        for w in prefs:
            if w in present and w != col:
                return w
        if calendar_ok and "duration_seconds" in present:
            return "duration_seconds"
        return None

    for col in columns:
        if col in aggregations:
            how = aggregations[col]
            if how not in _AGGREGATIONS:
                raise ValueError(
                    f"aggregations[{col!r}]={how!r}; expected one of {_AGGREGATIONS}"
                )
            plan[col] = (how, None) if how != "drop" else ("drop", None)
            continue
        if col in _RULES:
            exact, prefs, calendar_ok = _RULES[col]
            fn = next(
                (f for basis, f in exact if all(b in present for b in basis)), None
            )
            if fn is not None:
                plan[col] = ("exact", fn)
                continue
            w = weight_for(col, prefs, calendar_ok)
            if w is not None:
                plan[col] = ("wmean", w)
            else:
                plan[col] = ("mean", None)
                unbased.append(col)
            continue
        if col == "oee_pct":
            plan[col] = ("oee", None)  # resolved after the components
            continue
        kind = _classify(col)
        if kind == "ratio":
            w = "duration_seconds" if "duration_seconds" in present else None
            if w is not None:
                plan[col] = ("wmean", w)
            else:
                plan[col] = ("mean", None)
                unbased.append(col)
            continue
        if kind == "sum":
            plan[col] = ("sum", None)
            continue
        plan[col] = ("drop", None)
        dropped.append(col)

    if "oee_pct" in plan and plan["oee_pct"][0] == "oee":
        if all(plan.get(c, ("", None))[0] == "exact" for c in _OEE_PARTS):
            plan["oee_pct"] = ("oee", None)
        else:
            w = weight_for("oee_pct", ["planned_seconds"], True)
            if w is not None:
                plan["oee_pct"] = ("wmean", w)
            else:
                plan["oee_pct"] = ("mean", None)
                unbased.append("oee_pct")
    return plan, unbased, dropped


def _rollup_group(
    grp: pd.DataFrame,
    plan: dict[str, tuple[str, Any]],
    weights: pd.DataFrame,
) -> dict[str, Any]:
    """One period's aggregated metrics from its daily rows."""
    numeric = grp.select_dtypes(include="number")
    sums: Sums = {c: float(numeric[c].sum()) for c in numeric.columns}
    for c in weights.columns:
        sums.setdefault(c, float(weights[c].sum()))
    row: dict[str, Any] = {}
    for col, (kind, arg) in plan.items():
        series = grp[col]
        if kind == "drop" or kind == "oee":
            continue
        if kind == "sum":
            val = series.sum()
            row[col] = int(val) if pd.api.types.is_integer_dtype(series) else val
        elif kind == "exact":
            row[col] = float(arg(sums))
        elif kind == "wmean":
            w = grp[arg] if arg in grp.columns else weights[arg]
            ok = series.notna() & w.notna() & (w > 0)
            total = float(w[ok].sum())
            row[col] = (
                float((series[ok] * w[ok]).sum() / total)
                if total > 0
                else float(series.mean())
            )
        else:
            row[col] = getattr(series, kind)()
    if plan.get("oee_pct", ("",))[0] == "oee":
        row["oee_pct"] = (
            row["availability_pct"] * row["performance_pct"] * row["quality_pct"]
        ) / 10000.0
    return {c: row[c] for c in plan if c in row}


class PeriodSummaryEvents(Base):
    """Aggregate daily metrics into weekly/monthly summaries.

    Merge keys: [week_start, week_end] for weekly, [year, month] for monthly.

    Two usage patterns:

    **Pattern A — from raw signals:**

        summary = PeriodSummaryEvents(df)
        weekly = summary.weekly_summary(counter_uuid='prod')

    **Pattern B — pipeline (compose from upstream modules):**

        # Build a daily DataFrame from any combination of upstream modules
        quality = QualityEvents(df).daily_quality_summary('ok', 'nok')
        downtime = DowntimeEvents(df).availability_trend('state', window='1D')

        # Merge on date
        daily = quality.merge(downtime.rename(columns={'period': 'date'}), on='date')

        # Roll up to weekly/monthly
        weekly = PeriodSummaryEvents.from_daily_data(daily, window='W')
        monthly = PeriodSummaryEvents.from_daily_data(daily, window='MS')

    Example:
        A part counter that rises by 100 a day:

        >>> import ts_shape
        >>> df = ts_shape.make_timeseries(["counter"], n_points=14, freq="1D",
        ...     start="2025-01-06", baseline=0, noise=0, drift=1300,
        ...     value_column="value_integer")
        >>> weekly = ts_shape.PeriodSummaryEvents(df).weekly_summary("counter")
        >>> weekly[["start", "end", "total_production", "daily_avg"]]
               start        end  total_production   daily_avg
        0 2025-01-06 2025-01-13               600   85.714286
        1 2025-01-13 2025-01-20               700  100.000000
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        time_column: str = "systime",
    ) -> None:
        super().__init__(dataframe, time_column=time_column)
        self.time_column = time_column

    def _daily_counter(
        self,
        uuid: str,
        value_column: str = "value_integer",
        *,
        handle_resets: bool = True,
    ) -> pd.DataFrame:
        """Get daily counter deltas.

        When ``handle_resets`` is True a decreasing counter is treated as a
        reset and the post-reset production is counted, and a ``resets`` column
        with the per-day reset count is added. The increments are computed once
        over the whole signal and booked to the day of the reading that reports
        them, so the days add up to the signal total (the production between
        the last reading of one day and the first of the next counts in the
        next day). When False each day is ``max(0, last - first)`` of its own
        readings.
        """
        data = (
            self.dataframe[self.dataframe["uuid"] == uuid]
            .copy()
            .sort_values(self.time_column)
        )
        if data.empty:
            cols = ["date", "quantity"]
            if handle_resets:
                cols.append("resets")
            return pd.DataFrame(columns=cols)

        data[self.time_column] = pd.to_datetime(data[self.time_column])
        data["date"] = data[self.time_column].dt.date

        # Increments over the whole signal, booked to the day of the reading
        # that reports them, so the days add up to the signal total.
        totals = counter_totals(
            data, value_column, ["date"], handle_resets=handle_resets
        )
        totals["quantity"] = totals["quantity"].astype(int)
        cols = ["date", "quantity", "resets"] if handle_resets else ["date", "quantity"]
        return totals[cols]

    # ------------------------------------------------------------------
    # Pipeline entry-point: roll up any daily DataFrame
    # ------------------------------------------------------------------

    @staticmethod
    @renamed_kwargs(freq="window")
    def from_daily_data(
        daily_df: pd.DataFrame,
        *,
        window: str = "W",
        date_column: str = "date",
        aggregations: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        """Roll up a pre-computed daily DataFrame to weekly or monthly.

        This is the **pipeline-friendly** entry-point.  It accepts any daily
        DataFrame (e.g. merged output of QualityEvents, DowntimeEvents,
        OEEEvents, ReliabilityEvents, the energy EnPI / SEC tables) and
        aggregates its numeric columns by a declared rule per column. A ratio
        KPI of a period is recomputed from the period's summed basis columns
        (ISO 22400-2), never summed and, when its basis is present, not
        averaged either:

        ========================  ==========================================
        column                    period value
        ========================  ==========================================
        counts, durations,        sum (names containing ``parts``,
        energy                    ``quantity``, ``production``, ``count``,
                                  ``failures``, ``resets``, ``seconds``,
                                  ``minutes``, ``hours``, ``kwh``,
                                  ``energy``, ``units``, ``output``, ...)
        ``quality_pct``,          100 x sum(good) / sum(total), good =
        ``first_pass_yield_pct``  ``good_parts`` or ``ok_parts``, total =
                                  ``total_parts`` or ok + nok / good +
                                  reject
        ``nok_rate_pct``          100 x sum(nok) / sum(total)
        ``availability_pct``      100 x sum(run) / sum(planned): run /
                                  planned = ``run_seconds`` /
                                  ``planned_seconds`` (OEE), ``uptime_*`` /
                                  (uptime + downtime) or ``total_seconds``
                                  (downtime), ``operating_hours`` /
                                  (operating + repair) (reliability)
        ``performance_pct``       100 x sum(``actual_parts``) /
                                  sum(``ideal_parts``)
        ``oee_pct``               availability x performance x quality of
                                  the period (each from its summed bases)
        ``mttr_hours``            sum(``repair_hours``) / sum(``failures``)
        ``mtbf_hours``            sum(``operating_hours``) / sum(``failures``)
        ``mtbf_iso22400_hours``   sum(operating) / (sum(failures) + 1)
        ``failure_rate_per_hour`` sum(failures) / sum(operating_hours)
        ``enpi``                  sum(``energy_kwh``) /
                                  sum(``units_produced``) (ISO 50006)
        ``sec``                   sum(``total_energy``) / sum(``total_output``)
        ``efficiency``            sum(``units``) / sum(``energy``)
        ========================  ==========================================

        When a rule's basis columns are missing the ratio is the mean of the
        daily values weighted by the ratio's denominator if that column is
        present (``total_parts`` for quality, ``planned_seconds`` /
        ``total_seconds`` for availability and OEE, ``ideal_parts`` /
        ``run_seconds`` for performance, ``failures`` for MTTR/MTBF,
        ``units_produced`` for EnPI, ...), which is exact. Availability, OEE
        and other ``*_pct`` columns fall back to the mean weighted by the
        daily rows' ``duration_seconds`` (time-weighted). With no weight
        either, the plain mean is used and a
        :class:`~ts_shape.errors.DataQualityWarning` names the columns whose
        basis is missing. Other intensive columns (``avg``, ``mean``,
        ``rate``, ``ratio``, cycle times) are averaged the same way. Numeric
        columns that match no rule are dropped with a ``DataQualityWarning``
        naming them, instead of being summed; pass ``aggregations`` to keep
        them.

        The output always includes ``start``, ``end``, and
        ``days`` columns.

        Args:
            daily_df: DataFrame with a date column and numeric metric columns.
                Expected merge key: [date].
            window: Pandas frequency string. ``'W'`` (pandas' weeks ending
                Sunday, same as ``'W-SUN'``) gives Monday-to-Sunday weeks:
                ``start`` is the Monday, ``end`` the following Sunday
                (inclusive), even when the data covers only part of the week.
                An anchored ``'W-<DAY>'`` gives weeks ending on ``<DAY>``
                (e.g. ``'W-SAT'`` = Sunday-to-Saturday). ``'MS'`` = calendar
                months (``end`` = last day of the month); any other frequency
                is passed to :class:`pandas.Grouper` and ``end`` is the last
                date present in the group.
            date_column: Name of the date column.
            aggregations: Optional per-column override, ``{column: how}``
                with ``how`` one of ``"sum"``, ``"mean"``, ``"min"``,
                ``"max"``, ``"first"``, ``"last"`` or ``"drop"``. It wins over
                the built-in rules.

        Returns:
            DataFrame with start, end, days, and aggregated
            metric columns.

        Example::

            quality = QualityEvents(df).daily_quality_summary('ok', 'nok')
            weekly = PeriodSummaryEvents.from_daily_data(quality, window='W')
            # → [start, end, days, ok_parts, nok_parts,
            #    total_parts, quality_pct, ...]
        """
        if daily_df.empty:
            return pd.DataFrame(columns=["start", "end", "days"])

        # Window bounds of the daily rows are not metrics: never sum them.
        # The day length is kept aside as a time weight.
        weight_cols = [c for c in ("duration_seconds",) if c in daily_df.columns]
        data = daily_df.drop(
            columns=[
                c for c in ("start", "end", "duration_seconds") if c != date_column
            ],
            errors="ignore",
        )
        data[date_column] = pd.to_datetime(data[date_column])
        weights = daily_df[weight_cols].set_axis(data[date_column])
        data = data.set_index(date_column)

        numeric_cols = data.select_dtypes(include="number").columns.tolist()
        plan, unbased, dropped = _plan_rollup(
            numeric_cols, weight_cols, dict(aggregations or {})
        )
        if unbased:
            warnings.warn(
                "PeriodSummaryEvents.from_daily_data(): no basis columns for "
                f"{unbased}; their period value is the plain mean of the daily "
                "values, which is exact only when every day has the same "
                "basis. Include the basis columns (see the docstring) for an "
                "exact roll-up.",
                DataQualityWarning,
                stacklevel=caller_stacklevel(),
            )
        if dropped:
            warnings.warn(
                "PeriodSummaryEvents.from_daily_data(): no aggregation rule for "
                f"numeric column(s) {dropped}; they were dropped rather than "
                "summed. Pass aggregations={column: 'sum' | 'mean' | ...} to "
                "keep them.",
                DataQualityWarning,
                stacklevel=caller_stacklevel(),
            )

        if window.startswith("W"):
            days = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
            anchor = window.split("-", 1)[1].upper() if "-" in window else "SUN"
            if anchor not in days:
                raise ValueError(f"Unknown weekly anchor in window={window!r}")
            # A week ending on <anchor> starts the day after it.
            first_day = days[(days.index(anchor) + 1) % 7]
            grouper = pd.Grouper(freq=f"W-{first_day}", label="left", closed="left")
        else:
            grouper = pd.Grouper(freq=window)

        results = []
        for start, grp in data.groupby(grouper):
            if grp.empty:
                continue

            row = {"start": start.date(), "days": len(grp)}

            if window.startswith("W"):
                row["end"] = (start + pd.Timedelta(days=6)).date()
            elif window == "MS":
                row["end"] = (start + pd.offsets.MonthEnd(0)).date()
            else:
                row["end"] = grp.index.max().date()

            row.update(_rollup_group(grp, plan, weights.loc[grp.index]))
            results.append(row)

        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Raw-signal entry-points
    # ------------------------------------------------------------------

    @keyword_only(2)
    def weekly_summary(
        self,
        counter_uuid: str,
        ok_counter_uuid: str | None = None,
        nok_counter_uuid: str | None = None,
        *,
        value_column: str = "value_integer",
        handle_resets: bool = True,
    ) -> pd.DataFrame:
        """Roll up daily production to weekly summaries from raw signals.

        For pipeline usage with pre-computed DataFrames, use
        :meth:`from_daily_data` instead.

        Args:
            counter_uuid: UUID of production counter.
            ok_counter_uuid: UUID of good parts counter (optional).
            nok_counter_uuid: UUID of the NOK part counter (optional).
            value_column: Column containing counter values.
            handle_resets: When True (default), treat a decreasing counter as a reset and
                count the production after the reset instead of losing it (a
                daily boundary is a common reset point). Increments are booked
                to the day of the reading that reports them, so no production
                is lost between days. The result gains a
                ``resets`` column with the number of resets observed each week.

        Returns:
            DataFrame with columns:
            - start, end, duration_seconds: the week window (Monday midnight to
                the following Monday midnight, tz-aware in the data's tz)
            - week_start, week_end, total_production, daily_avg,
                production_days, [ok_parts, nok_parts, quality_pct]
            - resets: Number of counter resets in the week (only when
                ``handle_resets=True``)
        """
        daily = self._daily_counter(
            counter_uuid, value_column, handle_resets=handle_resets
        )
        if daily.empty:
            cols = [
                "week_start",
                "week_end",
                "total_production",
                "daily_avg",
                "production_days",
            ]
            if ok_counter_uuid:
                cols.extend(["ok_parts", "nok_parts", "quality_pct"])
            if handle_resets:
                cols.append("resets")
            return add_day_window(pd.DataFrame(columns=cols), None)

        daily["date"] = pd.to_datetime(daily["date"])

        # Quality data
        ok_daily = None
        nok_daily = None
        if ok_counter_uuid:
            ok_daily = self._daily_counter(
                ok_counter_uuid, value_column, handle_resets=handle_resets
            )
            if not ok_daily.empty:
                ok_daily["date"] = pd.to_datetime(ok_daily["date"])
        if nok_counter_uuid:
            nok_daily = self._daily_counter(
                nok_counter_uuid, value_column, handle_resets=handle_resets
            )
            if not nok_daily.empty:
                nok_daily["date"] = pd.to_datetime(nok_daily["date"])

        daily = daily.set_index("date")
        results = []

        for week_start, grp in daily.groupby(
            pd.Grouper(freq="W-MON", label="left", closed="left")
        ):
            if grp.empty:
                continue

            total_prod = grp["quantity"].sum()
            days = len(grp)
            daily_avg = total_prod / days if days > 0 else 0

            week_end = week_start + pd.Timedelta(days=6)

            row = {
                "week_start": week_start.date(),
                "week_end": week_end.date(),
                "total_production": int(total_prod),
                "daily_avg": daily_avg,
                "production_days": days,
            }

            if ok_daily is not None and not ok_daily.empty:
                ok_week = ok_daily[
                    (ok_daily["date"] >= week_start) & (ok_daily["date"] <= week_end)
                ]
                row["ok_parts"] = (
                    int(ok_week["quantity"].sum()) if not ok_week.empty else 0
                )

            if nok_daily is not None and not nok_daily.empty:
                nok_week = nok_daily[
                    (nok_daily["date"] >= week_start) & (nok_daily["date"] <= week_end)
                ]
                row["nok_parts"] = (
                    int(nok_week["quantity"].sum()) if not nok_week.empty else 0
                )

            if "ok_parts" in row and "nok_parts" in row:
                total_q = row["ok_parts"] + row["nok_parts"]
                row["quality_pct"] = (
                    row["ok_parts"] / total_q * 100 if total_q > 0 else 0
                )

            if handle_resets:
                row["resets"] = int(grp["resets"].sum())

            results.append(row)

        return add_day_window(
            pd.DataFrame(results),
            data_tz(self.dataframe[self.time_column]),
            date_col="week_start",
            offset=pd.DateOffset(days=7),
        )

    @keyword_only(2)
    def monthly_summary(
        self,
        counter_uuid: str,
        ok_counter_uuid: str | None = None,
        nok_counter_uuid: str | None = None,
        *,
        value_column: str = "value_integer",
        handle_resets: bool = True,
    ) -> pd.DataFrame:
        """Roll up daily production to monthly summaries from raw signals.

        For pipeline usage with pre-computed DataFrames, use
        :meth:`from_daily_data` instead.

        Args:
            counter_uuid: UUID of production counter.
            ok_counter_uuid: UUID of good parts counter (optional).
            nok_counter_uuid: UUID of the NOK part counter (optional).
            value_column: Column containing counter values.
            handle_resets: When True (default), treat a decreasing counter as a reset and
                count the production after the reset instead of losing it.
                Increments are booked to the day of the reading that reports
                them, so no production is lost between days. The
                result gains a ``resets`` column with the number of resets
                observed each month.

        Returns:
            DataFrame with columns:
            - start, end, duration_seconds: the calendar-month window (first
                of the month midnight to first of the next month midnight,
                tz-aware in the data's tz)
            - year, month, total_production, daily_avg,
                production_days, [ok_parts, nok_parts, quality_pct]
            - resets: Number of counter resets in the month (only when
                ``handle_resets=True``)
        """
        daily = self._daily_counter(
            counter_uuid, value_column, handle_resets=handle_resets
        )
        if daily.empty:
            cols = ["year", "month", "total_production", "daily_avg", "production_days"]
            if ok_counter_uuid:
                cols.extend(["ok_parts", "nok_parts", "quality_pct"])
            if handle_resets:
                cols.append("resets")
            return add_day_window(pd.DataFrame(columns=cols), None)

        daily["date"] = pd.to_datetime(daily["date"])

        ok_daily = None
        nok_daily = None
        if ok_counter_uuid:
            ok_daily = self._daily_counter(
                ok_counter_uuid, value_column, handle_resets=handle_resets
            )
            if not ok_daily.empty:
                ok_daily["date"] = pd.to_datetime(ok_daily["date"])
        if nok_counter_uuid:
            nok_daily = self._daily_counter(
                nok_counter_uuid, value_column, handle_resets=handle_resets
            )
            if not nok_daily.empty:
                nok_daily["date"] = pd.to_datetime(nok_daily["date"])

        daily = daily.set_index("date")
        results = []

        for month_start, grp in daily.groupby(pd.Grouper(freq="MS")):
            if grp.empty:
                continue

            total_prod = grp["quantity"].sum()
            days = len(grp)
            daily_avg = total_prod / days if days > 0 else 0

            month_end = month_start + pd.offsets.MonthEnd(0)

            row = {
                "year": month_start.year,
                "month": month_start.month,
                "total_production": int(total_prod),
                "daily_avg": daily_avg,
                "production_days": days,
            }

            if ok_daily is not None and not ok_daily.empty:
                ok_month = ok_daily[
                    (ok_daily["date"] >= month_start) & (ok_daily["date"] <= month_end)
                ]
                row["ok_parts"] = (
                    int(ok_month["quantity"].sum()) if not ok_month.empty else 0
                )

            if nok_daily is not None and not nok_daily.empty:
                nok_month = nok_daily[
                    (nok_daily["date"] >= month_start)
                    & (nok_daily["date"] <= month_end)
                ]
                row["nok_parts"] = (
                    int(nok_month["quantity"].sum()) if not nok_month.empty else 0
                )

            if "ok_parts" in row and "nok_parts" in row:
                total_q = row["ok_parts"] + row["nok_parts"]
                row["quality_pct"] = (
                    row["ok_parts"] / total_q * 100 if total_q > 0 else 0
                )

            if handle_resets:
                row["resets"] = int(grp["resets"].sum())

            results.append(row)

        out = pd.DataFrame(results)
        out["_month_start"] = pd.to_datetime(
            {"year": out["year"], "month": out["month"], "day": 1}
        )
        out = add_day_window(
            out,
            data_tz(self.dataframe[self.time_column]),
            date_col="_month_start",
            offset=pd.DateOffset(months=1),
        )
        return out.drop(columns="_month_start")

    @renamed_kwargs(period1="period_a", period2="period_b")
    def compare_periods(
        self,
        counter_uuid: str,
        period_a: tuple[str, str],
        period_b: tuple[str, str],
        *,
        value_column: str = "value_integer",
        handle_resets: bool = True,
    ) -> pd.DataFrame:
        """Compare production between two time periods.

        Args:
            counter_uuid: UUID of production counter.
            period_a: Tuple of (start_date, end_date) for first period.
            period_b: Tuple of (start_date, end_date) for second period.
            value_column: Column containing counter values.
            handle_resets: When True (default), treat a decreasing counter as a reset and
                count the production after the reset instead of losing it, so the
                compared metrics are not understated for days containing a reset.

        Returns:
            DataFrame with columns:
            - metric, period_a_value, period_b_value, change, change_pct.
              ``period1_value`` / ``period2_value`` are deprecated twins of
              ``period_a_value`` / ``period_b_value`` (each sits right before
              its replacement); they are removed in the next major release.
        """
        daily = self._daily_counter(
            counter_uuid, value_column, handle_resets=handle_resets
        )
        if daily.empty:
            return pd.DataFrame(
                columns=[
                    "metric",
                    "period1_value",
                    "period_a_value",
                    "period2_value",
                    "period_b_value",
                    "change",
                    "change_pct",
                ]
            )

        daily["date"] = pd.to_datetime(daily["date"]).dt.date

        p1_start = pd.to_datetime(period_a[0]).date()
        p1_end = pd.to_datetime(period_a[1]).date()
        p2_start = pd.to_datetime(period_b[0]).date()
        p2_end = pd.to_datetime(period_b[1]).date()

        p1_data = daily[(daily["date"] >= p1_start) & (daily["date"] <= p1_end)]
        p2_data = daily[(daily["date"] >= p2_start) & (daily["date"] <= p2_end)]

        def _metrics(data: pd.DataFrame) -> dict[str, float]:
            if data.empty:
                return {
                    "total_production": 0,
                    "daily_avg": 0.0,
                    "production_days": 0,
                    "min_daily": 0,
                    "max_daily": 0,
                }
            return {
                "total_production": int(data["quantity"].sum()),
                "daily_avg": data["quantity"].mean(),
                "production_days": len(data),
                "min_daily": int(data["quantity"].min()),
                "max_daily": int(data["quantity"].max()),
            }

        m1 = _metrics(p1_data)
        m2 = _metrics(p2_data)

        rows = []
        for metric in m1:
            v1 = m1[metric]
            v2 = m2[metric]
            change = v2 - v1
            change_pct = (change / v1 * 100) if v1 != 0 else 0.0

            rows.append(
                {
                    "metric": metric,
                    "period1_value": v1,
                    "period_a_value": v1,
                    "period2_value": v2,
                    "period_b_value": v2,
                    "change": change,
                    "change_pct": change_pct,
                }
            )

        return pd.DataFrame(rows)


#: Deprecated name, removed in the next major release.
if TYPE_CHECKING:

    @deprecated("PeriodSummary is deprecated, use PeriodSummaryEvents")
    class PeriodSummary(PeriodSummaryEvents): ...

else:
    PeriodSummary = deprecated_class(PeriodSummaryEvents, "PeriodSummary")
