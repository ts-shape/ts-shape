"""Per-facility local time: production days, shifts and comparable periods.

Timeseries arrives in UTC. Plants report in local time. Between the two sit the
three failure modes that make cross-facility reports wrong in ways nobody
notices until an auditor asks:

* **A "day" is not a day.** Grouping UTC timestamps by calendar date books
  Shanghai's evening shift on the following day and cuts Detroit's night shift
  in half.
* **DST.** Two days a year a local day is 23 or 25 hours long. A fixed offset
  quietly shifts a whole shift across the boundary.
* **Shift patterns differ.** Sites do not agree on when a day starts, so
  "yesterday's output" needs each site's own shift definition.

:class:`FacilityCalendar` resolves all three from the
:class:`~ts_shape.reporting.facility.FacilityRegistry`: it converts to each
site's local wall clock, assigns that site's shifts and production day, and
maps local reporting periods back to auditable UTC windows.

Example::

    from ts_shape.reporting import FacilityCalendar

    cal = FacilityCalendar(registry)
    local = cal.localize(df)                       # + local_time, utc_offset_hours
    shifted = cal.assign_shifts(df)                # + shift, production_date
    daily = cal.local_periods(df, freq="D")        # + period, period_start_utc, ...
    grid = cal.period_grid("2026-01-01", "2026-01-07")   # expected calendar
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import pandas as pd

from ts_shape.reporting.facility import FACILITY_COLUMN, FacilityRegistry
from ts_shape.utils.shifts import assign_shift, production_dates

__all__ = ["FacilityCalendar"]

logger = logging.getLogger(__name__)

#: pandas offset aliases that ``Series.dt.to_period`` does not accept, mapped
#: to the period alias meaning the same thing.
_PERIOD_ALIASES: dict[str, str] = {
    "MS": "M",
    "ME": "M",
    "QS": "Q",
    "QE": "Q",
    "YS": "Y",
    "YE": "Y",
    "AS": "Y",
    "A": "Y",
}


def _period_alias(freq: str) -> str:
    """Translate a date-range offset alias to a period alias."""
    return _PERIOD_ALIASES.get(freq.upper(), freq)


class FacilityCalendar:
    """Local-time calendar operations driven by facility master data.

    Args:
        registry: The :class:`~ts_shape.reporting.facility.FacilityRegistry`
            supplying timezones, shift patterns and working days.
        assume_timezone: Timezone attributed to tz-naive input timestamps.
            ts-shape loaders return UTC, which is the default.
        facility_column: Default column holding the site id.
        time_column: Default column holding the timestamp.
    """

    def __init__(
        self,
        registry: FacilityRegistry,
        *,
        assume_timezone: str = "UTC",
        facility_column: str = FACILITY_COLUMN,
        time_column: str = "systime",
    ) -> None:
        if not isinstance(registry, FacilityRegistry):
            raise TypeError(f"Expected FacilityRegistry, got {type(registry).__name__}")
        self.registry = registry
        self.assume_timezone = assume_timezone
        self.facility_column = facility_column
        self.time_column = time_column

    def __repr__(self) -> str:
        return (
            f"FacilityCalendar(facilities={len(self.registry)}, "
            f"assume_timezone={self.assume_timezone!r})"
        )

    # -- internals -------------------------------------------------------

    def _prepare(
        self,
        dataframe: pd.DataFrame,
        facility_column: str | None,
        time_column: str | None,
    ) -> tuple[pd.DataFrame, str, str]:
        facility_col = facility_column or self.facility_column
        time_col = time_column or self.time_column
        for column in (facility_col, time_col):
            if column not in dataframe.columns:
                raise ValueError(
                    f"Column {column!r} not found. "
                    f"Available: {list(dataframe.columns)}"
                )
        work = dataframe.copy()
        timestamps = pd.to_datetime(work[time_col], errors="coerce", utc=False)
        if timestamps.dt.tz is None:
            timestamps = timestamps.dt.tz_localize(
                self.assume_timezone, ambiguous="NaT", nonexistent="shift_forward"
            )
        work[time_col] = timestamps
        return work, facility_col, time_col

    def _unknown_facilities(self, values: pd.Series) -> list[str]:
        seen = {str(v) for v in values.dropna().unique()}
        return sorted(seen - set(self.registry.ids))

    # -- local time ------------------------------------------------------

    def localize(
        self,
        dataframe: pd.DataFrame,
        *,
        facility_column: str | None = None,
        time_column: str | None = None,
        output_column: str = "local_time",
        strict: bool = False,
    ) -> pd.DataFrame:
        """Add each row's timestamp in its facility's local wall-clock time.

        Args:
            dataframe: Frame with a facility key and a timestamp column.
            facility_column: Overrides the instance default.
            time_column: Overrides the instance default.
            output_column: Name for the tz-naive local timestamp column.
            strict: Raise when a facility is absent from the registry. When
                ``False`` (default) those rows keep the assumed timezone and a
                warning is logged, so one unmapped site cannot fail a
                fleet-wide report.

        Returns:
            A copy of ``dataframe`` with ``output_column`` (tz-naive local
            time), ``timezone`` and ``utc_offset_hours`` added. The original
            timestamp column is left untouched and tz-aware.

        Raises:
            ValueError: If a column is missing, or ``strict`` and a facility is
                unknown.
        """
        work, facility_col, time_col = self._prepare(
            dataframe, facility_column, time_column
        )
        if work.empty:
            work[output_column] = pd.Series(dtype="datetime64[ns]")
            work["timezone"] = pd.Series(dtype="object")
            work["utc_offset_hours"] = pd.Series(dtype="float64")
            return work

        unknown = self._unknown_facilities(work[facility_col])
        if unknown:
            message = (
                f"Facilities {unknown} are not in the registry; "
                f"treating their timestamps as {self.assume_timezone}."
            )
            if strict:
                raise ValueError(message)
            logger.warning(message)

        local = pd.Series(pd.NaT, index=work.index, dtype="datetime64[ns]")
        tz_names = pd.Series("", index=work.index, dtype="object")
        offsets = pd.Series(float("nan"), index=work.index, dtype="float64")

        for site_id, group in work.groupby(work[facility_col].astype(str), sort=False):
            timezone = (
                self.registry[site_id].timezone
                if site_id in self.registry
                else self.assume_timezone
            )
            converted = group[time_col].dt.tz_convert(timezone)
            local.loc[group.index] = converted.dt.tz_localize(None)
            tz_names.loc[group.index] = timezone
            offsets.loc[group.index] = (
                converted.map(
                    lambda ts: (
                        float("nan")
                        if pd.isna(ts) or ts.utcoffset() is None
                        else ts.utcoffset().total_seconds() / 3600.0
                    )
                )
            ).astype("float64")

        work[output_column] = local
        work["timezone"] = tz_names
        work["utc_offset_hours"] = offsets
        return work

    def assign_shifts(
        self,
        dataframe: pd.DataFrame,
        *,
        facility_column: str | None = None,
        time_column: str | None = None,
        shift_column: str = "shift",
        date_column: str = "production_date",
        production_day: bool = True,
    ) -> pd.DataFrame:
        """Assign each row its facility's local shift and production day.

        Each site's own shift pattern is applied to its own local time, so a
        night shift is one contiguous group per site even when the sites sit on
        different continents.

        Args:
            dataframe: Frame with a facility key and a timestamp column.
            facility_column: Overrides the instance default.
            time_column: Overrides the instance default.
            shift_column: Output column for the shift name.
            date_column: Output column for the production date.
            production_day: When ``True`` (default) the post-midnight portion
                of an overnight shift is attributed to the shift's start date
                (see :func:`ts_shape.utils.shifts.production_dates`). Set
                ``False`` for a raw local calendar date.

        Returns:
            A copy of ``dataframe`` with ``local_time``, ``timezone``,
            ``utc_offset_hours``, ``shift_column`` and ``date_column`` added.
        """
        work = self.localize(
            dataframe, facility_column=facility_column, time_column=time_column
        )
        facility_col = facility_column or self.facility_column
        if work.empty:
            work[shift_column] = pd.Series(dtype="object")
            work[date_column] = pd.Series(dtype="object")
            return work

        shifts = pd.Series("unknown", index=work.index, dtype="object")
        dates = pd.Series(pd.NaT, index=work.index, dtype="object")

        for site_id, group in work.groupby(work[facility_col].astype(str), sort=False):
            definitions = (
                dict(self.registry[site_id].shifts) if site_id in self.registry else {}
            )
            local_times = group["local_time"]
            if not definitions:
                shifts.loc[group.index] = "unknown"
                dates.loc[group.index] = local_times.dt.date
                continue
            assigned = local_times.map(
                lambda ts: "unknown" if pd.isna(ts) else assign_shift(ts, definitions)
            )
            shifts.loc[group.index] = assigned
            if production_day:
                dates.loc[group.index] = production_dates(
                    local_times, assigned, definitions
                )
            else:
                dates.loc[group.index] = local_times.dt.date

        work[shift_column] = shifts
        work[date_column] = dates
        return work

    # -- reporting periods -----------------------------------------------

    def local_periods(
        self,
        dataframe: pd.DataFrame,
        *,
        freq: str = "D",
        facility_column: str | None = None,
        time_column: str | None = None,
        period_column: str = "period",
        with_utc_bounds: bool = True,
    ) -> pd.DataFrame:
        """Label each row with its facility-local reporting period.

        Args:
            dataframe: Frame with a facility key and a timestamp column.
            freq: Period frequency -- ``"D"``, ``"W"`` (or ``"W-MON"``),
                ``"MS"``/``"M"``, ``"QS"``/``"Q"``, ``"YS"``/``"Y"``.
            facility_column: Overrides the instance default.
            time_column: Overrides the instance default.
            period_column: Output column holding the local period start.
            with_utc_bounds: Also emit ``period_start_utc`` and
                ``period_end_utc`` -- the exact UTC window the local period
                covers at that site, DST included. These make a consolidated
                figure reproducible against the raw data.

        Returns:
            A copy of ``dataframe`` with the local-time columns plus the period
            columns added.
        """
        work = self.localize(
            dataframe, facility_column=facility_column, time_column=time_column
        )
        facility_col = facility_column or self.facility_column
        if work.empty:
            work[period_column] = pd.Series(dtype="datetime64[ns]")
            if with_utc_bounds:
                work["period_start_utc"] = pd.Series(dtype="object")
                work["period_end_utc"] = pd.Series(dtype="object")
            return work

        alias = _period_alias(freq)
        periods = work["local_time"].dt.to_period(alias)
        work[period_column] = periods.dt.start_time

        if not with_utc_bounds:
            return work

        starts = pd.Series(pd.NaT, index=work.index, dtype="object")
        ends = pd.Series(pd.NaT, index=work.index, dtype="object")
        local_end = periods.dt.end_time.dt.ceil("s")
        for site_id, group in work.groupby(work[facility_col].astype(str), sort=False):
            timezone = (
                self.registry[site_id].timezone
                if site_id in self.registry
                else self.assume_timezone
            )
            starts.loc[group.index] = _to_utc(
                work.loc[group.index, period_column], timezone
            )
            ends.loc[group.index] = _to_utc(local_end.loc[group.index], timezone)
        work["period_start_utc"] = starts
        work["period_end_utc"] = ends
        return work

    def period_grid(
        self,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        *,
        freq: str = "D",
        facilities: Sequence[str] | None = None,
        working_days_only: bool = False,
    ) -> pd.DataFrame:
        """Build the expected reporting calendar: every facility x period.

        This is the denominator of a coverage check -- what *should* have been
        reported -- and the join target that makes a silent site visible as a
        missing row instead of an absent one.

        Args:
            start: First local period (inclusive), as a date or timestamp.
            end: Last local period (inclusive).
            freq: Period frequency, as in :meth:`local_periods`.
            facilities: Restrict to these site ids. Defaults to the whole
                registry.
            working_days_only: Drop periods that start on a weekday the site
                does not run (daily frequencies only).

        Returns:
            DataFrame with columns ``facility``, ``period``, ``period_start_utc``,
            ``period_end_utc``, ``timezone``, ``is_working_day``.

        Raises:
            KeyError: If a requested facility is not registered.
        """
        columns = [
            FACILITY_COLUMN,
            "period",
            "period_start_utc",
            "period_end_utc",
            "timezone",
            "is_working_day",
        ]
        site_ids = list(facilities) if facilities is not None else self.registry.ids
        if not site_ids:
            return pd.DataFrame(columns=columns)

        alias = _period_alias(freq)
        local_periods = pd.period_range(
            start=pd.Timestamp(start), end=pd.Timestamp(end), freq=alias
        )
        if len(local_periods) == 0:
            return pd.DataFrame(columns=columns)

        starts = local_periods.start_time
        ends = local_periods.end_time.ceil("s")

        frames: list[pd.DataFrame] = []
        for site_id in site_ids:
            facility = self.registry.get(site_id)
            frame = pd.DataFrame(
                {
                    FACILITY_COLUMN: site_id,
                    "period": starts,
                    "period_start_utc": _to_utc(pd.Series(starts), facility.timezone),
                    "period_end_utc": _to_utc(pd.Series(ends), facility.timezone),
                    "timezone": facility.timezone,
                }
            )
            frame["is_working_day"] = frame["period"].dt.weekday.isin(
                facility.working_days
            )
            if working_days_only:
                frame = frame[frame["is_working_day"]]
            frames.append(frame)

        grid = pd.concat(frames, ignore_index=True)
        return (
            grid[columns]
            .sort_values([FACILITY_COLUMN, "period"])
            .reset_index(drop=True)
        )

    def utc_window(
        self,
        facility_id: str,
        local_start: str | pd.Timestamp,
        local_end: str | pd.Timestamp,
    ) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Translate a site-local window into the UTC window to query for.

        Use it to pull exactly one plant's local day, week or shift out of a
        UTC-indexed store::

            start, end = cal.utc_window("CN-SHA", "2026-03-01", "2026-03-02")
            loader.load_data_by_time_range(start, end)

        Args:
            facility_id: Registered site id.
            local_start: Window start in the site's local time (inclusive).
            local_end: Window end in the site's local time (exclusive).

        Returns:
            ``(start_utc, end_utc)`` as tz-aware UTC timestamps.

        Raises:
            KeyError: If the facility is not registered.
        """
        timezone = self.registry.get(facility_id).timezone
        bounds = _to_utc(
            pd.Series([pd.Timestamp(local_start), pd.Timestamp(local_end)]), timezone
        )
        return pd.Timestamp(bounds.iloc[0]), pd.Timestamp(bounds.iloc[1])


def _to_utc(local: pd.Series, timezone: str) -> pd.Series:
    """Localize naive local timestamps to ``timezone`` and convert to UTC.

    Ambiguous wall-clock times (the repeated hour when DST ends) resolve to the
    DST reading; times that do not exist (the skipped hour when DST starts)
    shift forward. Both keep a period boundary monotonic rather than raising in
    the middle of a fleet report.
    """
    naive = pd.to_datetime(local, errors="coerce")
    if getattr(naive.dt, "tz", None) is not None:
        return naive.dt.tz_convert("UTC")
    return naive.dt.tz_localize(
        timezone, ambiguous=True, nonexistent="shift_forward"
    ).dt.tz_convert("UTC")
