"""Power (kW) to energy (kWh) integration for ts-shape.

Active power is a *rate*; energy is its integral over time::

    E[kWh] = INTEGRAL P[kW] dt

Historians log on change / deadband, so power samples are irregular. That
makes both of the obvious shortcuts wrong:

* ``sum()`` over the raw samples is meaningless -- double the log rate and the
  "energy" doubles. This is what happens when a ``*.Total_Active_Power-kW``
  signal is fed straight into :class:`~ts_shape.events.energy.consumption_analysis.EnergyConsumptionEvents`,
  whose methods assume the values are already energy deltas in kWh.
* ``mean()`` over the raw samples is a *sample-count* average, not a
  *time-weighted* one. A machine that sits at 200 kW for 59 minutes and then
  logs 40 samples around a 900 kW spike in the last minute reads far too high.

:class:`PowerIntegrator` does the integration properly: every sample is
weighted by how long it was valid, segments are split at window boundaries,
and logging outages are capped so a single stale sample cannot bill hours of
phantom energy.

Typical use -- one number per hour from the raw ts-shape schema::

    from ts_shape.transform.calculator.power_integration import PowerIntegrator

    PowerIntegrator.energy_by_window(df, window="1h", max_gap="15min")
    #                  start                 end  duration_seconds    kwh  avg_kw  coverage  sample_count

Or emit ts-shape-schema energy deltas and hand off to the energy pack::

    deltas = PowerIntegrator.to_energy_deltas(df, window="1h", max_gap="15min")
    EnergyConsumptionEvents(deltas).consumption_by_window(meter_uuid, window="1h")

Note:
    If the meter also exposes a cumulative kWh register, prefer it -- a
    counter difference is billing-accurate and needs no integration. Use this
    class when instantaneous power is all you have.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from pandas.tseries.offsets import Tick

from ts_shape.errors import DataQualityWarning
from ts_shape.utils.base import Base
from ts_shape.utils.coerce import to_numeric_checked
from ts_shape.utils.deprecation import caller_stacklevel, renamed_kwargs
from ts_shape.utils.frames import (
    as_datetime,
    localize_like,
    require_columns,
    window_offset,
)
from ts_shape.utils.time_accounting import grouper_periods

logger = logging.getLogger(__name__)

# Multipliers onto kW. Energy is always returned in kWh regardless of input.
_POWER_UNITS: dict[str, float] = {
    "w": 0.001,
    "kw": 1.0,
    "mw": 1000.0,
    "watt": 0.001,
    "kilowatt": 1.0,
    "megawatt": 1000.0,
}

_SEGMENT_COLUMNS = [
    "uuid",
    "window_start",
    "start",
    "end",
    "duration_seconds",
    "kw_start",
    "kw_end",
    "kwh",
]

_WINDOW_COLUMNS = [
    "start",
    "end",
    "duration_seconds",
    "kwh",
    "avg_kw",
    "coverage",
    "sample_count",
]


def _time_dtype(dataframe: pd.DataFrame, time_column: str) -> object:
    """dtype the time columns of the result get (the input's, as datetime64)."""
    if time_column not in dataframe.columns:
        return np.dtype("datetime64[ns]")
    values = dataframe[time_column]
    if isinstance(values.dtype, pd.ArrowDtype) or values.dtype == object:
        if not values.notna().any():
            return np.dtype("datetime64[ns]")
        try:
            values = as_datetime(values, where="PowerIntegrator")
        except ValueError:
            return np.dtype("datetime64[ns]")
    if pd.api.types.is_datetime64_any_dtype(values.dtype):
        return values.dtype
    return np.dtype("datetime64[ns]")


def _empty_segments(time_dtype: object) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "uuid": pd.Series(dtype=object),
            "window_start": pd.Series(dtype=time_dtype),
            "start": pd.Series(dtype=time_dtype),
            "end": pd.Series(dtype=time_dtype),
            "duration_seconds": pd.Series(dtype="float64"),
            "kw_start": pd.Series(dtype="float64"),
            "kw_end": pd.Series(dtype="float64"),
            "kwh": pd.Series(dtype="float64"),
        }
    )


def _empty_windows(time_dtype: object, total: bool) -> pd.DataFrame:
    columns: dict[str, pd.Series] = {} if total else {"uuid": pd.Series(dtype=object)}
    columns.update(
        {
            "start": pd.Series(dtype=time_dtype),
            "end": pd.Series(dtype=time_dtype),
            "duration_seconds": pd.Series(dtype="float64"),
            "kwh": pd.Series(dtype="float64"),
            "avg_kw": pd.Series(dtype="float64"),
            "coverage": pd.Series(dtype="float64"),
            "sample_count": pd.Series(dtype="int64"),
        }
    )
    return pd.DataFrame(columns)


def _is_fixed(offset: object) -> bool:
    """Whether a frequency has a fixed length (Tick: h, min, s, ...)."""
    return isinstance(offset, Tick)


def _calendar_grid(t0: pd.Timestamp, t1: pd.Timestamp, window: str) -> pd.DatetimeIndex:
    """Boundaries of the ``resample(window)`` bins covering ``[t0, t1]``.

    The true bin bounds (:func:`~ts_shape.utils.time_accounting.grouper_periods`),
    not the resample labels: a right-labeled frequency (``'W'``, ``'ME'``,
    ``'QE'``, ``'YE'``) labels a bin with its *last* day.
    """
    periods = grouper_periods(t0, t1, window)
    if periods.empty:
        raise ValueError(f"window {window!r} gives no bins for the data range")
    starts = pd.DatetimeIndex(periods["start"])
    return starts.append(pd.DatetimeIndex([periods["end"].iloc[-1]]))


def _window_ends(starts: pd.Series, window: str) -> pd.Series:
    """End of each window that starts at ``starts`` (calendar-correct)."""
    offset = window_offset(window)
    if _is_fixed(offset) or starts.empty:
        return starts + offset
    grid = _calendar_grid(starts.min(), starts.max(), window)
    pos = grid.searchsorted(pd.DatetimeIndex(starts), side="left")
    return pd.Series(grid[pos + 1], index=starts.index)


class PowerIntegrator(Base):
    """Transform: integrate an instantaneous power signal into energy.

    Stateless classmethods -- each call is a pure function of its inputs, in
    line with the calculator/transform pattern described in
    :mod:`ts_shape.utils.base`.

    Methods:
    - energy_by_window: kWh (and time-weighted mean kW) per time window.
    - to_energy_deltas: the same energy, emitted in the standard ts-shape
      schema so the ``events.energy`` detectors can consume it.
    - segment_energy: the per-sample audit trail the two above are built on.

    Interpolation (``method``):

    - ``"locf"`` (default) -- last-observation-carried-forward / step hold.
      Correct for on-change PLC and meter logging, where a value is by
      definition valid until the next one is written.
    - ``"linear"`` -- trapezoidal. Correct for a smoothly varying analog
      signal sampled on a fixed cycle. On a 0 -> 100 kW ramp over one hour,
      ``linear`` gives exactly 50 kWh while ``locf`` gives slightly less; that
      is the interpolation bias, not an error.

    Negative power (PV export, regenerative drives) is integrated as-is, so
    ``kwh`` is *net* energy. Filter the input first if gross import is wanted.

    Example:
        100 kW for half an hour, then 50 kW, then off (``locf`` holds each value):

        >>> import pandas as pd
        >>> from ts_shape import PowerIntegrator
        >>> df = pd.DataFrame({
        ...     "systime": pd.to_datetime([
        ...         "2025-01-01 00:00", "2025-01-01 00:30",
        ...         "2025-01-01 01:00", "2025-01-01 02:00"]),
        ...     "uuid": "press",
        ...     "value_double": [100.0, 50.0, 0.0, 0.0],
        ... })
        >>> PowerIntegrator.energy_by_window(df, window="1h")[["start", "kwh", "avg_kw"]]
                        start   kwh  avg_kw
        0 2025-01-01 00:00:00  75.0    75.0
        1 2025-01-01 01:00:00   0.0     0.0
    """

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _power_scale(power_unit: str) -> float:
        """Return the multiplier that converts ``power_unit`` to kW."""
        key = str(power_unit).strip().lower()
        if key not in _POWER_UNITS:
            raise ValueError(
                f"Unknown power_unit {power_unit!r}. "
                f"Supported: {sorted(set(_POWER_UNITS))}"
            )
        return _POWER_UNITS[key]

    @classmethod
    def _prepare(
        cls,
        dataframe: pd.DataFrame,
        *,
        time_column: str,
        uuid_column: str,
        value_column: str,
        power_unit: str,
    ) -> pd.DataFrame:
        """Return a tidy, sorted ``uuid | systime | kw`` frame."""
        for col in (time_column, uuid_column, value_column):
            cls._validate_column(dataframe, col)

        # Reset first: the three columns are recombined into a new frame, so
        # they must share one index or a non-default input index silently
        # mispairs them.
        src = dataframe.reset_index(drop=True)
        raw_times = src[time_column]
        if isinstance(raw_times.dtype, pd.ArrowDtype):
            raw_times = as_datetime(raw_times, where="PowerIntegrator")
        times = pd.to_datetime(raw_times, errors="coerce")
        bad_times = int((times.isna() & src[time_column].notna()).sum())
        if bad_times:
            warnings.warn(
                f"{bad_times} value(s) in time column {time_column!r} are not "
                "timestamps; those rows are skipped.",
                DataQualityWarning,
                stacklevel=caller_stacklevel(),
            )
        power = to_numeric_checked(
            src[value_column], what=f"power column {value_column!r}"
        )
        df = pd.DataFrame(
            {
                "uuid": src[uuid_column],
                "ts": times,
                "kw": power * cls._power_scale(power_unit),
            }
        )
        df = df.dropna(subset=["ts", "kw"])
        # Two readings with the same timestamp span zero time; the later one
        # wins, matching last-value-hold semantics.
        df = df.sort_values(["uuid", "ts"]).drop_duplicates(
            subset=["uuid", "ts"], keep="last"
        )
        return df.reset_index(drop=True)

    @staticmethod
    def _hold_segments(
        df: pd.DataFrame,
        *,
        max_gap: str | None,
        data_end: pd.Timestamp | str | None,
    ) -> pd.DataFrame:
        """Pair each sample with the next one to form hold segments."""
        grouped = df.groupby("uuid", sort=False)
        df = df.copy()
        df["end"] = grouped["ts"].shift(-1)
        df["kw_end"] = grouped["kw"].shift(-1)

        if data_end is not None:
            # Close the trailing segment of every series at a known end of
            # data (e.g. the query window end) instead of dropping it.
            end_ts = localize_like(data_end, df["ts"])
            tail = df["end"].isna()
            df.loc[tail, "end"] = end_ts
            # Nothing is known past the last sample, so hold it flat.
            df.loc[tail, "kw_end"] = df.loc[tail, "kw"]

        df = df[df["end"].notna()].copy()
        df["seconds"] = (df["end"] - df["ts"]).dt.total_seconds()
        df = df[df["seconds"] > 0]

        if max_gap is not None:
            # Repo convention (see OEEEvents.max_gap): an over-long gap is
            # a logging outage, not a state that persisted. Drop it entirely
            # rather than billing energy for a machine whose power is unknown.
            cap = pd.to_timedelta(max_gap).total_seconds()
            dropped = int((df["seconds"] > cap).sum())
            if dropped:
                logger.debug(
                    "PowerIntegrator: dropped %d segment(s) longer than "
                    "max_gap=%s as logging outages",
                    dropped,
                    max_gap,
                )
            df = df[df["seconds"] <= cap]

        return df.reset_index(drop=True)

    @staticmethod
    def _split_on_windows(segments: pd.DataFrame, window: str) -> pd.DataFrame:
        """Cut every hold segment at window boundaries.

        A sample at 00:55 held until 01:05 belongs 5 minutes to hour 00 and 5
        minutes to hour 01. Flooring the sample timestamp instead would book
        all ten minutes to hour 00.
        """
        offset = window_offset(window)
        start = segments["ts"]
        end = segments["end"]
        if start.empty:
            return _empty_segments(start.dtype)
        if _is_fixed(offset):
            first = start.min().floor(window)
            if first + offset <= first:
                raise ValueError(f"window must be a positive frequency, got {window!r}")
            grid = pd.date_range(first, end.max(), freq=window)
            grid = grid.append(pd.DatetimeIndex([grid[-1] + offset]))
            while grid[-1] < end.max():
                grid = grid.append(pd.DatetimeIndex([grid[-1] + offset]))
        else:
            # Calendar windows ('1D', 'W', 'MS', 'ME', ...) on the calendar of
            # the data's timezone: a DST day lasts 23 h / 25 h, a month its
            # real length, and right-labeled bins start after their label.
            grid = _calendar_grid(start.min(), end.max(), window)

        # Window of each segment's start, and the first boundary at/after its
        # (exclusive) end: a segment ending exactly on a boundary does not
        # reach into the next window.
        i0 = grid.searchsorted(start, side="right") - 1
        i1 = grid.searchsorted(end, side="left")
        if (i0 < 0).any() or (i1 >= len(grid)).any():
            raise RuntimeError(
                f"PowerIntegrator: window grid for {window!r} does not cover "
                "the data (internal error)"
            )
        n_buckets = np.maximum(i1 - i0, 0)

        idx = np.repeat(np.arange(len(segments)), n_buckets)
        if idx.size == 0:
            return _empty_segments(start.dtype)

        # Position of each row within its parent segment: 0, 1, 2, ...
        offsets = np.arange(idx.size) - np.repeat(
            np.cumsum(n_buckets) - n_buckets, n_buckets
        )
        bucket = np.repeat(i0, n_buckets) + offsets

        def take(col: str) -> pd.Series:
            return segments[col].take(idx).reset_index(drop=True)

        # Stay in pandas so tz-aware timestamps keep their dtype.
        window_start = pd.Series(grid[bucket])
        window_end = pd.Series(grid[bucket + 1])
        seg_start = take("ts")
        seg_end = take("end")
        sub_start = seg_start.where(seg_start > window_start, window_start)
        sub_end = seg_end.where(seg_end < window_end, window_end)

        return pd.DataFrame(
            {
                "uuid": take("uuid"),
                "window_start": window_start,
                "start": sub_start,
                "end": sub_end,
                "duration_seconds": (sub_end - sub_start).dt.total_seconds(),
                "kw": take("kw"),
                "kw_next": take("kw_end"),
                "seg_start": seg_start,
                "seg_seconds": take("seconds"),
            }
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    @renamed_kwargs(freq="window")
    def segment_energy(
        cls,
        dataframe: pd.DataFrame,
        *,
        window: str = "1h",
        method: str = "locf",
        max_gap: str | None = None,
        power_unit: str = "kW",
        data_end: pd.Timestamp | str | None = None,
        time_column: str = "systime",
        uuid_column: str = "uuid",
        value_column: str = "value_double",
    ) -> pd.DataFrame:
        """Energy contribution of every sample, split at window boundaries.

        The audit trail behind :meth:`energy_by_window`: one row per
        (sample, window) pair, so a surprising hourly total can be traced back
        to the samples that produced it. The ``kwh`` column sums exactly to the
        windowed result.

        Args:
            dataframe: Long-format power data (ts-shape schema by default).
            window: Window the segments are cut on ('1h', '15min', '1D', 'W',
                'MS', 'ME'); see :meth:`energy_by_window`.
            method: 'locf' (step hold) or 'linear' (trapezoidal).
            max_gap: Gaps longer than this (e.g. '15min') are treated as
                logging outages: the segment is dropped entirely and counts
                towards neither energy nor coverage, because the power over
                that span is unknown. Without a cap, one sample before a
                three-hour outage bills three hours of energy at that power
                level; that is what ``None`` (default) does. Set it above the signal's normal logging cadence -- an
                on-change signal on a steady machine can legitimately stay silent
                for a long time, and too tight a cap silently discards real
                consumption (watch ``coverage``).
            power_unit: Unit of ``value_column`` -- 'W', 'kW' or 'MW'.
            data_end: Optional end of the data window. The last sample of each
                series is held until this timestamp instead of being dropped.
            time_column: Timestamp column.
            uuid_column: Column identifying the signal.
            value_column: Column holding instantaneous power.

        Returns:
            DataFrame with columns: uuid, window_start, start, end,
            duration_seconds, kw_start, kw_end, kwh.
        """
        if method not in ("locf", "linear"):
            raise ValueError(f"method must be 'locf' or 'linear', got {method!r}")
        require_columns(
            dataframe,
            [time_column, uuid_column, value_column],
            where="PowerIntegrator",
        )
        empty = _empty_segments(_time_dtype(dataframe, time_column))
        if dataframe.empty:
            return empty

        prepared = cls._prepare(
            dataframe,
            time_column=time_column,
            uuid_column=uuid_column,
            value_column=value_column,
            power_unit=power_unit,
        )
        segments = cls._hold_segments(prepared, max_gap=max_gap, data_end=data_end)
        if segments.empty:
            return empty

        parts = cls._split_on_windows(segments, window)
        if parts.empty:
            return empty

        if method == "locf":
            parts["kw_start"] = parts["kw"]
            parts["kw_end"] = parts["kw"]
        else:
            # Interpolate the parent segment's ramp at the sub-segment bounds
            # so splitting a window boundary stays exact.
            slope = (parts["kw_next"] - parts["kw"]) / parts["seg_seconds"]
            offset_start = (parts["start"] - parts["seg_start"]).dt.total_seconds()
            offset_end = (parts["end"] - parts["seg_start"]).dt.total_seconds()
            parts["kw_start"] = parts["kw"] + slope * offset_start
            parts["kw_end"] = parts["kw"] + slope * offset_end

        parts["kwh"] = (
            (parts["kw_start"] + parts["kw_end"]) / 2.0 * parts["duration_seconds"]
        ) / 3600.0

        return parts[_SEGMENT_COLUMNS].reset_index(drop=True)

    @classmethod
    @renamed_kwargs(freq="window")
    def energy_by_window(
        cls,
        dataframe: pd.DataFrame,
        *,
        window: str = "1h",
        method: str = "locf",
        max_gap: str | None = None,
        power_unit: str = "kW",
        total: bool = True,
        data_end: pd.Timestamp | str | None = None,
        time_column: str = "systime",
        uuid_column: str = "uuid",
        value_column: str = "value_double",
    ) -> pd.DataFrame:
        """Integrate power into energy per time window.

        Args:
            dataframe: Long-format power data (ts-shape schema by default).
            window: Window size -- '1h' gives kWh per hour.
            method: 'locf' (step hold, default) or 'linear' (trapezoidal).
            max_gap: Cap on how long one sample may be held (e.g. '15min').
                Longer gaps are dropped as logging outages -- see
                :meth:`segment_energy`. ``None`` (default) holds every sample
                until the next one, however long the gap.
            power_unit: Unit of ``value_column`` -- 'W', 'kW' or 'MW'.
            total: When True (default), sum across every signal in the frame --
                energy is additive across machines. When False, return one row
                per signal per window with a ``uuid`` column. Never average
                across signals: power averages, energy adds.
            data_end: Optional end of the data window; the last sample of each
                series is held until then.
            time_column: Timestamp column.
            uuid_column: Column identifying the signal.
            value_column: Column holding instantaneous power.

        Returns:
            DataFrame with columns: start, end, duration_seconds, kwh, avg_kw,
            coverage, sample_count (plus ``uuid`` when ``total=False``).

            - ``kwh`` -- integrated energy in the window. Net of any negative
                (exported/regenerated) power.
            - ``avg_kw`` -- time-weighted mean power over the *covered* time,
                i.e. ``kwh`` divided by the hours actually logged. With full
                coverage and ``window='1h'`` this equals ``kwh``.
            - ``coverage`` -- fraction of the window backed by data (with
                ``total=True``, that of the least-covered signal; a signal
                with no data in the window counts as 0). Anything below 1.0
                means the window is under-reported.
            - ``sample_count`` -- sample intervals contributing to the window.
                One interval straddling a boundary counts in both windows.

            Windows with no data at all are omitted rather than reported as
            0.0 kWh, so "not measured" never renders as "used nothing".

        Note:
            ``window`` is a fixed frequency ('15min', '1h') or a calendar one
            ('1D', 'W', 'MS', 'ME', ...). Fixed windows are anchored on the
            floored sample timestamp and last exactly that long. Calendar
            windows are the ``resample`` bins on the local calendar of the
            data's timezone: ``start`` / ``end`` are the true bin bounds
            (also for right-labeled frequencies such as 'W' and 'ME', whose
            resample label is the bin's last day), so a ``'1D'`` window is
            the real local day -- 23 h or 25 h on a daylight-saving change
            -- and a month window its real number of days. ``coverage`` and
            ``duration_seconds`` use that real length.
        """
        extra = [] if total else ["uuid"]

        parts = cls.segment_energy(
            dataframe,
            window=window,
            method=method,
            max_gap=max_gap,
            power_unit=power_unit,
            data_end=data_end,
            time_column=time_column,
            uuid_column=uuid_column,
            value_column=value_column,
        )
        if parts.empty:
            return _empty_windows(parts["start"].dtype, total)

        def window_seconds(starts: pd.Series) -> pd.Series:
            # Real window length: calendar windows are 23 h / 25 h on DST days,
            # a month its real number of days.
            return (_window_ends(starts, window) - starts).dt.total_seconds()

        per_uuid = (
            parts.groupby(["uuid", "window_start"], sort=True)
            .agg(
                kwh=("kwh", "sum"),
                covered_seconds=("duration_seconds", "sum"),
                sample_count=("kwh", "size"),
            )
            .reset_index()
        )
        per_uuid["coverage"] = per_uuid["covered_seconds"] / window_seconds(
            per_uuid["window_start"]
        )

        if total:
            # A meter with no data in a window covers none of it: the total's
            # coverage is the minimum over every meter, absent ones as 0.
            grid = pd.MultiIndex.from_product(
                [per_uuid["uuid"].unique(), per_uuid["window_start"].unique()],
                names=["uuid", "window_start"],
            )
            coverage = (
                per_uuid.set_index(["uuid", "window_start"])["coverage"]
                .reindex(grid, fill_value=0.0)
                .groupby(level="window_start", sort=True)
                .min()
                .rename("coverage")
                .reset_index()
            )
            grouped = (
                per_uuid.groupby("window_start", sort=True)
                .agg(
                    kwh=("kwh", "sum"),
                    sample_count=("sample_count", "sum"),
                )
                .reset_index()
                .merge(coverage, on="window_start", how="left")
            )
            # Sum of the per-signal mean demands, not a mean across signals.
            avg = (
                per_uuid.assign(
                    avg_kw=np.where(
                        per_uuid["covered_seconds"] > 0,
                        per_uuid["kwh"] / (per_uuid["covered_seconds"] / 3600.0),
                        np.nan,
                    )
                )
                .groupby("window_start", sort=True)["avg_kw"]
                .sum()
                .reset_index()
            )
            out = grouped.merge(avg, on="window_start", how="left")
        else:
            out = per_uuid.copy()
            out["avg_kw"] = np.where(
                out["covered_seconds"] > 0,
                out["kwh"] / (out["covered_seconds"] / 3600.0),
                np.nan,
            )

        out = out.rename(columns={"window_start": "start"})
        out["end"] = _window_ends(out["start"], window)
        out["duration_seconds"] = (out["end"] - out["start"]).dt.total_seconds()
        return out[extra + _WINDOW_COLUMNS].reset_index(drop=True)

    @classmethod
    @renamed_kwargs(freq="window")
    def to_energy_deltas(
        cls,
        dataframe: pd.DataFrame,
        *,
        window: str = "1h",
        method: str = "locf",
        max_gap: str | None = None,
        power_unit: str = "kW",
        total: bool = False,
        event_uuid: str = "energy:kwh",
        data_end: pd.Timestamp | str | None = None,
        time_column: str = "systime",
        uuid_column: str = "uuid",
        value_column: str = "value_double",
    ) -> pd.DataFrame:
        """Emit integrated energy in the standard ts-shape schema.

        The bridge into the ``events.energy`` pack, whose detectors assume
        their input values are already energy deltas in kWh::

            deltas = PowerIntegrator.to_energy_deltas(power_df, max_gap="15min")
            EnergyConsumptionEvents(deltas).consumption_by_window(
                meter_uuid, window="1h", agg="sum"
            )

        Args:
            dataframe: Long-format power data.
            window: Delta size -- one row per signal per window.
            method: 'locf' (default) or 'linear'.
            max_gap: Cap on how long one sample may be held; longer gaps
                are dropped as logging outages. ``None`` (default) holds every
                sample until the next one, however long the gap.
            power_unit: Unit of ``value_column`` -- 'W', 'kW' or 'MW'.
            total: When True, collapse every signal into one plant-level
                series identified by ``event_uuid``.
            event_uuid: UUID assigned to the summed series when ``total=True``.
            data_end: Optional end of the data window.
            time_column: Timestamp column.
            uuid_column: Column identifying the signal.
            value_column: Column holding instantaneous power.

        Returns:
            DataFrame with columns: systime, uuid, value_double, is_delta --
            ``value_double`` being kWh consumed in the window starting at
            ``systime``.
        """
        windows = cls.energy_by_window(
            dataframe,
            window=window,
            method=method,
            max_gap=max_gap,
            power_unit=power_unit,
            total=total,
            data_end=data_end,
            time_column=time_column,
            uuid_column=uuid_column,
            value_column=value_column,
        )
        if windows.empty:
            return pd.DataFrame(
                {
                    "systime": pd.Series(dtype=windows["start"].dtype),
                    "uuid": pd.Series(dtype=object),
                    "value_double": pd.Series(dtype="float64"),
                    "is_delta": pd.Series(dtype=bool),
                }
            )

        out = pd.DataFrame(
            {
                "systime": windows["start"],
                "uuid": (
                    event_uuid if total else windows["uuid"].astype(str).to_numpy()
                ),
                "value_double": windows["kwh"].to_numpy(),
                "is_delta": True,
            }
        )
        return out.sort_values(["systime", "uuid"]).reset_index(drop=True)
