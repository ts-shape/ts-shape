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

    PowerIntegrator.energy_by_window(df, freq="1h", max_gap="15min")
    #                  start                 end  duration_seconds    kwh  avg_kw  coverage  sample_count

Or emit ts-shape-schema energy deltas and hand off to the energy pack::

    deltas = PowerIntegrator.to_energy_deltas(df, freq="1h", max_gap="15min")
    EnergyConsumptionEvents(deltas).consumption_by_window(meter_uuid, window="1h")

Note:
    If the meter also exposes a cumulative kWh register, prefer it -- a
    counter difference is billing-accurate and needs no integration. Use this
    class when instantaneous power is all you have.
"""

from __future__ import annotations

import logging

import numpy as np  # type: ignore
import pandas as pd
from pandas.tseries.frequencies import to_offset
from pandas.tseries.offsets import Tick  # type: ignore

from ts_shape.utils.base import Base

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


def _is_fixed(offset: object) -> bool:
    """Whether a frequency has a fixed length (Tick: h, min, s, ...)."""
    return isinstance(offset, Tick)


def _calendar_floor(ts: pd.Timestamp, freq: str) -> pd.Timestamp:
    """Start of the calendar period containing ``ts`` (e.g. local midnight)."""
    period_start = (
        pd.Series(1, index=pd.DatetimeIndex([ts])).resample(freq).sum().index[0]
    )
    return period_start


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
        df = pd.DataFrame(
            {
                "uuid": src[uuid_column],
                "ts": pd.to_datetime(src[time_column], errors="coerce"),
                "kw": pd.to_numeric(src[value_column], errors="coerce")
                * cls._power_scale(power_unit),
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
            end_ts = pd.Timestamp(data_end)
            tail = df["end"].isna()
            df.loc[tail, "end"] = end_ts
            # Nothing is known past the last sample, so hold it flat.
            df.loc[tail, "kw_end"] = df.loc[tail, "kw"]

        df = df[df["end"].notna()].copy()
        df["seconds"] = (df["end"] - df["ts"]).dt.total_seconds()
        df = df[df["seconds"] > 0]

        if max_gap is not None:
            # Repo convention (see OEECalculator.max_gap): an over-long gap is
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
    def _split_on_windows(segments: pd.DataFrame, freq: str) -> pd.DataFrame:
        """Cut every hold segment at window boundaries.

        A sample at 00:55 held until 01:05 belongs 5 minutes to hour 00 and 5
        minutes to hour 01. Flooring the sample timestamp instead would book
        all ten minutes to hour 00.
        """
        offset = to_offset(freq)
        start = segments["ts"]
        end = segments["end"]
        if start.empty:
            return pd.DataFrame(columns=_SEGMENT_COLUMNS)
        first = (
            start.min().floor(freq)
            if _is_fixed(offset)
            else _calendar_floor(start.min(), freq)
        )
        if first + offset <= first:
            raise ValueError(f"freq must be a positive frequency, got {freq!r}")
        # Window boundaries on the calendar of the data's timezone: hourly
        # windows stay one real hour across DST, daily windows are 23 h / 25 h.
        grid = pd.date_range(first, end.max(), freq=freq)
        grid = grid.append(pd.DatetimeIndex([grid[-1] + offset]))
        while grid[-1] < end.max():
            grid = grid.append(pd.DatetimeIndex([grid[-1] + offset]))

        # Bucket of each segment's start, and the first boundary at/after its
        # (exclusive) end: a segment ending exactly on a boundary does not
        # reach into the next window.
        i0 = grid.searchsorted(start, side="right") - 1
        i1 = grid.searchsorted(end, side="left")
        n_buckets = np.maximum(i1 - i0, 0)

        idx = np.repeat(np.arange(len(segments)), n_buckets)
        if idx.size == 0:
            return pd.DataFrame(columns=_SEGMENT_COLUMNS)

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
    def segment_energy(
        cls,
        dataframe: pd.DataFrame,
        *,
        freq: str = "1h",
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
            freq: Fixed window size the segments are cut on ('1h', '15min', '1D').
            method: 'locf' (step hold) or 'linear' (trapezoidal).
            max_gap: Gaps longer than this (e.g. '15min') are treated as
                logging outages: the segment is dropped entirely and counts
                towards neither energy nor coverage, because the power over
                that span is unknown. Without a cap, one sample before a
                three-hour outage bills three hours of energy at that power
                level. Set it above the signal's normal logging cadence -- an
                on-change tag on a steady machine can legitimately stay silent
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
        if dataframe.empty:
            return pd.DataFrame(columns=_SEGMENT_COLUMNS)

        prepared = cls._prepare(
            dataframe,
            time_column=time_column,
            uuid_column=uuid_column,
            value_column=value_column,
            power_unit=power_unit,
        )
        segments = cls._hold_segments(prepared, max_gap=max_gap, data_end=data_end)
        if segments.empty:
            return pd.DataFrame(columns=_SEGMENT_COLUMNS)

        parts = cls._split_on_windows(segments, freq)
        if parts.empty:
            return pd.DataFrame(columns=_SEGMENT_COLUMNS)

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
    def energy_by_window(
        cls,
        dataframe: pd.DataFrame,
        *,
        freq: str = "1h",
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
            freq: Window size -- '1h' gives kWh per hour.
            method: 'locf' (step hold, default) or 'linear' (trapezoidal).
            max_gap: Cap on how long one sample may be held (e.g. '15min').
                Longer gaps are dropped as logging outages -- see
                :meth:`segment_energy`.
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
                coverage and ``freq='1h'`` this equals ``kwh``.
            - ``coverage`` -- fraction of the window backed by data (with
                ``total=True``, that of the least-covered signal). Anything below
                1.0 means the window is under-reported.
            - ``sample_count`` -- sample intervals contributing to the window.
                One interval straddling a boundary counts in both windows.

            Windows with no data at all are omitted rather than reported as
            0.0 kWh, so "not measured" never renders as "used nothing".

        Note:
            ``freq`` must be a *fixed* frequency ('15min', '1h', '1D'), and
            windows are fixed-length buckets anchored on the floored sample
            timestamp. Elapsed time and total energy are always correct on
            tz-aware input, but a ``'1D'`` bucket spans 24 elapsed hours
            rather than the 23/25-hour local day, so bucket labels drift by an
            hour after a daylight-saving changeover. Use ``'1h'`` and
            aggregate to local days afterwards if that matters.
        """
        extra = [] if total else ["uuid"]
        empty = pd.DataFrame(columns=extra + _WINDOW_COLUMNS)

        parts = cls.segment_energy(
            dataframe,
            freq=freq,
            method=method,
            max_gap=max_gap,
            power_unit=power_unit,
            data_end=data_end,
            time_column=time_column,
            uuid_column=uuid_column,
            value_column=value_column,
        )
        if parts.empty:
            return empty

        offset = to_offset(freq)

        def window_seconds(starts: pd.Series) -> pd.Series:
            # Real window length: calendar windows are 23 h / 25 h on DST days.
            return (starts + offset - starts).dt.total_seconds()

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
            grouped = (
                per_uuid.groupby("window_start", sort=True)
                .agg(
                    kwh=("kwh", "sum"),
                    coverage=("coverage", "min"),
                    sample_count=("sample_count", "sum"),
                )
                .reset_index()
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
        out["end"] = out["start"] + offset
        out["duration_seconds"] = window_seconds(out["start"])
        return out[extra + _WINDOW_COLUMNS].reset_index(drop=True)

    @classmethod
    def to_energy_deltas(
        cls,
        dataframe: pd.DataFrame,
        *,
        freq: str = "1h",
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
            freq: Delta size -- one row per signal per window.
            method: 'locf' (default) or 'linear'.
            max_gap: Cap on how long one sample may be held; longer gaps
                are dropped as logging outages.
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
        cols = ["systime", "uuid", "value_double", "is_delta"]
        windows = cls.energy_by_window(
            dataframe,
            freq=freq,
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
            return pd.DataFrame(columns=cols)

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
