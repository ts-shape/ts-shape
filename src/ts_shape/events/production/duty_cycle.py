import logging
from collections.abc import Collection
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ts_shape.events._output import empty_event_df, twin_columns
from ts_shape.events._windows import floor_local, window_bounds
from ts_shape.utils.base import Base
from ts_shape.utils.coerce import run_bool
from ts_shape.utils.deprecation import deprecated_method, keyword_only
from ts_shape.utils.frames import localize_like
from ts_shape.utils.intervals import state_intervals
from ts_shape.utils.time_accounting import (
    EXCLUDED,
    floor_periods,
    reading_intervals,
    time_by_period,
)

if TYPE_CHECKING:
    from warnings import deprecated

logger = logging.getLogger(__name__)

#: State code of a row whose on/off value is missing (unknown time).
_UNKNOWN = -1
#: Interval state of time after a sample with a missing value.
_UNKNOWN_STATE = "unknown"


class DutyCycleEvents(Base):
    """Production: Duty Cycle Analysis

    Analyze on/off patterns from a boolean signal: duty cycle percentage,
    interval listing, transition counts, and excessive cycling detection.

    Methods:
    - duty_cycle_by_window: On-time percentage per time window.
    - on_off_intervals: List every on and off interval with duration.
    - cycle_count: Number of on/off transitions per window.
    - excessive_cycling: Flag windows with too many transitions.
    """

    #: Methods read only the constructor's signals (``Base`` keeps their rows).
    _reads_own_signals_only = True

    def __init__(
        self,
        dataframe: pd.DataFrame,
        signal_uuid: str,
        *,
        event_uuid: str = "prod:duty_cycle",
        value_column: str = "value_bool",
        time_column: str = "systime",
        interval_end: str = "next_change",
        data_end: pd.Timestamp | None = None,
        run_states: Collection[Any] | None = None,
        value_range: tuple[float | None, float | None] | None = None,
    ) -> None:
        """Initialize the duty-cycle analyzer.

        Args:
            dataframe: Long-format input data.
            signal_uuid: UUID of the boolean on/off signal (True = on).
            event_uuid: UUID assigned to generated events.
            value_column: Column holding the on/off state. A missing value
                (``None``, ``NaN``, ``pd.NA``) is unknown, not off: the time
                from such a row to the next sample is in neither ``on_time``
                nor ``off_time``, no on/off interval covers it, and it is no
                transition. Ratios use observed (known-state) time.
            time_column: Timestamp column.
            interval_end: How on/off intervals are closed.
                ``"next_change"`` (default) closes each interval at the next
                transition, which measures both dense and change-encoded
                (``is_delta``) signals. ``"last_sample"`` ends each interval at
                its last sample; on a change-encoded signal (one row per
                transition) every on/off interval then has zero duration.
            data_end: Optional bound for the final, still-open interval when
                ``interval_end="next_change"``; :meth:`duty_cycle_by_window`
                (time-weighted) and :meth:`cycle_count` then also report the
                windows up to ``data_end``. A naive value or text on tz-aware
                data is local time of the data's timezone.
            run_states: Values of the run signal that mean running (e.g.
                ``{2}`` for a mode signal where 2 is automatic, or
                ``{"RUN"}``); every other known value is not running.
                ``None`` (default) uses ``value_range`` or reads the signal
                as a bool, with a ``DataQualityWarning`` when it is not one.
            value_range: ``(lower, upper)`` inclusive bounds; a value inside
                counts as running (either bound may be ``None``). Used when
                ``run_states`` is ``None``.
        """
        super().__init__(dataframe, time_column=time_column)
        self.signal_uuid = signal_uuid
        self.event_uuid = event_uuid
        self.value_column = value_column
        self.run_states = None if run_states is None else frozenset(run_states)
        self.value_range = value_range
        self.time_column = time_column
        self.interval_end = interval_end
        self.data_end = data_end

        self.series = (
            self.dataframe[self.dataframe["uuid"] == self.signal_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        self.series[self.time_column] = pd.to_datetime(self.series[self.time_column])

    def on_off_intervals(
        self, *, max_gap: str | pd.Timedelta | None = None
    ) -> pd.DataFrame:
        """List every on and off interval with duration.

        Args:
            max_gap: Longest time between two samples an interval may span
                (e.g. ``"15min"``). A longer gap is "no data": the interval
                before it ends at its last sample and a new interval starts
                after the gap, even if the state is unchanged. ``None``
                (default): no limit.

        Returns:
            DataFrame with columns: start, end, duration_seconds (float),
            uuid (event uuid), source_uuid (signal uuid), state (on/off),
            duration (Timedelta).
        """
        cols = ["state", "duration"]
        if self.series.empty:
            return empty_event_df("interval", cols)

        s = self.series[[self.time_column, self.value_column]].copy()
        s["state"] = self._state_codes(s[self.value_column])
        if max_gap is None:
            intervals = state_intervals(
                s,
                time_column=self.time_column,
                state_column="state",
                interval_end=self.interval_end,
                data_end=self.data_end,
            )
        else:
            intervals = self._intervals_split_at_gaps(s, pd.Timedelta(max_gap))
        # Time after a missing reading is unknown: no on/off interval.
        intervals = intervals[(intervals["state"] != _UNKNOWN).to_numpy()]
        if intervals.empty:
            return empty_event_df("interval", cols)

        out = pd.DataFrame(
            {
                "start": intervals["start"].to_numpy(),
                "end": intervals["end"].to_numpy(),
                "state": ["on" if st == 1 else "off" for st in intervals["state"]],
                "duration": pd.to_timedelta(intervals["duration_seconds"], unit="s"),
            }
        )
        out.insert(
            2,
            "duration_seconds",
            (
                pd.to_datetime(out["end"]) - pd.to_datetime(out["start"])
            ).dt.total_seconds(),
        )
        out.insert(3, "uuid", self.event_uuid)
        out.insert(4, "source_uuid", self.signal_uuid)
        return out

    def _known(self, values: pd.Series) -> pd.Series:
        """``True`` where the on/off value is present."""
        return values.notna()

    def _on(self, values: pd.Series) -> pd.Series:
        """The on/off signal as ``bool`` (see :func:`ts_shape.utils.coerce.run_mask`)."""
        return run_bool(
            values,
            run_states=self.run_states,
            value_range=self.value_range,
            what=f"DutyCycleEvents: on/off signal {self.signal_uuid!r}",
        )

    def _state_codes(self, values: pd.Series) -> pd.Series:
        """``1`` on, ``0`` off, ``-1`` unknown (missing value), as ``int64``."""
        codes = self._on(values).astype("int64")
        return codes.where(self._known(values), _UNKNOWN)

    def _data_end(self) -> pd.Timestamp | None:
        """``data_end`` in the data's timezone, or None."""
        if self.data_end is None:
            return None
        return localize_like(self.data_end, self.series[self.time_column])

    def _intervals_split_at_gaps(
        self, s: pd.DataFrame, max_gap: pd.Timedelta
    ) -> pd.DataFrame:
        """:func:`state_intervals`, with a new interval after each long gap."""
        s = s.reset_index(drop=True)
        times = s[self.time_column]
        gap_after = ((times.shift(-1) - times) > max_gap).to_numpy(dtype=bool)
        gap_before = np.r_[False, gap_after[:-1]]
        changed = (s["state"] != s["state"].shift()).to_numpy(dtype=bool)
        s["_segment"] = np.cumsum(changed | gap_before)
        intervals = state_intervals(
            s,
            time_column=self.time_column,
            state_column="_segment",
            interval_end=self.interval_end,
            data_end=self.data_end,
        )
        grouped = s.groupby("_segment", sort=True)
        last_sample = grouped[self.time_column].last().reset_index(drop=True)
        ends_at_gap = (
            pd.Series(gap_after).groupby(s["_segment"]).last().to_numpy(dtype=bool)
        )
        intervals["state"] = grouped["state"].first().to_numpy()
        intervals["end"] = intervals["end"].where(~ends_at_gap, last_sample)
        intervals["duration_seconds"] = (
            intervals["end"] - intervals["start"]
        ).dt.total_seconds()
        return intervals

    if TYPE_CHECKING:

        @deprecated(
            "DutyCycleEvents.duty_cycle_per_window() is deprecated, use .duty_cycle_by_window()"
        )
        def duty_cycle_per_window(
            self, window: str = "1h", *, time_weighted: bool = True
        ) -> pd.DataFrame: ...

    else:
        duty_cycle_per_window = deprecated_method("duty_cycle_by_window")

    @keyword_only(1)
    @twin_columns({"on_time": "on_time_seconds", "off_time": "off_time_seconds"})
    def duty_cycle_by_window(
        self,
        window: str = "1h",
        *,
        time_weighted: bool = True,
        max_gap: Any = "auto",
    ) -> pd.DataFrame:
        """Percentage of time the signal is True per window.

        Args:
            window: Resample window.
            time_weighted: When True (default), each sample is weighted by
                the gap to the next sample, so the duty cycle reflects
                wall-clock on-time on change-encoded and irregularly sampled
                signals; ``on_time`` and ``off_time`` are the covered seconds
                in the window. When False, the duty cycle is the mean of the
                boolean samples in the window, which equals the time share only
                under uniform sampling (``On@09:00, Off@09:59`` reads 50 %).
            max_gap: Time-weighted only. A gap between two samples longer
                than this (``"15min"``, ``Timedelta``) is a logging gap: its
                time is ``unknown_seconds``, neither on nor off. ``"auto"``
                (default) uses the signal's inferred ``max_gap`` (five times
                the sampling period of a sampled signal; no cap for a
                change-encoded one, whose long gaps are real holds -- pass a
                value when the export has holes). ``None`` applies no cap.

        Each interval between two samples is split at the window
        boundaries, so ``on_time + off_time + unknown_seconds`` never
        exceeds the window's length (time-weighted).

        Samples with a missing value are unknown time: in neither ``on_time``
        nor ``off_time`` (time-weighted) or not averaged (sample mean), and
        ``duty_cycle_pct`` is the on share of the known time (NaN for a window
        without any). With the constructor's ``data_end`` (time-weighted),
        the last sample's state is held until ``data_end`` and the windows up
        to it are reported.

        Returns:
            DataFrame with columns: start, end (start + window),
            duration_seconds, uuid, source_uuid, on_time, on_time_seconds,
            off_time, off_time_seconds, duty_cycle_pct, unknown_seconds. On
            and off time are in seconds; ``unknown_seconds`` is the time in
            the window with a missing value or inside a gap longer than
            ``max_gap`` (time-weighted; 0 for the sample mean).

            Canonical twin columns: ``on_time_seconds`` (after ``on_time``),
            ``off_time_seconds`` (after ``off_time``); each holds the same
            values as the deprecated column right before it, which is removed
            in the next major release.
        """
        cols = ["start", "on_time", "off_time", "duty_cycle_pct", "unknown_seconds"]
        if self.series.empty:
            return self._with_window(pd.DataFrame(columns=cols), window)

        s = self.series[[self.time_column, self.value_column]].copy()
        s[self.time_column] = pd.to_datetime(s[self.time_column])
        s = s.sort_values(self.time_column)
        known = self._known(s[self.value_column])
        s["state"] = self._on(s[self.value_column]).astype(float).where(known)

        events: list[dict[str, Any]] = []
        if time_weighted:
            gap = self._max_gap(max_gap)
            real = len(s)
            s = self._hold_to_data_end(s[[self.time_column, "state"]], window)
            if len(s) < 2:
                return self._with_window(pd.DataFrame(columns=cols), window)
            labels = s["state"].map({1.0: "on", 0.0: "off"}).fillna(_UNKNOWN_STATE)
            intervals = reading_intervals(s[self.time_column], labels)
            if gap is not None:
                # Only a gap between two real samples is a logging outage;
                # the rows held until data_end are known time.
                between_real = np.arange(len(intervals)) < real - 1
                over = (intervals["end"] - intervals["start"]) > pd.to_timedelta(gap)
                intervals.loc[over.to_numpy() & between_real, "state"] = EXCLUDED
            times = s[self.time_column]
            periods = floor_periods(times.iloc[0], times.iloc[-1], window)
            table = time_by_period(
                intervals,
                periods,
                keys=["period"],
                states=["on", "off", _UNKNOWN_STATE, EXCLUDED],
            )
            for start, on_time, off_time, unknown, gap in table[
                ["start", "on", "off", _UNKNOWN_STATE, EXCLUDED]
            ].itertuples(index=False, name=None):
                total = on_time + off_time
                pct = (on_time / total) if total > 0 else float("nan")
                events.append(
                    {
                        "start": start,
                        "on_time": float(on_time),
                        "off_time": float(off_time),
                        "duty_cycle_pct": pct * 100,
                        "unknown_seconds": float(unknown + gap),
                    }
                )
        else:
            resampled = s.set_index(self.time_column)["state"].resample(window).mean()
            # The window's real length (a DST day lasts 23 h or 25 h).
            w_start, w_end = window_bounds(resampled.index.to_series(), window)
            window_s = (w_end - w_start).dt.total_seconds().tolist()
            for (ts, pct), span_s in zip(resampled.items(), window_s, strict=True):
                if pd.isna(pct):
                    continue
                on_time = pct * span_s
                off_time = (1 - pct) * span_s
                events.append(
                    {
                        "start": ts,
                        "on_time": on_time,
                        "off_time": off_time,
                        "duty_cycle_pct": pct * 100,
                        "unknown_seconds": 0.0,
                    }
                )

        return self._with_window(pd.DataFrame(events, columns=cols), window)

    def _max_gap(self, max_gap: Any) -> Any:
        """``max_gap``, with ``"auto"`` resolved from the signal's inferred spec."""
        if not (isinstance(max_gap, str) and max_gap == "auto"):
            return max_gap
        if len(self.series) < 3:
            return None
        from ts_shape.signals.infer import infer_spec

        return infer_spec(self.series, self.signal_uuid, self.time_column).max_gap

    def _hold_to_data_end(self, s: pd.DataFrame, window: str) -> pd.DataFrame:
        """``s`` with the last state held until ``data_end``.

        Adds a row with the last state at every window start after the last
        sample and before ``data_end``, and a closing row at ``data_end``, so
        the held time is booked to the windows it falls in.
        """
        data_end = self._data_end()
        if data_end is None or s.empty:
            return s
        last_time = s[self.time_column].iloc[-1]
        if not data_end > last_time:
            return s
        grid = floor_periods(last_time, data_end, window)["start"]
        starts = grid[(grid > last_time) & (grid < data_end)]
        held = pd.DataFrame(
            {
                self.time_column: pd.concat(
                    [starts, pd.Series([data_end])], ignore_index=True
                ),
                "state": s["state"].iloc[-1],
            }
        )
        held[self.time_column] = held[self.time_column].astype(
            s[self.time_column].dtype
        )
        return pd.concat([s, held], ignore_index=True)

    @keyword_only(1)
    def cycle_count(self, window: str = "1h") -> pd.DataFrame:
        """Number of on/off transitions per window.

        A transition is a change between two known states; samples with a
        missing value are skipped. With the constructor's ``data_end`` the
        windows up to ``data_end`` are reported (with zero transitions after
        the last sample).

        Args:
            window: Resample window.

        Returns:
            DataFrame with columns: start, end (start + window),
            duration_seconds, uuid, source_uuid, on_count, off_count,
            total_transitions.
        """
        cols = ["start", "on_count", "off_count", "total_transitions"]
        if self.series.empty:
            return self._with_window(pd.DataFrame(columns=cols), window)

        s = self.series[[self.time_column, self.value_column]].copy()
        s = s[self._known(s[self.value_column]).to_numpy()]
        s["state"] = self._on(s[self.value_column])
        s["transition"] = s["state"] != s["state"].shift()
        s["to_on"] = s["transition"] & s["state"]
        s["to_off"] = s["transition"] & ~s["state"]
        data_end = self._data_end()
        if data_end is not None and (s.empty or data_end > s[self.time_column].max()):
            # A non-transition row in the window holding the last instant
            # before data_end extends the windows to it.
            last_instant = pd.Series([data_end - pd.Timedelta(1, "ns")])
            tail = pd.DataFrame(
                {
                    self.time_column: floor_local(last_instant, window),
                    "to_on": [False],
                    "to_off": [False],
                }
            )
            tail[self.time_column] = tail[self.time_column].astype(
                s[self.time_column].dtype
            )
            s = pd.concat(
                [s[[self.time_column, "to_on", "to_off"]], tail], ignore_index=True
            )
        if s.empty:  # no known state
            return self._with_window(pd.DataFrame(columns=cols), window)
        s = s.set_index(self.time_column)

        resampled = s.resample(window).agg(
            on_count=("to_on", "sum"),
            off_count=("to_off", "sum"),
        )
        resampled["total_transitions"] = resampled["on_count"] + resampled["off_count"]

        result = resampled.reset_index().rename(columns={self.time_column: "start"})
        result["on_count"] = result["on_count"].astype(int)
        result["off_count"] = result["off_count"].astype(int)
        result["total_transitions"] = result["total_transitions"].astype(int)

        return self._with_window(result[cols], window)

    @keyword_only(1)
    @twin_columns(
        {
            "avg_on_duration": "avg_on_duration_seconds",
            "avg_off_duration": "avg_off_duration_seconds",
        }
    )
    def excessive_cycling(
        self, max_transitions: int = 20, window: str = "1h"
    ) -> pd.DataFrame:
        """Flag windows where transition count exceeds threshold.

        Excessive cycling indicates hunting, instability, or wear risk.

        Args:
            max_transitions: Threshold for flagging.
            window: Resample window.

        Returns:
            DataFrame with columns: start, end (start + window),
            duration_seconds, uuid, source_uuid, transition_count,
            avg_on_duration, avg_on_duration_seconds, avg_off_duration,
            avg_off_duration_seconds. The averages are in seconds.

            Canonical twin columns: ``avg_on_duration_seconds`` (after
            ``avg_on_duration``), ``avg_off_duration_seconds`` (after
            ``avg_off_duration``); each holds the same values as the
            deprecated column right before it, which is removed in the next
            major release.
        """
        cols = [
            "start",
            "transition_count",
            "avg_on_duration",
            "avg_off_duration",
        ]

        counts = self.cycle_count(window=window)
        if counts.empty:
            return self._with_window(pd.DataFrame(columns=cols), window)

        excessive = counts[counts["total_transitions"] >= max_transitions].copy()
        if excessive.empty:
            return self._with_window(pd.DataFrame(columns=cols), window)

        intervals = self.on_off_intervals()

        events: list[dict[str, Any]] = []
        for _, row in excessive.iterrows():
            # The window's real end (a DST day lasts 23 h or 25 h).
            w_start, w_end = row["start"], row["end"]

            win_intervals = intervals[
                (intervals["start"] >= w_start) & (intervals["start"] < w_end)
            ]

            on_intervals = win_intervals[win_intervals["state"] == "on"]
            off_intervals = win_intervals[win_intervals["state"] == "off"]

            avg_on = (
                on_intervals["duration"].mean().total_seconds()
                if not on_intervals.empty
                else 0.0
            )
            avg_off = (
                off_intervals["duration"].mean().total_seconds()
                if not off_intervals.empty
                else 0.0
            )

            events.append(
                {
                    "start": w_start,
                    "end": w_end,
                    "transition_count": int(row["total_transitions"]),
                    "avg_on_duration": avg_on,
                    "avg_off_duration": avg_off,
                }
            )

        # ``start`` / ``end`` are already the real bounds (from cycle_count).
        return self._with_window(
            pd.DataFrame(events, columns=[cols[0], "end", *cols[1:]]), window
        )

    def _with_window(self, df: pd.DataFrame, window: str) -> pd.DataFrame:
        """Insert end (start + window), duration_seconds, uuid and source_uuid.

        A frame that already has ``end`` holds real bounds and keeps them.
        """
        out = df.copy()
        if "end" in out.columns:
            span = pd.to_datetime(out["end"]) - pd.to_datetime(out["start"])
            seconds = span.dt.total_seconds() if len(out) else pd.Series(dtype=float)
            out.insert(2, "duration_seconds", seconds.astype("float64"))
        elif out.empty:
            out.insert(1, "end", pd.Series(dtype="object"))
            out.insert(2, "duration_seconds", pd.Series(dtype="float64"))
        else:
            start, end = window_bounds(out["start"], window)
            out["start"] = start
            out.insert(1, "end", end)
            out.insert(2, "duration_seconds", (out["end"] - start).dt.total_seconds())
        out.insert(3, "uuid", self.event_uuid)
        out.insert(4, "source_uuid", self.signal_uuid)
        return out
