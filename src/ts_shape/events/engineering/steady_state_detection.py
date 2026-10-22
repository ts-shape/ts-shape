import logging
from typing import Any, Literal, overload

import pandas as pd

from ts_shape.events._compat import summary_bounds, warn_dict_return
from ts_shape.utils.base import Base
from ts_shape.utils.deprecation import keyword_only
from ts_shape.utils.intervals import interval_end_times
from ts_shape.utils.runs import true_runs

logger = logging.getLogger(__name__)


class SteadyStateDetectionEvents(Base):
    """Engineering: Steady-State Detection

    Identify when a process signal has settled into a steady operating state
    (low variance, no trend) versus transient/dynamic periods.

    Methods:
    - detect_steady_state: Intervals where rolling std stays below threshold.
    - detect_transient_periods: Inverse — intervals where signal is changing.
    - steady_state_statistics: Summary of steady vs transient time (one row).
    - steady_state_value_bands: Operating band (mean +/- std) per steady interval.
    """

    #: Methods read only the constructor's signals (``Base`` keeps their rows).
    _reads_own_signals_only = True

    def __init__(
        self,
        dataframe: pd.DataFrame,
        signal_uuid: str,
        *,
        event_uuid: str = "eng:steady_state",
        value_column: str = "value_double",
        time_column: str = "systime",
    ) -> None:
        """Initialize the steady-state detector.

        Args:
            dataframe: Long-format timeseries frame (one row per sample, ``uuid``
                identifies the signal).
            signal_uuid: UUID of the signal to analyze.
            event_uuid: UUID stamped on emitted events.
            value_column: Column holding the signal values.
            time_column: Timestamp column.
        """
        super().__init__(dataframe, time_column=time_column)
        self.signal_uuid = signal_uuid
        self.event_uuid = event_uuid
        self.value_column = value_column
        self.time_column = time_column

        self.signal = (
            self.dataframe[self.dataframe["uuid"] == self.signal_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        self.signal[self.time_column] = pd.to_datetime(self.signal[self.time_column])

    def _rolling_std(self, window: str) -> pd.Series:
        """Compute rolling std indexed by time.

        A window with fewer than two samples has no std (NaN) and is neither
        steady nor transient. (It used to be filled with 0 and read as
        perfectly steady, so sparse data looked steady between samples.)
        """
        sig = self.signal.set_index(self.time_column)[self.value_column]
        td = pd.Timedelta(window)
        return sig.rolling(td, min_periods=2).std()

    def _intervalize(
        self,
        mask: pd.Series,
        min_duration: str = "0s",
        interval_end: str = "next_change",
    ) -> list[dict[str, Any]]:
        """Convert a boolean mask (indexed by timestamp) into intervals.

        Each interval also carries ``first`` / ``stop``: the positions of its
        flagged samples (``mask.iloc[first:stop]``), which exclude the
        closing sample a ``"next_change"`` end points at.
        """
        if mask.empty or not mask.any():
            return []

        min_td = pd.Timedelta(min_duration)
        ends = interval_end_times(pd.Series(mask.index), interval_end)
        runs = true_runs(mask.to_numpy(dtype=bool))
        starts = pd.Series(mask.index[runs.starts])
        stops = pd.Series(ends.array[runs.last])
        keep = ((stops - starts) >= min_td).to_numpy()
        return [
            {"start": a, "end": b, "first": int(f), "stop": int(e)}
            for a, b, f, e in zip(
                starts[keep],
                stops[keep],
                runs.starts[keep],
                runs.ends[keep],
                strict=True,
            )
        ]

    @keyword_only(1)
    def detect_steady_state(
        self,
        window: str = "5min",
        std_threshold: float = 1.0,
        min_duration: str = "10min",
        *,
        interval_end: str = "next_change",
    ) -> pd.DataFrame:
        """Detect intervals where signal is in steady state.

        Args:
            window: Rolling window for std computation.
            std_threshold: Maximum rolling std to consider steady.
            min_duration: Minimum duration of a steady interval.
            interval_end: ``"next_change"`` (default) ends an interval at the
                next sample (the transition), so the interval spans the time the
                condition held and change-encoded signals -- one row per change --
                keep their duration. ``"last_sample"`` ends it at its last flagged
                sample. See :func:`ts_shape.utils.intervals.interval_end_times`.

        Returns:
            DataFrame with columns: start, end, uuid, source_uuid, is_delta,
            mean_value, std_value, duration_seconds. ``mean_value`` and
            ``std_value`` (sample std; 0.0 for a single sample) are taken over
            the steady samples only -- the samples whose rolling std is below
            ``std_threshold`` -- not over the sample that ends the interval
            with ``interval_end="next_change"``.
        """
        cols = [
            "start",
            "end",
            "uuid",
            "source_uuid",
            "is_delta",
            "mean_value",
            "std_value",
            "duration_seconds",
        ]
        if self.signal.empty or len(self.signal) < 2:
            return pd.DataFrame(columns=cols)

        rolling_std = self._rolling_std(window)
        steady_mask = rolling_std < std_threshold
        intervals = self._intervalize(
            steady_mask, min_duration, interval_end=interval_end
        )

        if not intervals:
            return pd.DataFrame(columns=cols)

        values = self.signal[self.value_column].reset_index(drop=True)
        events: list[dict[str, Any]] = []
        for iv in intervals:
            # Only the steady samples: the closing (transition) sample that a
            # "next_change" end points at is not part of the steady state.
            segment = values.iloc[iv["first"] : iv["stop"]]
            events.append(
                {
                    "start": iv["start"],
                    "end": iv["end"],
                    "uuid": self.event_uuid,
                    "source_uuid": self.signal_uuid,
                    "is_delta": False,
                    "mean_value": float(segment.mean()),
                    "std_value": float(segment.std()) if len(segment) > 1 else 0.0,
                    "duration_seconds": (iv["end"] - iv["start"]).total_seconds(),
                }
            )

        return pd.DataFrame(events, columns=cols)

    @keyword_only(1)
    def detect_transient_periods(
        self,
        window: str = "5min",
        std_threshold: float = 1.0,
        *,
        interval_end: str = "next_change",
    ) -> pd.DataFrame:
        """Detect intervals where signal is in transient/dynamic state.

        Args:
            window: Rolling window for std computation.
            std_threshold: Minimum rolling std to consider transient.
            interval_end: ``"next_change"`` (default) ends an interval at the
                next sample (the transition), so the interval spans the time the
                condition held and change-encoded signals -- one row per change --
                keep their duration. ``"last_sample"`` ends it at its last flagged
                sample. See :func:`ts_shape.utils.intervals.interval_end_times`.

        Returns:
            DataFrame with columns: start, end, uuid, source_uuid, is_delta,
            max_std, duration_seconds.
        """
        cols = [
            "start",
            "end",
            "uuid",
            "source_uuid",
            "is_delta",
            "max_std",
            "duration_seconds",
        ]
        if self.signal.empty or len(self.signal) < 2:
            return pd.DataFrame(columns=cols)

        rolling_std = self._rolling_std(window)
        transient_mask = rolling_std >= std_threshold
        intervals = self._intervalize(transient_mask, interval_end=interval_end)

        if not intervals:
            return pd.DataFrame(columns=cols)

        events: list[dict[str, Any]] = []
        for iv in intervals:
            seg_std = rolling_std.loc[iv["start"] : iv["end"]]
            events.append(
                {
                    "start": iv["start"],
                    "end": iv["end"],
                    "uuid": self.event_uuid,
                    "source_uuid": self.signal_uuid,
                    "is_delta": False,
                    "max_std": float(seg_std.max()),
                    "duration_seconds": (iv["end"] - iv["start"]).total_seconds(),
                }
            )

        return pd.DataFrame(events, columns=cols)

    @overload
    def steady_state_statistics(
        self,
        window: str = ...,
        std_threshold: float = ...,
        min_duration: str = ...,
        *,
        as_frame: Literal[True] = ...,
    ) -> pd.DataFrame: ...

    @overload
    def steady_state_statistics(
        self,
        window: str = ...,
        std_threshold: float = ...,
        min_duration: str = ...,
        *,
        as_frame: Literal[False],
    ) -> dict[str, Any]: ...

    @keyword_only(1)
    def steady_state_statistics(
        self,
        window: str = "5min",
        std_threshold: float = 1.0,
        min_duration: str = "10min",
        *,
        as_frame: bool = True,
    ) -> pd.DataFrame | dict[str, Any]:
        """Summary statistics of steady vs transient time.

        Args:
            window: Rolling window for the std computation (pandas offset string),
                passed to :meth:`detect_steady_state`.
            std_threshold: Maximum rolling std, in signal units, to consider the
                signal steady.
            min_duration: Minimum duration of a steady interval.
            as_frame: When True (default), return a one-row DataFrame (see
                below). ``as_frame=False`` returns the dictionary form; it is
                deprecated, emits a ``DeprecationWarning`` and will be removed
                in the next major release.

        Returns:
            With ``as_frame=True`` (default), a one-row summary DataFrame with
            columns: start, end, duration_seconds (the span of the signal;
            NaT / 0 without data), uuid (event uuid), source_uuid (signal
            uuid), total_steady_seconds, total_transient_seconds, steady_pct,
            num_steady_periods, avg_steady_duration_seconds. ``steady_pct``
            is NaN when the signal has no time span (no data, one sample, or
            all samples at one instant) and ``avg_steady_duration_seconds``
            is NaN when there is no steady period to average.

            With ``as_frame=False`` (deprecated), a dict with the same keys
            from total_steady_seconds on.
        """
        if not as_frame:
            warn_dict_return("SteadyStateDetectionEvents.steady_state_statistics")
        stats = self._steady_state_stats(window, std_threshold, min_duration)
        if not as_frame:
            return stats
        start, end, span = summary_bounds(self.signal[self.time_column])
        return pd.DataFrame(
            [
                {
                    "start": start,
                    "end": end,
                    "duration_seconds": span,
                    "uuid": self.event_uuid,
                    "source_uuid": self.signal_uuid,
                    **stats,
                }
            ]
        )

    def _steady_state_stats(
        self, window: str, std_threshold: float, min_duration: str
    ) -> dict[str, Any]:
        if self.signal.empty or len(self.signal) < 2:
            # No span to take a share of: the ratios are undefined.
            return {
                "total_steady_seconds": 0.0,
                "total_transient_seconds": 0.0,
                "steady_pct": float("nan"),
                "num_steady_periods": 0,
                "avg_steady_duration_seconds": float("nan"),
            }

        steady = self.detect_steady_state(
            window=window, std_threshold=std_threshold, min_duration=min_duration
        )
        total_span = (
            self.signal[self.time_column].max() - self.signal[self.time_column].min()
        ).total_seconds()

        total_steady = (
            float(steady["duration_seconds"].sum()) if not steady.empty else 0.0
        )
        total_transient = max(total_span - total_steady, 0.0)
        n = len(steady)

        return {
            "total_steady_seconds": total_steady,
            "total_transient_seconds": total_transient,
            "steady_pct": (
                (total_steady / total_span * 100) if total_span > 0 else float("nan")
            ),
            "num_steady_periods": n,
            "avg_steady_duration_seconds": (
                (total_steady / n) if n > 0 else float("nan")
            ),
        }

    @keyword_only(1)
    def steady_state_value_bands(
        self,
        window: str = "5min",
        std_threshold: float = 1.0,
        min_duration: str = "10min",
    ) -> pd.DataFrame:
        """Operating band (mean +/- std) for each steady-state interval.

        Args:
            window: Rolling window for the std computation (pandas offset string),
                passed to :meth:`detect_steady_state`.
            std_threshold: Maximum rolling std, in signal units, to consider the
                signal steady.
            min_duration: Minimum duration of a steady interval.

        Returns:
            DataFrame with columns: start, end, mean_value, lower_band,
            upper_band, duration_seconds.
        """
        cols = [
            "start",
            "end",
            "mean_value",
            "lower_band",
            "upper_band",
            "duration_seconds",
        ]
        steady = self.detect_steady_state(
            window=window, std_threshold=std_threshold, min_duration=min_duration
        )
        if steady.empty:
            return pd.DataFrame(columns=cols)

        result = steady[["start", "end", "mean_value", "duration_seconds"]].copy()
        result["lower_band"] = steady["mean_value"] - steady["std_value"]
        result["upper_band"] = steady["mean_value"] + steady["std_value"]
        return result[cols].reset_index(drop=True)
