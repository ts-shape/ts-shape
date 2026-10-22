import logging
from collections.abc import Mapping
from typing import Any, Literal, overload

import numpy as np
import pandas as pd

from ts_shape.events._compat import summary_bounds, warn_dict_return
from ts_shape.events._output import twin_columns
from ts_shape.signals.primitives import intervals as spec_intervals
from ts_shape.signals.spec import SignalSpec
from ts_shape.utils.base import Base
from ts_shape.utils.coerce import bool_or_false
from ts_shape.utils.deprecation import keyword_only
from ts_shape.utils.intervals import state_intervals
from ts_shape.utils.runs import total_seconds

logger = logging.getLogger(__name__)

#: State code of a row whose run/idle value is missing (unknown time).
_UNKNOWN = -1


class MachineStateEvents(Base):
    """Production: Machine State

    Detect run/idle transitions and intervals from a boolean state signal.

    Classes:
    - MachineStateEvents: Run/idle state intervals and transitions.
      - detect_run_idle: Intervalize run/idle states with optional min duration filter.
      - transition_events: Point events on state changes (idle->run, run->idle).

    Example:
        >>> import pandas as pd, ts_shape
        >>> t = pd.date_range("2025-01-01 08:00", periods=6, freq="10min")
        >>> df = pd.DataFrame({"systime": t, "uuid": "run",
        ...                    "value_bool": [True, True, False, True, True, False]})
        >>> out = ts_shape.MachineStateEvents(df, "run").detect_run_idle()
        >>> out[["start", "state", "duration_seconds"]]
                        start state  duration_seconds
        0 2025-01-01 08:00:00   run            1200.0
        1 2025-01-01 08:20:00  idle             600.0
        2 2025-01-01 08:30:00   run            1200.0
        3 2025-01-01 08:50:00  idle               0.0
    """

    #: Methods read only the constructor's signals (``Base`` keeps their rows).
    _reads_own_signals_only = True

    def __init__(
        self,
        dataframe: pd.DataFrame,
        run_state_uuid: str,
        *,
        event_uuid: str = "prod:run_idle",
        value_column: str = "value_bool",
        time_column: str = "systime",
        value_range: tuple[float | None, float | None] | None = None,
        specs: Mapping[str, SignalSpec] | None = None,
    ) -> None:
        """Initialize the machine state detector.

        Args:
            dataframe: Long-format timeseries frame (one row per sample, ``uuid``
                identifies the signal).
            run_state_uuid: UUID of the run-state signal; must be present in a non-empty
                ``dataframe``.
            event_uuid: UUID stamped on emitted events.
            value_column: Column holding the run state.
            time_column: Timestamp column.
            value_range: ``(lower, upper)`` inclusive bounds; a value inside counts as
                running (either bound may be ``None``). ``None`` (default) casts the
                value to bool.
            specs: Optional signal semantics (see :mod:`ts_shape.signals`). When the
                run-state signal has a spec, interval closing and gap handling follow
                it.
        """
        super().__init__(dataframe, time_column=time_column)
        self.run_state_uuid = run_state_uuid
        # Optional signal semantics (see ts_shape.signals). When the run-state
        # signal has a spec, interval closing and gap handling follow it.
        self.spec: SignalSpec | None = (specs or {}).get(run_state_uuid)
        self.event_uuid = event_uuid
        self.value_column = value_column
        self.time_column = time_column
        self.value_range = value_range
        # An empty frame has no signals at all: return empty results rather
        # than raising. A non-empty frame without the uuid is a caller error.
        if (
            "uuid" in self.dataframe.columns
            and not self.dataframe.empty
            and run_state_uuid not in self.dataframe["uuid"].values
        ):
            raise ValueError(
                f"UUID '{run_state_uuid}' not found in dataframe. "
                f"Available UUIDs: {list(self.dataframe['uuid'].unique())}"
            )
        self.series = (
            self.dataframe[self.dataframe["uuid"] == self.run_state_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        self.series[self.time_column] = pd.to_datetime(self.series[self.time_column])
        self._state_groups: pd.Series | None = None
        self._compute_state_groups()

    def _as_state(self, col: pd.Series) -> pd.Series:
        """Convert a value series to a boolean running/idle state.

        Uses value_range when set (inclusive on both ends); otherwise casts to
        bool. A missing value (``None``, ``NaN``, ``pd.NA``) is unknown, not
        idle: it is ``pd.NA`` in the returned nullable ``boolean`` Series.
        """
        known = col.notna()
        if self.value_range is not None:
            lower, upper = self.value_range
            values = pd.to_numeric(col, errors="coerce")
            mask = pd.Series(True, index=col.index)
            if lower is not None:
                mask &= (values >= lower).fillna(False).astype(bool)
            if upper is not None:
                mask &= (values <= upper).fillna(False).astype(bool)
            state = mask
        else:
            state = bool_or_false(col)
        return state.astype("boolean").where(known, pd.NA)

    def _state_codes(self, col: pd.Series) -> pd.Series:
        """``1`` run, ``0`` idle, ``-1`` unknown (missing value), as ``int64``."""
        state = self._as_state(col)
        codes = state.astype("Int64").fillna(_UNKNOWN)
        return codes.astype("int64")

    @keyword_only(1)
    def detect_run_idle(
        self,
        min_duration: str = "0s",
        *,
        interval_end: str | None = None,
        data_end: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Return intervals labeled as 'run' or 'idle'.

        - min_duration: discard intervals shorter than this duration.
        - interval_end: how to close each interval. ``"next_change"`` closes
          each interval at the next transition instant, which measures both
          dense and change-encoded (``is_delta``) signals. ``"last_sample"``
          ends an interval at its last sample; on a change-encoded signal
          (one row per transition) every interval then has zero duration.
          Default (``None``): follow the signal's spec when ``specs=`` was
          given (held states close at the next change, and gaps longer than
          the spec's ``max_gap`` are "no data"), otherwise ``"next_change"``.
        - data_end: optional bound for the final, still-open interval when
          intervals close at the next change (e.g. the end of the data window).
        - A missing state value (``None``, ``NaN``, ``pd.NA``) is unknown, not
          idle: the time from such a row to the next known row is in neither a
          run nor an idle interval (like ReliabilityEvents' "no data").
        Columns: start, end, uuid, source_uuid, is_delta, state, duration_seconds
        """
        columns = [
            "start",
            "end",
            "uuid",
            "source_uuid",
            "is_delta",
            "state",
            "duration_seconds",
        ]
        if self.series.empty:
            return pd.DataFrame(columns=columns)
        s = self.series[[self.time_column, self.value_column]].copy()
        s["state"] = self._state_codes(s[self.value_column])
        if interval_end is None and self.spec is not None:
            # Spec-driven: the primitive closes held states at the next change
            # and splits at gaps longer than max_gap.
            run_spec = self.spec.with_(value_column="value_integer")
            frame = pd.DataFrame(
                {
                    "systime": s[self.time_column].to_numpy(),
                    "uuid": run_spec.uuid,
                    "value_integer": s["state"].to_numpy(),
                }
            )
            intervals = spec_intervals(frame, run_spec, data_end=data_end)
        else:
            intervals = state_intervals(
                s,
                time_column=self.time_column,
                state_column="state",
                interval_end=interval_end or "next_change",
                data_end=data_end,
            )
        min_td = pd.to_timedelta(min_duration)
        keep = pd.to_timedelta(intervals["duration_seconds"], unit="s") >= min_td
        intervals = intervals[keep & (intervals["state"] != _UNKNOWN).to_numpy()]
        if intervals.empty:
            return pd.DataFrame(columns=columns)
        result = pd.DataFrame(
            {
                "start": intervals["start"].to_numpy(),
                "end": intervals["end"].to_numpy(),
                "uuid": self.event_uuid,
                "source_uuid": self.run_state_uuid,
                "is_delta": True,
                "state": ["run" if st == 1 else "idle" for st in intervals["state"]],
                "duration_seconds": intervals["duration_seconds"].to_numpy(),
            }
        )
        return result

    def transition_events(self) -> pd.DataFrame:
        """Return point events at state transitions.

        Rows with a missing state are skipped: a transition is a change
        between two known states.

        Columns: systime, uuid, source_uuid, is_delta, transition ('idle_to_run'|'run_to_idle'), time_since_last_transition_seconds
        """
        if self.series.empty:
            return pd.DataFrame(
                columns=[
                    "systime",
                    "uuid",
                    "source_uuid",
                    "is_delta",
                    "transition",
                    "time_since_last_transition_seconds",
                ]
            )
        s = self.series[[self.time_column, self.value_column]].copy()
        state = self._as_state(s[self.value_column])
        s = s[state.notna().to_numpy()]
        s["state"] = state[state.notna()].astype(bool)
        s["prev"] = s["state"].shift()
        changes = s[s["state"] != s["prev"]].dropna(subset=["prev"])  # ignore first row
        if changes.empty:
            return pd.DataFrame(
                columns=[
                    "systime",
                    "uuid",
                    "source_uuid",
                    "is_delta",
                    "transition",
                    "time_since_last_transition_seconds",
                ]
            )
        changes = changes.rename(columns={self.time_column: "systime"})
        changes["transition"] = np.where(
            changes["state"] & ~changes["prev"].astype(bool),
            "idle_to_run",
            "run_to_idle",
        )
        changes["time_since_last_transition_seconds"] = (
            changes["systime"].diff().dt.total_seconds()
        )
        return pd.DataFrame(
            {
                "systime": changes["systime"],
                "uuid": self.event_uuid,
                "source_uuid": self.run_state_uuid,
                "is_delta": True,
                "transition": changes["transition"],
                "time_since_last_transition_seconds": changes[
                    "time_since_last_transition_seconds"
                ],
            }
        )

    def _compute_state_groups(self) -> None:
        """Compute and cache state change groups for performance."""
        if self.series.empty:
            self._state_groups = None
            return
        s = self.series[[self.time_column, self.value_column]].copy()
        s["state"] = self._state_codes(s[self.value_column])
        self._state_groups = (s["state"] != s["state"].shift()).cumsum()

    @keyword_only(1)
    def detect_rapid_transitions(
        self, threshold: str = "5s", min_count: int = 3
    ) -> pd.DataFrame:
        """Identify suspicious rapid state changes.

        - threshold: time window to look for rapid transitions
        - min_count: minimum number of transitions within threshold to be considered rapid
        Returns: DataFrame with systime (= start, the first transition of the
        burst), uuid (event uuid), source_uuid (run-state uuid), start, end,
        transition_count, duration_seconds
        """
        cols = [
            "systime",
            "uuid",
            "source_uuid",
            "start",
            "end",
            "transition_count",
            "duration_seconds",
        ]
        transitions = self.transition_events()
        if transitions.empty or len(transitions) < min_count:
            return pd.DataFrame(columns=cols)

        threshold_td = pd.to_timedelta(threshold)
        times = transitions["systime"].reset_index(drop=True)
        if min_count < 1 or bool(times.isna().any()):
            rapid_events = self._rapid_transitions_loop(
                transitions, threshold_td, min_count
            )
        else:
            # Every (i, j) with j >= i + min_count - 1 and t_j - t_i <=
            # threshold, in (i, j) order; times are sorted, so the j for one i
            # run up to the first later transition beyond the threshold.
            n = len(times)
            first_j = np.arange(n, dtype=np.int64) + min_count - 1
            stop_j = times.searchsorted(times + threshold_td, side="right")
            count = np.clip(stop_j - first_j, 0, None)
            count[first_j >= n] = 0
            i = np.repeat(np.arange(n, dtype=np.int64), count)
            offset = np.arange(i.size, dtype=np.int64) - np.repeat(
                np.cumsum(count) - count, count
            )
            j = first_j[i] + offset
            t = times.array
            start, end = t[i], t[j]
            rapid_events = pd.DataFrame(
                {
                    "start": start,
                    "end": end,
                    "transition_count": j - i + 1,
                    "duration_seconds": total_seconds(end - start),
                }
            )

        if len(rapid_events) == 0:
            return pd.DataFrame(columns=cols)
        out = pd.DataFrame(rapid_events)
        out.insert(0, "systime", out["start"])
        out.insert(1, "uuid", self.event_uuid)
        out.insert(2, "source_uuid", self.run_state_uuid)
        return out

    @staticmethod
    def _rapid_transitions_loop(
        transitions: pd.DataFrame, threshold_td: pd.Timedelta, min_count: int
    ) -> list[dict[str, Any]]:
        """Reference pairwise scan (a missing transition time, min_count < 1)."""
        rapid_events: list[dict[str, Any]] = []
        for i in range(len(transitions) - min_count + 1):
            start = transitions.iloc[i]["systime"]
            for j in range(i + min_count - 1, len(transitions)):
                end = transitions.iloc[j]["systime"]
                duration = end - start
                if duration <= threshold_td:
                    rapid_events.append(
                        {
                            "start": start,
                            "end": end,
                            "transition_count": j - i + 1,
                            "duration_seconds": duration.total_seconds(),
                        }
                    )
                else:
                    break
        return rapid_events

    def _detect_data_gaps(self, max_gap: str = "5min") -> int:
        """Detect missing data gaps in the time series.

        - max_gap: maximum acceptable gap between data points
        Returns: count of data gaps detected
        """
        if self.series.empty or len(self.series) < 2:
            return 0

        max_gap_td = pd.to_timedelta(max_gap)
        time_diffs = self.series[self.time_column].diff()
        gaps = time_diffs[time_diffs > max_gap_td]
        return len(gaps)

    @overload
    def state_quality_metrics(
        self,
        *,
        interval_end: str | None = ...,
        data_end: pd.Timestamp | None = ...,
        as_frame: Literal[True] = ...,
    ) -> pd.DataFrame: ...

    @overload
    def state_quality_metrics(
        self,
        *,
        interval_end: str | None = ...,
        data_end: pd.Timestamp | None = ...,
        as_frame: Literal[False],
    ) -> dict[str, Any]: ...

    @twin_columns(
        {
            "avg_run_duration": "avg_run_duration_seconds",
            "avg_idle_duration": "avg_idle_duration_seconds",
            "run_idle_ratio": "run_idle_factor",
        }
    )
    def state_quality_metrics(
        self,
        *,
        interval_end: str | None = None,
        data_end: pd.Timestamp | None = None,
        as_frame: bool = True,
    ) -> pd.DataFrame | dict[str, Any]:
        """Return quality metrics for the state data.

        Args:
            interval_end: Forwarded to :meth:`detect_run_idle`. Default
                ``None`` resolves exactly as in :meth:`detect_run_idle`:
                follow the signal's spec when ``specs=`` was given, otherwise
                ``"next_change"`` (run/idle durations are measured to the next
                transition).
            data_end: Forwarded to :meth:`detect_run_idle`; optional bound for
                the final, still-open interval.
            as_frame: When True (default), return a one-row DataFrame (see
                below). ``as_frame=False`` returns the dictionary form; it is
                deprecated, emits a ``DeprecationWarning`` and will be removed
                in the next major release.

        Returns:
            With ``as_frame=True`` (default), a one-row summary DataFrame with
            columns start, end, duration_seconds (the span of the state
            signal; NaT / 0 without data), uuid (event uuid), source_uuid
            (run-state uuid), followed by the metrics below.

            Canonical twin columns: ``avg_run_duration_seconds`` (after
            ``avg_run_duration``), ``avg_idle_duration_seconds`` (after
            ``avg_idle_duration``), ``run_idle_factor`` (after
            ``run_idle_ratio``, which is not a 0-1 fraction); each holds the
            same values as the deprecated column right before it, which is
            removed in the next major release. The dictionary form keeps the
            old keys.

            With ``as_frame=False`` (deprecated), a dictionary with:

            - total_transitions: total number of state transitions
            - avg_run_duration: average duration of run states in seconds;
                NaN when there is no run interval
            - avg_idle_duration: average duration of idle states in seconds;
                NaN when there is no idle interval
            Missing state values are unknown time, in neither average nor the
            ratio.
            - run_idle_ratio: ratio of total run time to total idle time;
                NaN when there is no idle time (including no intervals at all),
                since the ratio is undefined
            - data_gaps_detected: number of data gaps found
            - rapid_transitions_detected: number of rapid transition events
        """
        if not as_frame:
            warn_dict_return("MachineStateEvents.state_quality_metrics")
        transitions = self.transition_events()
        intervals = self.detect_run_idle(interval_end=interval_end, data_end=data_end)

        total_transitions = len(transitions)

        # Durations are of observed (known-state) time only; a mean over no
        # intervals is undefined (NaN), not 0.
        run_seconds = intervals.loc[intervals["state"] == "run", "duration_seconds"]
        idle_seconds = intervals.loc[intervals["state"] == "idle", "duration_seconds"]
        avg_run_duration = float(run_seconds.mean()) if len(run_seconds) else np.nan
        avg_idle_duration = float(idle_seconds.mean()) if len(idle_seconds) else np.nan
        total_run_time = float(run_seconds.sum())
        total_idle_time = float(idle_seconds.sum())
        run_idle_ratio = (
            total_run_time / total_idle_time if total_idle_time > 0 else float("nan")
        )

        data_gaps_detected = self._detect_data_gaps()
        rapid_transitions_detected = len(self.detect_rapid_transitions())

        metrics = {
            "total_transitions": total_transitions,
            "avg_run_duration": avg_run_duration,
            "avg_idle_duration": avg_idle_duration,
            "run_idle_ratio": float(run_idle_ratio),
            "data_gaps_detected": data_gaps_detected,
            "rapid_transitions_detected": rapid_transitions_detected,
        }
        if not as_frame:
            return metrics
        start, end, span = summary_bounds(self.series[self.time_column])
        return pd.DataFrame(
            [
                {
                    "start": start,
                    "end": end,
                    "duration_seconds": span,
                    "uuid": self.event_uuid,
                    "source_uuid": self.run_state_uuid,
                    **metrics,
                }
            ]
        )
