import logging
from collections.abc import Mapping
from typing import Any, Literal, overload

import numpy as np
import pandas as pd

from ts_shape.events._compat import summary_bounds, warn_dict_return
from ts_shape.signals.primitives import intervals as spec_intervals
from ts_shape.signals.spec import SignalSpec
from ts_shape.utils.base import Base
from ts_shape.utils.deprecation import keyword_only
from ts_shape.utils.intervals import state_intervals

logger = logging.getLogger(__name__)


class MachineStateEvents(Base):
    """Production: Machine State

    Detect run/idle transitions and intervals from a boolean state signal.

    Classes:
    - MachineStateEvents: Run/idle state intervals and transitions.
      - detect_run_idle: Intervalize run/idle states with optional min duration filter.
      - transition_events: Point events on state changes (idle->run, run->idle).
    """

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

        Uses value_range when set (inclusive on both ends); otherwise casts to bool.
        """
        if self.value_range is not None:
            lower, upper = self.value_range
            mask = pd.Series(True, index=col.index)
            if lower is not None:
                mask &= col >= lower
            if upper is not None:
                mask &= col <= upper
            return mask
        return col.fillna(False).astype(bool)

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
        s["state"] = self._as_state(s[self.value_column])
        if interval_end is None and self.spec is not None:
            # Spec-driven: the primitive closes held states at the next change
            # and splits at gaps longer than max_gap.
            run_spec = self.spec.with_(value_column="value_bool")
            frame = pd.DataFrame(
                {
                    "systime": s[self.time_column].to_numpy(),
                    "uuid": run_spec.uuid,
                    "value_bool": s["state"].to_numpy(),
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
        intervals = intervals[keep]
        if intervals.empty:
            return pd.DataFrame(columns=columns)
        result = pd.DataFrame(
            {
                "start": intervals["start"].to_numpy(),
                "end": intervals["end"].to_numpy(),
                "uuid": self.event_uuid,
                "source_uuid": self.run_state_uuid,
                "is_delta": True,
                "state": ["run" if bool(st) else "idle" for st in intervals["state"]],
                "duration_seconds": intervals["duration_seconds"].to_numpy(),
            }
        )
        return result

    def transition_events(self) -> pd.DataFrame:
        """Return point events at state transitions.

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
        s["state"] = self._as_state(s[self.value_column])
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
        s["state"] = self._as_state(s[self.value_column])
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
        rapid_events: list[dict[str, Any]] = []

        for i in range(len(transitions) - min_count + 1):
            start = transitions.iloc[i]["systime"]
            for j in range(i + min_count - 1, len(transitions)):
                end = transitions.iloc[j]["systime"]
                duration = end - start
                if duration <= threshold_td:
                    transition_count = j - i + 1
                    rapid_events.append(
                        {
                            "start": start,
                            "end": end,
                            "transition_count": transition_count,
                            "duration_seconds": duration.total_seconds(),
                        }
                    )
                else:
                    break

        if not rapid_events:
            return pd.DataFrame(columns=cols)
        out = pd.DataFrame(rapid_events)
        out.insert(0, "systime", out["start"])
        out.insert(1, "uuid", self.event_uuid)
        out.insert(2, "source_uuid", self.run_state_uuid)
        return out

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

            With ``as_frame=False`` (deprecated), a dictionary with:

            - total_transitions: total number of state transitions
            - avg_run_duration: average duration of run states in seconds
            - avg_idle_duration: average duration of idle states in seconds
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

        if intervals.empty:
            avg_run_duration = 0.0
            avg_idle_duration = 0.0
            run_idle_ratio = float("nan")
        else:
            run_intervals = intervals[intervals["state"] == "run"]
            idle_intervals = intervals[intervals["state"] == "idle"]

            avg_run_duration = (
                run_intervals["duration_seconds"].mean()
                if not run_intervals.empty
                else 0.0
            )
            avg_idle_duration = (
                idle_intervals["duration_seconds"].mean()
                if not idle_intervals.empty
                else 0.0
            )

            total_run_time = (
                run_intervals["duration_seconds"].sum()
                if not run_intervals.empty
                else 0.0
            )
            total_idle_time = (
                idle_intervals["duration_seconds"].sum()
                if not idle_intervals.empty
                else 0.0
            )
            run_idle_ratio = (
                total_run_time / total_idle_time
                if total_idle_time > 0
                else float("nan")
            )

        data_gaps_detected = self._detect_data_gaps()
        rapid_transitions_detected = len(self.detect_rapid_transitions())

        metrics = {
            "total_transitions": total_transitions,
            "avg_run_duration": (
                float(avg_run_duration) if not np.isnan(avg_run_duration) else 0.0
            ),
            "avg_idle_duration": (
                float(avg_idle_duration) if not np.isnan(avg_idle_duration) else 0.0
            ),
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
