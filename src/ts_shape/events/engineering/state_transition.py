"""Engineering: State transition analysis.

Where :class:`~ts_shape.events.production.machine_state.MachineStateEvents`
measures *how long* each state lasts, :class:`StateTransitionEvents` measures
*how a state signal moves between states* — which transitions happen, how
often, and how long each state is held before leaving. Built directly on
:func:`ts_shape.utils.intervals.state_intervals`, so dense and change-encoded
(``is_delta=True``) signals both measure correctly.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from ts_shape.events._output import empty_event_df, finalize_point_df
from ts_shape.utils.base import Base
from ts_shape.utils.coerce import bool_or_false
from ts_shape.utils.intervals import state_intervals

logger = logging.getLogger(__name__)

_MATRIX_COLS = [
    "from_state",
    "to_state",
    "sample_count",
    "total_dwell_seconds",
    "mean_dwell_seconds",
    "share_of_transitions_pct",
]
_TRANSITION_COLS = [
    "systime",
    "uuid",
    "source_uuid",
    "from_state",
    "to_state",
    "dwell_seconds",
]


class StateTransitionEvents(Base):
    """Engineering: State transition matrix and per-transition events.

    Methods:
    - transitions: Point events, one per transition (``systime`` = the
      transition instant, carrying ``from_state`` / ``to_state`` and the
      dwell time in the state being left).
    - transition_matrix: ``from_state`` × ``to_state`` summary with
      ``sample_count``, ``total_dwell_seconds``, ``mean_dwell_seconds`` and
      ``share_of_transitions_pct``.
    - hold_intervals: The underlying collapsed intervals
      (``start`` / ``end`` / ``state`` / ``duration_seconds``) used by
      the other methods.

    Example:
        >>> import pandas as pd, ts_shape
        >>> df = ts_shape.make_id_signal(
        ...     "state",
        ...     ["Idle", "Run", "Idle", "Alarm", "Idle"],
        ...     hold=10, freq="1min",
        ... )
        >>> ev = ts_shape.StateTransitionEvents(df, "state")
        >>> ev.transition_matrix()[["from_state", "to_state", "sample_count"]]
          from_state to_state  sample_count
        0      Alarm     Idle             1
        1       Idle    Alarm             1
        2       Idle      Run             1
        3        Run     Idle             1
    """

    #: Methods read only the constructor's signals (``Base`` keeps their rows).
    _reads_own_signals_only = True

    def __init__(
        self,
        dataframe: pd.DataFrame,
        run_state_uuid: str,
        *,
        event_uuid: str = "eng:state_transition",
        value_column: str = "value_string",
        time_column: str = "systime",
        interval_end: str = "next_change",
        data_end: pd.Timestamp | None = None,
    ) -> None:
        """Initialize the state transition analyzer.

        Args:
            dataframe: Long-format timeseries frame.
            run_state_uuid: UUID of the state signal (any hashable state works
                — strings, booleans, integer run states). Named with the
                ``run_state_uuid`` idiom (docs/naming.md) because the class is
                most often used on a run/idle or machine-state signal; any
                state signal is accepted.
            event_uuid: UUID stamped on emitted transition events.
            value_column: Column holding the state values. The default
                ``value_string`` matches a text state signal; a boolean
                run/idle signal typically uses ``value_bool``.
            time_column: Timestamp column.
            interval_end: How each state's hold interval is closed; see
                :func:`ts_shape.utils.intervals.state_intervals`.
                ``"next_change"`` (default) is correct for both dense and
                change-encoded signals.
            data_end: Optional bound for the final still-open interval when
                ``interval_end="next_change"``.
        """
        super().__init__(dataframe, time_column=time_column)
        self.run_state_uuid = run_state_uuid
        self.event_uuid = event_uuid
        self.value_column = value_column
        self.time_column = time_column
        self.interval_end = interval_end
        self.data_end = data_end

    def _intervals(self) -> pd.DataFrame:
        """Collapse the state signal into one row per held interval."""
        if "uuid" in self.dataframe.columns:
            series = self.dataframe[
                self.dataframe["uuid"] == self.run_state_uuid
            ].copy()
        else:
            series = self.dataframe.iloc[0:0].copy()
        if series.empty or self.value_column not in series.columns:
            return pd.DataFrame(columns=["start", "end", "state", "duration_seconds"])
        series = series.sort_values(self.time_column)
        series[self.time_column] = pd.to_datetime(series[self.time_column])
        # Keep the state label as given (string / int / bool); use ``bool_or_false``
        # only for a boolean run signal where we want falsy values collapsed.
        state_series = series[self.value_column]
        if state_series.dtype == bool or self.value_column == "value_bool":
            state_series = bool_or_false(state_series)
        prepared = pd.DataFrame(
            {
                self.time_column: series[self.time_column].to_numpy(),
                "state": state_series.to_numpy(),
            }
        )
        return state_intervals(
            prepared,
            time_column=self.time_column,
            state_column="state",
            interval_end=self.interval_end,
            data_end=self.data_end,
        )

    def transitions(self) -> pd.DataFrame:
        """One point event per transition between states.

        Returns:
            DataFrame with the canonical point columns plus
            ``from_state``, ``to_state`` and ``dwell_seconds`` (how long the
            state being left was held). Transitions where the state does not
            actually change are dropped. The event's ``systime`` is the
            transition instant.
        """
        intervals = self._intervals()
        if intervals.empty or len(intervals) < 2:
            return empty_event_df(
                "point", extra_cols=["from_state", "to_state", "dwell_seconds"]
            )
        from_state = intervals["state"].iloc[:-1].to_numpy()
        to_state = intervals["state"].iloc[1:].to_numpy()
        changes = from_state != to_state
        if not changes.any():
            return empty_event_df(
                "point", extra_cols=["from_state", "to_state", "dwell_seconds"]
            )
        instants = intervals["end"].iloc[:-1].to_numpy()
        dwell = intervals["duration_seconds"].iloc[:-1].to_numpy(dtype=float)
        frame = pd.DataFrame(
            {
                "systime": instants,
                "from_state": from_state,
                "to_state": to_state,
                "dwell_seconds": dwell,
            }
        ).loc[changes].reset_index(drop=True)
        return finalize_point_df(
            frame, uuid=self.event_uuid, source_uuid=self.run_state_uuid
        )

    def transition_matrix(self) -> pd.DataFrame:
        """Count and dwell per ``(from_state, to_state)`` pair.

        Returns:
            DataFrame with columns (:data:`_MATRIX_COLS`):

            - ``from_state`` / ``to_state``.
            - ``sample_count``: how many transitions of this kind occurred.
            - ``total_dwell_seconds``: total time spent in ``from_state``
              across all its transitions to ``to_state``.
            - ``mean_dwell_seconds``: ``total_dwell_seconds / sample_count``.
            - ``share_of_transitions_pct``: percentage of all transitions.

            Rows sorted ``from_state`` ascending, then ``to_state`` ascending.
            Empty frame when the signal never changes state.
        """
        events = self.transitions()
        if events.empty:
            return pd.DataFrame(
                columns=_MATRIX_COLS,
            )
        grouped = (
            events.groupby(["from_state", "to_state"], dropna=False)["dwell_seconds"]
            .agg([("sample_count", "size"), ("total_dwell_seconds", "sum")])
            .reset_index()
        )
        grouped["mean_dwell_seconds"] = (
            grouped["total_dwell_seconds"] / grouped["sample_count"]
        )
        total = int(grouped["sample_count"].sum())
        grouped["share_of_transitions_pct"] = (
            grouped["sample_count"].astype(float) / total * 100.0 if total else 0.0
        )
        return grouped.sort_values(
            ["from_state", "to_state"], kind="stable"
        ).reset_index(drop=True)[_MATRIX_COLS]

    # Documented helpers that may evolve: expose the raw intervals view.
    def hold_intervals(self) -> pd.DataFrame:
        """The underlying collapsed intervals (``start``, ``end``, ``state``, ...)."""
        return self._intervals()


__all__: list[str] = ["StateTransitionEvents"]
