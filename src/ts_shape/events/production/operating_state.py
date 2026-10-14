import logging
from typing import Any

import numpy as np
import pandas as pd

from ts_shape.utils.base import Base
from ts_shape.utils.deprecation import deprecated_method, keyword_only

logger = logging.getLogger(__name__)


class OperatingStateEvents(Base):
    """Production: Operating State Inference

    Infer discrete operating states from continuous signals (vibration,
    current, power, temperature) when no explicit boolean state signal
    is available.  Answers the question "what state is the machine in?"

    Unlike :class:`MachineStateEvents` which requires a boolean run/idle
    signal, this class clusters continuous values to discover states
    automatically.

    Methods:
    - infer_states: Cluster the signal into N operating states.
    - state_duration_statistics: How long does each state typically last?
    - unusual_state_sequence: Did states occur in an unexpected order?
    - dwell_time_violation: Did the machine stay in a state too long?
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        signal_uuid: str,
        *,
        event_uuid: str = "prod:operating_state",
        value_column: str = "value_double",
        time_column: str = "systime",
    ) -> None:
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

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    @keyword_only(1)
    def infer_states(
        self,
        n_states: int = 2,
        min_duration: str = "0s",
    ) -> pd.DataFrame:
        """Cluster the signal into N operating states.

        Uses quantile-based binning to assign each sample to a state,
        then intervalizes consecutive same-state runs.

        Args:
            n_states: Number of states to infer (2 = low/high,
                3 = low/mid/high, etc.).
            min_duration: Discard intervals shorter than this.

        Returns:
            DataFrame with columns: start, end, state, state_label,
            mean_value, duration_seconds, uuid (event uuid), source_uuid
            (signal uuid). Samples without a value are ignored; with fewer
            valued samples than ``n_states`` the result is empty.
        """
        cols = [
            "start",
            "end",
            "state",
            "state_label",
            "mean_value",
            "duration_seconds",
            "uuid",
            "source_uuid",
        ]
        # Samples without a value carry no state; an all-NaN signal has none.
        sig = (
            self.signal[[self.time_column, self.value_column]]
            .dropna(subset=[self.value_column])
            .reset_index(drop=True)
        )
        if sig.empty or len(sig) < n_states:
            return pd.DataFrame(columns=cols)

        values = sig[self.value_column].values

        # Quantile-based binning
        bins = self._quantile_bins(values, n_states)
        sig["state"] = np.digitize(values, bins)
        sig["state"] = sig["state"].clip(0, n_states - 1)

        # Generate labels
        labels = self._state_labels(n_states)

        # Intervalize
        min_td = pd.to_timedelta(min_duration)
        state_change = (sig["state"] != sig["state"].shift()).cumsum()

        # One row per run of equal states: positions of each run's first and
        # last sample (runs are contiguous, in time order).
        first_pos = np.flatnonzero(state_change.diff().fillna(1).to_numpy() != 0)
        last_pos = np.r_[first_pos[1:] - 1, len(sig) - 1]
        times = sig[self.time_column]
        start = times.iloc[first_pos].reset_index(drop=True)
        end = times.iloc[last_pos].reset_index(drop=True)
        duration = end - start
        keep = ~(duration < min_td).to_numpy()
        if not keep.any():
            return pd.DataFrame(columns=cols)
        state_ids = sig["state"].iloc[first_pos[keep]].astype("int64")
        means = sig.groupby(state_change, sort=True)[self.value_column].mean()
        return pd.DataFrame(
            {
                "start": start[keep].reset_index(drop=True),
                "end": end[keep].reset_index(drop=True),
                "state": state_ids.to_numpy(),
                "state_label": [labels[i] for i in state_ids.tolist()],
                "mean_value": means.to_numpy(dtype="float64")[keep],
                "duration_seconds": duration[keep].dt.total_seconds().to_numpy(),
                "uuid": self.event_uuid,
                "source_uuid": self.signal_uuid,
            },
            columns=cols,
        )

    state_duration_stats = deprecated_method("state_duration_statistics")

    @keyword_only(1)
    def state_duration_statistics(self, n_states: int = 2) -> pd.DataFrame:
        """How long does each inferred state typically last?

        Args:
            n_states: Number of states to infer.

        Returns:
            DataFrame with columns: state, state_label, count,
            mean_duration, std_duration, min_duration, max_duration,
            total_duration.
        """
        cols = [
            "state",
            "state_label",
            "count",
            "mean_duration",
            "std_duration",
            "min_duration",
            "max_duration",
            "total_duration",
        ]
        intervals = self.infer_states(n_states=n_states)
        if intervals.empty:
            return pd.DataFrame(columns=cols)

        events: list[dict[str, Any]] = []
        for state_id, group in intervals.groupby("state"):
            durations = group["duration_seconds"]
            events.append(
                {
                    "state": int(state_id),
                    "state_label": group["state_label"].iloc[0],
                    "count": len(group),
                    "mean_duration": float(durations.mean()),
                    "std_duration": float(durations.std()) if len(group) > 1 else 0.0,
                    "min_duration": float(durations.min()),
                    "max_duration": float(durations.max()),
                    "total_duration": float(durations.sum()),
                }
            )

        return (
            pd.DataFrame(events, columns=cols) if events else pd.DataFrame(columns=cols)
        )

    @keyword_only(1)
    def unusual_state_sequence(
        self, n_states: int = 2, expected_order: list[int] | None = None
    ) -> pd.DataFrame:
        """Detect state transitions that violate an expected order.

        By default, the expected order is the most common transition
        sequence observed in the data.  Supply ``expected_order`` to
        override (e.g. ``[0, 1, 2, 1, 0]`` for a warm-up/run/cool-down
        cycle).

        Args:
            n_states: Number of states to infer.
            expected_order: List of state IDs defining the expected
                cycle.  If None, the most frequent bigram transitions
                are treated as "normal".

        Returns:
            DataFrame with columns: transition_time, from_state,
            to_state, from_label, to_label, is_unusual.
        """
        cols = [
            "transition_time",
            "from_state",
            "to_state",
            "from_label",
            "to_label",
            "is_unusual",
        ]
        intervals = self.infer_states(n_states=n_states)
        if len(intervals) < 2:
            return pd.DataFrame(columns=cols)

        labels = self._state_labels(n_states)
        states = intervals["state"].tolist()

        # Build normal transition set
        if expected_order is not None:
            normal_transitions = set()
            for i in range(len(expected_order) - 1):
                normal_transitions.add((expected_order[i], expected_order[i + 1]))
        else:
            # Count bigrams, take transitions occurring at least 20% of the time
            transition_counts: dict[tuple, int] = {}
            for i in range(len(states) - 1):
                pair = (states[i], states[i + 1])
                transition_counts[pair] = transition_counts.get(pair, 0) + 1
            total = sum(transition_counts.values())
            threshold = max(1, total * 0.1)
            normal_transitions = {
                pair for pair, count in transition_counts.items() if count >= threshold
            }

        def _label(state_id: int) -> str:
            return labels[state_id] if state_id < len(labels) else f"state_{state_id}"

        from_states = [int(x) for x in states[:-1]]
        to_states = [int(x) for x in states[1:]]
        return pd.DataFrame(
            {
                "transition_time": intervals["start"].iloc[1:].reset_index(drop=True),
                "from_state": from_states,
                "to_state": to_states,
                "from_label": [_label(x) for x in from_states],
                "to_label": [_label(x) for x in to_states],
                "is_unusual": [
                    (a, b) not in normal_transitions
                    for a, b in zip(from_states, to_states, strict=True)
                ],
            },
            columns=cols,
        )

    @keyword_only(3)
    def dwell_time_violation(
        self,
        state: int,
        max_duration: str,
        n_states: int = 2,
    ) -> pd.DataFrame:
        """Find intervals where the machine stayed in a state too long.

        Args:
            state: State ID to check (0-based).
            max_duration: Maximum allowed duration (e.g. ``'5min'``).
            n_states: Number of states to infer.

        Returns:
            DataFrame with columns: start, end, state_label,
            duration_seconds, excess_seconds, uuid (event uuid), source_uuid
            (signal uuid).
        """
        cols = [
            "start",
            "end",
            "state_label",
            "duration_seconds",
            "excess_seconds",
            "uuid",
            "source_uuid",
        ]
        intervals = self.infer_states(n_states=n_states)
        if intervals.empty:
            return pd.DataFrame(columns=cols)

        max_td = pd.to_timedelta(max_duration).total_seconds()
        state_intervals = intervals[intervals["state"] == state]

        events: list[dict[str, Any]] = []
        for _, row in state_intervals.iterrows():
            dur = row["duration_seconds"]
            if dur > max_td:
                events.append(
                    {
                        "start": row["start"],
                        "end": row["end"],
                        "state_label": row["state_label"],
                        "duration_seconds": dur,
                        "excess_seconds": dur - max_td,
                        "uuid": self.event_uuid,
                        "source_uuid": self.signal_uuid,
                    }
                )

        return (
            pd.DataFrame(events, columns=cols) if events else pd.DataFrame(columns=cols)
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _quantile_bins(values: np.ndarray, n_states: int) -> np.ndarray:
        """Compute bin edges that split values into n_states quantile groups."""
        quantiles = np.linspace(0, 100, n_states + 1)[1:-1]
        bins = np.percentile(values[~np.isnan(values)], quantiles)
        # Ensure unique bins
        bins = np.unique(bins)
        return bins

    @staticmethod
    def _state_labels(n_states: int) -> list[str]:
        """Generate human-readable labels for N states."""
        if n_states == 2:
            return ["low", "high"]
        if n_states == 3:
            return ["low", "mid", "high"]
        return [f"state_{i}" for i in range(n_states)]
