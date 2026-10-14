import logging
from typing import Any

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from ts_shape.utils.base import Base

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
    - state_duration_stats: How long does each state typically last?
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
        super().__init__(dataframe, column_name=time_column)
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
            mean_value, duration_seconds.
        """
        cols = [
            "start",
            "end",
            "state",
            "state_label",
            "mean_value",
            "duration_seconds",
        ]
        if self.signal.empty or len(self.signal) < n_states:
            return pd.DataFrame(columns=cols)

        sig = (
            self.signal[[self.time_column, self.value_column]]
            .copy()
            .reset_index(drop=True)
        )
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

        events: list[dict[str, Any]] = []
        for _, seg in sig.groupby(state_change):
            state_id = int(seg["state"].iloc[0])
            start = seg[self.time_column].iloc[0]
            end = seg[self.time_column].iloc[-1]
            duration = end - start
            if duration < min_td:
                continue
            events.append(
                {
                    "start": start,
                    "end": end,
                    "state": state_id,
                    "state_label": labels[state_id],
                    "mean_value": round(float(seg[self.value_column].mean()), 4),
                    "duration_seconds": round(duration.total_seconds(), 3),
                }
            )

        return (
            pd.DataFrame(events, columns=cols) if events else pd.DataFrame(columns=cols)
        )

    def state_duration_stats(self, n_states: int = 2) -> pd.DataFrame:
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
                    "mean_duration": round(float(durations.mean()), 3),
                    "std_duration": (
                        round(float(durations.std()), 3) if len(group) > 1 else 0.0
                    ),
                    "min_duration": round(float(durations.min()), 3),
                    "max_duration": round(float(durations.max()), 3),
                    "total_duration": round(float(durations.sum()), 3),
                }
            )

        return (
            pd.DataFrame(events, columns=cols) if events else pd.DataFrame(columns=cols)
        )

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

        events: list[dict[str, Any]] = []
        for i in range(len(intervals) - 1):
            from_state = int(intervals.iloc[i]["state"])
            to_state = int(intervals.iloc[i + 1]["state"])
            is_unusual = (from_state, to_state) not in normal_transitions
            events.append(
                {
                    "transition_time": intervals.iloc[i + 1]["start"],
                    "from_state": from_state,
                    "to_state": to_state,
                    "from_label": (
                        labels[from_state]
                        if from_state < len(labels)
                        else f"state_{from_state}"
                    ),
                    "to_label": (
                        labels[to_state]
                        if to_state < len(labels)
                        else f"state_{to_state}"
                    ),
                    "is_unusual": is_unusual,
                }
            )

        return (
            pd.DataFrame(events, columns=cols) if events else pd.DataFrame(columns=cols)
        )

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
            duration_seconds, excess_seconds.
        """
        cols = [
            "start",
            "end",
            "state_label",
            "duration_seconds",
            "excess_seconds",
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
                        "excess_seconds": round(dur - max_td, 3),
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
