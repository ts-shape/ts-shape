import logging
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from typing import List, Dict, Any, Optional

from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class WarmUpCoolDownEvents(Base):
    """Engineering: Warm-Up / Cool-Down Analysis

    Detect and characterize warm-up and cool-down curves — common for ovens,
    extruders, molds, hydraulic systems. Analyzes the shape, consistency,
    and timing of monotonic temperature/pressure ramps.

    Methods:
    - detect_warmup: Intervals where signal rises by at least min_rise.
    - detect_cooldown: Intervals where signal falls by at least min_fall.
    - warmup_consistency: Compare warm-up durations and rates for consistency.
    - time_to_target: Time from warmup start until target value is reached.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        signal_uuid: str,
        *,
        event_uuid: str = "eng:warmup",
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

    def _detect_ramps(
        self, direction: str, min_change: float, min_duration: str = "1m"
    ) -> List[Dict[str, Any]]:
        """Detect monotonic ramp intervals (rising or falling).

        Uses a smoothed diff to identify sustained directional movement.
        Merges short gaps to handle noise in the signal.
        """
        if self.signal.empty or len(self.signal) < 3:
            return []

        min_td = pd.Timedelta(min_duration)
        sig = (
            self.signal[[self.time_column, self.value_column]]
            .copy()
            .reset_index(drop=True)
        )
        vals = sig[self.value_column].values
        times = sig[self.time_column].values

        # Smoothed diff direction
        diff = np.diff(vals)
        if direction == "rising":
            moving = diff > 0
        else:
            moving = diff < 0

        # Intervalize the boolean mask over the diff indices
        intervals: List[Dict[str, Any]] = []
        in_ramp = False
        start_idx = 0

        for i in range(len(moving)):
            if moving[i] and not in_ramp:
                start_idx = i
                in_ramp = True
            elif not moving[i] and in_ramp:
                # End of ramp at index i (the point before reversal)
                end_idx = i
                in_ramp = False
                s_time = pd.Timestamp(times[start_idx])
                e_time = pd.Timestamp(times[end_idx])
                s_val = float(vals[start_idx])
                e_val = float(vals[end_idx])
                change = abs(e_val - s_val)
                dur = e_time - s_time
                if change >= min_change and dur >= min_td:
                    dur_s = dur.total_seconds()
                    intervals.append(
                        {
                            "start": s_time,
                            "end": e_time,
                            "start_value": s_val,
                            "end_value": e_val,
                            "change": change,
                            "duration_seconds": dur_s,
                            "avg_rate": change / dur_s if dur_s > 0 else 0.0,
                        }
                    )

        # Handle ramp that extends to end of data
        if in_ramp:
            end_idx = len(vals) - 1
            s_time = pd.Timestamp(times[start_idx])
            e_time = pd.Timestamp(times[end_idx])
            s_val = float(vals[start_idx])
            e_val = float(vals[end_idx])
            change = abs(e_val - s_val)
            dur = e_time - s_time
            if change >= min_change and dur >= min_td:
                dur_s = dur.total_seconds()
                intervals.append(
                    {
                        "start": s_time,
                        "end": e_time,
                        "start_value": s_val,
                        "end_value": e_val,
                        "change": change,
                        "duration_seconds": dur_s,
                        "avg_rate": change / dur_s if dur_s > 0 else 0.0,
                    }
                )

        return intervals

    def detect_warmup(
        self,
        min_rise: float,
        min_duration: str = "1m",
    ) -> pd.DataFrame:
        """Detect intervals where signal rises by at least min_rise.

        Args:
            min_rise: Minimum total value increase to qualify.
            min_duration: Minimum duration of the ramp.

        Returns:
            DataFrame with columns: start, end, uuid, is_delta,
            start_value, end_value, rise, duration_seconds, avg_rate.
        """
        cols = [
            "start",
            "end",
            "uuid",
            "is_delta",
            "start_value",
            "end_value",
            "rise",
            "duration_seconds",
            "avg_rate",
        ]
        ramps = self._detect_ramps("rising", min_rise, min_duration)
        if not ramps:
            return pd.DataFrame(columns=cols)

        events: List[Dict[str, Any]] = []
        for r in ramps:
            events.append(
                {
                    "start": r["start"],
                    "end": r["end"],
                    "uuid": self.event_uuid,
                    "is_delta": False,
                    "start_value": r["start_value"],
                    "end_value": r["end_value"],
                    "rise": r["change"],
                    "duration_seconds": r["duration_seconds"],
                    "avg_rate": r["avg_rate"],
                }
            )

        return pd.DataFrame(events, columns=cols)

    def detect_cooldown(
        self,
        min_fall: float,
        min_duration: str = "1m",
    ) -> pd.DataFrame:
        """Detect intervals where signal falls by at least min_fall.

        Args:
            min_fall: Minimum total value decrease to qualify (positive number).
            min_duration: Minimum duration of the ramp.

        Returns:
            DataFrame with columns: start, end, uuid, is_delta,
            start_value, end_value, fall, duration_seconds, avg_rate.
        """
        cols = [
            "start",
            "end",
            "uuid",
            "is_delta",
            "start_value",
            "end_value",
            "fall",
            "duration_seconds",
            "avg_rate",
        ]
        ramps = self._detect_ramps("falling", min_fall, min_duration)
        if not ramps:
            return pd.DataFrame(columns=cols)

        events: List[Dict[str, Any]] = []
        for r in ramps:
            events.append(
                {
                    "start": r["start"],
                    "end": r["end"],
                    "uuid": self.event_uuid,
                    "is_delta": False,
                    "start_value": r["start_value"],
                    "end_value": r["end_value"],
                    "fall": r["change"],
                    "duration_seconds": r["duration_seconds"],
                    "avg_rate": r["avg_rate"],
                }
            )

        return pd.DataFrame(events, columns=cols)

    def warmup_consistency(
        self,
        min_rise: float,
        min_duration: str = "1m",
    ) -> pd.DataFrame:
        """Compare warm-up curves for consistency in duration and rate.

        Returns:
            DataFrame with columns: warmup_index, start, duration_seconds,
            avg_rate, deviation_from_median_duration.
        """
        cols = [
            "warmup_index",
            "start",
            "duration_seconds",
            "avg_rate",
            "deviation_from_median_duration",
        ]
        warmups = self.detect_warmup(min_rise, min_duration)
        if warmups.empty:
            return pd.DataFrame(columns=cols)

        median_dur = warmups["duration_seconds"].median()
        result = pd.DataFrame(
            {
                "warmup_index": range(len(warmups)),
                "start": warmups["start"].values,
                "duration_seconds": warmups["duration_seconds"].values,
                "avg_rate": warmups["avg_rate"].values,
                "deviation_from_median_duration": (
                    warmups["duration_seconds"].values - median_dur
                ),
            }
        )
        return result

    def time_to_target(
        self,
        target_value: float,
        direction: str = "rising",
    ) -> pd.DataFrame:
        """Time from each ramp start until target value is reached.

        Args:
            target_value: The target value to reach.
            direction: 'rising' or 'falling'.

        Returns:
            DataFrame with columns: start, target_reached_at,
            time_to_target_seconds, overshoot.
        """
        cols = ["start", "target_reached_at", "time_to_target_seconds", "overshoot"]
        if self.signal.empty or len(self.signal) < 3:
            return pd.DataFrame(columns=cols)

        sig = (
            self.signal[[self.time_column, self.value_column]]
            .copy()
            .reset_index(drop=True)
        )

        # Detect ramp starts using a minimal change threshold
        ramps = self._detect_ramps(direction, min_change=0.01, min_duration="0s")
        if not ramps:
            return pd.DataFrame(columns=cols)

        events: List[Dict[str, Any]] = []
        for r in ramps:
            mask = sig[self.time_column] >= r["start"]
            segment = sig[mask].reset_index(drop=True)

            if direction == "rising":
                reached = segment[segment[self.value_column] >= target_value]
            else:
                reached = segment[segment[self.value_column] <= target_value]

            if reached.empty:
                continue

            reached_time = reached[self.time_column].iloc[0]
            time_to = (reached_time - r["start"]).total_seconds()

            # Overshoot: max beyond target after reaching it
            after_reach = segment[segment[self.time_column] >= reached_time]
            if direction == "rising":
                peak = float(after_reach[self.value_column].max())
                overshoot = max(peak - target_value, 0.0)
            else:
                trough = float(after_reach[self.value_column].min())
                overshoot = max(target_value - trough, 0.0)

            events.append(
                {
                    "start": r["start"],
                    "target_reached_at": reached_time,
                    "time_to_target_seconds": time_to,
                    "overshoot": overshoot,
                }
            )

        return pd.DataFrame(events, columns=cols)
