import logging
from typing import Any

import numpy as np
import pandas as pd

from ts_shape.utils.base import Base
from ts_shape.utils.deprecation import keyword_only

logger = logging.getLogger(__name__)


class FailurePredictionEvents(Base):
    """
    Predict remaining useful life and detect escalating failure patterns
    in time series signals from manufacturing/industrial IoT systems.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        signal_uuid: str,
        *,
        event_uuid: str = "maint:failure_pred",
        value_column: str = "value_double",
        time_column: str = "systime",
    ) -> None:
        super().__init__(dataframe, time_column=time_column)
        self.signal_uuid = signal_uuid
        self.event_uuid = event_uuid
        self.value_column = value_column
        self.time_column = time_column

        # Isolate signal series and ensure proper dtypes/sort
        self.signal = (
            self.dataframe[self.dataframe["uuid"] == self.signal_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        self.signal[self.time_column] = pd.to_datetime(self.signal[self.time_column])

    def remaining_useful_life(
        self,
        degradation_rate: float,
        failure_threshold: float,
    ) -> pd.DataFrame:
        """
        Estimate remaining useful life (RUL) at each point via linear
        extrapolation from the recent trend to a failure threshold.

        Args:
            degradation_rate: Expected rate of change per second (used as
                fallback if local slope cannot be computed).
            failure_threshold: The signal level that represents failure.
                The sign of ``degradation_rate`` gives the failure direction
                (positive: failure when the value rises to the threshold).

        Returns:
            DataFrame with columns: systime, uuid, source_uuid, is_delta,
            current_value, rul_seconds, rul_hours, confidence. ``rul_*`` is 0 once the
            threshold is reached or passed, and None when no failure is
            projected (flat trend, or a trend moving away from the threshold).
        """
        cols = [
            "systime",
            "uuid",
            "source_uuid",
            "is_delta",
            "current_value",
            "rul_seconds",
            "rul_hours",
            "confidence",
        ]
        if self.signal.empty:
            return pd.DataFrame(columns=cols)

        sig = (
            self.signal[[self.time_column, self.value_column]]
            .copy()
            .reset_index(drop=True)
        )
        sig["t_seconds"] = (
            sig[self.time_column] - sig[self.time_column].iloc[0]
        ).dt.total_seconds()

        rows: list[dict[str, Any]] = []
        for i in range(len(sig)):
            current_val = float(sig[self.value_column].iloc[i])
            t_now = sig[self.time_column].iloc[i]

            # Use recent points (up to last 10) for local slope estimate
            start_idx = max(0, i - 9)
            # NaN samples make polyfit return NaN silently; fit on the rest.
            recent = sig.iloc[start_idx : i + 1].dropna(subset=[self.value_column])

            if len(recent) >= 2:
                x = recent["t_seconds"].values
                y = recent[self.value_column].values
                try:
                    coeffs = np.polyfit(x, y, 1)
                    local_slope = coeffs[0]
                except (np.linalg.LinAlgError, ValueError):
                    logger.debug(
                        "Polyfit failed for failure prediction; using degradation_rate."
                    )
                    local_slope = degradation_rate

                # Confidence based on fit quality
                if len(recent) >= 3:
                    y_pred = np.polyval(coeffs, x)
                    ss_res = np.sum((y - y_pred) ** 2)
                    ss_tot = np.sum((y - y.mean()) ** 2)
                    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                    confidence = max(0.0, min(1.0, r2))
                else:
                    confidence = 0.5
            else:
                local_slope = degradation_rate
                confidence = 0.3

            # Use local slope if it has the right sign, else fallback
            if local_slope == 0:
                local_slope = degradation_rate
            effective_rate = local_slope if local_slope != 0 else degradation_rate

            # Calculate RUL
            distance_to_threshold = failure_threshold - current_val
            failure_dir = np.sign(degradation_rate) or np.sign(effective_rate)
            if np.isnan(current_val) or np.isnan(effective_rate):
                rul_seconds = None
                rul_hours = None
            elif failure_dir != 0 and failure_dir * -distance_to_threshold >= 0:
                # Threshold already reached or passed in the failure direction
                rul_seconds = 0.0
                rul_hours = 0.0
            elif effective_rate == 0 or distance_to_threshold / effective_rate < 0:
                # Flat, or trending away from the threshold: no failure projected.
                # (Previously reported as 0, i.e. imminent failure.)
                rul_seconds = None
                rul_hours = None
            else:
                rul_s = distance_to_threshold / effective_rate
                rul_seconds = float(rul_s)
                rul_hours = float(rul_s / 3600.0)

            rows.append(
                {
                    "systime": t_now,
                    "uuid": self.event_uuid,
                    "source_uuid": self.signal_uuid,
                    "is_delta": True,
                    "current_value": current_val,
                    "rul_seconds": rul_seconds,
                    "rul_hours": rul_hours,
                    "confidence": float(confidence),
                }
            )

        return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)

    @keyword_only(3)
    def detect_exceedance_pattern(
        self,
        warning_threshold: float,
        critical_threshold: float,
        window: str = "1h",
    ) -> pd.DataFrame:
        """
        Track frequency of threshold exceedances in rolling windows and
        flag escalating patterns (increasing exceedance counts).

        Args:
            warning_threshold: Warning level (absolute value comparison).
            critical_threshold: Critical level (absolute value comparison).
            window: Rolling window size.

        Returns:
            DataFrame with columns: start, end, duration_seconds, uuid,
            source_uuid, is_delta, warning_count, critical_count,
            escalation_detected. ``end`` is ``start`` plus ``window``.
        """
        cols = [
            "start",
            "end",
            "duration_seconds",
            "uuid",
            "source_uuid",
            "is_delta",
            "warning_count",
            "critical_count",
            "escalation_detected",
        ]
        if self.signal.empty:
            return pd.DataFrame(columns=cols)

        sig = (
            self.signal[[self.time_column, self.value_column]]
            .copy()
            .reset_index(drop=True)
        )
        window_td = pd.to_timedelta(window)

        # Determine direction: which threshold is higher tells us comparison direction
        if critical_threshold >= warning_threshold:
            sig["is_warning"] = sig[self.value_column] >= warning_threshold
            sig["is_critical"] = sig[self.value_column] >= critical_threshold
        else:
            sig["is_warning"] = sig[self.value_column] <= warning_threshold
            sig["is_critical"] = sig[self.value_column] <= critical_threshold

        # Generate non-overlapping windows
        t_min = sig[self.time_column].iloc[0]
        t_max = sig[self.time_column].iloc[-1]
        window_starts = pd.date_range(start=t_min, end=t_max, freq=window_td)

        rows: list[dict[str, Any]] = []
        prev_warning_count = 0
        prev_critical_count = 0

        for ws in window_starts:
            we = ws + window_td
            mask = (sig[self.time_column] >= ws) & (sig[self.time_column] < we)
            win = sig.loc[mask]
            if win.empty:
                continue

            warning_count = int(win["is_warning"].sum())
            critical_count = int(win["is_critical"].sum())

            # Escalation: current counts exceed previous window counts
            escalation = (warning_count > prev_warning_count) or (
                critical_count > prev_critical_count
            )

            rows.append(
                {
                    "start": ws,
                    "end": we,
                    "duration_seconds": (we - ws).total_seconds(),
                    "uuid": self.event_uuid,
                    "source_uuid": self.signal_uuid,
                    "is_delta": True,
                    "warning_count": warning_count,
                    "critical_count": critical_count,
                    "escalation_detected": escalation,
                }
            )

            prev_warning_count = warning_count
            prev_critical_count = critical_count

        return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)

    @keyword_only(2)
    def time_to_threshold(
        self,
        threshold: float,
        direction: str = "increasing",
    ) -> pd.DataFrame:
        """
        Estimate time to reach a threshold based on recent rate of change.

        Args:
            threshold: Target threshold value.
            direction: 'increasing' or 'decreasing'.

        Returns:
            DataFrame with columns: systime, uuid, source_uuid, is_delta,
            current_value, rate_of_change, estimated_time_seconds.
        """
        cols = [
            "systime",
            "uuid",
            "source_uuid",
            "is_delta",
            "current_value",
            "rate_of_change",
            "estimated_time_seconds",
        ]
        if self.signal.empty:
            return pd.DataFrame(columns=cols)

        sig = (
            self.signal[[self.time_column, self.value_column]]
            .copy()
            .reset_index(drop=True)
        )
        sig["t_seconds"] = (
            sig[self.time_column] - sig[self.time_column].iloc[0]
        ).dt.total_seconds()

        rows: list[dict[str, Any]] = []
        for i in range(len(sig)):
            current_val = float(sig[self.value_column].iloc[i])
            t_now = sig[self.time_column].iloc[i]

            # Use recent points for rate of change
            start_idx = max(0, i - 9)
            # NaN samples make polyfit return NaN silently; fit on the rest.
            recent = sig.iloc[start_idx : i + 1].dropna(subset=[self.value_column])

            if len(recent) >= 2:
                x = recent["t_seconds"].values
                y = recent[self.value_column].values
                try:
                    coeffs = np.polyfit(x, y, 1)
                    rate = float(coeffs[0])
                except (np.linalg.LinAlgError, ValueError):
                    rate = 0.0
            else:
                rate = 0.0

            # Check if rate is in the right direction
            distance = threshold - current_val
            if (
                rate == 0
                or (direction == "increasing" and rate <= 0)
                or (direction == "decreasing" and rate >= 0)
            ):
                estimated_time = None
            else:
                est = distance / rate
                estimated_time = float(est) if est > 0 else 0.0

            rows.append(
                {
                    "systime": t_now,
                    "uuid": self.event_uuid,
                    "source_uuid": self.signal_uuid,
                    "is_delta": True,
                    "current_value": current_val,
                    "rate_of_change": rate,
                    "estimated_time_seconds": estimated_time,
                }
            )

        return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)
