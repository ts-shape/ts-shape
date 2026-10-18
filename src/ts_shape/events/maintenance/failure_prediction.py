import logging

import numpy as np
import pandas as pd

from ts_shape.events.maintenance._rolling import (
    count_window_bounds,
    interval_bounds,
    window_moments,
)
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
        """Initialize the failure predictor.

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

        current = sig[self.value_column].to_numpy(dtype=float)
        # Local least-squares slope over the last 10 samples (NaN samples
        # skipped). A window without a usable slope -- fewer than two samples,
        # or all at one timestamp -- falls back to degradation_rate.
        lo, hi = count_window_bounds(len(sig), 10)
        mom = window_moments(sig["t_seconds"].to_numpy(), current, lo, hi)
        n = mom["n"]
        fitted = (n >= 2) & ~np.isnan(mom["slope"])
        local_slope = np.where(fitted, mom["slope"], degradation_rate)

        # Confidence based on fit quality: R^2 of the local fit.
        with np.errstate(invalid="ignore", divide="ignore"):
            ss_res = np.maximum(mom["syy"] - mom["sxy"] ** 2 / mom["sxx"], 0.0)
            r2 = np.where(mom["syy"] > 0, 1 - ss_res / mom["syy"], 0.0)
        confidence = np.where(fitted, np.where(n >= 3, np.clip(r2, 0.0, 1.0), 0.5), 0.3)

        # A zero local slope falls back to degradation_rate.
        effective_rate = np.where(local_slope == 0, degradation_rate, local_slope)

        # Calculate RUL
        distance = failure_threshold - current
        failure_dir = np.sign(degradation_rate) or np.sign(effective_rate)
        with np.errstate(invalid="ignore", divide="ignore"):
            projected = distance / effective_rate
        unknown = np.isnan(current) | np.isnan(effective_rate)
        # Threshold already reached or passed in the failure direction
        reached = (failure_dir != 0) & (failure_dir * -distance >= 0)
        # Flat, or trending away from the threshold: no failure projected.
        # (Previously reported as 0, i.e. imminent failure.)
        away = (effective_rate == 0) | (projected < 0)
        rul = np.where(reached, 0.0, projected)
        none = unknown | (~reached & away)
        rul_seconds = _floats_or_none(rul, none)
        rul_hours = _floats_or_none(rul / 3600.0, none)

        result = pd.DataFrame(
            {
                "systime": sig[self.time_column],
                "uuid": self.event_uuid,
                "source_uuid": self.signal_uuid,
                "is_delta": True,
                "current_value": current,
                "rul_seconds": rul_seconds,
                "rul_hours": rul_hours,
                "confidence": confidence.astype(float),
            },
            columns=cols,
        )
        return result

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

        # Per-window counts over [ws, ws + window) from prefix sums; empty
        # windows are skipped, and escalation compares with the previous
        # non-empty window.
        lo, hi = interval_bounds(sig[self.time_column], window_starts, window_td)
        keep = hi > lo
        cw = np.concatenate(([0], np.cumsum(sig["is_warning"].to_numpy(dtype=bool))))
        cc = np.concatenate(([0], np.cumsum(sig["is_critical"].to_numpy(dtype=bool))))
        warning_count = (cw[hi] - cw[lo])[keep]
        critical_count = (cc[hi] - cc[lo])[keep]
        prev_w = np.concatenate(([0], warning_count[:-1]))
        prev_c = np.concatenate(([0], critical_count[:-1]))
        ws = window_starts[keep]
        we = ws + window_td
        if not len(ws):
            return pd.DataFrame(columns=cols)
        return pd.DataFrame(
            {
                "start": ws,
                "end": we,
                "duration_seconds": (we - ws).total_seconds(),
                "uuid": self.event_uuid,
                "source_uuid": self.signal_uuid,
                "is_delta": True,
                "warning_count": warning_count.astype(int),
                "critical_count": critical_count.astype(int),
                "escalation_detected": (warning_count > prev_w)
                | (critical_count > prev_c),
            },
            columns=cols,
        )

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

        current = sig[self.value_column].to_numpy(dtype=float)
        # Rate of change: least-squares slope over the last 10 samples (NaN
        # samples skipped); 0 when it cannot be computed (fewer than two
        # samples, or all at one timestamp).
        lo, hi = count_window_bounds(len(sig), 10)
        mom = window_moments(sig["t_seconds"].to_numpy(), current, lo, hi)
        rate = np.where((mom["n"] >= 2) & ~np.isnan(mom["slope"]), mom["slope"], 0.0)

        # Check if rate is in the right direction
        distance = threshold - current
        wrong_way = (
            (rate == 0)
            | ((direction == "increasing") & (rate <= 0))
            | ((direction == "decreasing") & (rate >= 0))
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            est = np.where(distance / rate > 0, distance / rate, 0.0)
        # A NaN current value has no estimate (was reported as 0, i.e. "at the
        # threshold now").
        estimated = _floats_or_none(est, wrong_way | np.isnan(current))

        result = pd.DataFrame(
            {
                "systime": sig[self.time_column],
                "uuid": self.event_uuid,
                "source_uuid": self.signal_uuid,
                "is_delta": True,
                "current_value": current,
                "rate_of_change": rate.astype(float),
                "estimated_time_seconds": estimated,
            },
            columns=cols,
        )
        return result


def _floats_or_none(values: np.ndarray, none: np.ndarray) -> list[float | None]:
    """Python floats, ``None`` where ``none`` is set (the historical column type)."""
    return [None if skip else float(v) for v, skip in zip(values, none)]
