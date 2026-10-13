import logging
from typing import Any

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from ts_shape.utils.base import Base
from ts_shape.utils.intervals import interval_end_times

logger = logging.getLogger(__name__)


class AnomalyClassificationEvents(Base):
    """Quality: Anomaly Classification

    Classify anomalies in a numeric signal by type: spike, drift,
    oscillation, flatline, or level shift.

    Methods:
    - classify_anomalies: Detect and classify anomalous windows.
    - detect_flatline: Signal stuck at constant value.
    - detect_oscillation: Periodic instability detection.
    - detect_drift: Short-term slope-based drift events.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        signal_uuid: str,
        *,
        event_uuid: str = "quality:anomaly_type",
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

    def detect_flatline(
        self,
        min_duration: str = "1min",
        tolerance: float = 1e-6,
        *,
        interval_end: str = "last_sample",
    ) -> pd.DataFrame:
        """Detect intervals where the signal is stuck at a constant value.

        A run starts at its first sample (the value was already stuck there)
        and continues while each sample is within ``tolerance`` of the
        previous one.

        Args:
            min_duration: Minimum duration to qualify as a flatline.
            tolerance: Maximum step between consecutive samples to count as flat.
            interval_end: ``"last_sample"`` (default) ends a run at its last
                sample and needs at least two samples. ``"next_change"`` ends
                it at the next differing sample, so a historian with a
                deadband -- one row, then nothing until the value moves --
                still reports a stuck sensor.

        Returns:
            DataFrame with columns: start_time, end_time, duration, stuck_value.
        """
        cols = ["start", "end", "duration", "stuck_value"]
        if self.signal.empty:
            return pd.DataFrame(columns=cols)

        sig = (
            self.signal[[self.time_column, self.value_column]]
            .copy()
            .reset_index(drop=True)
        )
        min_td = pd.to_timedelta(min_duration)
        ends = interval_end_times(sig[self.time_column], interval_end)

        values = sig[self.value_column]
        # A new run starts wherever the step from the previous sample exceeds
        # the tolerance (or there is no comparable previous sample).
        new_run = ~(values.diff().abs() <= tolerance)
        groups = new_run.cumsum()

        events: list[dict[str, Any]] = []
        for _, run in sig.groupby(groups):
            stuck = run[self.value_column].iloc[0]
            if pd.isna(stuck):
                continue
            if interval_end == "last_sample" and len(run) < 2:
                continue
            start = run[self.time_column].iloc[0]
            end = ends.loc[run.index[-1]]
            duration = end - start
            if duration >= min_td:
                events.append(
                    {
                        "start": start,
                        "end": end,
                        "duration": duration,
                        "stuck_value": float(stuck),
                    }
                )

        return (
            pd.DataFrame(events, columns=cols) if events else pd.DataFrame(columns=cols)
        )

    def detect_oscillation(
        self, window: str = "1min", min_crossings: int = 6
    ) -> pd.DataFrame:
        """Detect windows with excessive zero-crossings of the detrended signal.

        Args:
            window: Window size for analysis.
            min_crossings: Minimum zero-crossings to flag oscillation.

        Returns:
            DataFrame with columns: start_time, end_time, crossing_count, amplitude.
        """
        cols = ["start", "end", "crossing_count", "amplitude"]
        if self.signal.empty:
            return pd.DataFrame(columns=cols)

        sig = (
            self.signal[[self.time_column, self.value_column]]
            .copy()
            .reset_index(drop=True)
        )
        sig = sig.set_index(self.time_column)
        window_td = pd.to_timedelta(window)

        events: list[dict[str, Any]] = []
        start_time = sig.index[0]
        end_time = sig.index[-1]
        current = start_time

        while current + window_td <= end_time:
            win_end = current + window_td
            win = sig.loc[current:win_end, self.value_column]
            if len(win) >= 3:
                detrended = win.values - np.linspace(
                    win.values[0], win.values[-1], len(win)
                )
                signs = np.sign(detrended)
                signs[signs == 0] = 1
                crossings = int(np.sum(np.abs(np.diff(signs)) > 0))

                if crossings >= min_crossings:
                    amplitude = float(np.max(win.values) - np.min(win.values))
                    events.append(
                        {
                            "start": current,
                            "end": win_end,
                            "crossing_count": crossings,
                            "amplitude": amplitude,
                        }
                    )
            current = win_end

        return (
            pd.DataFrame(events, columns=cols) if events else pd.DataFrame(columns=cols)
        )

    def detect_drift(self, window: str = "1h", min_slope: float = 0.01) -> pd.DataFrame:
        """Detect windows where the rolling slope exceeds a threshold.

        Args:
            window: Window size for slope computation.
            min_slope: Minimum absolute slope to flag.

        Returns:
            DataFrame with columns: start_time, end_time, slope, direction.
        """
        cols = ["start", "end", "slope", "direction"]
        if self.signal.empty:
            return pd.DataFrame(columns=cols)

        sig = (
            self.signal[[self.time_column, self.value_column]]
            .copy()
            .reset_index(drop=True)
        )
        sig = sig.set_index(self.time_column)
        window_td = pd.to_timedelta(window)

        events: list[dict[str, Any]] = []
        start_time = sig.index[0]
        end_time = sig.index[-1]
        current = start_time

        while current + window_td <= end_time:
            win_end = current + window_td
            win = sig.loc[current:win_end, self.value_column]
            if len(win) >= 3:
                x = np.arange(len(win), dtype=float)
                try:
                    slope = float(np.polyfit(x, win.values, 1)[0])
                except (np.linalg.LinAlgError, ValueError):
                    slope = 0.0

                if abs(slope) >= min_slope:
                    events.append(
                        {
                            "start": current,
                            "end": win_end,
                            "slope": slope,
                            "direction": "increasing" if slope > 0 else "decreasing",
                        }
                    )
            current = win_end

        return (
            pd.DataFrame(events, columns=cols) if events else pd.DataFrame(columns=cols)
        )

    def classify_anomalies(
        self, window: str = "10min", z_threshold: float = 3.0
    ) -> pd.DataFrame:
        """Detect anomalous windows and classify by type.

        Types: spike, drift, oscillation, flatline, level_shift.

        Args:
            window: Window size for analysis.
            z_threshold: Z-score threshold for spike detection.

        Returns:
            DataFrame with columns: start_time, end_time, anomaly_type, severity, details.
        """
        cols = ["start", "end", "anomaly_type", "severity", "details"]
        if self.signal.empty:
            return pd.DataFrame(columns=cols)

        sig = (
            self.signal[[self.time_column, self.value_column]]
            .copy()
            .reset_index(drop=True)
        )
        sig = sig.set_index(self.time_column)
        window_td = pd.to_timedelta(window)

        global_mean = sig[self.value_column].mean()
        global_std = sig[self.value_column].std()
        if global_std == 0 or np.isnan(global_std):
            global_std = 1e-10

        events: list[dict[str, Any]] = []
        start_time = sig.index[0]
        end_time = sig.index[-1]
        current = start_time

        while current + window_td <= end_time:
            win_end = current + window_td
            win = sig.loc[current:win_end, self.value_column]
            if len(win) < 3:
                current = win_end
                continue

            values = win.values
            win_std = float(np.std(values))
            _win_mean = float(np.mean(values))

            # Check flatline
            if win_std < 1e-8:
                events.append(
                    {
                        "start": current,
                        "end": win_end,
                        "anomaly_type": "flatline",
                        "severity": "warning",
                        "details": f"stuck_value={values[0]:.4f}",
                    }
                )
                current = win_end
                continue

            # Check spikes
            z_scores = np.abs((values - global_mean) / global_std)
            spike_count = int(np.sum(z_scores > z_threshold))
            if spike_count > 0 and spike_count <= max(3, len(values) * 0.1):
                max_z = float(np.max(z_scores))
                severity = "critical" if max_z > z_threshold * 2 else "warning"
                events.append(
                    {
                        "start": current,
                        "end": win_end,
                        "anomaly_type": "spike",
                        "severity": severity,
                        "details": f"spike_count={spike_count}, max_z={max_z:.2f}",
                    }
                )
                current = win_end
                continue

            # Check level shift (mean of first half vs second half)
            mid = len(values) // 2
            first_half_mean = float(np.mean(values[:mid]))
            second_half_mean = float(np.mean(values[mid:]))
            shift = abs(second_half_mean - first_half_mean)
            if shift > global_std * 2:
                events.append(
                    {
                        "start": current,
                        "end": win_end,
                        "anomaly_type": "level_shift",
                        "severity": "warning",
                        "details": f"shift={shift:.4f}, from={first_half_mean:.4f}, to={second_half_mean:.4f}",
                    }
                )
                current = win_end
                continue

            # Check drift
            x = np.arange(len(values), dtype=float)
            try:
                slope = float(np.polyfit(x, values, 1)[0])
            except (np.linalg.LinAlgError, ValueError):
                slope = 0.0

            if abs(slope) > global_std * 0.1:
                events.append(
                    {
                        "start": current,
                        "end": win_end,
                        "anomaly_type": "drift",
                        "severity": "warning",
                        "details": f"slope={slope:.6f}, direction={'increasing' if slope > 0 else 'decreasing'}",
                    }
                )
                current = win_end
                continue

            # Check oscillation
            detrended = values - np.linspace(values[0], values[-1], len(values))
            signs = np.sign(detrended)
            signs[signs == 0] = 1
            crossings = int(np.sum(np.abs(np.diff(signs)) > 0))
            if crossings > len(values) * 0.4:
                amplitude = float(np.max(values) - np.min(values))
                events.append(
                    {
                        "start": current,
                        "end": win_end,
                        "anomaly_type": "oscillation",
                        "severity": "warning",
                        "details": f"crossings={crossings}, amplitude={amplitude:.4f}",
                    }
                )

            current = win_end

        return (
            pd.DataFrame(events, columns=cols) if events else pd.DataFrame(columns=cols)
        )
