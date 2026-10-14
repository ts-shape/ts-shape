import logging
from typing import Any

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from ts_shape.events._windows import finish_events
from ts_shape.utils.base import Base
from ts_shape.utils.deprecation import keyword_only, renamed_kwargs
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

    @keyword_only(1)
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
            DataFrame with columns: start, end, duration_seconds, uuid, source_uuid,
                duration, stuck_value. ``duration`` is a Timedelta;
                ``duration_seconds`` is the same span in float seconds.
        """
        cols = ["start", "end", "duration", "stuck_value"]
        if self.signal.empty:
            return finish_events(
                [], cols, "interval", uuid=self.event_uuid, source_uuid=self.signal_uuid
            )

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

        return finish_events(
            events, cols, "interval", uuid=self.event_uuid, source_uuid=self.signal_uuid
        )

    @keyword_only(1)
    def detect_oscillation(
        self, window: str = "1min", min_crossings: int = 6
    ) -> pd.DataFrame:
        """Detect windows with excessive zero-crossings of the detrended signal.

        Args:
            window: Window size for analysis.
            min_crossings: Minimum zero-crossings to flag oscillation.

        Returns:
            DataFrame with columns: start, end, duration_seconds, uuid,
                source_uuid, crossing_count, amplitude.
        """
        cols = ["start", "end", "crossing_count", "amplitude"]
        if self.signal.empty:
            return finish_events(
                [], cols, "interval", uuid=self.event_uuid, source_uuid=self.signal_uuid
            )

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

        return finish_events(
            events, cols, "interval", uuid=self.event_uuid, source_uuid=self.signal_uuid
        )

    @staticmethod
    def _check_slope_per(slope_per: str) -> None:
        if slope_per not in ("sample", "second"):
            raise ValueError(
                f"slope_per must be 'sample' or 'second', got {slope_per!r}"
            )

    @staticmethod
    def _slope_x(index: pd.Index, slope_per: str) -> np.ndarray:
        """x-axis for a slope fit: sample index or elapsed seconds."""
        if slope_per == "second":
            times = pd.DatetimeIndex(index)
            return np.asarray((times - times[0]).total_seconds(), dtype=float)
        return np.arange(len(index), dtype=float)

    @keyword_only(1)
    def detect_drift(
        self, window: str = "1h", min_slope: float = 0.01, slope_per: str = "sample"
    ) -> pd.DataFrame:
        """Detect windows where the rolling slope exceeds a threshold.

        A least-squares line is fitted to each non-overlapping window. With
        the default ``slope_per="sample"`` the x-axis is the sample index,
        so ``slope`` is in signal units per *sample* and the meaning of
        ``min_slope`` depends on the sampling rate. With
        ``slope_per="second"`` the x-axis is elapsed time, so ``slope`` and
        ``min_slope`` are in signal units per second regardless of the
        sampling rate (and irregular sampling is handled correctly).

        Args:
            window: Window size for slope computation.
            min_slope: Minimum absolute slope to flag, in signal units per
                sample or per second (see ``slope_per``).
            slope_per: ``"sample"`` (default, previous behavior) or
                ``"second"``.

        Returns:
            DataFrame with columns: start, end, duration_seconds, uuid,
                source_uuid, slope, direction.
        """
        self._check_slope_per(slope_per)
        cols = ["start", "end", "slope", "direction"]
        if self.signal.empty:
            return finish_events(
                [], cols, "interval", uuid=self.event_uuid, source_uuid=self.signal_uuid
            )

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
                x = self._slope_x(win.index, slope_per)
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

        return finish_events(
            events, cols, "interval", uuid=self.event_uuid, source_uuid=self.signal_uuid
        )

    @keyword_only(1)
    @renamed_kwargs(z_threshold="n_sigma")
    def classify_anomalies(
        self, window: str = "10min", n_sigma: float = 3.0, slope_per: str = "sample"
    ) -> pd.DataFrame:
        """Detect anomalous windows and classify by type.

        Types: spike, drift, oscillation, flatline, level_shift.

        A window is a drift when its least-squares slope exceeds 0.1 × the
        signal's overall std per sample. With the default
        ``slope_per="sample"`` the slope is fitted against the sample index,
        so the test depends on the sampling rate. With
        ``slope_per="second"`` it is fitted against elapsed time and compared
        with 0.1 × std divided by the signal's median sampling interval
        (seconds): on regularly sampled data the same windows are flagged,
        irregular sampling is handled correctly, and the ``slope`` reported
        in ``details`` is in signal units per second.

        Args:
            window: Window size for analysis.
            n_sigma: Z-score threshold for spike detection.
            slope_per: ``"sample"`` (default, previous behavior) or
                ``"second"``.

        Returns:
            DataFrame with columns: start, end, duration_seconds, uuid,
                source_uuid, anomaly_type, severity, details.
        """
        self._check_slope_per(slope_per)
        cols = ["start", "end", "anomaly_type", "severity", "details"]
        if self.signal.empty:
            return finish_events(
                [], cols, "interval", uuid=self.event_uuid, source_uuid=self.signal_uuid
            )

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
        drift_threshold = global_std * 0.1
        if slope_per == "second":
            dt = pd.Series(sig.index).diff().dt.total_seconds()
            median_dt = float(dt[dt > 0].median()) if (dt > 0).any() else np.nan
            drift_threshold = (
                drift_threshold / median_dt if median_dt > 0 else drift_threshold
            )

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
            spike_count = int(np.sum(z_scores > n_sigma))
            if spike_count > 0 and spike_count <= max(3, len(values) * 0.1):
                max_z = float(np.max(z_scores))
                severity = "critical" if max_z > n_sigma * 2 else "warning"
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
            x = self._slope_x(win.index, slope_per)
            try:
                slope = float(np.polyfit(x, values, 1)[0])
            except (np.linalg.LinAlgError, ValueError):
                slope = 0.0

            if abs(slope) > drift_threshold:
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

        return finish_events(
            events, cols, "interval", uuid=self.event_uuid, source_uuid=self.signal_uuid
        )
