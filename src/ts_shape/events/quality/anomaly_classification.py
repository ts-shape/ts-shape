import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd

from ts_shape.errors import DataQualityWarning
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
        interval_end: str = "next_change",
        max_gap: str | pd.Timedelta | None = None,
    ) -> pd.DataFrame:
        """Detect intervals where the signal is stuck at a constant value.

        A run starts at its first sample (the value was already stuck there)
        and continues while each sample is within ``tolerance`` of the
        previous one. Samples at the same timestamp (duplicates) add no
        duration: a run whose samples all share one timestamp counts as a
        single sample.

        Args:
            min_duration: Minimum duration to qualify as a flatline.
            tolerance: Maximum step between consecutive samples to count as flat.
            interval_end: ``"next_change"`` (default) ends a run at the
                next differing sample, so a historian with a deadband -- one
                row, then nothing until the value moves -- still reports a
                stuck sensor. A single-sample run counts only when the gap
                after it is longer than the signal's median sampling
                interval. ``"last_sample"`` ends a run at its own last
                sample and needs at least two samples.
            max_gap: Longest time between two samples a run may span (e.g.
                ``"15min"``). A longer gap is "no data": the run before it
                ends at its last sample and a new run starts after it, even
                if the value is unchanged. ``None`` (default): no limit.

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
        times = sig[self.time_column]
        ends = interval_end_times(times, interval_end)
        steps_all = times.diff()

        values = sig[self.value_column]
        # A new run starts wherever the step from the previous sample exceeds
        # the tolerance (or there is no comparable previous sample).
        new_run = ~(values.diff().abs() <= tolerance)
        if max_gap is not None:
            # A gap longer than max_gap is "no data": split the run there and
            # end the run before it at its own last sample.
            gap_before = (steps_all > pd.Timedelta(max_gap)).to_numpy()
            new_run |= gap_before
            gap_after = np.append(gap_before[1:], False)
            ends = ends.where(~gap_after, times)

        # A one-sample run is only a flatline when the silence after it is
        # longer than the signal's usual sampling interval (a deadband
        # historian); otherwise every sample of a slow signal would qualify.
        # Two samples give no sampling interval to compare against; duplicate
        # timestamps (zero steps) are no sampling interval either.
        steps = steps_all[steps_all > pd.Timedelta(0)]
        typical_step = steps.median() if len(steps) >= 2 else pd.Timedelta(0)

        # Runs as index ranges [first, last] (vectorized; no per-run groupby).
        first = np.flatnonzero(new_run.to_numpy())
        last = np.append(first[1:], len(sig)) - 1
        stuck = values.to_numpy(dtype=float)[first]
        start = times.iloc[first].reset_index(drop=True)
        end = ends.iloc[last].reset_index(drop=True)
        duration = end - start
        # A run whose samples share one timestamp is a single sample.
        t_np = times.to_numpy()
        single = t_np[last] == t_np[first]
        keep = ~np.isnan(stuck)
        if interval_end == "last_sample":
            keep &= ~single
        keep &= ~(single & ~(duration > typical_step).to_numpy())
        keep &= (duration >= min_td).to_numpy()
        events: list[dict[str, Any]] = [
            {
                "start": st,
                "end": en,
                "duration": du,
                "stuck_value": float(v),
            }
            for st, en, du, v in zip(
                start[keep], end[keep], duration[keep], stuck[keep]
            )
        ]

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

        if window_td <= pd.Timedelta(0):
            raise ValueError(f"window must be positive, got {window!r}")

        # Consecutive windows [start + k*window, start + (k+1)*window] (both
        # ends included, as label slicing does) as index ranges, evaluated
        # together: crossings of the line joining each window's first and
        # last value (computed as np.linspace does), and its peak-to-peak.
        t_ns = pd.DatetimeIndex(sig.index).as_unit("ns").asi8  # UTC for tz-aware
        w_ns = int(window_td.value)
        n_win = int((t_ns[-1] - t_ns[0]) // w_ns)
        events: list[dict[str, Any]] = []
        if n_win > 0:
            starts_ns = t_ns[0] + w_ns * np.arange(n_win, dtype=np.int64)
            lo = np.searchsorted(t_ns, starts_ns, side="left")
            hi = np.searchsorted(t_ns, starts_ns + w_ns, side="right")
            lens = hi - lo
            ok = np.flatnonzero(lens >= 3)
            if len(ok):
                lo_k, lens_k = lo[ok], lens[ok]
                offs = np.concatenate(([0], np.cumsum(lens_k)[:-1]))
                idx = np.arange(int(lens_k.sum())) + np.repeat(lo_k - offs, lens_k)
                v = sig[self.value_column].to_numpy(dtype=float)
                vals = v[idx]
                pos = idx - np.repeat(lo_k, lens_k)  # position within window
                first_v = v[lo_k]
                last_v = v[lo_k + lens_k - 1]
                step = (last_v - first_v) / (lens_k - 1)
                line = pos * np.repeat(step, lens_k) + np.repeat(first_v, lens_k)
                is_last = pos == np.repeat(lens_k - 1, lens_k)
                line[is_last] = np.repeat(last_v, lens_k)[is_last]
                signs = np.sign(vals - line)
                signs[signs == 0] = 1
                change = np.abs(np.diff(signs)) > 0
                # Only steps inside a window count (not across two windows).
                change &= ~is_last[:-1]
                crossings = np.add.reduceat(
                    np.append(change, False).astype(np.int64), offs
                )
                amplitude = np.maximum.reduceat(vals, offs) - np.minimum.reduceat(
                    vals, offs
                )
                hit = crossings >= min_crossings
                for k, c, amp in zip(ok[hit], crossings[hit], amplitude[hit]):
                    current = sig.index[0] + k * window_td
                    events.append(
                        {
                            "start": current,
                            "end": current + window_td,
                            "crossing_count": int(c),
                            "amplitude": float(amp),
                        }
                    )

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
            # (times - times[0]).total_seconds(), on the integer ticks directly
            # (same division, without building a TimedeltaIndex per window).
            times = pd.DatetimeIndex(index)
            ticks = times.asi8
            per_second = {"s": 1, "ms": 10**3, "us": 10**6, "ns": 10**9}[times.unit]
            return np.asarray((ticks - ticks[0]) / per_second, dtype=float)
        return np.arange(len(index), dtype=float)

    def _fit_slope(
        self, win: pd.Series, slope_per: str, stacklevel: int
    ) -> float | None:
        """Least-squares slope of ``win`` (NaN values left out).

        Returns ``None`` -- after a :class:`DataQualityWarning` -- when the
        window cannot be fitted.
        """
        x = self._slope_x(win.index, slope_per)
        y = win.to_numpy(dtype=float)
        finite = np.isfinite(y)
        if finite.sum() < 3:
            return None
        try:
            return float(np.polyfit(x[finite], y[finite], 1)[0])
        except (np.linalg.LinAlgError, ValueError) as exc:
            warnings.warn(
                f"{self.signal_uuid}: slope fit failed for the window starting "
                f"{win.index[0]} ({exc}); window skipped",
                DataQualityWarning,
                stacklevel=stacklevel,
            )
            return None

    @keyword_only(1)
    def detect_drift(
        self, window: str = "1h", min_slope: float = 0.01, slope_per: str = "second"
    ) -> pd.DataFrame:
        """Detect windows where the rolling slope exceeds a threshold.

        A least-squares line is fitted to each non-overlapping window. With
        the default ``slope_per="second"`` the x-axis is elapsed time, so
        ``slope`` and ``min_slope`` are in signal units per second regardless
        of the sampling rate (and irregular sampling is handled correctly).
        With ``slope_per="sample"`` the x-axis is the sample index, so
        ``slope`` is in signal units per *sample* and the meaning of
        ``min_slope`` depends on the sampling rate.

        NaN values are left out of the fit. A window whose fit fails (e.g.
        all of its samples share one timestamp) is skipped with a
        :class:`~ts_shape.errors.DataQualityWarning`.

        Args:
            window: Window size for slope computation.
            min_slope: Minimum absolute slope to flag, in signal units per
                sample or per second (see ``slope_per``).
            slope_per: ``"second"`` (default) or ``"sample"``.

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
            slope = self._fit_slope(win, slope_per, 5) if len(win) >= 3 else None
            if slope is not None and abs(slope) >= min_slope:
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
        self, window: str = "10min", n_sigma: float = 3.0, slope_per: str = "second"
    ) -> pd.DataFrame:
        """Detect anomalous windows and classify by type.

        Types: spike, drift, oscillation, flatline, level_shift.

        A window is a drift when its least-squares slope exceeds 0.1 × the
        signal's overall std per sample. With the default
        ``slope_per="second"`` the slope is fitted against elapsed time and
        compared with 0.1 × std divided by the signal's median sampling
        interval (seconds): irregular sampling is handled correctly and the
        ``slope`` reported in ``details`` is in signal units per second. With
        ``slope_per="sample"`` the slope is fitted against the sample index,
        so the test depends on the sampling rate; on regularly sampled data
        both settings flag the same windows.

        Args:
            window: Window size for analysis.
            n_sigma: Z-score threshold for spike detection.
            slope_per: ``"second"`` (default) or ``"sample"``.

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
            fitted = self._fit_slope(win, slope_per, 6)
            slope = 0.0 if fitted is None else fitted

            if fitted is not None and abs(slope) > drift_threshold:
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
