import logging
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from scipy.stats import kurtosis  # type: ignore
from typing import List, Dict, Any

from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class VibrationAnalysisEvents(Base):
    """
    Analyse vibration signals from industrial equipment: RMS exceedance,
    amplitude growth, and bearing health indicators (kurtosis, crest factor).
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        signal_uuid: str,
        *,
        event_uuid: str = "maint:vibration",
        value_column: str = "value_double",
        time_column: str = "systime",
    ) -> None:
        super().__init__(dataframe, column_name=time_column)
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

    def detect_rms_exceedance(
        self,
        baseline_rms: float,
        threshold_factor: float = 1.5,
        window: str = "1m",
    ) -> pd.DataFrame:
        """
        Compute rolling RMS and flag intervals exceeding baseline_rms * threshold_factor.

        Args:
            baseline_rms: Known baseline RMS value for healthy equipment.
            threshold_factor: Multiplier above baseline to trigger alarm.
            window: Rolling window size.

        Returns:
            DataFrame with columns: start, end, uuid, is_delta,
            rms_value, baseline_rms, ratio, duration_seconds.
        """
        cols = [
            "start",
            "end",
            "uuid",
            "is_delta",
            "rms_value",
            "baseline_rms",
            "ratio",
            "duration_seconds",
        ]
        if self.signal.empty:
            return pd.DataFrame(columns=cols)

        sig = (
            self.signal[[self.time_column, self.value_column]]
            .copy()
            .reset_index(drop=True)
        )
        window_td = pd.to_timedelta(window)
        threshold = baseline_rms * threshold_factor

        # Compute rolling RMS using time-based windows
        rms_values = []
        for i in range(len(sig)):
            t_end = sig[self.time_column].iloc[i]
            t_start = t_end - window_td
            mask = (sig[self.time_column] > t_start) & (sig[self.time_column] <= t_end)
            win = sig.loc[mask, self.value_column]
            if len(win) == 0:
                rms_values.append(np.nan)
            else:
                rms_values.append(float(np.sqrt(np.mean(win.values**2))))

        sig["rms"] = rms_values
        exceeded = (sig["rms"] >= threshold).fillna(False)

        if not exceeded.any():
            return pd.DataFrame(columns=cols)

        # Group contiguous exceeded intervals
        group_id = (exceeded != exceeded.shift()).cumsum()
        events: List[Dict[str, Any]] = []
        for gid, seg in sig.groupby(group_id):
            seg_exc = exceeded.loc[seg.index]
            if not seg_exc.iloc[0]:
                continue
            start_time = seg[self.time_column].iloc[0]
            end_time = seg[self.time_column].iloc[-1]
            avg_rms = float(seg["rms"].mean())
            events.append(
                {
                    "start": start_time,
                    "end": end_time,
                    "uuid": self.event_uuid,
                    "is_delta": True,
                    "rms_value": avg_rms,
                    "baseline_rms": baseline_rms,
                    "ratio": avg_rms / baseline_rms if baseline_rms > 0 else None,
                    "duration_seconds": (end_time - start_time).total_seconds(),
                }
            )

        return (
            pd.DataFrame(events, columns=cols) if events else pd.DataFrame(columns=cols)
        )

    def detect_amplitude_growth(
        self,
        window: str = "1h",
        growth_threshold: float = 0.1,
    ) -> pd.DataFrame:
        """
        Track peak-to-peak amplitude in non-overlapping windows and flag
        windows where amplitude grows beyond growth_threshold relative to baseline.

        Args:
            window: Window size for amplitude measurement.
            growth_threshold: Minimum fractional growth (e.g. 0.1 = 10%) to flag.

        Returns:
            DataFrame with columns: window_start, uuid, is_delta,
            amplitude, baseline_amplitude, growth_pct.
        """
        cols = [
            "window_start",
            "uuid",
            "is_delta",
            "amplitude",
            "baseline_amplitude",
            "growth_pct",
        ]
        if self.signal.empty:
            return pd.DataFrame(columns=cols)

        sig = (
            self.signal[[self.time_column, self.value_column]]
            .copy()
            .reset_index(drop=True)
        )
        window_td = pd.to_timedelta(window)

        # Generate non-overlapping windows
        t_min = sig[self.time_column].iloc[0]
        t_max = sig[self.time_column].iloc[-1]
        window_starts = pd.date_range(start=t_min, end=t_max, freq=window_td)

        amplitudes: List[Dict[str, Any]] = []
        for ws in window_starts:
            we = ws + window_td
            mask = (sig[self.time_column] >= ws) & (sig[self.time_column] < we)
            win = sig.loc[mask, self.value_column]
            if len(win) < 2:
                continue
            amp = float(win.max() - win.min())
            amplitudes.append({"window_start": ws, "amplitude": amp})

        if not amplitudes:
            return pd.DataFrame(columns=cols)

        # Use first window as baseline
        baseline_amp = amplitudes[0]["amplitude"]
        if baseline_amp == 0:
            baseline_amp = np.finfo(float).eps

        rows: List[Dict[str, Any]] = []
        for a in amplitudes:
            growth_pct = (a["amplitude"] - baseline_amp) / baseline_amp
            rows.append(
                {
                    "window_start": a["window_start"],
                    "uuid": self.event_uuid,
                    "is_delta": True,
                    "amplitude": a["amplitude"],
                    "baseline_amplitude": amplitudes[0]["amplitude"],
                    "growth_pct": round(growth_pct, 6),
                }
            )

        return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)

    def bearing_health_indicators(
        self,
        window: str = "5m",
    ) -> pd.DataFrame:
        """
        Compute bearing health indicators per window: RMS, peak value,
        crest factor (peak/RMS), and kurtosis.

        Args:
            window: Window size for indicator computation.

        Returns:
            DataFrame with columns: window_start, uuid, is_delta,
            rms, peak, crest_factor, kurtosis.
        """
        cols = [
            "window_start",
            "uuid",
            "is_delta",
            "rms",
            "peak",
            "crest_factor",
            "kurtosis",
        ]
        if self.signal.empty:
            return pd.DataFrame(columns=cols)

        sig = (
            self.signal[[self.time_column, self.value_column]]
            .copy()
            .reset_index(drop=True)
        )
        window_td = pd.to_timedelta(window)

        # Generate non-overlapping windows
        t_min = sig[self.time_column].iloc[0]
        t_max = sig[self.time_column].iloc[-1]
        window_starts = pd.date_range(start=t_min, end=t_max, freq=window_td)

        rows: List[Dict[str, Any]] = []
        for ws in window_starts:
            we = ws + window_td
            mask = (sig[self.time_column] >= ws) & (sig[self.time_column] < we)
            win = sig.loc[mask, self.value_column]
            if len(win) < 2:
                continue

            values = win.values
            rms_val = float(np.sqrt(np.mean(values**2)))
            peak_val = float(np.max(np.abs(values)))
            crest_factor = peak_val / rms_val if rms_val > 0 else None
            kurt_val = float(kurtosis(values, fisher=True))

            rows.append(
                {
                    "window_start": ws,
                    "uuid": self.event_uuid,
                    "is_delta": True,
                    "rms": round(rms_val, 6),
                    "peak": round(peak_val, 6),
                    "crest_factor": (
                        round(crest_factor, 6) if crest_factor is not None else None
                    ),
                    "kurtosis": round(kurt_val, 6),
                }
            )

        return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)
