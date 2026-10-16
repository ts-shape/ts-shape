"""Value distribution: shape and stability of a signal's values.

Detects mode changes and bimodality, tests normality per window, and
tracks percentiles over time: the distribution checks behind capability
indices and SPC limits.
"""

import logging
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ts_shape.errors import DataQualityWarning
from ts_shape.events._output import twin_columns
from ts_shape.events._windows import finish_events
from ts_shape.utils.base import Base
from ts_shape.utils.deprecation import caller_stacklevel, keyword_only, renamed_kwargs

logger = logging.getLogger(__name__)


class ValueDistributionEvents(Base):
    """Quality: Value Distribution Analysis

    Answer the question "is my process behaving normally?" by examining
    the statistical distribution of a numeric signal over time.

    Methods:
    - detect_mode_changes: Detect shifts between distinct operating modes.
    - detect_bimodal: Test whether the signal has a bimodal distribution.
    - normality_windows: Flag time windows with non-normal distributions.
    - percentile_tracking: Track selected percentiles over time windows.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        signal_uuid: str,
        *,
        event_uuid: str = "quality:distribution",
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
    def detect_mode_changes(
        self,
        window: str = "1h",
        n_modes: int = 2,
        min_separation: float = 1.0,
    ) -> pd.DataFrame:
        """Detect shifts between distinct operating modes.

        Splits the signal into time windows, computes the histogram peaks
        (modes) in each window, and flags when the dominant mode shifts.

        Args:
            window: Time window for analysis.
            n_modes: Maximum number of modes to look for per window.
            min_separation: Minimum separation between modes in units of
                the signal's standard deviation to count as distinct.

        Returns:
            DataFrame with columns: start, end, duration_seconds, uuid,
            source_uuid, dominant_mode, n_modes_detected, mode_values,
            mode_changed. ``end`` is ``start`` plus one ``window``.
        """
        cols = [
            "start",
            "dominant_mode",
            "n_modes_detected",
            "mode_values",
            "mode_changed",
        ]
        if self.signal.empty:
            return finish_events(
                [],
                cols,
                "interval",
                uuid=self.event_uuid,
                source_uuid=self.signal_uuid,
                window=window,
            )

        sig = (
            self.signal[[self.time_column, self.value_column]]
            .copy()
            .set_index(self.time_column)
        )
        global_std = float(sig[self.value_column].std())
        if global_std == 0:
            global_std = 1.0
        sep_abs = min_separation * global_std

        events: list[dict[str, Any]] = []
        prev_mode = None

        for ts, group in sig.resample(window):
            vals = group[self.value_column].dropna()
            if len(vals) < 5:
                continue

            modes = self._find_modes(vals.values, n_modes, sep_abs)
            dominant = modes[0] if modes else float(vals.median())
            changed = prev_mode is not None and (
                len(modes) == 0 or abs(dominant - prev_mode) > sep_abs
            )
            prev_mode = dominant

            events.append(
                {
                    "start": ts,
                    "dominant_mode": float(dominant),
                    "n_modes_detected": len(modes),
                    "mode_values": [float(m) for m in modes],
                    "mode_changed": changed,
                }
            )

        return finish_events(
            events,
            cols,
            "interval",
            uuid=self.event_uuid,
            source_uuid=self.signal_uuid,
            window=window,
        )

    @keyword_only(1)
    @twin_columns({"n_samples": "sample_count"})
    def detect_bimodal(self, min_samples: int = 30) -> pd.DataFrame:
        """Test whether the overall signal distribution is bimodal.

        Uses Hartigan's dip test statistic approximated via the sorted
        data's inter-mode gap relative to overall range.  A simpler,
        dependency-free alternative to the full dip test.

        Args:
            min_samples: Minimum sample count required to run the test.

        Returns:
            Single-row DataFrame with columns: is_bimodal, dip_score,
            n_samples, mode_1, mode_2, valley (empty when there are fewer
            than ``min_samples`` values). A constant signal is unimodal with
            ``mode_1`` at its value.

            Canonical twin columns: ``sample_count`` (after ``n_samples``); each holds
            the same values as the deprecated column right before it, which is removed
            in the next major release.
        """
        from scipy import stats as sp_stats

        cols = ["is_bimodal", "dip_score", "n_samples", "mode_1", "mode_2", "valley"]
        vals = self.signal[self.value_column].dropna().values
        if len(vals) < min_samples:
            return pd.DataFrame(columns=cols)

        def unimodal(mode: float | None) -> pd.DataFrame:
            return pd.DataFrame(
                [
                    {
                        "is_bimodal": False,
                        "dip_score": 0.0,
                        "n_samples": len(vals),
                        "mode_1": mode,
                        "mode_2": None,
                        "valley": None,
                    }
                ],
                columns=cols,
            )

        # A constant signal has a single mode at its value.
        if np.ptp(vals) == 0:
            return unimodal(float(vals[0]))

        # Kernel density estimation to find peaks
        try:
            kde = sp_stats.gaussian_kde(vals)
        except np.linalg.LinAlgError:
            warnings.warn(
                f"{self.signal_uuid}: kernel density estimate is singular; "
                "reporting a single mode at the median",
                DataQualityWarning,
                stacklevel=caller_stacklevel(),
            )
            return unimodal(float(np.median(vals)))

        x_grid = np.linspace(float(vals.min()), float(vals.max()), 512)
        density = kde(x_grid)

        # Find local maxima
        peaks: list[int] = []
        for i in range(1, len(density) - 1):
            if density[i] > density[i - 1] and density[i] > density[i + 1]:
                peaks.append(i)

        if len(peaks) < 2:
            return unimodal(float(x_grid[peaks[0]]) if peaks else None)

        # Take the two tallest peaks
        peak_heights = [(density[p], p) for p in peaks]
        peak_heights.sort(reverse=True)
        p1_idx = peak_heights[0][1]
        p2_idx = peak_heights[1][1]
        if p1_idx > p2_idx:
            p1_idx, p2_idx = p2_idx, p1_idx

        # Valley = minimum density between the two peaks
        valley_idx = p1_idx + int(np.argmin(density[p1_idx : p2_idx + 1]))
        valley_depth = density[valley_idx]
        min_peak_height = min(density[p1_idx], density[p2_idx])

        # Dip score: how deep the valley is relative to the shorter peak
        dip_score = (
            1.0 - (valley_depth / min_peak_height) if min_peak_height > 0 else 0.0
        )

        return pd.DataFrame(
            [
                {
                    "is_bimodal": dip_score > 0.5,
                    "dip_score": float(dip_score),
                    "n_samples": len(vals),
                    "mode_1": float(x_grid[p1_idx]),
                    "mode_2": float(x_grid[p2_idx]),
                    "valley": float(x_grid[valley_idx]),
                }
            ],
            columns=cols,
        )

    @keyword_only(1)
    @renamed_kwargs(freq="window")
    @twin_columns({"n_samples": "sample_count"})
    def normality_windows(
        self,
        window: str = "1h",
        alpha: float = 0.05,
        min_samples: int = 20,
    ) -> pd.DataFrame:
        """Flag time windows whose values are not normally distributed.

        Uses the Shapiro-Wilk test per window.

        Args:
            window: Resample window.
            alpha: Significance level for the normality test.
            min_samples: Minimum samples per window to run the test.

        Returns:
            DataFrame with columns: start, end, duration_seconds, uuid,
            source_uuid, n_samples, is_normal, p_value, skewness, kurtosis.
            ``end`` is ``start`` plus one ``window``.

            Canonical twin columns: ``sample_count`` (after ``n_samples``); each holds
            the same values as the deprecated column right before it, which is removed
            in the next major release.
        """
        from scipy import stats as sp_stats

        cols = [
            "start",
            "n_samples",
            "is_normal",
            "p_value",
            "skewness",
            "kurtosis",
        ]
        if self.signal.empty:
            return finish_events(
                [],
                cols,
                "summary",
                uuid=self.event_uuid,
                source_uuid=self.signal_uuid,
                window=window,
            )

        sig = (
            self.signal[[self.time_column, self.value_column]]
            .copy()
            .set_index(self.time_column)
        )

        events: list[dict[str, Any]] = []
        for ts, group in sig.resample(window):
            vals = group[self.value_column].dropna()
            if len(vals) < min_samples:
                continue

            # Shapiro-Wilk test (limit to 5000 samples for performance)
            sample = vals.values
            if len(sample) > 5000:
                rng = np.random.default_rng(seed=42)
                sample = rng.choice(sample, size=5000, replace=False)

            # Shapiro-Wilk needs at least three values; scipy either raises
            # or returns NaN for fewer.
            reason = None
            if len(sample) < 3:
                reason = f"{len(sample)} value(s), the test needs at least 3"
            else:
                try:
                    _, p_value = sp_stats.shapiro(sample)
                except ValueError as exc:
                    reason = str(exc)
                else:
                    if np.isnan(p_value):
                        reason = "the test returned NaN"
            if reason is not None:
                warnings.warn(
                    f"{self.signal_uuid}: Shapiro-Wilk test skipped for the "
                    f"window starting {ts} ({reason})",
                    DataQualityWarning,
                    stacklevel=caller_stacklevel(),
                )
                continue

            events.append(
                {
                    "start": ts,
                    "n_samples": len(vals),
                    "is_normal": p_value >= alpha,
                    "p_value": float(p_value),
                    "skewness": float(sp_stats.skew(vals.values)),
                    "kurtosis": float(sp_stats.kurtosis(vals.values)),
                }
            )

        return finish_events(
            events,
            cols,
            "summary",
            uuid=self.event_uuid,
            source_uuid=self.signal_uuid,
            window=window,
        )

    @keyword_only(1)
    @renamed_kwargs(freq="window")
    @twin_columns({"n_samples": "sample_count"})
    def percentile_tracking(
        self,
        percentiles: Sequence[float] = (5, 25, 50, 75, 95),
        window: str = "1h",
    ) -> pd.DataFrame:
        """Track selected percentiles over time windows.

        Useful for spotting distribution spread changes even when the
        mean stays constant.

        Args:
            percentiles: Percentile values to track (0-100).
            window: Resample window.

        Returns:
            DataFrame with columns: start, end, duration_seconds, uuid,
            source_uuid, n_samples, plus one column per percentile (e.g.
            ``p5``, ``p95``). ``end`` is ``start`` plus one ``window``.

            Canonical twin columns: ``sample_count`` (after ``n_samples``); each holds
            the same values as the deprecated column right before it, which is removed
            in the next major release.
        """
        pct_cols = [f"p{int(p)}" for p in percentiles]
        cols = ["start", "n_samples"] + pct_cols
        if self.signal.empty:
            return finish_events(
                [],
                cols,
                "summary",
                uuid=self.event_uuid,
                source_uuid=self.signal_uuid,
                window=window,
            )

        sig = (
            self.signal[[self.time_column, self.value_column]]
            .copy()
            .set_index(self.time_column)
        )

        events: list[dict[str, Any]] = []
        for ts, group in sig.resample(window):
            vals = group[self.value_column].dropna()
            if len(vals) < 2:
                continue

            row: dict[str, Any] = {
                "start": ts,
                "n_samples": len(vals),
            }
            computed = np.percentile(vals.values, list(percentiles))
            for label, val in zip(pct_cols, computed):
                row[label] = float(val)

            events.append(row)

        return finish_events(
            events,
            cols,
            "summary",
            uuid=self.event_uuid,
            source_uuid=self.signal_uuid,
            window=window,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_modes(
        values: np.ndarray, max_modes: int, min_separation: float
    ) -> list[float]:
        """Find up to *max_modes* peaks in the value distribution via KDE."""
        from scipy import stats as sp_stats

        if len(values) < 5:
            return []

        # A singular KDE means the values are (numerically) constant: one
        # mode, at the median.
        try:
            kde = sp_stats.gaussian_kde(values)
        except np.linalg.LinAlgError:
            return [float(np.median(values))]

        x_grid = np.linspace(float(values.min()), float(values.max()), 256)
        density = kde(x_grid)

        # Local maxima
        peaks: list[int] = []
        for i in range(1, len(density) - 1):
            if density[i] > density[i - 1] and density[i] > density[i + 1]:
                peaks.append(i)

        if not peaks:
            return [float(np.median(values))]

        # Sort by density height, keep modes separated by min_separation
        peaks.sort(key=lambda idx: density[idx], reverse=True)
        modes: list[float] = []
        for p in peaks:
            val = float(x_grid[p])
            if all(abs(val - m) >= min_separation for m in modes):
                modes.append(val)
            if len(modes) >= max_modes:
                break

        # Return sorted by value
        modes.sort()
        return modes
