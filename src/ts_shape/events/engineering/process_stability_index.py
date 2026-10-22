import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ts_shape.events._output import twin_columns
from ts_shape.events._windows import window_summary
from ts_shape.utils.base import Base
from ts_shape.utils.deprecation import deprecated_class, keyword_only

if TYPE_CHECKING:
    from typing_extensions import deprecated

logger = logging.getLogger(__name__)


class ProcessStabilityIndexEvents(Base):
    """Engineering: Process Stability Index

    Generate a single 0-100 stability score per shift/day for a signal.
    Answers 'is my process running well today?' with one number.

    Methods:
    - stability_score: Composite 0-100 score per window.
    - score_trend: Is stability improving or degrading?
    - worst_periods: N worst-scoring windows.
    - stability_comparison: Compare each window to best-observed.
    """

    #: Methods read only the constructor's signals (``Base`` keeps their rows).
    _reads_own_signals_only = True

    def __init__(
        self,
        dataframe: pd.DataFrame,
        signal_uuid: str,
        *,
        target: float | None = None,
        lower_spec: float | None = None,
        upper_spec: float | None = None,
        event_uuid: str = "eng:stability_index",
        value_column: str = "value_double",
        time_column: str = "systime",
    ) -> None:
        """Initialize the stability index.

        Args:
            dataframe: Long-format timeseries frame (one row per sample, ``uuid``
                identifies the signal).
            signal_uuid: UUID of the signal to analyze.
            target: Target value. ``None`` (default) uses the signal mean.
            lower_spec: Lower specification limit. ``None`` (default) uses mean - 3 std.
            upper_spec: Upper specification limit. ``None`` (default) uses mean + 3 std.
            event_uuid: UUID stamped on emitted events.
            value_column: Column holding the signal values.
            time_column: Timestamp column.
        """
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

        # Compute defaults from data
        vals = self.signal[self.value_column].dropna()
        if not vals.empty:
            self._global_mean = float(vals.mean())
            self._global_std = float(vals.std()) if len(vals) > 1 else 1.0
        else:
            self._global_mean = 0.0
            self._global_std = 1.0

        # Unset target/spec limits default to the data mean and mean +/- 3 sigma.
        self.target: float = self._global_mean if target is None else float(target)
        self.lower_spec: float = (
            self._global_mean - 3 * self._global_std
            if lower_spec is None
            else float(lower_spec)
        )
        self.upper_spec: float = (
            self._global_mean + 3 * self._global_std
            if upper_spec is None
            else float(upper_spec)
        )

    @keyword_only(1)
    def stability_score(self, window: str = "8h") -> pd.DataFrame:
        """Composite 0-100 stability score per window.

        Four sub-scores (each 0-25):
        - variance_score: low std = high score
        - bias_score: on-target = high score
        - excursion_score: in-spec = high score
        - smoothness_score: low point-to-point jitter = high score

        Args:
            window: Scoring window (pandas offset string, e.g. '8h' per shift).
                Windows with fewer than two samples are skipped.

        Returns:
            DataFrame with columns: start, end, duration_seconds, uuid,
            source_uuid, stability_score,
            variance_score, bias_score, excursion_score, smoothness_score,
            grade.
        """
        cols = [
            "start",
            "stability_score",
            "variance_score",
            "bias_score",
            "excursion_score",
            "smoothness_score",
            "grade",
        ]
        if self.signal.empty:
            return self._summary([], cols, window)

        indexed = self.signal.set_index(self.time_column)[self.value_column]
        groups = indexed.resample(window)
        spec_range = self.upper_spec - self.lower_spec
        half_range = spec_range / 2.0 if spec_range > 0 else 1.0

        events: list[dict[str, Any]] = []
        for start, group in groups:
            vals = group.dropna()
            if len(vals) < 2:
                continue

            std = float(vals.std())
            mean = float(vals.mean())

            # Variance score: 25 * (1 - clamp(std / half_range, 0, 1))
            var_ratio = min(std / half_range, 1.0) if half_range > 0 else 1.0
            variance_score = 25.0 * (1.0 - var_ratio)

            # Bias score: 25 * (1 - clamp(|mean - target| / half_range, 0, 1))
            bias_ratio = (
                min(abs(mean - self.target) / half_range, 1.0)
                if half_range > 0
                else 1.0
            )
            bias_score = 25.0 * (1.0 - bias_ratio)

            # Excursion score: 25 * (1 - pct_out_of_spec / 100)
            in_spec = ((vals >= self.lower_spec) & (vals <= self.upper_spec)).sum()
            pct_in = in_spec / len(vals)
            excursion_score = 25.0 * pct_in

            # Smoothness score: 25 * (1 - clamp(mean_abs_diff / half_range, 0, 1))
            abs_diff = vals.diff().abs().mean()
            smooth_ratio = min(abs_diff / half_range, 1.0) if half_range > 0 else 1.0
            smoothness_score = 25.0 * (1.0 - smooth_ratio)

            total = variance_score + bias_score + excursion_score + smoothness_score

            if total >= 90:
                grade = "A"
            elif total >= 75:
                grade = "B"
            elif total >= 60:
                grade = "C"
            elif total >= 40:
                grade = "D"
            else:
                grade = "F"

            events.append(
                {
                    "start": start,
                    "stability_score": float(total),
                    "variance_score": float(variance_score),
                    "bias_score": float(bias_score),
                    "excursion_score": float(excursion_score),
                    "smoothness_score": float(smoothness_score),
                    "grade": grade,
                }
            )

        return self._summary(events, cols, window)

    def _summary(
        self, rows: pd.DataFrame | list[dict[str, Any]], cols: list[str], window: str
    ) -> pd.DataFrame:
        return window_summary(
            rows, cols, window, uuid=self.event_uuid, source_uuid=self.signal_uuid
        )

    @keyword_only(1)
    def score_trend(
        self,
        window: str = "8h",
        n_windows: int = 7,
    ) -> pd.DataFrame:
        """Track whether stability is improving or degrading.

        Args:
            window: Scoring window passed to :meth:`stability_score`.
            n_windows: Number of windows in the rolling average of the score.

        Returns:
            DataFrame with columns: start, end, duration_seconds, uuid,
            source_uuid, stability_score,
            rolling_avg, trend_direction, score_change.
        """
        cols = [
            "start",
            "stability_score",
            "rolling_avg",
            "trend_direction",
            "score_change",
        ]
        scores = self.stability_score(window=window)
        if scores.empty or len(scores) < 2:
            return self._summary([], cols, window)

        result = scores[["start", "stability_score"]].copy()
        result["rolling_avg"] = (
            result["stability_score"].rolling(n_windows, min_periods=1).mean()
        )
        result["score_change"] = result["stability_score"].diff()
        result["trend_direction"] = np.where(
            result["score_change"] > 2,
            "improving",
            np.where(result["score_change"] < -2, "degrading", "stable"),
        )

        return self._summary(
            result[cols].dropna(subset=["score_change"]).reset_index(drop=True),
            cols,
            window,
        )

    @keyword_only(1)
    def worst_periods(
        self,
        window: str = "1h",
        n: int = 5,
    ) -> pd.DataFrame:
        """Return the N worst-scoring windows.

        Args:
            window: Scoring window passed to :meth:`stability_score`.
            n: Number of lowest-scoring windows to return.

        Returns:
            DataFrame with columns: start, end, duration_seconds, uuid,
            source_uuid, stability_score,
            variance_score, bias_score, excursion_score, smoothness_score,
            primary_issue.
        """
        cols = [
            "start",
            "stability_score",
            "variance_score",
            "bias_score",
            "excursion_score",
            "smoothness_score",
            "primary_issue",
        ]
        scores = self.stability_score(window=window)
        if scores.empty:
            return self._summary([], cols, window)

        sorted_scores = scores.sort_values("stability_score").head(n).copy()

        # Identify primary issue = lowest sub-score
        sub_cols = [
            "variance_score",
            "bias_score",
            "excursion_score",
            "smoothness_score",
        ]
        issue_map = {
            "variance_score": "high_variance",
            "bias_score": "off_target",
            "excursion_score": "excursions",
            "smoothness_score": "rough_signal",
        }
        sorted_scores["primary_issue"] = (
            sorted_scores[sub_cols].idxmin(axis=1).map(issue_map)
        )

        return self._summary(sorted_scores[cols].reset_index(drop=True), cols, window)

    @keyword_only(1)
    @twin_columns({"pct_of_best": "share_of_best_pct"})
    def stability_comparison(self, window: str = "8h") -> pd.DataFrame:
        """Compare each window to the best-observed window.

        Args:
            window: Scoring window passed to :meth:`stability_score`.

        Returns:
            DataFrame with columns: start, end, duration_seconds, uuid,
            source_uuid, stability_score,
            best_score, gap_to_best, pct_of_best.

            Canonical twin columns: ``share_of_best_pct`` (after ``pct_of_best``); each holds
            the same values as the deprecated column right before it, which is removed
            in the next major release.
        """
        cols = [
            "start",
            "stability_score",
            "best_score",
            "gap_to_best",
            "pct_of_best",
        ]
        scores = self.stability_score(window=window)
        if scores.empty:
            return self._summary([], cols, window)

        best = float(scores["stability_score"].max())
        result = scores[["start", "stability_score"]].copy()
        result["best_score"] = best
        result["gap_to_best"] = best - result["stability_score"]
        result["pct_of_best"] = (
            result["stability_score"] / best * 100 if best > 0 else 0.0
        )

        return self._summary(result[cols].reset_index(drop=True), cols, window)


#: Deprecated name, removed in the next major release.
if TYPE_CHECKING:

    @deprecated("ProcessStabilityIndex is deprecated, use ProcessStabilityIndexEvents")
    class ProcessStabilityIndex(ProcessStabilityIndexEvents): ...

else:
    ProcessStabilityIndex = deprecated_class(
        ProcessStabilityIndexEvents, "ProcessStabilityIndex"
    )
