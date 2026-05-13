import logging
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from typing import List, Dict, Any, Optional

from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class ProcessStabilityIndex(Base):
    """Engineering: Process Stability Index

    Generate a single 0-100 stability score per shift/day for a signal.
    Answers 'is my process running well today?' with one number.

    Methods:
    - stability_score: Composite 0-100 score per window.
    - score_trend: Is stability improving or degrading?
    - worst_periods: N worst-scoring windows.
    - stability_comparison: Compare each window to best-observed.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        signal_uuid: str,
        *,
        target: Optional[float] = None,
        lower_spec: Optional[float] = None,
        upper_spec: Optional[float] = None,
        event_uuid: str = "eng:stability_index",
        value_column: str = "value_double",
        time_column: str = "systime",
    ) -> None:
        super().__init__(dataframe, column_name=time_column)
        self.signal_uuid = signal_uuid
        self.target = target
        self.lower_spec = lower_spec
        self.upper_spec = upper_spec
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

        if self.target is None:
            self.target = self._global_mean
        if self.lower_spec is None:
            self.lower_spec = self._global_mean - 3 * self._global_std
        if self.upper_spec is None:
            self.upper_spec = self._global_mean + 3 * self._global_std

    def stability_score(self, window: str = "8h") -> pd.DataFrame:
        """Composite 0-100 stability score per window.

        Four sub-scores (each 0-25):
        - variance_score: low std = high score
        - bias_score: on-target = high score
        - excursion_score: in-spec = high score
        - smoothness_score: low point-to-point jitter = high score

        Returns:
            DataFrame with columns: window_start, stability_score,
            variance_score, bias_score, excursion_score, smoothness_score,
            grade.
        """
        cols = [
            "window_start",
            "stability_score",
            "variance_score",
            "bias_score",
            "excursion_score",
            "smoothness_score",
            "grade",
        ]
        if self.signal.empty:
            return pd.DataFrame(columns=cols)

        indexed = self.signal.set_index(self.time_column)[self.value_column]
        groups = indexed.resample(window)
        spec_range = self.upper_spec - self.lower_spec
        half_range = spec_range / 2.0 if spec_range > 0 else 1.0

        events: List[Dict[str, Any]] = []
        for window_start, group in groups:
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
                    "window_start": window_start,
                    "stability_score": round(total, 1),
                    "variance_score": round(variance_score, 1),
                    "bias_score": round(bias_score, 1),
                    "excursion_score": round(excursion_score, 1),
                    "smoothness_score": round(smoothness_score, 1),
                    "grade": grade,
                }
            )

        return pd.DataFrame(events, columns=cols)

    def score_trend(
        self,
        window: str = "8h",
        n_windows: int = 7,
    ) -> pd.DataFrame:
        """Track whether stability is improving or degrading.

        Returns:
            DataFrame with columns: window_start, stability_score,
            rolling_avg, trend_direction, score_change.
        """
        cols = [
            "window_start",
            "stability_score",
            "rolling_avg",
            "trend_direction",
            "score_change",
        ]
        scores = self.stability_score(window)
        if scores.empty or len(scores) < 2:
            return pd.DataFrame(columns=cols)

        result = scores[["window_start", "stability_score"]].copy()
        result["rolling_avg"] = (
            result["stability_score"].rolling(n_windows, min_periods=1).mean()
        )
        result["score_change"] = result["stability_score"].diff()
        result["trend_direction"] = np.where(
            result["score_change"] > 2,
            "improving",
            np.where(result["score_change"] < -2, "degrading", "stable"),
        )

        return result[cols].dropna(subset=["score_change"]).reset_index(drop=True)

    def worst_periods(
        self,
        window: str = "1h",
        n: int = 5,
    ) -> pd.DataFrame:
        """Return the N worst-scoring windows.

        Returns:
            DataFrame with columns: window_start, stability_score,
            variance_score, bias_score, excursion_score, smoothness_score,
            primary_issue.
        """
        cols = [
            "window_start",
            "stability_score",
            "variance_score",
            "bias_score",
            "excursion_score",
            "smoothness_score",
            "primary_issue",
        ]
        scores = self.stability_score(window)
        if scores.empty:
            return pd.DataFrame(columns=cols)

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

        return sorted_scores[cols].reset_index(drop=True)

    def stability_comparison(self, window: str = "8h") -> pd.DataFrame:
        """Compare each window to the best-observed window.

        Returns:
            DataFrame with columns: window_start, stability_score,
            best_score, gap_to_best, pct_of_best.
        """
        cols = [
            "window_start",
            "stability_score",
            "best_score",
            "gap_to_best",
            "pct_of_best",
        ]
        scores = self.stability_score(window)
        if scores.empty:
            return pd.DataFrame(columns=cols)

        best = float(scores["stability_score"].max())
        result = scores[["window_start", "stability_score"]].copy()
        result["best_score"] = best
        result["gap_to_best"] = best - result["stability_score"]
        result["pct_of_best"] = (
            result["stability_score"] / best * 100 if best > 0 else 0.0
        )

        return result[cols].reset_index(drop=True)
