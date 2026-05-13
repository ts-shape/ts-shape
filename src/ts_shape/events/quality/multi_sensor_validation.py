import logging
import pandas as pd  # type: ignore
from itertools import combinations
from typing import List, Dict, Any

from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class MultiSensorValidationEvents(Base):
    """Quality: Multi-Sensor Cross-Validation

    Cross-validate redundant inline sensors measuring the same process
    variable. Detects disagreement, identifies drifting sensors, and
    assesses measurement consensus.

    Methods:
    - detect_disagreement: Flag windows where sensor spread exceeds threshold.
    - pairwise_bias: Mean difference between each sensor pair per window.
    - consensus_score: Per-window measurement consensus across sensors.
    - identify_outlier_sensor: Find the sensor furthest from the group.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        sensor_uuids: List[str],
        *,
        value_column: str = "value_double",
        event_uuid: str = "quality:multi_sensor",
        time_column: str = "systime",
    ) -> None:
        super().__init__(dataframe, column_name=time_column)
        if len(sensor_uuids) < 2:
            raise ValueError("At least 2 sensor UUIDs are required")
        self.sensor_uuids = sensor_uuids
        self.value_column = value_column
        self.event_uuid = event_uuid
        self.time_column = time_column

        # Build a pivoted DataFrame: time × sensor
        self._pivot = self._build_pivot()

    def _build_pivot(self) -> pd.DataFrame:
        """Pivot sensor data so each sensor is a column aligned by time."""
        mask = self.dataframe["uuid"].isin(self.sensor_uuids)
        df = self.dataframe.loc[
            mask, [self.time_column, "uuid", self.value_column]
        ].copy()
        if df.empty:
            return pd.DataFrame()

        df[self.time_column] = pd.to_datetime(df[self.time_column])
        pivot = df.pivot_table(
            index=self.time_column,
            columns="uuid",
            values=self.value_column,
            aggfunc="mean",
        )
        # Forward-fill small gaps so sensors align
        pivot = pivot.sort_index().ffill(limit=5)
        return pivot

    def detect_disagreement(self, threshold: float, window: str = "5m") -> pd.DataFrame:
        """Flag windows where sensor spread exceeds threshold.

        Args:
            threshold: Maximum acceptable spread (max - min) across sensors.
            window: Resample window size.

        Returns:
            DataFrame with columns: window_start, window_end, max_spread,
            sensor_high, sensor_low, duration.
        """
        cols = [
            "window_start",
            "window_end",
            "max_spread",
            "sensor_high",
            "sensor_low",
            "duration",
        ]
        if self._pivot.empty:
            return pd.DataFrame(columns=cols)

        events: List[Dict[str, Any]] = []
        for ts, group in self._pivot.resample(window):
            group = group.dropna(how="all")
            if group.empty or len(group) < 1:
                continue

            means = group.mean()
            valid_means = means.dropna()
            if len(valid_means) < 2:
                continue

            spread = float(valid_means.max() - valid_means.min())
            if spread > threshold:
                window_end = ts + pd.to_timedelta(window)
                events.append(
                    {
                        "window_start": ts,
                        "window_end": window_end,
                        "max_spread": round(spread, 6),
                        "sensor_high": valid_means.idxmax(),
                        "sensor_low": valid_means.idxmin(),
                        "duration": window_end - ts,
                    }
                )

        return (
            pd.DataFrame(events, columns=cols) if events else pd.DataFrame(columns=cols)
        )

    def pairwise_bias(self, window: str = "1h") -> pd.DataFrame:
        """Mean difference between each sensor pair per window.

        Args:
            window: Resample window size.

        Returns:
            DataFrame with columns: window_start, sensor_a, sensor_b,
            bias, abs_bias.
        """
        cols = ["window_start", "sensor_a", "sensor_b", "bias", "abs_bias"]
        if self._pivot.empty:
            return pd.DataFrame(columns=cols)

        pairs = list(combinations(self.sensor_uuids, 2))
        events: List[Dict[str, Any]] = []

        for ts, group in self._pivot.resample(window):
            group = group.dropna(how="all")
            if group.empty:
                continue

            means = group.mean()
            for a, b in pairs:
                if a not in means.index or b not in means.index:
                    continue
                if pd.isna(means[a]) or pd.isna(means[b]):
                    continue
                bias = float(means[a] - means[b])
                events.append(
                    {
                        "window_start": ts,
                        "sensor_a": a,
                        "sensor_b": b,
                        "bias": round(bias, 6),
                        "abs_bias": round(abs(bias), 6),
                    }
                )

        return (
            pd.DataFrame(events, columns=cols) if events else pd.DataFrame(columns=cols)
        )

    def consensus_score(self, window: str = "1h") -> pd.DataFrame:
        """Per-window measurement consensus across sensors.

        Score = 1.0 means perfect agreement, 0.0 means high disagreement.

        Args:
            window: Resample window size.

        Returns:
            DataFrame with columns: window_start, consensus_mean,
            spread_std, consensus_score.
        """
        cols = ["window_start", "consensus_mean", "spread_std", "consensus_score"]
        if self._pivot.empty:
            return pd.DataFrame(columns=cols)

        events: List[Dict[str, Any]] = []
        for ts, group in self._pivot.resample(window):
            group = group.dropna(how="all")
            if group.empty:
                continue

            means = group.mean().dropna()
            if len(means) < 2:
                continue

            consensus_mean = float(means.mean())
            spread_std = float(means.std())

            # Normalize: score = 1 - spread_std / |consensus_mean|
            if abs(consensus_mean) > 1e-10:
                score = max(0.0, 1.0 - spread_std / abs(consensus_mean))
            else:
                score = max(0.0, 1.0 - spread_std) if spread_std < 1.0 else 0.0

            events.append(
                {
                    "window_start": ts,
                    "consensus_mean": round(consensus_mean, 6),
                    "spread_std": round(spread_std, 6),
                    "consensus_score": round(min(1.0, score), 4),
                }
            )

        return (
            pd.DataFrame(events, columns=cols) if events else pd.DataFrame(columns=cols)
        )

    def identify_outlier_sensor(
        self, window: str = "1h", method: str = "median"
    ) -> pd.DataFrame:
        """Identify the sensor furthest from the group per window.

        Args:
            window: Resample window size.
            method: 'median' or 'mean' — central tendency for comparison.

        Returns:
            DataFrame with columns: window_start, outlier_sensor,
            deviation, other_sensors_mean.
        """
        cols = ["window_start", "outlier_sensor", "deviation", "other_sensors_mean"]
        if self._pivot.empty:
            return pd.DataFrame(columns=cols)

        events: List[Dict[str, Any]] = []
        for ts, group in self._pivot.resample(window):
            group = group.dropna(how="all")
            if group.empty:
                continue

            means = group.mean().dropna()
            if len(means) < 2:
                continue

            if method == "median":
                center = float(means.median())
            else:
                center = float(means.mean())

            deviations = (means - center).abs()
            outlier = deviations.idxmax()
            deviation = float(deviations[outlier])

            # Mean of all other sensors
            others = means.drop(outlier)
            other_mean = float(others.mean())

            events.append(
                {
                    "window_start": ts,
                    "outlier_sensor": outlier,
                    "deviation": round(deviation, 6),
                    "other_sensors_mean": round(other_mean, 6),
                }
            )

        return (
            pd.DataFrame(events, columns=cols) if events else pd.DataFrame(columns=cols)
        )
