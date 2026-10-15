import logging
import warnings
from itertools import combinations
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ts_shape.errors import DataQualityWarning
from ts_shape.events._windows import finish_events
from ts_shape.utils.base import Base
from ts_shape.utils.deprecation import (
    REMOVAL,
    caller_stacklevel,
    deprecated_method,
    keyword_only,
    renamed_kwargs,
)

if TYPE_CHECKING:
    from typing_extensions import deprecated

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
    - detect_outlier_sensor: Find the sensor furthest from the group.
    - sensor_uuids: Deprecated read-only alias of ``signal_uuids``.
    """

    @renamed_kwargs(sensor_uuids="signal_uuids")
    def __init__(
        self,
        dataframe: pd.DataFrame,
        signal_uuids: list[str],
        *,
        value_column: str = "value_double",
        event_uuid: str = "quality:multi_sensor",
        time_column: str = "systime",
    ) -> None:
        super().__init__(dataframe, time_column=time_column)
        if len(signal_uuids) < 2:
            raise ValueError("At least 2 sensor UUIDs are required")
        self.signal_uuids = signal_uuids
        self.value_column = value_column
        self.event_uuid = event_uuid
        self.time_column = time_column

        # Build a pivoted DataFrame: time × sensor
        self._pivot = self._build_pivot()

    @property
    def sensor_uuids(self) -> list[str]:
        """Deprecated alias of :attr:`signal_uuids` (read-only)."""
        warnings.warn(
            "MultiSensorValidationEvents.sensor_uuids is deprecated, use "
            f".signal_uuids; the old name will be removed in {REMOVAL}",
            DeprecationWarning,
            stacklevel=caller_stacklevel(),
        )
        return self.signal_uuids

    def _build_pivot(self) -> pd.DataFrame:
        """Pivot sensor data so each sensor is a column aligned by time."""
        mask = self.dataframe["uuid"].isin(self.signal_uuids)
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

    @keyword_only(2)
    def detect_disagreement(
        self, threshold: float, window: str = "5min"
    ) -> pd.DataFrame:
        """Flag windows where sensor spread exceeds threshold.

        Args:
            threshold: Maximum acceptable spread (max - min) across sensors.
            window: Resample window size.

        Returns:
            DataFrame with columns: start, end, duration_seconds, uuid,
            source_uuid, max_spread, sensor_high, sensor_low, duration.
            ``duration`` is a Timedelta; ``duration_seconds`` is the same
            span in float seconds. ``source_uuid`` is the comma-joined
            sensor uuids.
        """
        cols = [
            "start",
            "end",
            "max_spread",
            "sensor_high",
            "sensor_low",
            "duration",
        ]
        if self._pivot.empty:
            return finish_events(
                [],
                cols,
                "interval",
                uuid=self.event_uuid,
                source_uuid=",".join(self.signal_uuids),
            )

        # Per-window sensor means for all windows at once; a window needs at
        # least two sensors with data.
        means = self._pivot.resample(window).mean()
        means = means[means.notna().sum(axis=1) >= 2]
        spread = means.max(axis=1) - means.min(axis=1)
        hit = spread > threshold
        means, spread = means[hit], spread[hit]
        events: list[dict[str, Any]] = []
        if not means.empty:
            win_td = pd.to_timedelta(window)
            for ts, sp, hi, lo in zip(
                means.index,
                spread,
                means.idxmax(axis=1),
                means.idxmin(axis=1),
            ):
                end = ts + win_td
                events.append(
                    {
                        "start": ts,
                        "end": end,
                        "max_spread": float(sp),
                        "sensor_high": hi,
                        "sensor_low": lo,
                        "duration": end - ts,
                    }
                )

        return finish_events(
            events,
            cols,
            "interval",
            uuid=self.event_uuid,
            source_uuid=",".join(self.signal_uuids),
        )

    @keyword_only(1)
    def pairwise_bias(self, window: str = "1h") -> pd.DataFrame:
        """Mean difference between each sensor pair per window.

        Args:
            window: Resample window size.

        Returns:
            DataFrame with columns: start, sensor_a, sensor_b,
            bias, abs_bias.
        """
        cols = ["start", "sensor_a", "sensor_b", "bias", "abs_bias"]
        if self._pivot.empty:
            return pd.DataFrame(columns=cols)

        pairs = list(combinations(self.signal_uuids, 2))
        events: list[dict[str, Any]] = []

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
                        "start": ts,
                        "sensor_a": a,
                        "sensor_b": b,
                        "bias": float(bias),
                        "abs_bias": float(abs(bias)),
                    }
                )

        return (
            pd.DataFrame(events, columns=cols) if events else pd.DataFrame(columns=cols)
        )

    @keyword_only(1)
    def consensus_score(self, window: str = "1h") -> pd.DataFrame:
        """Per-window measurement consensus across sensors.

        Score = 1.0 means perfect agreement, 0.0 means high disagreement.

        Args:
            window: Resample window size.

        Returns:
            DataFrame with columns: start, end, duration_seconds, uuid,
            source_uuid, consensus_mean, spread_std, consensus_score.
            ``end`` is ``start`` plus one ``window``; ``source_uuid`` is the
            comma-joined sensor uuids.
        """
        cols = ["start", "consensus_mean", "spread_std", "consensus_score"]
        if self._pivot.empty:
            return finish_events(
                [],
                cols,
                "summary",
                uuid=self.event_uuid,
                source_uuid=",".join(self.signal_uuids),
                window=window,
            )

        events: list[dict[str, Any]] = []
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
                    "start": ts,
                    "consensus_mean": float(consensus_mean),
                    "spread_std": float(spread_std),
                    "consensus_score": float(min(1.0, score)),
                }
            )

        return finish_events(
            events,
            cols,
            "summary",
            uuid=self.event_uuid,
            source_uuid=",".join(self.signal_uuids),
            window=window,
        )

    if TYPE_CHECKING:

        @deprecated(
            "MultiSensorValidationEvents.identify_outlier_sensor() is deprecated, use .detect_outlier_sensor()"
        )
        def identify_outlier_sensor(
            self, window: str = "1h", method: str = "median"
        ) -> pd.DataFrame: ...

    else:
        identify_outlier_sensor = deprecated_method("detect_outlier_sensor")

    @keyword_only(1)
    def detect_outlier_sensor(
        self, window: str = "1h", method: str = "median"
    ) -> pd.DataFrame:
        """Identify the sensor furthest from the group per window.

        Telling which sensor is off needs at least three: with two, each is
        as far from the other as the other is from it. Windows with fewer
        than three sensors reporting are skipped with one
        :class:`~ts_shape.errors.DataQualityWarning`. When several sensors are
        equally far from the center, the first one in ``signal_uuids`` wins.

        Args:
            window: Resample window size.
            method: 'median' or 'mean' — central tendency for comparison.

        Returns:
            DataFrame with columns: start, outlier_sensor,
            deviation, other_sensors_mean.
        """
        cols = ["start", "outlier_sensor", "deviation", "other_sensors_mean"]
        if self._pivot.empty:
            return pd.DataFrame(columns=cols)

        events: list[dict[str, Any]] = []
        undecidable = 0
        for ts, group in self._pivot.resample(window):
            group = group.dropna(how="all")
            if group.empty:
                continue

            means = group.mean().dropna()
            if len(means) < 3:
                undecidable += 1
                continue

            if method == "median":
                center = float(means.median())
            else:
                center = float(means.mean())

            deviations = (means - center).abs()
            largest = deviations.max()
            tied = deviations.index[np.isclose(deviations, largest, rtol=1e-12)]
            order = [u for u in self.signal_uuids if u in tied]
            outlier = order[0] if order else tied[0]
            deviation = float(deviations[outlier])

            # Mean of all other sensors
            others = means.drop(outlier)
            other_mean = float(others.mean())

            events.append(
                {
                    "start": ts,
                    "outlier_sensor": outlier,
                    "deviation": float(deviation),
                    "other_sensors_mean": float(other_mean),
                }
            )

        if undecidable:
            warnings.warn(
                f"{type(self).__name__}.detect_outlier_sensor: {undecidable} "
                "window(s) had fewer than three sensors reporting; an outlier "
                "sensor cannot be identified there and they were skipped.",
                DataQualityWarning,
                stacklevel=caller_stacklevel(),
            )
        return (
            pd.DataFrame(events, columns=cols) if events else pd.DataFrame(columns=cols)
        )
