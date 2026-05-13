import logging
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from typing import List, Dict, Any, Optional

from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class AnomalyCorrelationEvents(Base):
    """Correlation: Anomaly Correlation Analysis

    Correlate anomaly events across multiple signals to find coincident
    patterns, cascading failures, and root cause candidates.

    Methods:
    - coincident_anomalies: Find anomalies that co-occur within a time window.
    - cascade_detection: Detect anomaly cascades (A precedes B within a window).
    - root_cause_ranking: Rank signals by how often their anomalies precede others.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        event_uuid: str = "corr:anomaly",
        value_column: str = "value_double",
        time_column: str = "systime",
    ) -> None:
        super().__init__(dataframe, column_name=time_column)
        self.event_uuid = event_uuid
        self.value_column = value_column
        self.time_column = time_column

    def _detect_signal_anomalies(
        self,
        signal_uuid: str,
        *,
        z_threshold: float = 3.0,
    ) -> pd.DataFrame:
        """Internal: detect anomalies in a single signal using Z-score."""
        s = (
            self.dataframe[self.dataframe["uuid"] == signal_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        if s.empty or len(s) < 3:
            return pd.DataFrame(
                columns=[self.time_column, "uuid", self.value_column, "z_score"]
            )

        s[self.time_column] = pd.to_datetime(s[self.time_column])
        values = s[self.value_column]
        mean = values.mean()
        std = values.std()
        if std == 0:
            return pd.DataFrame(
                columns=[self.time_column, "uuid", self.value_column, "z_score"]
            )

        s["z_score"] = ((values - mean) / std).abs()
        anomalies = s[s["z_score"] >= z_threshold][
            [self.time_column, "uuid", self.value_column, "z_score"]
        ]
        return anomalies.reset_index(drop=True)

    def coincident_anomalies(
        self,
        signal_uuids: List[str],
        *,
        z_threshold: float = 3.0,
        coincidence_window: str = "5min",
        min_signals: int = 2,
    ) -> pd.DataFrame:
        """Find anomalies that co-occur across multiple signals within a time window.

        Args:
            signal_uuids: List of signal UUIDs to analyze.
            z_threshold: Z-score threshold for anomaly detection per signal.
            coincidence_window: Time window for considering anomalies coincident.
            min_signals: Minimum number of signals with anomalies to flag.

        Returns:
            DataFrame: window_start, window_end, uuid, is_delta,
                       anomaly_count, signal_uuids_involved
        """
        all_anomalies = []
        for uid in signal_uuids:
            anom = self._detect_signal_anomalies(uid, z_threshold=z_threshold)
            if not anom.empty:
                all_anomalies.append(anom)

        if not all_anomalies:
            return pd.DataFrame(
                columns=[
                    "window_start",
                    "window_end",
                    "uuid",
                    "is_delta",
                    "anomaly_count",
                    "signal_uuids_involved",
                ]
            )

        combined = pd.concat(all_anomalies, ignore_index=True)
        combined = combined.sort_values(self.time_column)

        window_td = pd.to_timedelta(coincidence_window)
        rows: List[Dict[str, Any]] = []
        processed = set()

        for i, row in combined.iterrows():
            if i in processed:
                continue
            t = row[self.time_column]
            window_mask = (combined[self.time_column] >= t) & (
                combined[self.time_column] <= t + window_td
            )
            window_data = combined[window_mask]
            unique_signals = window_data["uuid"].unique()

            if len(unique_signals) >= min_signals:
                rows.append(
                    {
                        "window_start": t,
                        "window_end": t + window_td,
                        "uuid": self.event_uuid,
                        "is_delta": True,
                        "anomaly_count": len(window_data),
                        "signal_uuids_involved": ",".join(sorted(unique_signals)),
                    }
                )
                processed.update(window_data.index.tolist())

        return pd.DataFrame(rows)

    def cascade_detection(
        self,
        leader_uuid: str,
        follower_uuid: str,
        *,
        z_threshold: float = 3.0,
        max_delay: str = "10min",
    ) -> pd.DataFrame:
        """Detect anomaly cascades: leader anomaly followed by follower anomaly.

        Identifies cases where an anomaly in signal A is followed by an
        anomaly in signal B within the max_delay window.

        Args:
            leader_uuid: UUID of the leading signal.
            follower_uuid: UUID of the following signal.
            z_threshold: Z-score threshold for anomaly detection.
            max_delay: Maximum time between leader and follower anomaly.

        Returns:
            DataFrame: leader_time, follower_time, uuid, is_delta,
                       leader_uuid, follower_uuid, delay_seconds
        """
        leader_anom = self._detect_signal_anomalies(
            leader_uuid, z_threshold=z_threshold
        )
        follower_anom = self._detect_signal_anomalies(
            follower_uuid, z_threshold=z_threshold
        )

        if leader_anom.empty or follower_anom.empty:
            return pd.DataFrame(
                columns=[
                    "leader_time",
                    "follower_time",
                    "uuid",
                    "is_delta",
                    "leader_uuid",
                    "follower_uuid",
                    "delay_seconds",
                ]
            )

        max_delay_td = pd.to_timedelta(max_delay)
        rows: List[Dict[str, Any]] = []
        used_followers = set()

        for _, lrow in leader_anom.iterrows():
            lt = lrow[self.time_column]
            candidates = follower_anom[
                (follower_anom[self.time_column] > lt)
                & (follower_anom[self.time_column] <= lt + max_delay_td)
            ]
            for fidx, frow in candidates.iterrows():
                if fidx not in used_followers:
                    rows.append(
                        {
                            "leader_time": lt,
                            "follower_time": frow[self.time_column],
                            "uuid": self.event_uuid,
                            "is_delta": True,
                            "leader_uuid": leader_uuid,
                            "follower_uuid": follower_uuid,
                            "delay_seconds": (
                                frow[self.time_column] - lt
                            ).total_seconds(),
                        }
                    )
                    used_followers.add(fidx)
                    break  # one follower per leader

        return pd.DataFrame(rows)

    def root_cause_ranking(
        self,
        signal_uuids: List[str],
        *,
        z_threshold: float = 3.0,
        max_delay: str = "10min",
    ) -> pd.DataFrame:
        """Rank signals by how often their anomalies precede others.

        For each pair of signals, counts how many times signal A's anomaly
        precedes signal B's anomaly within max_delay. Signals that frequently
        lead are potential root causes.

        Args:
            signal_uuids: List of signal UUIDs.
            z_threshold: Z-score threshold.
            max_delay: Maximum delay for cascade detection.

        Returns:
            DataFrame: signal_uuid, leader_count, follower_count,
                       leader_ratio, rank
        """
        if len(signal_uuids) < 2:
            return pd.DataFrame(
                columns=[
                    "signal_uuid",
                    "leader_count",
                    "follower_count",
                    "leader_ratio",
                    "rank",
                ]
            )

        leader_counts: Dict[str, int] = {uid: 0 for uid in signal_uuids}
        follower_counts: Dict[str, int] = {uid: 0 for uid in signal_uuids}

        for i, uid_a in enumerate(signal_uuids):
            for uid_b in signal_uuids[i + 1 :]:
                cascades_ab = self.cascade_detection(
                    uid_a, uid_b, z_threshold=z_threshold, max_delay=max_delay
                )
                cascades_ba = self.cascade_detection(
                    uid_b, uid_a, z_threshold=z_threshold, max_delay=max_delay
                )
                leader_counts[uid_a] += len(cascades_ab)
                follower_counts[uid_b] += len(cascades_ab)
                leader_counts[uid_b] += len(cascades_ba)
                follower_counts[uid_a] += len(cascades_ba)

        rows = []
        for uid in signal_uuids:
            total = leader_counts[uid] + follower_counts[uid]
            rows.append(
                {
                    "signal_uuid": uid,
                    "leader_count": leader_counts[uid],
                    "follower_count": follower_counts[uid],
                    "leader_ratio": leader_counts[uid] / total if total > 0 else 0.0,
                }
            )

        result = pd.DataFrame(rows)
        result = result.sort_values("leader_ratio", ascending=False).reset_index(
            drop=True
        )
        result["rank"] = range(1, len(result) + 1)
        return result
