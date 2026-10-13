import logging
from typing import Any

import pandas as pd  # type: ignore

from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class BottleneckDetectionEvents(Base):
    """Production: Bottleneck Detection

    Identify which station constrains a production line by analyzing
    utilization of multiple boolean run-state signals.

    Methods:
    - station_utilization: Per-station uptime percentage per window.
    - detect_bottleneck: Identify the bottleneck station per window.
    - shifting_bottleneck: Track when the bottleneck moves between stations.
    - throughput_constraint_summary: Summary statistics across all stations.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        time_column: str = "systime",
        event_uuid: str = "prod:bottleneck",
        value_column: str = "value_bool",
    ) -> None:
        super().__init__(dataframe, column_name=time_column)
        self.time_column = time_column
        self.event_uuid = event_uuid
        self.value_column = value_column

    def station_utilization(
        self,
        station_uuids: list[str],
        window: str = "1h",
        *,
        time_weighted: bool = False,
    ) -> pd.DataFrame:
        """Per-station uptime percentage per time window.

        Args:
            station_uuids: List of UUID strings identifying station run-state signals.
            window: Resample window (e.g. '1h', '30m').
            time_weighted: When False (default), utilization is the mean of the
                boolean samples in the window -- correct only under uniform
                sampling. On change-encoded / irregular signals the result is
                driven by how many rows land in the window, so a station
                emitting one ``True`` per hour reads 100 %. When True,
                utilization is time-weighted: each sample is weighted by the gap
                to the next sample (as in ``RuntimeAccountingEvents``), so it
                reflects wall-clock time in the running state.

        Returns:
            DataFrame with columns: start, uuid, utilization_pct.
        """
        rows: list[dict[str, Any]] = []

        for uid in station_uuids:
            station = self.dataframe[self.dataframe["uuid"] == uid].copy()
            if station.empty:
                continue
            station[self.time_column] = pd.to_datetime(station[self.time_column])
            station = station.sort_values(self.time_column)
            station["state"] = station[self.value_column].fillna(False).astype(float)

            if time_weighted:
                # Weight each sample by the gap to the next sample, bucketed by
                # the sample's window. The trailing sample has no successor.
                station["dur"] = (
                    station[self.time_column].shift(-1) - station[self.time_column]
                ).dt.total_seconds()
                station = station[station["dur"].notna()]
                if station.empty:
                    continue
                station["win"] = station[self.time_column].dt.floor(window)
                station["run_dur"] = station["state"] * station["dur"]
                grouped = station.groupby("win").agg(
                    run=("run_dur", "sum"), total=("dur", "sum")
                )
                series = grouped.apply(
                    lambda r: (r["run"] / r["total"]) if r["total"] > 0 else 0.0,
                    axis=1,
                )
            else:
                series = (
                    station.set_index(self.time_column)["state"].resample(window).mean()
                )

            for ts, pct in series.items():
                if pd.notna(pct):
                    rows.append(
                        {
                            "start": ts,
                            "uuid": uid,
                            "utilization_pct": round(pct * 100, 2),
                        }
                    )

        return (
            pd.DataFrame(rows)
            if rows
            else pd.DataFrame(columns=["start", "uuid", "utilization_pct"])
        )

    def detect_bottleneck(
        self,
        station_uuids: list[str],
        window: str = "1h",
        *,
        time_weighted: bool = False,
    ) -> pd.DataFrame:
        """Identify the bottleneck station per window.

        The bottleneck is the station with the highest utilization —
        it is always running because it is the constraint.

        Args:
            station_uuids: List of station run-state UUIDs.
            window: Resample window.
            time_weighted: Forwarded to :meth:`station_utilization` -- pass True
                for change-encoded/irregular signals.

        Returns:
            DataFrame with columns: start, end, bottleneck_uuid, utilization_pct.
        """
        util = self.station_utilization(
            station_uuids, window, time_weighted=time_weighted
        )
        if util.empty:
            return pd.DataFrame(
                columns=[
                    "start",
                    "end",
                    "bottleneck_uuid",
                    "utilization_pct",
                ]
            )

        idx = util.groupby("start")["utilization_pct"].idxmax()
        bottlenecks = util.loc[idx].copy()
        window_td = pd.to_timedelta(window)

        return pd.DataFrame(
            {
                "start": bottlenecks["start"].array,
                "end": bottlenecks["start"].array + window_td,
                "bottleneck_uuid": bottlenecks["uuid"].values,
                "utilization_pct": bottlenecks["utilization_pct"].values,
            }
        )

    def shifting_bottleneck(
        self,
        station_uuids: list[str],
        window: str = "1h",
        *,
        time_weighted: bool = False,
    ) -> pd.DataFrame:
        """Track when the bottleneck identity changes between stations.

        Args:
            station_uuids: List of station run-state UUIDs.
            window: Resample window.
            time_weighted: Forwarded to :meth:`station_utilization`.

        Returns:
            DataFrame with columns: systime, from_uuid, to_uuid,
            previous_utilization, new_utilization.
        """
        bottlenecks = self.detect_bottleneck(
            station_uuids, window, time_weighted=time_weighted
        )
        if bottlenecks.empty or len(bottlenecks) < 2:
            return pd.DataFrame(
                columns=[
                    "systime",
                    "from_uuid",
                    "to_uuid",
                    "previous_utilization",
                    "new_utilization",
                ]
            )

        shifts: list[dict[str, Any]] = []
        prev_uuid = bottlenecks.iloc[0]["bottleneck_uuid"]
        prev_util = bottlenecks.iloc[0]["utilization_pct"]

        for i in range(1, len(bottlenecks)):
            curr_uuid = bottlenecks.iloc[i]["bottleneck_uuid"]
            curr_util = bottlenecks.iloc[i]["utilization_pct"]
            if curr_uuid != prev_uuid:
                shifts.append(
                    {
                        "systime": bottlenecks.iloc[i]["start"],
                        "from_uuid": prev_uuid,
                        "to_uuid": curr_uuid,
                        "previous_utilization": prev_util,
                        "new_utilization": curr_util,
                    }
                )
            prev_uuid = curr_uuid
            prev_util = curr_util

        return (
            pd.DataFrame(shifts)
            if shifts
            else pd.DataFrame(
                columns=[
                    "systime",
                    "from_uuid",
                    "to_uuid",
                    "previous_utilization",
                    "new_utilization",
                ]
            )
        )

    def throughput_constraint_summary(
        self,
        station_uuids: list[str],
        window: str = "1h",
        *,
        time_weighted: bool = False,
    ) -> dict[str, Any]:
        """Summary statistics for bottleneck analysis.

        Args:
            station_uuids: List of station run-state UUIDs.
            window: Resample window.
            time_weighted: Forwarded to :meth:`station_utilization`.

        Returns:
            Dict with: bottleneck_counts, bottleneck_percentages,
            most_frequent_bottleneck, avg_utilization_per_station.
        """
        bottlenecks = self.detect_bottleneck(
            station_uuids, window, time_weighted=time_weighted
        )
        util = self.station_utilization(
            station_uuids, window, time_weighted=time_weighted
        )

        if bottlenecks.empty:
            return {
                "bottleneck_counts": {},
                "bottleneck_percentages": {},
                "most_frequent_bottleneck": None,
                "avg_utilization_per_station": {},
            }

        counts = bottlenecks["bottleneck_uuid"].value_counts().to_dict()
        total = len(bottlenecks)
        percentages = {k: round(v / total * 100, 2) for k, v in counts.items()}
        most_frequent = bottlenecks["bottleneck_uuid"].value_counts().idxmax()

        avg_util = {}
        if not util.empty:
            avg_util = util.groupby("uuid")["utilization_pct"].mean().round(2).to_dict()

        return {
            "bottleneck_counts": counts,
            "bottleneck_percentages": percentages,
            "most_frequent_bottleneck": most_frequent,
            "avg_utilization_per_station": avg_util,
        }
