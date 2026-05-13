"""Value-based traceability across stations.

Given multiple station UUIDs that each carry a string signal (any shared
identifier — serial number, order number, batch ID, lot number, etc.),
build an end-to-end timeline showing when each identifier was at each
station, with durations and total lead times.

Typical use case: every machine on a production line writes the current
identifier to its own UUID.  This module joins those signals to answer
"when was identifier X at station A, then B, then C?"
"""

import logging
import pandas as pd  # type: ignore
import numpy as np
from typing import List, Dict, Any, Optional

from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class ValueTraceabilityEvents(Base):
    """Trace a shared identifier across multiple stations.

    Each station has its own UUID that carries a string value (the identifier
    currently being processed — serial number, order ID, batch code, etc.).
    This module detects when a given identifier appears at each station and
    builds a timeline.

    Example usage:
        trace = ValueTraceabilityEvents(
            df,
            station_uuids={
                'station_a_uuid': 'Station A',
                'station_b_uuid': 'Station B',
                'station_c_uuid': 'Station C',
            },
        )

        # Full timeline of every identifier across all stations
        timeline = trace.build_timeline()

        # Lead time from first to last station
        lead = trace.lead_time()

        # Where is each identifier right now?
        status = trace.current_status()

        # Station dwell-time statistics
        dwell = trace.station_dwell_statistics()
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        station_uuids: Dict[str, str],
        *,
        event_uuid: str = "prod:value_trace",
        value_column: str = "value_string",
        time_column: str = "systime",
    ) -> None:
        """Initialize value traceability.

        Args:
            dataframe: Input DataFrame with timeseries data.
            station_uuids: Mapping of UUID -> human-readable station name.
                           e.g. {'uuid_abc': 'Station A', 'uuid_def': 'Station B'}
            event_uuid: UUID to tag derived events with.
            value_column: Column holding the identifier string value.
            time_column: Name of timestamp column.
        """
        super().__init__(dataframe, column_name=time_column)
        self.station_uuids = station_uuids
        self.event_uuid = event_uuid
        self.value_column = value_column
        self.time_column = time_column

        # Pre-filter and parse per-station data
        self._station_data: Dict[str, pd.DataFrame] = {}
        for uuid, name in self.station_uuids.items():
            sdf = (
                self.dataframe[self.dataframe["uuid"] == uuid]
                .copy()
                .sort_values(self.time_column)
            )
            sdf[self.time_column] = pd.to_datetime(sdf[self.time_column])
            self._station_data[uuid] = sdf

    # ------------------------------------------------------------------
    # Internal: detect intervals per station (batch-tracking style)
    # ------------------------------------------------------------------

    def _detect_intervals(self, uuid: str) -> pd.DataFrame:
        """Detect contiguous intervals of each identifier at one station."""
        sdf = self._station_data.get(uuid, pd.DataFrame())
        if sdf.empty:
            return pd.DataFrame(
                columns=[
                    "identifier",
                    "station_uuid",
                    "station_name",
                    "start",
                    "end",
                    "duration_seconds",
                    "sample_count",
                ]
            )

        s = sdf[[self.time_column, self.value_column]].copy()
        s["identifier"] = s[self.value_column].fillna("")
        s["group"] = (s["identifier"] != s["identifier"].shift()).cumsum()

        station_name = self.station_uuids[uuid]
        rows: List[Dict[str, Any]] = []
        for _, seg in s.groupby("group"):
            identifier = seg["identifier"].iloc[0]
            if identifier == "":
                continue
            start = seg[self.time_column].iloc[0]
            end = seg[self.time_column].iloc[-1]
            rows.append(
                {
                    "identifier": identifier,
                    "station_uuid": uuid,
                    "station_name": station_name,
                    "start": start,
                    "end": end,
                    "duration_seconds": (end - start).total_seconds(),
                    "sample_count": len(seg),
                }
            )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # build_timeline
    # ------------------------------------------------------------------

    def build_timeline(self) -> pd.DataFrame:
        """Build a full timeline of every identifier at every station.

        Returns:
            DataFrame with columns:
            - identifier: The shared identifier string.
            - station_uuid: UUID of the station signal.
            - station_name: Human-readable station name.
            - start: Timestamp when the identifier appeared at the station.
            - end: Timestamp of the last sample at the station.
            - duration_seconds: Time spent at the station.
            - sample_count: Number of data points.
            - station_sequence: Order of visit (1-based) per identifier.
            - uuid: Event UUID.
        """
        all_intervals: List[pd.DataFrame] = []
        for uuid in self.station_uuids:
            intervals = self._detect_intervals(uuid)
            if not intervals.empty:
                all_intervals.append(intervals)

        if not all_intervals:
            return pd.DataFrame(
                columns=[
                    "identifier",
                    "station_uuid",
                    "station_name",
                    "start",
                    "end",
                    "duration_seconds",
                    "sample_count",
                    "station_sequence",
                    "uuid",
                ]
            )

        timeline = pd.concat(all_intervals, ignore_index=True)
        timeline = timeline.sort_values(["identifier", "start"]).reset_index(drop=True)

        # Assign station visit sequence per identifier
        timeline["station_sequence"] = timeline.groupby("identifier").cumcount() + 1
        timeline["uuid"] = self.event_uuid

        return timeline

    # ------------------------------------------------------------------
    # lead_time
    # ------------------------------------------------------------------

    def lead_time(self) -> pd.DataFrame:
        """Compute end-to-end lead time per identifier.

        Returns:
            DataFrame with columns:
            - identifier
            - first_station: Name of first station visited.
            - last_station: Name of last station visited.
            - first_seen: Earliest timestamp across all stations.
            - last_seen: Latest timestamp across all stations.
            - lead_time_seconds: Total time from first seen to last seen.
            - stations_visited: Number of distinct stations visited.
            - station_path: Arrow-separated ordered station names.
            - uuid: Event UUID.
        """
        timeline = self.build_timeline()

        if timeline.empty:
            return pd.DataFrame(
                columns=[
                    "identifier",
                    "first_station",
                    "last_station",
                    "first_seen",
                    "last_seen",
                    "lead_time_seconds",
                    "stations_visited",
                    "station_path",
                    "uuid",
                ]
            )

        rows: List[Dict[str, Any]] = []
        for identifier, grp in timeline.groupby("identifier"):
            grp = grp.sort_values("start")
            first_seen = grp["start"].iloc[0]
            last_seen = grp["end"].iloc[-1]
            station_path = " -> ".join(grp["station_name"].tolist())
            rows.append(
                {
                    "identifier": identifier,
                    "first_station": grp["station_name"].iloc[0],
                    "last_station": grp["station_name"].iloc[-1],
                    "first_seen": first_seen,
                    "last_seen": last_seen,
                    "lead_time_seconds": (last_seen - first_seen).total_seconds(),
                    "stations_visited": grp["station_name"].nunique(),
                    "station_path": station_path,
                    "uuid": self.event_uuid,
                }
            )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # current_status
    # ------------------------------------------------------------------

    def current_status(self) -> pd.DataFrame:
        """Determine the last-known station for each identifier.

        Returns:
            DataFrame with columns:
            - identifier
            - current_station: Name of the last station where seen.
            - current_station_uuid: UUID of that station.
            - arrived_at: When the identifier arrived at the current station.
            - time_at_station_seconds: Dwell time at the current station so far.
            - uuid: Event UUID.
        """
        timeline = self.build_timeline()

        if timeline.empty:
            return pd.DataFrame(
                columns=[
                    "identifier",
                    "current_station",
                    "current_station_uuid",
                    "arrived_at",
                    "time_at_station_seconds",
                    "uuid",
                ]
            )

        # Latest visit per identifier
        latest = (
            timeline.sort_values("start").groupby("identifier").last().reset_index()
        )

        return pd.DataFrame(
            {
                "identifier": latest["identifier"],
                "current_station": latest["station_name"],
                "current_station_uuid": latest["station_uuid"],
                "arrived_at": latest["start"],
                "time_at_station_seconds": latest["duration_seconds"],
                "uuid": self.event_uuid,
            }
        )

    # ------------------------------------------------------------------
    # station_dwell_statistics
    # ------------------------------------------------------------------

    def station_dwell_statistics(self) -> pd.DataFrame:
        """Compute dwell-time statistics per station (across all identifiers).

        Returns:
            DataFrame with columns:
            - station_name
            - station_uuid
            - identifier_count: Number of distinct identifiers seen.
            - min_dwell_seconds
            - avg_dwell_seconds
            - max_dwell_seconds
            - total_dwell_seconds
        """
        timeline = self.build_timeline()

        if timeline.empty:
            return pd.DataFrame(
                columns=[
                    "station_name",
                    "station_uuid",
                    "identifier_count",
                    "min_dwell_seconds",
                    "avg_dwell_seconds",
                    "max_dwell_seconds",
                    "total_dwell_seconds",
                ]
            )

        stats = (
            timeline.groupby(["station_name", "station_uuid"])
            .agg(
                identifier_count=("identifier", "nunique"),
                min_dwell_seconds=("duration_seconds", "min"),
                avg_dwell_seconds=("duration_seconds", "mean"),
                max_dwell_seconds=("duration_seconds", "max"),
                total_dwell_seconds=("duration_seconds", "sum"),
            )
            .reset_index()
        )

        for col in [
            "min_dwell_seconds",
            "avg_dwell_seconds",
            "max_dwell_seconds",
            "total_dwell_seconds",
        ]:
            stats[col] = stats[col].round(2)

        return stats


# Backwards-compatible alias
OrderTraceabilityEvents = ValueTraceabilityEvents
