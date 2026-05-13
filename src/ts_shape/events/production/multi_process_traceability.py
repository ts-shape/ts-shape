"""Multi-process traceability across parallel and sequential stations.

Handles real production topologies where:
- Each station / process segment has its own ID UUID (carrying the current
  order / serial number).
- Multiple handover signals exist between station pairs (fires when an
  item transfers).
- Stations can run in parallel (e.g., two welding cells feeding one
  painting station).

Example topology::

    Welding Cell 1 ─┐
                     ├─► Painting ─► Assembly ─► Test
    Welding Cell 2 ─┘

Each cell has its own ``id_uuid`` that carries the serial number currently
being processed.  Handover signals between cells confirm the transfer.
"""

import logging
import pandas as pd  # type: ignore
from typing import List, Dict, Any, Optional

from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class MultiProcessTraceabilityEvents(Base):
    """Trace items across a multi-station topology with parallel paths.

    Example usage::

        trace = MultiProcessTraceabilityEvents(
            df,
            processes=[
                {"id_uuid": "serial_weld1",  "station": "Welding Cell 1"},
                {"id_uuid": "serial_weld2",  "station": "Welding Cell 2"},
                {"id_uuid": "serial_paint",  "station": "Painting"},
                {"id_uuid": "serial_assy",   "station": "Assembly"},
            ],
            handovers=[
                {"uuid": "ho_w1_paint", "from_station": "Welding Cell 1", "to_station": "Painting"},
                {"uuid": "ho_w2_paint", "from_station": "Welding Cell 2", "to_station": "Painting"},
                {"uuid": "ho_paint_assy", "from_station": "Painting", "to_station": "Assembly"},
            ],
        )

        timeline = trace.build_timeline()
        lead = trace.lead_time()
        parallel = trace.parallel_activity()
        handover_log = trace.handover_log()
        paths = trace.routing_paths()
        station_stats = trace.station_statistics()
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        processes: List[Dict[str, str]],
        *,
        handovers: Optional[List[Dict[str, str]]] = None,
        event_uuid: str = "prod:multi_process_trace",
        id_value_column: str = "value_string",
        handover_value_column: str = "value_integer",
        time_column: str = "systime",
    ) -> None:
        """Initialize multi-process traceability.

        Args:
            dataframe: Input DataFrame with timeseries data.
            processes: List of process definitions, each a dict with:
                - id_uuid: UUID of the signal carrying item IDs at this station.
                - station: Human-readable station name.
            handovers: Optional list of handover signal definitions, each a dict:
                - uuid: UUID of the handover signal.
                - from_station: Station name the item leaves.
                - to_station: Station name the item arrives at.
                Handover signal value > 0 (or True) indicates a transfer event.
            event_uuid: UUID to tag derived events with.
            id_value_column: Column holding item ID strings in process signals.
            handover_value_column: Column holding handover trigger values.
            time_column: Name of timestamp column.
        """
        super().__init__(dataframe, column_name=time_column)
        self.processes = processes
        self.handovers = handovers or []
        self.event_uuid = event_uuid
        self.id_value_column = id_value_column
        self.handover_value_column = handover_value_column
        self.time_column = time_column

        # Build station lookup: station_name -> list of id_uuids (parallel cells)
        self._station_uuids: Dict[str, List[str]] = {}
        for proc in self.processes:
            station = proc["station"]
            self._station_uuids.setdefault(station, []).append(proc["id_uuid"])

        # Pre-filter process data per uuid
        self._process_data: Dict[str, pd.DataFrame] = {}
        for proc in self.processes:
            uid = proc["id_uuid"]
            sdf = (
                self.dataframe[self.dataframe["uuid"] == uid]
                .copy()
                .sort_values(self.time_column)
            )
            sdf[self.time_column] = pd.to_datetime(sdf[self.time_column])
            self._process_data[uid] = sdf

        # Pre-filter handover data per uuid
        self._handover_data: Dict[str, pd.DataFrame] = {}
        for ho in self.handovers:
            uid = ho["uuid"]
            hdf = (
                self.dataframe[self.dataframe["uuid"] == uid]
                .copy()
                .sort_values(self.time_column)
            )
            hdf[self.time_column] = pd.to_datetime(hdf[self.time_column])
            self._handover_data[uid] = hdf

    # ------------------------------------------------------------------
    # Internal: detect intervals at one process UUID
    # ------------------------------------------------------------------

    def _detect_intervals(self, id_uuid: str, station_name: str) -> pd.DataFrame:
        """Detect contiguous item-ID intervals at a single process UUID."""
        sdf = self._process_data.get(id_uuid, pd.DataFrame())
        if sdf.empty:
            return pd.DataFrame(
                columns=[
                    "item_id",
                    "station",
                    "id_uuid",
                    "start",
                    "end",
                    "duration_seconds",
                    "sample_count",
                ]
            )

        s = sdf[[self.time_column, self.id_value_column]].copy()
        s["item_id"] = s[self.id_value_column].fillna("")

        # Split on value changes OR large time gaps (>3x median sample interval).
        # This handles the case where the same item revisits a station after
        # being elsewhere (rework loops).
        value_change = s["item_id"] != s["item_id"].shift()
        dt = s[self.time_column].diff()
        median_dt = dt.median()
        if pd.notna(median_dt) and median_dt > pd.Timedelta(0):
            gap = dt > median_dt * 3
        else:
            gap = pd.Series(False, index=s.index)
        s["group"] = (value_change | gap).cumsum()

        rows: List[Dict[str, Any]] = []
        for _, seg in s.groupby("group"):
            item_id = seg["item_id"].iloc[0]
            if item_id == "":
                continue
            start = seg[self.time_column].iloc[0]
            end = seg[self.time_column].iloc[-1]
            rows.append(
                {
                    "item_id": item_id,
                    "station": station_name,
                    "id_uuid": id_uuid,
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
        """Build a full timeline of every item at every station.

        Handles parallel stations: the same item may appear at overlapping
        time intervals at different stations (concurrent processing), or
        different items at parallel cells of the same station type.

        Returns:
            DataFrame with columns:
            - item_id: Order / serial number.
            - station: Human-readable station name.
            - id_uuid: Source UUID for this station's ID signal.
            - start: Interval start.
            - end: Interval end.
            - duration_seconds: Time at this station.
            - sample_count: Number of data points.
            - step_sequence: Visit order per item (1-based, by start time).
            - is_parallel: True if this interval overlaps with another
              station for the same item.
            - uuid: Event UUID.
        """
        all_intervals: List[pd.DataFrame] = []
        for proc in self.processes:
            intervals = self._detect_intervals(proc["id_uuid"], proc["station"])
            if not intervals.empty:
                all_intervals.append(intervals)

        if not all_intervals:
            return pd.DataFrame(
                columns=[
                    "item_id",
                    "station",
                    "id_uuid",
                    "start",
                    "end",
                    "duration_seconds",
                    "sample_count",
                    "step_sequence",
                    "is_parallel",
                    "uuid",
                ]
            )

        timeline = pd.concat(all_intervals, ignore_index=True)
        timeline = timeline.sort_values(["item_id", "start"]).reset_index(drop=True)

        # Assign step sequence per item
        timeline["step_sequence"] = timeline.groupby("item_id").cumcount() + 1

        # Detect parallel activity: for each item, check if any intervals overlap
        timeline["is_parallel"] = False
        for item_id, grp in timeline.groupby("item_id"):
            if len(grp) < 2:
                continue
            idxs = grp.index.tolist()
            for i, idx_i in enumerate(idxs):
                for idx_j in idxs[i + 1 :]:
                    row_i = timeline.loc[idx_i]
                    row_j = timeline.loc[idx_j]
                    # Overlap: start_i < end_j AND start_j < end_i
                    if row_i["start"] < row_j["end"] and row_j["start"] < row_i["end"]:
                        timeline.loc[idx_i, "is_parallel"] = True
                        timeline.loc[idx_j, "is_parallel"] = True

        timeline["uuid"] = self.event_uuid
        return timeline

    # ------------------------------------------------------------------
    # lead_time
    # ------------------------------------------------------------------

    def lead_time(self) -> pd.DataFrame:
        """Compute end-to-end lead time per item across all processes.

        Returns:
            DataFrame with columns:
            - item_id
            - first_station: First station visited.
            - last_station: Last station visited.
            - first_seen: Earliest timestamp.
            - last_seen: Latest timestamp.
            - lead_time_seconds: Total elapsed time.
            - stations_visited: Number of distinct stations.
            - station_path: Ordered station names (" -> ").
            - has_parallel: Whether item had parallel processing.
            - uuid: Event UUID.
        """
        timeline = self.build_timeline()

        if timeline.empty:
            return pd.DataFrame(
                columns=[
                    "item_id",
                    "first_station",
                    "last_station",
                    "first_seen",
                    "last_seen",
                    "lead_time_seconds",
                    "stations_visited",
                    "station_path",
                    "has_parallel",
                    "uuid",
                ]
            )

        rows: List[Dict[str, Any]] = []
        for item_id, grp in timeline.groupby("item_id"):
            grp = grp.sort_values("start")
            first_seen = grp["start"].iloc[0]
            last_seen = grp["end"].iloc[-1]
            station_path = " -> ".join(grp["station"].tolist())
            rows.append(
                {
                    "item_id": item_id,
                    "first_station": grp["station"].iloc[0],
                    "last_station": grp["station"].iloc[-1],
                    "first_seen": first_seen,
                    "last_seen": last_seen,
                    "lead_time_seconds": (last_seen - first_seen).total_seconds(),
                    "stations_visited": grp["station"].nunique(),
                    "station_path": station_path,
                    "has_parallel": grp["is_parallel"].any(),
                    "uuid": self.event_uuid,
                }
            )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # parallel_activity
    # ------------------------------------------------------------------

    def parallel_activity(self) -> pd.DataFrame:
        """Find items that were processed at multiple stations simultaneously.

        Returns:
            DataFrame with columns:
            - item_id
            - station_a: First overlapping station.
            - station_b: Second overlapping station.
            - overlap_start: Start of overlap period.
            - overlap_end: End of overlap period.
            - overlap_seconds: Duration of overlap.
            - uuid: Event UUID.
        """
        timeline = self.build_timeline()

        if timeline.empty:
            return pd.DataFrame(
                columns=[
                    "item_id",
                    "station_a",
                    "station_b",
                    "overlap_start",
                    "overlap_end",
                    "overlap_seconds",
                    "uuid",
                ]
            )

        rows: List[Dict[str, Any]] = []
        for item_id, grp in timeline.groupby("item_id"):
            if len(grp) < 2:
                continue
            grp = grp.sort_values("start")
            records = grp.to_dict("records")
            for i in range(len(records)):
                for j in range(i + 1, len(records)):
                    ri, rj = records[i], records[j]
                    # Check overlap
                    overlap_start = max(ri["start"], rj["start"])
                    overlap_end = min(ri["end"], rj["end"])
                    if overlap_start < overlap_end:
                        rows.append(
                            {
                                "item_id": item_id,
                                "station_a": ri["station"],
                                "station_b": rj["station"],
                                "overlap_start": overlap_start,
                                "overlap_end": overlap_end,
                                "overlap_seconds": (
                                    overlap_end - overlap_start
                                ).total_seconds(),
                                "uuid": self.event_uuid,
                            }
                        )

        return (
            pd.DataFrame(rows)
            if rows
            else pd.DataFrame(
                columns=[
                    "item_id",
                    "station_a",
                    "station_b",
                    "overlap_start",
                    "overlap_end",
                    "overlap_seconds",
                    "uuid",
                ]
            )
        )

    # ------------------------------------------------------------------
    # handover_log
    # ------------------------------------------------------------------

    def handover_log(self) -> pd.DataFrame:
        """Extract handover events and correlate with item IDs.

        For each handover signal, detects trigger points (value > 0 or
        value changes to a truthy state) and resolves which item was being
        transferred based on the from-station's ID signal at that time.

        Returns:
            DataFrame with columns:
            - timestamp: When the handover fired.
            - item_id: Item being transferred (from from_station's ID signal).
            - from_station: Source station.
            - to_station: Destination station.
            - handover_uuid: UUID of the handover signal.
            - handover_value: Raw value of the handover signal.
            - uuid: Event UUID.
        """
        if not self.handovers:
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "item_id",
                    "from_station",
                    "to_station",
                    "handover_uuid",
                    "handover_value",
                    "uuid",
                ]
            )

        rows: List[Dict[str, Any]] = []
        for ho in self.handovers:
            ho_uuid = ho["uuid"]
            from_station = ho["from_station"]
            to_station = ho["to_station"]

            hdf = self._handover_data.get(ho_uuid, pd.DataFrame())
            if hdf.empty:
                continue

            # Find trigger points: value > 0 after being 0 or NaN
            hvals = hdf[[self.time_column, self.handover_value_column]].copy()
            hvals["val"] = pd.to_numeric(
                hvals[self.handover_value_column], errors="coerce"
            ).fillna(0)
            hvals["prev"] = hvals["val"].shift(fill_value=0)
            triggers = hvals[(hvals["val"] > 0) & (hvals["prev"] <= 0)]

            if triggers.empty:
                continue

            # Resolve item_id from from_station's ID signals at trigger time
            from_uuids = self._station_uuids.get(from_station, [])
            if not from_uuids:
                continue

            # Merge all from-station ID data
            from_frames = []
            for fuid in from_uuids:
                fdf = self._process_data.get(fuid, pd.DataFrame())
                if not fdf.empty:
                    from_frames.append(fdf[[self.time_column, self.id_value_column]])

            if not from_frames:
                continue

            from_ids = pd.concat(from_frames).sort_values(self.time_column)

            trigger_times = triggers[[self.time_column]].copy()
            merged = pd.merge_asof(
                trigger_times,
                from_ids,
                on=self.time_column,
                direction="backward",
            )

            for _, row in merged.iterrows():
                item_id = row.get(self.id_value_column, "")
                if pd.isna(item_id) or item_id == "":
                    continue
                # Get the matching trigger row for handover_value
                t = row[self.time_column]
                trigger_match = triggers[triggers[self.time_column] == t]
                ho_val = trigger_match["val"].iloc[0] if not trigger_match.empty else 0
                rows.append(
                    {
                        "timestamp": t,
                        "item_id": str(item_id),
                        "from_station": from_station,
                        "to_station": to_station,
                        "handover_uuid": ho_uuid,
                        "handover_value": ho_val,
                        "uuid": self.event_uuid,
                    }
                )

        if not rows:
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "item_id",
                    "from_station",
                    "to_station",
                    "handover_uuid",
                    "handover_value",
                    "uuid",
                ]
            )

        return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)

    # ------------------------------------------------------------------
    # station_statistics
    # ------------------------------------------------------------------

    def station_statistics(self) -> pd.DataFrame:
        """Compute dwell-time statistics per station across all items.

        Distinguishes parallel cells of the same station type.

        Returns:
            DataFrame with columns:
            - station: Station name.
            - id_uuid: Specific cell UUID (for parallel cells).
            - item_count: Distinct items processed.
            - min_dwell_seconds
            - avg_dwell_seconds
            - max_dwell_seconds
            - total_dwell_seconds
        """
        timeline = self.build_timeline()

        if timeline.empty:
            return pd.DataFrame(
                columns=[
                    "station",
                    "id_uuid",
                    "item_count",
                    "min_dwell_seconds",
                    "avg_dwell_seconds",
                    "max_dwell_seconds",
                    "total_dwell_seconds",
                ]
            )

        stats = (
            timeline.groupby(["station", "id_uuid"])
            .agg(
                item_count=("item_id", "nunique"),
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

    # ------------------------------------------------------------------
    # routing_paths
    # ------------------------------------------------------------------

    def routing_paths(self) -> pd.DataFrame:
        """Analyze routing path frequencies across all items.

        Returns:
            DataFrame with columns:
            - station_path: Ordered station names (" -> ").
            - item_count: Items that followed this path.
            - avg_lead_time_seconds
            - min_lead_time_seconds
            - max_lead_time_seconds
            - has_parallel_steps: Whether path includes parallel processing.
        """
        lead = self.lead_time()

        if lead.empty:
            return pd.DataFrame(
                columns=[
                    "station_path",
                    "item_count",
                    "avg_lead_time_seconds",
                    "min_lead_time_seconds",
                    "max_lead_time_seconds",
                    "has_parallel_steps",
                ]
            )

        stats = (
            lead.groupby("station_path")
            .agg(
                item_count=("item_id", "nunique"),
                avg_lead_time_seconds=("lead_time_seconds", "mean"),
                min_lead_time_seconds=("lead_time_seconds", "min"),
                max_lead_time_seconds=("lead_time_seconds", "max"),
                has_parallel_steps=("has_parallel", "any"),
            )
            .reset_index()
        )

        for col in [
            "avg_lead_time_seconds",
            "min_lead_time_seconds",
            "max_lead_time_seconds",
        ]:
            stats[col] = stats[col].round(2)

        return stats.sort_values("item_count", ascending=False).reset_index(drop=True)
