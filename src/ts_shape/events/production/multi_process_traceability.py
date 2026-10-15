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
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd

from ts_shape.utils.base import Base
from ts_shape.utils.coerce import to_numeric_checked
from ts_shape.utils.runs import Runs, run_bounds, run_labels, total_seconds

logger = logging.getLogger(__name__)

_Ints: TypeAlias = npt.NDArray[np.int64]


def _grouped_searchsorted(
    groups: _Ints, times: _Ints, q_groups: _Ints, q_times: _Ints, side: str
) -> _Ints:
    """``searchsorted`` of ``(q_group, q_time)`` in data sorted by ``(group, time)``.

    Returns, per query, the global position of the first data element not
    less (``side="left"``) / greater (``side="right"``) than the query, so
    for a query in group ``g`` the result lies within ``g``'s data range.
    """
    n = groups.size
    # Ties: a "left" query sorts before equal data, a "right" one after it.
    q_tie, d_tie = (0, 1) if side == "left" else (1, 0)
    tie = np.r_[np.full(n, d_tie), np.full(q_groups.size, q_tie)]
    order = np.lexsort((tie, np.r_[times, q_times], np.r_[groups, q_groups]))
    is_data = order < n
    data_before: _Ints = np.cumsum(is_data, dtype=np.int64) - is_data
    out = np.empty(q_groups.size, dtype=np.int64)
    out[order[~is_data] - n] = data_before[~is_data]
    return out


def _overlap_bounds(
    timeline: pd.DataFrame, *, allow_ties: bool
) -> tuple[_Ints, _Ints, _Ints, _Ints, _Ints] | None:
    """Per-row item label, start/end (int ns), ``p`` (the first row of the
    same item whose start is not before this row's end) and the item's first
    row.

    ``timeline`` is sorted by ``(item_id, start)``. Returns None when a start
    or end is missing (NaT), or -- unless ``allow_ties`` -- when two rows of
    one item share a start (a per-item ``sort_values`` may then order them
    differently); callers fall back to their pairwise loops.
    """
    start, end = timeline["start"], timeline["end"]
    if bool(start.isna().any()) or bool(end.isna().any()):
        return None
    runs = run_bounds(timeline["item_id"])
    gid = run_labels(runs)
    s_ns = np.asarray(start.array.asi8, dtype=np.int64)
    e_ns = np.asarray(end.array.asi8, dtype=np.int64)
    if not allow_ties and bool(((gid[1:] == gid[:-1]) & (s_ns[1:] == s_ns[:-1])).any()):
        return None
    p = _grouped_searchsorted(gid, s_ns, gid, e_ns, "left")
    return gid, s_ns, e_ns, p, runs.starts[gid]


def _parallel_flags(timeline: pd.DataFrame) -> npt.NDArray[np.bool_] | None:
    """Whether each row overlaps another row of the same item.

    Rows ``i`` and ``j`` overlap when ``start_i < end_j`` and
    ``start_j < end_i``. Within an item the rows are sorted by start, so the
    rows starting before ``end_i`` are a prefix: an earlier one overlaps if
    the running maximum of their ends exceeds ``start_i``, a later one unless
    it is a zero-length row at exactly ``start_i``.
    """
    bounds = _overlap_bounds(timeline, allow_ties=True)
    if bounds is None:
        return None
    gid, s_ns, e_ns, p, g0 = bounds
    n = gid.size
    pos = np.arange(n, dtype=np.int64)
    # Earlier rows j < min(i, p_i) all start before end_i.
    run_max = pd.Series(e_ns).groupby(gid).cummax().to_numpy()
    k = np.minimum(pos, p) - 1
    earlier = (k >= g0) & (run_max[np.maximum(k, 0)] > s_ns)
    # Later rows j in (i, p_i) overlap unless zero-length at start_i (those
    # have start == start_i, so they sit before the first larger start).
    b = _grouped_searchsorted(gid, s_ns, gid, s_ns, "right")
    zero = np.r_[0, np.cumsum(s_ns == e_ns)]
    hi = np.maximum(np.minimum(p, b), pos + 1)
    later = (p - pos - 1) - (zero[hi] - zero[pos + 1]) > 0
    return np.asarray(earlier | later, dtype=bool)


def _overlap_pairs(timeline: pd.DataFrame) -> tuple[_Ints, _Ints] | None:
    """Row pairs ``i < j`` of one item whose intervals overlap, in loop order.

    Row ``j > i`` starts after row ``i`` (no tied starts), so the pair
    overlaps when ``j`` starts before ``end_i`` (``j < p_i``) and is not
    zero-length.
    """
    bounds = _overlap_bounds(timeline, allow_ties=False)
    if bounds is None:
        return None
    _, s_ns, e_ns, p, _ = bounds
    pos = np.arange(s_ns.size, dtype=np.int64)
    count = np.maximum(p - pos - 1, 0)
    i = np.repeat(pos, count)
    j = (
        i
        + 1
        + np.arange(i.size, dtype=np.int64)
        - np.repeat(np.cumsum(count) - count, count)
    )
    keep = s_ns[j] < e_ns[j]
    return i[keep], j[keep]


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
        processes: list[dict[str, str]],
        *,
        handovers: list[dict[str, str]] | None = None,
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
        super().__init__(dataframe, time_column=time_column)
        self.processes = processes
        self.handovers = handovers or []
        self.event_uuid = event_uuid
        self.id_value_column = id_value_column
        self.handover_value_column = handover_value_column
        self.time_column = time_column

        # Build station lookup: station_name -> list of id_uuids (parallel cells)
        self._station_uuids: dict[str, list[str]] = {}
        for proc in self.processes:
            station = proc["station"]
            self._station_uuids.setdefault(station, []).append(proc["id_uuid"])

        # Pre-filter process data per uuid
        self._process_data: dict[str, pd.DataFrame] = {}
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
        self._handover_data: dict[str, pd.DataFrame] = {}
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
        # A segment starts at every value change or time gap.
        first = np.flatnonzero((value_change | gap).to_numpy(dtype=bool))
        runs = Runs(first, np.r_[first[1:], len(s)], np.diff(np.r_[first, len(s)]))
        ids = s["item_id"].iloc[runs.starts]
        keep = (ids != "").to_numpy(dtype=bool)
        if not keep.any():
            return pd.DataFrame([])
        times = s[self.time_column].array
        start = times[runs.starts[keep]]
        end = times[runs.last[keep]]
        return pd.DataFrame(
            {
                "item_id": ids[keep].tolist(),
                "station": station_name,
                "id_uuid": id_uuid,
                "start": start,
                "end": end,
                "duration_seconds": total_seconds(end - start),
                "sample_count": runs.lengths[keep],
            }
        )

    # ------------------------------------------------------------------
    # build_timeline
    # ------------------------------------------------------------------

    def build_timeline(self, *, infer_dwell_from_next: bool = False) -> pd.DataFrame:
        """Build a full timeline of every item at every station.

        Handles parallel stations: the same item may appear at overlapping
        time intervals at different stations (concurrent processing), or
        different items at parallel cells of the same station type.

        Args:
            infer_dwell_from_next: When False (default), an interval's ``end`` is
                the last sample of the item at that station, so a station that
                logs the item once on arrival reports a dwell of ``0``. When
                True, each *sequential* (non-parallel) visit's ``end`` is set to
                the item's next station arrival, giving the honest
                arrival-here → arrival-at-next-station dwell. Parallel /
                overlapping visits keep their own span, since a "next arrival"
                is not meaningful while the item is at two stations at once.

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
        all_intervals: list[pd.DataFrame] = []
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
        flags = _parallel_flags(timeline)
        if flags is not None:
            timeline["is_parallel"] = flags
        else:
            self._parallel_flags_loop(timeline)

        timeline["uuid"] = self.event_uuid

        if infer_dwell_from_next:
            # Only sequential visits get next-arrival dwell; parallel visits
            # (item at two stations at once) keep their own span.
            next_start = timeline.groupby("item_id")["start"].shift(-1)
            infer = (~timeline["is_parallel"]) & next_start.notna()
            timeline.loc[infer, "end"] = next_start[infer]
            timeline["duration_seconds"] = (
                timeline["end"] - timeline["start"]
            ).dt.total_seconds()

        return timeline

    @staticmethod
    def _parallel_flags_loop(timeline: pd.DataFrame) -> None:
        """Pairwise ``is_parallel`` (fallback for rows with a NaT bound)."""
        timeline["is_parallel"] = False
        for _, grp in timeline.groupby("item_id"):
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

        rows: list[dict[str, Any]] = []
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

        pairs = _overlap_pairs(timeline)
        if pairs is None:
            rows = self._parallel_activity_loop(timeline)
            if rows:
                return pd.DataFrame(rows)
        elif pairs[0].size:
            i, j = pairs
            start, end = timeline["start"].array, timeline["end"].array
            # Rows are sorted by start without ties, so row j starts later.
            e_ns = np.asarray(end.asi8, dtype=np.int64)
            ov_start = start[j]
            ov_end = end[np.where(e_ns[j] < e_ns[i], j, i)]
            station = timeline["station"]
            return pd.DataFrame(
                {
                    "item_id": timeline["item_id"].iloc[i].tolist(),
                    "station_a": station.iloc[i].tolist(),
                    "station_b": station.iloc[j].tolist(),
                    "overlap_start": ov_start,
                    "overlap_end": ov_end,
                    "overlap_seconds": total_seconds(ov_end - ov_start),
                    "uuid": self.event_uuid,
                }
            )
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

    def _parallel_activity_loop(self, timeline: pd.DataFrame) -> list[dict[str, Any]]:
        """Pairwise overlaps (fallback for NaT bounds or tied starts)."""
        rows: list[dict[str, Any]] = []
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

        return rows

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

        rows: list[dict[str, Any]] = []
        for ho in self.handovers:
            ho_uuid = ho["uuid"]
            from_station = ho["from_station"]
            to_station = ho["to_station"]

            hdf = self._handover_data.get(ho_uuid, pd.DataFrame())
            if hdf.empty:
                continue

            # Find trigger points: value > 0 after being 0 or NaN
            hvals = hdf[[self.time_column, self.handover_value_column]].copy()
            hvals["val"] = to_numeric_checked(
                hvals[self.handover_value_column],
                what=f"handover {ho_uuid!r} ({self.handover_value_column!r})",
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

    def station_statistics(
        self, *, infer_dwell_from_next: bool = False
    ) -> pd.DataFrame:
        """Compute dwell-time statistics per station across all items.

        Distinguishes parallel cells of the same station type.

        Args:
            infer_dwell_from_next: Forwarded to :meth:`build_timeline`; pass True
                when stations log an item once on arrival so dwell is measured
                to the item's next station arrival instead of collapsing to zero.

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
        timeline = self.build_timeline(infer_dwell_from_next=infer_dwell_from_next)

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

        return stats.sort_values("item_count", ascending=False).reset_index(drop=True)
