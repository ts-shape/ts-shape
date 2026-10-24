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
from typing import TYPE_CHECKING, Any

import pandas as pd

from ts_shape.utils.base import Base
from ts_shape.utils.deprecation import deprecated_class
from ts_shape.utils.runs import run_bounds, total_seconds

from ._trace_objects import (
    Exclude,
    IntervalEnd,
    check_interval_end,
    exclude_readings,
    intervals_frame,
    next_change_ends,
)

if TYPE_CHECKING:
    from typing_extensions import deprecated

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

    #: Methods read only the constructor's signals (``Base`` keeps their rows).
    _reads_own_signals_only = True

    def __init__(
        self,
        dataframe: pd.DataFrame,
        station_uuids: dict[str, str],
        *,
        event_uuid: str = "prod:value_trace",
        value_column: str = "value_string",
        time_column: str = "systime",
        exclude: Exclude = None,
    ) -> None:
        """Initialize value traceability.

        Args:
            dataframe: Input DataFrame with timeseries data.
            station_uuids: Mapping of UUID -> human-readable station name.
                           e.g. {'uuid_abc': 'Station A', 'uuid_def': 'Station B'}
            event_uuid: UUID to tag derived events with.
            value_column: Column holding the identifier string value.
            time_column: Name of timestamp column.
            exclude: Readings that are not identifiers, read like
                :attr:`ts_shape.eventlog.ObjectSpec.exclude`: ``"misreads"``
                (:func:`ts_shape.utils.id_codes.is_misread`: ``NOREAD``,
                ``#N/A``, ``0``, ``65535``, blanks, ...), a collection of
                values, or a callable returning a boolean mask over a
                station's readings in time order. Between two readings of the
                same identifier an excluded reading is a gap (the identifier
                is held across it); anywhere else it is an empty reading.
                ``None`` (default) keeps every reading; ``"misreads"`` is
                planned to become the default in the next major release.
        """
        super().__init__(dataframe, time_column=time_column)
        self.station_uuids = station_uuids
        self.exclude = exclude
        self.event_uuid = event_uuid
        self.value_column = value_column
        self.time_column = time_column

        # Pre-filter and parse per-station data
        self._station_data: dict[str, pd.DataFrame] = {}
        for uuid in self.station_uuids:
            # Stable: readings at the same instant keep their input order.
            sdf = (
                self.dataframe[self.dataframe["uuid"] == uuid]
                .copy()
                .sort_values(self.time_column, kind="stable")
            )
            sdf[self.time_column] = pd.to_datetime(sdf[self.time_column])
            if exclude is not None and value_column in sdf.columns:
                sdf[value_column] = exclude_readings(sdf[value_column], exclude, uuid)
            self._station_data[uuid] = sdf

    # ------------------------------------------------------------------
    # Internal: detect intervals per station (batch-tracking style)
    # ------------------------------------------------------------------

    def _data_end(self) -> Any:
        """Last timestamp of the station signals (``None`` without data)."""
        ends = [sdf[self.time_column].max() for sdf in self._station_data.values()]
        ends = [t for t in ends if pd.notna(t)]
        return max(ends) if ends else None

    def _detect_intervals(
        self, uuid: str, interval_end: IntervalEnd = "last_sample"
    ) -> pd.DataFrame:
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

        identifier = sdf[self.value_column].fillna("")
        runs = run_bounds(identifier)
        ids = identifier.iloc[runs.starts]
        # An empty identifier (missing value) is a gap between visits.
        keep = (ids != "").to_numpy(dtype=bool)
        if not keep.any():
            return pd.DataFrame([])
        times = sdf[self.time_column].array
        start = times[runs.starts[keep]]
        end = times[runs.last[keep]]
        if interval_end == "next_change":
            # Held until the next reading on the signal (next identifier or
            # an empty one); the signal's last run until the end of the data.
            end = next_change_ends(times[runs.starts], self._data_end())[keep].array
        return pd.DataFrame(
            {
                "identifier": ids[keep].tolist(),
                "station_uuid": uuid,
                "station_name": self.station_uuids[uuid],
                "start": start,
                "end": end,
                "duration_seconds": total_seconds(end - start),
                "sample_count": runs.lengths[keep],
            }
        )

    # ------------------------------------------------------------------
    # build_timeline
    # ------------------------------------------------------------------

    def build_timeline(
        self,
        *,
        infer_dwell_from_next: bool = False,
        interval_end: IntervalEnd = "last_sample",
    ) -> pd.DataFrame:
        """Build a full timeline of every identifier at every station.

        Args:
            infer_dwell_from_next: When False (default), an interval's ``end``
                is the last sample of the identifier at that station, so a
                station that logs the identifier only once on arrival (the
                common event/change-encoded case) reports a dwell of ``0``.
                When True, each visit's ``end`` is set to the same identifier's
                *next* station arrival, so the dwell is the honest
                arrival-here → arrival-at-next-station time. The last visit per
                identifier stays open (keeps its own last sample).
            interval_end: Where a visit ends on its station signal.
                ``"last_sample"`` (default): at the identifier's last sample
                there. ``"next_change"``: at the station's next different
                reading (the next identifier, or an empty / missing one), so
                the identifier is held until it changes, like
                :func:`ts_shape.eventlog.object_intervals`; a station's last
                visit is held until the last timestamp of the station signals.
                ``infer_dwell_from_next`` is applied afterwards.

        Returns:
            DataFrame with columns:
            - identifier: The shared identifier string.
            - station_uuid: UUID of the station signal.
            - station_name: Human-readable station name.
            - start: Timestamp when the identifier appeared at the station.
            - end: Timestamp of the last sample at the station (see
                ``interval_end``).
            - duration_seconds: Time spent at the station.
            - sample_count: Number of data points.
            - station_sequence: Order of visit (1-based) per identifier.
            - uuid: Event UUID.
            - source_uuid: UUID of the station signal (same as station_uuid).
        """
        check_interval_end(interval_end, "ValueTraceabilityEvents.build_timeline")
        all_intervals: list[pd.DataFrame] = []
        for uuid in self.station_uuids:
            intervals = self._detect_intervals(uuid, interval_end)
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
                    "source_uuid",
                ]
            )

        timeline = pd.concat(all_intervals, ignore_index=True)
        timeline = timeline.sort_values(["identifier", "start"]).reset_index(drop=True)

        # Assign station visit sequence per identifier
        timeline["station_sequence"] = timeline.groupby("identifier").cumcount() + 1
        timeline["uuid"] = self.event_uuid
        timeline["source_uuid"] = timeline["station_uuid"]

        if infer_dwell_from_next:
            next_start = timeline.groupby("identifier")["start"].shift(-1)
            timeline["end"] = next_start.fillna(timeline["end"])
            timeline["duration_seconds"] = (
                timeline["end"] - timeline["start"]
            ).dt.total_seconds()

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
            - last_seen: Latest timestamp across all stations (the latest
                visit end, also when visits overlap).
            - lead_time_seconds: Total time from first seen to last seen.
            - stations_visited: Number of distinct stations visited.
            - station_path: Arrow-separated ordered station names.
            - uuid: Event UUID.
            - start: Same as first_seen.
            - end: Same as last_seen.
            - duration_seconds: Same as lead_time_seconds (float seconds).
            - source_uuid: Comma-joined station UUIDs in visit order
                (each listed once).
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
                    "start",
                    "end",
                    "duration_seconds",
                    "source_uuid",
                ]
            )

        rows: list[dict[str, Any]] = []
        for identifier, grp in timeline.groupby("identifier"):
            grp = grp.sort_values("start")
            first_seen = grp["start"].iloc[0]
            # The latest leave: an earlier visit can end after a later one.
            last_seen = grp["end"].max()
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
                    "start": first_seen,
                    "end": last_seen,
                    "duration_seconds": float((last_seen - first_seen).total_seconds()),
                    "source_uuid": ",".join(dict.fromkeys(grp["station_uuid"])),
                }
            )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # current_status
    # ------------------------------------------------------------------

    def current_status(
        self,
        *,
        infer_dwell_from_next: bool = False,
        interval_end: IntervalEnd = "last_sample",
    ) -> pd.DataFrame:
        """Determine the last-known station for each identifier.

        Args:
            infer_dwell_from_next: Forwarded to :meth:`build_timeline`. Note the
                *current* (last) station's dwell is still open, so this mainly
                affects identifiers whose latest visit has a later arrival.
            interval_end: Forwarded to :meth:`build_timeline`.

        Returns:
            DataFrame with columns:
            - identifier
            - current_station: Name of the last station where seen.
            - current_station_uuid: UUID of that station.
            - arrived_at: When the identifier arrived at the current station.
            - time_at_station_seconds: Dwell time at the current station so far.
            - uuid: Event UUID.
        """
        timeline = self.build_timeline(
            infer_dwell_from_next=infer_dwell_from_next, interval_end=interval_end
        )

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

    def station_dwell_statistics(
        self,
        *,
        infer_dwell_from_next: bool = False,
        interval_end: IntervalEnd = "last_sample",
    ) -> pd.DataFrame:
        """Compute dwell-time statistics per station (across all identifiers).

        Args:
            infer_dwell_from_next: Forwarded to :meth:`build_timeline`. Pass
                True when stations log an identifier once on arrival, so dwell
                is measured arrival-here → arrival-at-next-station instead of
                collapsing to zero.
            interval_end: Forwarded to :meth:`build_timeline`;
                ``"next_change"`` measures dwell to the station's next reading.

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
        timeline = self.build_timeline(
            infer_dwell_from_next=infer_dwell_from_next, interval_end=interval_end
        )

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

        return stats

    # ------------------------------------------------------------------
    # to_intervals
    # ------------------------------------------------------------------

    def to_intervals(
        self,
        *,
        object_type: str = "serial",
        interval_end: IntervalEnd = "last_sample",
    ) -> pd.DataFrame:
        """The timeline as an object interval table (one row per visit).

        The result has the columns of
        :func:`ts_shape.eventlog.object_intervals` plus ``station``, so it
        feeds :func:`~ts_shape.eventlog.presence_events`,
        :func:`~ts_shape.eventlog.object_kpis`,
        :func:`~ts_shape.eventlog.wip_over_time`,
        :func:`~ts_shape.eventlog.attach_counts`,
        :func:`~ts_shape.eventlog.relate` and
        :func:`~ts_shape.eventlog.link_events_to_objects` directly.

        Args:
            object_type: Object type of the identifiers (``"serial"``,
                ``"work_order"``, ``"batch"``, ...).
            interval_end: Forwarded to :meth:`build_timeline`.
                ``"next_change"`` gives the presences
                :func:`~ts_shape.eventlog.object_intervals` gives for the
                station signals (every non-empty value an id).

        Returns:
            DataFrame with columns:
            - oid: The identifier (text; ``12.0`` renders as ``"12"``).
            - type: ``object_type``.
            - start / end: The visit, in UTC (naive input is read as UTC).
            - duration_seconds
            - uuid: The station signal.
            - source_uuid: The station signal.
            - station: The station name; the analysis functions use it as
                the station (to :func:`~ts_shape.eventlog.relate` it is an
                object attribute, recorded at each visit).
            Ordered by identifier, then start. No visit gives an empty frame
            with these columns.
        """
        tl = self.build_timeline(interval_end=interval_end)
        return intervals_frame(
            oid=tl["identifier"],
            object_type=object_type,
            start=tl["start"],
            end=tl["end"],
            uuid=tl["station_uuid"],
            source_uuid=tl["station_uuid"],
            station=tl["station_name"],
        )


#: Deprecated name, removed in the next major release.
if TYPE_CHECKING:

    @deprecated("OrderTraceabilityEvents is deprecated, use ValueTraceabilityEvents")
    class OrderTraceabilityEvents(ValueTraceabilityEvents): ...

else:
    OrderTraceabilityEvents = deprecated_class(
        ValueTraceabilityEvents, "OrderTraceabilityEvents"
    )
