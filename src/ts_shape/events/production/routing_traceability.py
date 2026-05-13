"""Routing-based traceability using ID + state/handover signals.

In this pattern a single UUID carries the current item identifier (serial
code, order number, etc.) and a separate UUID carries a state / routing
signal whose value encodes which process step or station the item is at.

The ``state_map`` parameter defines the mapping from signal values to
station/step names.  This is required because the signal values alone
don't tell the module what they mean — they could be PLC step numbers,
recipe phases, station codes, or anything else.

Example::

    State signal value 10 → "Heating"
    State signal value 20 → "Holding"
    State signal value 30 → "Cooling"

Or::

    State signal value 1 → "Welding"
    State signal value 2 → "Painting"
    State signal value 3 → "Assembly"

This module correlates both signals to reconstruct the full routing path.
"""

import logging
import pandas as pd  # type: ignore
from typing import List, Dict, Any, Optional, Union

from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class RoutingTraceabilityEvents(Base):
    """Trace item routing using an ID signal paired with a state/routing signal.

    Two UUIDs work together:
    - **id_uuid**: string signal carrying the current identifier
      (serial number, order ID, batch code, etc.).
    - **routing_uuid**: signal whose value encodes the current process
      step or station.

    A ``state_map`` translates signal values to human-readable station
    or step names.  Without it, raw values are used as labels.

    Example usage::

        # PLC step numbers mapped to process steps
        trace = RoutingTraceabilityEvents(
            df,
            id_uuid='serial_code_signal',
            routing_uuid='step_chain_signal',
            state_map={
                10: 'Heating',
                20: 'Holding',
                30: 'Cooling',
                40: 'Discharge',
            },
        )

        # Station handover signal mapped to stations
        trace = RoutingTraceabilityEvents(
            df,
            id_uuid='serial_code_signal',
            routing_uuid='handover_signal',
            state_map={1: 'Welding', 2: 'Painting', 3: 'Assembly'},
        )

        timeline = trace.build_routing_timeline()
        lead = trace.lead_time()
        stats = trace.station_statistics()
        paths = trace.routing_paths()
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        id_uuid: str,
        routing_uuid: str,
        *,
        state_map: Optional[Dict[Union[int, float, str], str]] = None,
        station_map: Optional[Dict[Union[int, float, str], str]] = None,
        event_uuid: str = "prod:routing_trace",
        id_value_column: str = "value_string",
        routing_value_column: str = "value_integer",
        time_column: str = "systime",
    ) -> None:
        """Initialize routing traceability.

        Args:
            dataframe: Input DataFrame with timeseries data.
            id_uuid: UUID of the signal carrying item identifiers.
            routing_uuid: UUID of the state/routing signal.
            state_map: Mapping from signal value to station/step name.
                       e.g. {10: 'Heating', 20: 'Holding', 30: 'Cooling'}
                       or {1: 'Welding', 2: 'Painting', 3: 'Assembly'}
                       Values can be int, float, or str keys.
                       If not provided, raw signal values are used as labels.
            station_map: Deprecated alias for state_map (for backwards
                         compatibility). Use state_map instead.
            event_uuid: UUID to tag derived events with.
            id_value_column: Column holding the item ID string.
            routing_value_column: Column holding the state/routing value.
            time_column: Name of timestamp column.
        """
        super().__init__(dataframe, column_name=time_column)
        self.id_uuid = id_uuid
        self.routing_uuid = routing_uuid
        # state_map takes priority; fall back to station_map for backwards compat
        self.state_map: Dict[Union[int, float, str], str] = (
            state_map or station_map or {}
        )
        self.event_uuid = event_uuid
        self.id_value_column = id_value_column
        self.routing_value_column = routing_value_column
        self.time_column = time_column

        # Pre-filter signals
        self.id_data = (
            self.dataframe[self.dataframe["uuid"] == self.id_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        self.id_data[self.time_column] = pd.to_datetime(self.id_data[self.time_column])

        self.routing_data = (
            self.dataframe[self.dataframe["uuid"] == self.routing_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        self.routing_data[self.time_column] = pd.to_datetime(
            self.routing_data[self.time_column]
        )

    def _resolve_state(self, value: Any) -> str:
        """Resolve a routing/state signal value to a station/step name.

        Lookup order:
        1. Try the value as-is in state_map.
        2. Try converting to int.
        3. Try converting to str.
        4. Fall back to "State {value}".
        """
        if value in self.state_map:
            return self.state_map[value]
        try:
            v = int(value)
            if v in self.state_map:
                return self.state_map[v]
        except (ValueError, TypeError):
            pass
        s = str(value)
        if s in self.state_map:
            return self.state_map[s]
        # Format cleanly: 5.0 -> "State 5", 3.14 -> "State 3.14"
        try:
            v_float = float(value)
            if v_float == int(v_float):
                return f"State {int(v_float)}"
            return f"State {v_float}"
        except (ValueError, TypeError):
            return f"State {value}"

    # ------------------------------------------------------------------
    # build_routing_timeline
    # ------------------------------------------------------------------

    def build_routing_timeline(self) -> pd.DataFrame:
        """Correlate the ID signal with the state/routing signal to build a timeline.

        For each sample of the routing signal, the *current* item ID is
        determined via backward-fill merge (most recent ID at that timestamp).
        Contiguous intervals where the same (item_id, state) pair holds
        are grouped into single events.

        Returns:
            DataFrame with columns:
            - item_id: Identifier string.
            - routing_value: Raw signal value.
            - station_name: Resolved station/step name from state_map.
            - start: First timestamp of interval.
            - end: Last timestamp of interval.
            - duration_seconds: Time at this state.
            - sample_count: Number of routing samples in interval.
            - station_sequence: Visit order per item (1-based).
            - uuid: Event UUID.
        """
        if self.routing_data.empty or self.id_data.empty:
            return pd.DataFrame(
                columns=[
                    "item_id",
                    "routing_value",
                    "station_name",
                    "start",
                    "end",
                    "duration_seconds",
                    "sample_count",
                    "station_sequence",
                    "uuid",
                ]
            )

        # Merge: for each routing sample, attach the most-recent item ID
        routing_subset = self.routing_data[
            [self.time_column, self.routing_value_column]
        ].copy()
        id_subset = self.id_data[[self.time_column, self.id_value_column]].copy()

        merged = pd.merge_asof(
            routing_subset,
            id_subset,
            on=self.time_column,
            direction="backward",
        )

        merged = merged.rename(
            columns={
                self.id_value_column: "item_id",
                self.routing_value_column: "routing_value",
            }
        )
        merged = merged.dropna(subset=["item_id"])
        merged["item_id"] = merged["item_id"].astype(str)

        if merged.empty:
            return pd.DataFrame(
                columns=[
                    "item_id",
                    "routing_value",
                    "station_name",
                    "start",
                    "end",
                    "duration_seconds",
                    "sample_count",
                    "station_sequence",
                    "uuid",
                ]
            )

        # Detect contiguous intervals of (item_id, routing_value)
        merged["combo"] = merged["item_id"] + "|" + merged["routing_value"].astype(str)
        merged["group"] = (merged["combo"] != merged["combo"].shift()).cumsum()

        rows: List[Dict[str, Any]] = []
        for _, seg in merged.groupby("group"):
            item_id = seg["item_id"].iloc[0]
            routing_val = seg["routing_value"].iloc[0]
            start = seg[self.time_column].iloc[0]
            end = seg[self.time_column].iloc[-1]
            rows.append(
                {
                    "item_id": item_id,
                    "routing_value": routing_val,
                    "station_name": self._resolve_state(routing_val),
                    "start": start,
                    "end": end,
                    "duration_seconds": (end - start).total_seconds(),
                    "sample_count": len(seg),
                }
            )

        timeline = pd.DataFrame(rows)
        timeline = timeline.sort_values(["item_id", "start"]).reset_index(drop=True)
        timeline["station_sequence"] = timeline.groupby("item_id").cumcount() + 1
        timeline["uuid"] = self.event_uuid

        return timeline

    # ------------------------------------------------------------------
    # lead_time
    # ------------------------------------------------------------------

    def lead_time(self) -> pd.DataFrame:
        """Compute end-to-end lead time per item.

        Returns:
            DataFrame with columns:
            - item_id
            - first_station: Name of first station/step visited.
            - last_station: Name of last station/step visited.
            - first_seen: Earliest timestamp.
            - last_seen: Latest timestamp.
            - lead_time_seconds: Total elapsed time.
            - stations_visited: Number of distinct stations/steps.
            - routing_path: Ordered station names joined by " -> ".
            - uuid: Event UUID.
        """
        timeline = self.build_routing_timeline()

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
                    "routing_path",
                    "uuid",
                ]
            )

        rows: List[Dict[str, Any]] = []
        for item_id, grp in timeline.groupby("item_id"):
            grp = grp.sort_values("start")
            first_seen = grp["start"].iloc[0]
            last_seen = grp["end"].iloc[-1]
            routing_path = " -> ".join(grp["station_name"].tolist())
            rows.append(
                {
                    "item_id": item_id,
                    "first_station": grp["station_name"].iloc[0],
                    "last_station": grp["station_name"].iloc[-1],
                    "first_seen": first_seen,
                    "last_seen": last_seen,
                    "lead_time_seconds": (last_seen - first_seen).total_seconds(),
                    "stations_visited": grp["station_name"].nunique(),
                    "routing_path": routing_path,
                    "uuid": self.event_uuid,
                }
            )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # station_statistics
    # ------------------------------------------------------------------

    def station_statistics(self) -> pd.DataFrame:
        """Compute dwell-time statistics per station/step across all items.

        Returns:
            DataFrame with columns:
            - station_name
            - routing_value
            - item_count: Number of distinct items seen.
            - min_dwell_seconds
            - avg_dwell_seconds
            - max_dwell_seconds
            - total_dwell_seconds
        """
        timeline = self.build_routing_timeline()

        if timeline.empty:
            return pd.DataFrame(
                columns=[
                    "station_name",
                    "routing_value",
                    "item_count",
                    "min_dwell_seconds",
                    "avg_dwell_seconds",
                    "max_dwell_seconds",
                    "total_dwell_seconds",
                ]
            )

        stats = (
            timeline.groupby(["station_name", "routing_value"])
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
        """Analyze routing path frequencies -- which station sequences are most common.

        Returns:
            DataFrame with columns:
            - routing_path: Ordered station names joined by " -> ".
            - item_count: Number of items that followed this path.
            - avg_lead_time_seconds: Average lead time for items on this path.
            - min_lead_time_seconds
            - max_lead_time_seconds
        """
        lead = self.lead_time()

        if lead.empty:
            return pd.DataFrame(
                columns=[
                    "routing_path",
                    "item_count",
                    "avg_lead_time_seconds",
                    "min_lead_time_seconds",
                    "max_lead_time_seconds",
                ]
            )

        stats = (
            lead.groupby("routing_path")
            .agg(
                item_count=("item_id", "nunique"),
                avg_lead_time_seconds=("lead_time_seconds", "mean"),
                min_lead_time_seconds=("lead_time_seconds", "min"),
                max_lead_time_seconds=("lead_time_seconds", "max"),
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
