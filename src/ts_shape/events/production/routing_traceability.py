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
import math
import numbers
import re
import warnings
from typing import Any

import numpy as np
import pandas as pd

from ts_shape.utils.base import Base
from ts_shape.utils.deprecation import (
    REMOVAL,
    caller_stacklevel,
    deprecated_attribute,
    renamed_kwargs,
)

logger = logging.getLogger(__name__)

_NUMBER_RE = re.compile(r"\s*[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?\s*")


def _as_number(value: Any) -> float | None:
    """Return ``value`` as a finite float, or None when it is not a number.

    Accepts real numbers (booleans count as 0/1) and numeric strings such
    as ``"10"``.
    """
    is_numeric_string = isinstance(value, str) and bool(_NUMBER_RE.fullmatch(value))
    if not (isinstance(value, numbers.Real | np.bool_) or is_numeric_string):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _combo_run_starts(item_id: pd.Series, routing: pd.Series) -> np.ndarray:
    """Positions where ``item_id + "|" + routing.astype(str)`` changes.

    For numeric routing values and item IDs without ``"|"`` the key changes
    exactly when the item ID or the routing value's text changes, so both
    are compared directly (distinct floats have distinct text, except NaN
    with NaN; ``0.0`` and ``-0.0`` differ). Other inputs build the key.
    """
    ids = item_id.to_numpy()
    plain = routing.dtype.kind in "biuf" and not any(
        "|" in str(u) for u in pd.unique(item_id)
    )
    if not plain:
        combo = (item_id + "|" + routing.astype(str)).to_numpy()
        change = combo[1:] != combo[:-1]
    else:
        v = routing.to_numpy()
        if v.dtype.kind == "f":
            # Same text: both NaN, or equal with the same sign (0.0 / -0.0).
            same = (np.isnan(v[1:]) & np.isnan(v[:-1])) | (
                (v[1:] == v[:-1]) & (np.signbit(v[1:]) == np.signbit(v[:-1]))
            )
        else:
            same = v[1:] == v[:-1]
        change = (ids[1:] != ids[:-1]) | ~same
    return np.flatnonzero(np.r_[True, np.asarray(change, dtype=bool)])


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

    id_value_column = deprecated_attribute("value_column_id")
    routing_value_column = deprecated_attribute("value_column_routing")

    #: Methods read only the constructor's signals (``Base`` keeps their rows).
    _reads_own_signals_only = True

    @renamed_kwargs(
        id_value_column="value_column_id", routing_value_column="value_column_routing"
    )
    def __init__(
        self,
        dataframe: pd.DataFrame,
        id_uuid: str,
        routing_uuid: str,
        *,
        state_map: dict[int | float | str, str] | None = None,
        station_map: dict[int | float | str, str] | None = None,
        event_uuid: str = "prod:routing_trace",
        value_column_id: str = "value_string",
        value_column_routing: str = "value_integer",
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
            station_map: Deprecated alias for ``state_map``; passing it
                         emits a :class:`DeprecationWarning` and it will be
                         removed in the next major release. When both are passed,
                         ``state_map`` wins and a UserWarning is emitted.
            event_uuid: UUID to tag derived events with.
            value_column_id: Column holding the item ID string.
            value_column_routing: Column holding the state/routing value.
            time_column: Name of timestamp column.
        """
        super().__init__(dataframe, time_column=time_column)
        self.id_uuid = id_uuid
        self.routing_uuid = routing_uuid
        if station_map is not None:
            warnings.warn(
                "RoutingTraceabilityEvents(): 'station_map' is deprecated, use "
                f"'state_map'; the old name will be removed in {REMOVAL}",
                DeprecationWarning,
                stacklevel=caller_stacklevel(),
            )
        # state_map takes priority over the deprecated station_map.
        if state_map is not None and station_map is not None:
            warnings.warn(
                "Both state_map and station_map were passed; station_map is "
                "ignored and state_map is used.",
                UserWarning,
                stacklevel=caller_stacklevel(),
            )
        chosen = state_map if state_map is not None else station_map
        self.state_map: dict[int | float | str, str] = chosen or {}
        self.event_uuid = event_uuid
        self.value_column_id = value_column_id
        self.value_column_routing = value_column_routing
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
        2. Try the value as an integer (``10.0`` and ``"10"`` match key ``10``).
        3. Try the value as a string.
        4. Fall back to "State {value}" (``5.0`` -> "State 5").
        """
        if value in self.state_map:
            return self.state_map[value]
        number = _as_number(value)
        if number is not None and number.is_integer():
            as_int = int(number)
            if as_int in self.state_map:
                return self.state_map[as_int]
        s = str(value)
        if s in self.state_map:
            return self.state_map[s]
        if number is None:
            return f"State {value}"
        if number.is_integer():
            return f"State {int(number)}"
        return f"State {number}"

    def _resolve_states(self, values: list[Any]) -> list[str]:
        """:meth:`_resolve_state` for many values (each distinct value once)."""
        cache: dict[tuple[type, str, Any], str] = {}
        out: list[str] = []
        for value in values:
            # Type and str in the key: 1, 1.0 and True compare equal but can
            # resolve differently.
            try:
                key = (type(value), str(value), value)
                name = cache.get(key)
                if name is None:
                    name = cache[key] = self._resolve_state(value)
            except TypeError:  # unhashable value
                name = self._resolve_state(value)
            out.append(name)
        return out

    # ------------------------------------------------------------------
    # build_routing_timeline
    # ------------------------------------------------------------------

    @property
    def _source_uuid(self) -> str:
        return f"{self.id_uuid},{self.routing_uuid}"

    def build_routing_timeline(
        self, *, infer_dwell_from_next: bool = False
    ) -> pd.DataFrame:
        """Correlate the ID signal with the state/routing signal to build a timeline.

        For each sample of the routing signal, the *current* item ID is
        determined via backward-fill merge (most recent ID at that timestamp).
        Contiguous intervals where the same (item_id, state) pair holds
        are grouped into single events.

        Args:
            infer_dwell_from_next: When False (default), an interval's ``end`` is
                the last routing sample of the (item, state) segment, so a
                change-encoded routing signal (one row per change) reports a
                dwell of ``0``. When True, each visit's ``end`` is set to the
                item's next routing arrival, giving the honest
                arrival-here → arrival-at-next-state dwell. The last visit per
                item stays open.

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
            - source_uuid: ``"<id_uuid>,<routing_uuid>"`` (the two
                correlated input signals).
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
                    "source_uuid",
                ]
            )

        # Merge: for each routing sample, attach the most-recent item ID
        routing_subset = self.routing_data[
            [self.time_column, self.value_column_routing]
        ].copy()
        id_subset = self.id_data[[self.time_column, self.value_column_id]].copy()

        merged = pd.merge_asof(
            routing_subset,
            id_subset,
            on=self.time_column,
            direction="backward",
        )

        merged = merged.rename(
            columns={
                self.value_column_id: "item_id",
                self.value_column_routing: "routing_value",
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
                    "source_uuid",
                ]
            )

        # One row per contiguous run of the key ``item_id + "|" +
        # str(routing_value)``, in time order.
        n = len(merged)
        first_pos = _combo_run_starts(merged["item_id"], merged["routing_value"])
        last_pos = np.r_[first_pos[1:] - 1, n - 1]
        times = merged[self.time_column]
        start = times.iloc[first_pos].reset_index(drop=True)
        end = times.iloc[last_pos].reset_index(drop=True)
        routing_vals = merged["routing_value"].iloc[first_pos].tolist()
        timeline = pd.DataFrame(
            {
                "item_id": merged["item_id"].iloc[first_pos].tolist(),
                "routing_value": routing_vals,
                "station_name": self._resolve_states(routing_vals),
                "start": start,
                "end": end,
                "duration_seconds": (end - start).dt.total_seconds(),
                "sample_count": np.diff(np.r_[first_pos, n]).tolist(),
            }
        )
        timeline = timeline.sort_values(["item_id", "start"]).reset_index(drop=True)
        timeline["station_sequence"] = timeline.groupby("item_id").cumcount() + 1
        timeline["uuid"] = self.event_uuid
        timeline["source_uuid"] = self._source_uuid

        if infer_dwell_from_next:
            next_start = timeline.groupby("item_id")["start"].shift(-1)
            timeline["end"] = next_start.fillna(timeline["end"])
            timeline["duration_seconds"] = (
                timeline["end"] - timeline["start"]
            ).dt.total_seconds()

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
            - start: Same as first_seen.
            - end: Same as last_seen.
            - duration_seconds: Same as lead_time_seconds (float seconds).
            - source_uuid: ``"<id_uuid>,<routing_uuid>"``.
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
                    "start",
                    "end",
                    "duration_seconds",
                    "source_uuid",
                ]
            )

        rows: list[dict[str, Any]] = []
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
                    "start": first_seen,
                    "end": last_seen,
                    "duration_seconds": float((last_seen - first_seen).total_seconds()),
                    "source_uuid": self._source_uuid,
                }
            )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # station_statistics
    # ------------------------------------------------------------------

    def station_statistics(
        self, *, infer_dwell_from_next: bool = False
    ) -> pd.DataFrame:
        """Compute dwell-time statistics per station/step across all items.

        Args:
            infer_dwell_from_next: Forwarded to :meth:`build_routing_timeline`;
                pass True for change-encoded routing signals so dwell is measured
                to the item's next routing arrival instead of collapsing to zero.

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
        timeline = self.build_routing_timeline(
            infer_dwell_from_next=infer_dwell_from_next
        )

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

        return stats.sort_values("item_count", ascending=False).reset_index(drop=True)
