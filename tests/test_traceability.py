"""Tests for ValueTraceabilityEvents and RoutingTraceabilityEvents."""

import pandas as pd  # type: ignore
import pytest
from datetime import timedelta

from ts_shape.events.production import (
    ValueTraceabilityEvents,
    OrderTraceabilityEvents,  # backwards-compatible alias
    RoutingTraceabilityEvents,
)

# ============================================================================
# Shared helpers
# ============================================================================


def _empty_df():
    """Return an empty DataFrame with standard columns."""
    return pd.DataFrame(
        columns=[
            "systime",
            "uuid",
            "value_bool",
            "value_integer",
            "value_double",
            "value_string",
            "is_delta",
        ]
    )


# ============================================================================
# ValueTraceabilityEvents fixtures and tests
# ============================================================================


@pytest.fixture
def value_trace_data():
    """Create synthetic data: 3 stations, 2 identifiers flowing through them.

    Flow:
    - ORD-001: Station A (08:00-08:10) -> Station B (08:12-08:25) -> Station C (08:27-08:40)
    - ORD-002: Station A (08:15-08:30) -> Station B (08:32-08:45) -> Station C (08:47-08:55)
    """
    base = pd.Timestamp("2024-01-15 08:00:00")
    rows = []

    def _add_station_rows(uuid, identifier, start_min, end_min, freq_min=1):
        for m in range(start_min, end_min + 1, freq_min):
            rows.append(
                {
                    "systime": base + timedelta(minutes=m),
                    "uuid": uuid,
                    "value_bool": None,
                    "value_integer": None,
                    "value_double": None,
                    "value_string": identifier,
                    "is_delta": False,
                }
            )

    # ORD-001 at each station
    _add_station_rows("station_a", "ORD-001", 0, 10)
    _add_station_rows("station_b", "ORD-001", 12, 25)
    _add_station_rows("station_c", "ORD-001", 27, 40)

    # ORD-002 at each station
    _add_station_rows("station_a", "ORD-002", 15, 30)
    _add_station_rows("station_b", "ORD-002", 32, 45)
    _add_station_rows("station_c", "ORD-002", 47, 55)

    return pd.DataFrame(rows)


@pytest.fixture
def value_tracer(value_trace_data):
    return ValueTraceabilityEvents(
        value_trace_data,
        station_uuids={
            "station_a": "Station A",
            "station_b": "Station B",
            "station_c": "Station C",
        },
    )


class TestValueTraceabilityEvents:

    def test_build_timeline(self, value_tracer):
        timeline = value_tracer.build_timeline()
        assert not timeline.empty
        # 2 identifiers x 3 stations = 6 rows
        assert len(timeline) == 6
        assert set(timeline["identifier"]) == {"ORD-001", "ORD-002"}
        assert set(timeline["station_name"]) == {"Station A", "Station B", "Station C"}
        # Each identifier should have sequences 1, 2, 3
        for ident in ["ORD-001", "ORD-002"]:
            seqs = timeline[timeline["identifier"] == ident][
                "station_sequence"
            ].tolist()
            assert seqs == [1, 2, 3]

    def test_build_timeline_columns(self, value_tracer):
        timeline = value_tracer.build_timeline()
        expected_cols = {
            "identifier",
            "station_uuid",
            "station_name",
            "start",
            "end",
            "duration_seconds",
            "sample_count",
            "station_sequence",
            "uuid",
        }
        assert set(timeline.columns) == expected_cols

    def test_lead_time(self, value_tracer):
        lead = value_tracer.lead_time()
        assert len(lead) == 2

        ord1 = lead[lead["identifier"] == "ORD-001"].iloc[0]
        assert ord1["first_station"] == "Station A"
        assert ord1["last_station"] == "Station C"
        assert ord1["stations_visited"] == 3
        assert "Station A" in ord1["station_path"]
        assert "Station C" in ord1["station_path"]
        assert ord1["lead_time_seconds"] > 0

    def test_current_status(self, value_tracer):
        status = value_tracer.current_status()
        assert len(status) == 2
        # Both identifiers last seen at Station C
        assert set(status["current_station"]) == {"Station C"}

    def test_station_dwell_statistics(self, value_tracer):
        stats = value_tracer.station_dwell_statistics()
        assert len(stats) == 3  # 3 stations
        assert "avg_dwell_seconds" in stats.columns
        assert "identifier_count" in stats.columns
        # Each station saw 2 identifiers
        for _, row in stats.iterrows():
            assert row["identifier_count"] == 2

    def test_empty_data(self):
        tracer = ValueTraceabilityEvents(
            _empty_df(),
            station_uuids={"x": "X"},
        )
        assert tracer.build_timeline().empty
        assert tracer.lead_time().empty
        assert tracer.current_status().empty
        assert tracer.station_dwell_statistics().empty

    def test_single_station(self):
        """Identifier at only one station."""
        base = pd.Timestamp("2024-01-15 08:00:00")
        rows = []
        for m in range(5):
            rows.append(
                {
                    "systime": base + timedelta(minutes=m),
                    "uuid": "sta",
                    "value_bool": None,
                    "value_integer": None,
                    "value_double": None,
                    "value_string": "SN-100",
                    "is_delta": False,
                }
            )

        tracer = ValueTraceabilityEvents(
            pd.DataFrame(rows),
            station_uuids={"sta": "Only Station"},
        )
        timeline = tracer.build_timeline()
        assert len(timeline) == 1
        assert timeline.iloc[0]["identifier"] == "SN-100"

        lead = tracer.lead_time()
        assert len(lead) == 1
        assert lead.iloc[0]["stations_visited"] == 1

    def test_backwards_compat_alias(self, value_trace_data):
        """OrderTraceabilityEvents should work as an alias."""
        tracer = OrderTraceabilityEvents(
            value_trace_data,
            station_uuids={
                "station_a": "Station A",
                "station_b": "Station B",
                "station_c": "Station C",
            },
        )
        timeline = tracer.build_timeline()
        assert len(timeline) == 6
        assert "identifier" in timeline.columns


# ============================================================================
# RoutingTraceabilityEvents fixtures and tests
# ============================================================================


@pytest.fixture
def routing_trace_data():
    """Create synthetic data: ID signal + state/routing signal.

    Flow:
    - Serial "SN-AAA": state=10 (Heating), state=20 (Holding), state=30 (Cooling)
    - Serial "SN-BBB": state=10 (Heating), state=30 (Cooling) (skips Holding)
    """
    base = pd.Timestamp("2024-01-15 08:00:00")
    rows = []

    # ID signal: SN-AAA from 08:00 to 08:30, SN-BBB from 08:35 to 08:55
    for m in range(0, 31):
        rows.append(
            {
                "systime": base + timedelta(minutes=m),
                "uuid": "id_signal",
                "value_bool": None,
                "value_integer": None,
                "value_double": None,
                "value_string": "SN-AAA",
                "is_delta": False,
            }
        )
    for m in range(35, 56):
        rows.append(
            {
                "systime": base + timedelta(minutes=m),
                "uuid": "id_signal",
                "value_bool": None,
                "value_integer": None,
                "value_double": None,
                "value_string": "SN-BBB",
                "is_delta": False,
            }
        )

    # State signal: PLC step numbers
    def _add_state(start_min, end_min, state_val):
        for m in range(start_min, end_min + 1):
            rows.append(
                {
                    "systime": base + timedelta(minutes=m),
                    "uuid": "state_signal",
                    "value_bool": None,
                    "value_integer": state_val,
                    "value_double": None,
                    "value_string": None,
                    "is_delta": False,
                }
            )

    # SN-AAA states
    _add_state(0, 10, 10)  # Heating
    _add_state(12, 20, 20)  # Holding
    _add_state(22, 30, 30)  # Cooling

    # SN-BBB states
    _add_state(35, 45, 10)  # Heating
    _add_state(47, 55, 30)  # Cooling (skips Holding)

    return pd.DataFrame(rows)


@pytest.fixture
def routing_tracer(routing_trace_data):
    return RoutingTraceabilityEvents(
        routing_trace_data,
        id_uuid="id_signal",
        routing_uuid="state_signal",
        state_map={10: "Heating", 20: "Holding", 30: "Cooling"},
    )


class TestRoutingTraceabilityEvents:

    def test_build_routing_timeline(self, routing_tracer):
        timeline = routing_tracer.build_routing_timeline()
        assert not timeline.empty
        # SN-AAA: 3 steps, SN-BBB: 2 steps = 5 rows
        assert len(timeline) == 5

        aaa = timeline[timeline["item_id"] == "SN-AAA"]
        assert len(aaa) == 3
        assert aaa.iloc[0]["station_name"] == "Heating"
        assert aaa.iloc[1]["station_name"] == "Holding"
        assert aaa.iloc[2]["station_name"] == "Cooling"

        bbb = timeline[timeline["item_id"] == "SN-BBB"]
        assert len(bbb) == 2
        assert bbb.iloc[0]["station_name"] == "Heating"
        assert bbb.iloc[1]["station_name"] == "Cooling"

    def test_timeline_columns(self, routing_tracer):
        timeline = routing_tracer.build_routing_timeline()
        expected_cols = {
            "item_id",
            "routing_value",
            "station_name",
            "start",
            "end",
            "duration_seconds",
            "sample_count",
            "station_sequence",
            "uuid",
        }
        assert set(timeline.columns) == expected_cols

    def test_lead_time(self, routing_tracer):
        lead = routing_tracer.lead_time()
        assert len(lead) == 2

        aaa = lead[lead["item_id"] == "SN-AAA"].iloc[0]
        assert aaa["first_station"] == "Heating"
        assert aaa["last_station"] == "Cooling"
        assert aaa["stations_visited"] == 3
        assert aaa["lead_time_seconds"] > 0
        assert "Heating" in aaa["routing_path"]
        assert "Cooling" in aaa["routing_path"]

        bbb = lead[lead["item_id"] == "SN-BBB"].iloc[0]
        assert bbb["stations_visited"] == 2  # skipped Holding

    def test_station_statistics(self, routing_tracer):
        stats = routing_tracer.station_statistics()
        assert len(stats) == 3  # Heating, Holding, Cooling
        assert "item_count" in stats.columns

        heating = stats[stats["station_name"] == "Heating"].iloc[0]
        assert heating["item_count"] == 2  # Both items

        holding = stats[stats["station_name"] == "Holding"].iloc[0]
        assert holding["item_count"] == 1  # Only SN-AAA

    def test_routing_paths(self, routing_tracer):
        paths = routing_tracer.routing_paths()
        assert len(paths) == 2  # Two distinct paths
        assert "routing_path" in paths.columns
        assert "item_count" in paths.columns
        assert "avg_lead_time_seconds" in paths.columns

    def test_empty_data(self):
        tracer = RoutingTraceabilityEvents(
            _empty_df(),
            id_uuid="x",
            routing_uuid="y",
        )
        assert tracer.build_routing_timeline().empty
        assert tracer.lead_time().empty
        assert tracer.station_statistics().empty
        assert tracer.routing_paths().empty

    def test_no_state_map(self):
        """Without state_map, should use 'State N' naming."""
        base = pd.Timestamp("2024-01-15 08:00:00")
        rows = []
        for m in range(5):
            rows.append(
                {
                    "systime": base + timedelta(minutes=m),
                    "uuid": "id_sig",
                    "value_bool": None,
                    "value_integer": None,
                    "value_double": None,
                    "value_string": "ITEM-1",
                    "is_delta": False,
                }
            )
            rows.append(
                {
                    "systime": base + timedelta(minutes=m),
                    "uuid": "route_sig",
                    "value_bool": None,
                    "value_integer": 5,
                    "value_double": None,
                    "value_string": None,
                    "is_delta": False,
                }
            )

        tracer = RoutingTraceabilityEvents(
            pd.DataFrame(rows),
            id_uuid="id_sig",
            routing_uuid="route_sig",
        )
        timeline = tracer.build_routing_timeline()
        assert len(timeline) == 1
        assert timeline.iloc[0]["station_name"] == "State 5"

    def test_backwards_compat_station_map(self, routing_trace_data):
        """Old station_map parameter should still work."""
        tracer = RoutingTraceabilityEvents(
            routing_trace_data,
            id_uuid="id_signal",
            routing_uuid="state_signal",
            station_map={10: "Heating", 20: "Holding", 30: "Cooling"},
        )
        timeline = tracer.build_routing_timeline()
        assert len(timeline) == 5
        assert timeline.iloc[0]["station_name"] == "Heating"

    def test_station_sequence_per_item(self, routing_tracer):
        """Each item should have its own sequence numbering."""
        timeline = routing_tracer.build_routing_timeline()
        for item_id in ["SN-AAA", "SN-BBB"]:
            seqs = timeline[timeline["item_id"] == item_id]["station_sequence"].tolist()
            assert seqs == list(range(1, len(seqs) + 1))
