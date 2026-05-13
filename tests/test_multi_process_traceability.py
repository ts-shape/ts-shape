"""Tests for MultiProcessTraceabilityEvents.

Topology under test::

    Welding Cell 1 ─┐
                     ├─► Painting ─► Assembly
    Welding Cell 2 ─┘

Items:
- SN-001: Welding Cell 1 -> Painting -> Assembly (sequential)
- SN-002: Welding Cell 2 -> Painting -> Assembly (sequential)
- SN-003: Welding Cell 1 + Welding Cell 2 overlap -> Painting (parallel start)
"""

import pandas as pd  # type: ignore
import numpy as np
import pytest
from datetime import datetime, timedelta

from ts_shape.events.production import MultiProcessTraceabilityEvents

# ============================================================================
# Helpers
# ============================================================================


def _empty_df():
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


def _row(t, uuid, val_str=None, val_int=None):
    return {
        "systime": t,
        "uuid": uuid,
        "value_bool": None,
        "value_integer": val_int,
        "value_double": None,
        "value_string": val_str,
        "is_delta": False,
    }


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def parallel_data():
    """Build data for a parallel welding + sequential painting + assembly topology.

    Timeline (minutes from base):
    - SN-001: Weld1 [0-10], Painting [12-20], Assembly [22-30]
    - SN-002: Weld2 [5-15], Painting [21-29], Assembly [31-39]
      (Painting & Assembly are shared UUIDs, so items are sequential on them)
    - SN-003: Weld1 [40-48] AND Weld2 [42-50] (parallel overlap), Painting [52-60]

    Handover signals (fire as rising edge 0->1):
    - ho_w1_paint: fires at min 11 (SN-001), min 51 (SN-003)
    - ho_w2_paint: fires at min 20 (SN-002), min 51 (SN-003)
    - ho_paint_assy: fires at min 21 (SN-001), min 30 (SN-002)
    """
    base = pd.Timestamp("2024-01-15 08:00:00")
    rows = []

    def _add_id(uuid, item_id, start_min, end_min):
        for m in range(start_min, end_min + 1):
            rows.append(_row(base + timedelta(minutes=m), uuid, val_str=item_id))

    def _add_handover(uuid, fire_min):
        """Add a handover: 0 before, 1 at fire time, 0 after."""
        rows.append(_row(base + timedelta(minutes=fire_min - 1), uuid, val_int=0))
        rows.append(_row(base + timedelta(minutes=fire_min), uuid, val_int=1))
        rows.append(_row(base + timedelta(minutes=fire_min + 1), uuid, val_int=0))

    # SN-001 path (sequential: weld1 -> painting -> assembly)
    _add_id("serial_weld1", "SN-001", 0, 10)
    _add_id("serial_paint", "SN-001", 12, 20)
    _add_id("serial_assy", "SN-001", 22, 30)

    # SN-002 path (sequential: weld2 -> painting -> assembly, after SN-001 finishes)
    _add_id("serial_weld2", "SN-002", 5, 15)
    _add_id("serial_paint", "SN-002", 21, 29)
    _add_id("serial_assy", "SN-002", 31, 39)

    # SN-003: parallel welding (both cells simultaneously)
    _add_id("serial_weld1", "SN-003", 40, 48)
    _add_id("serial_weld2", "SN-003", 42, 50)
    _add_id("serial_paint", "SN-003", 52, 60)

    # Handovers
    _add_handover("ho_w1_paint", 11)
    _add_handover("ho_w2_paint", 20)
    _add_handover("ho_paint_assy", 21)
    _add_handover("ho_paint_assy", 30)
    _add_handover("ho_w1_paint", 51)
    _add_handover("ho_w2_paint", 51)

    return pd.DataFrame(rows)


@pytest.fixture
def tracer(parallel_data):
    return MultiProcessTraceabilityEvents(
        parallel_data,
        processes=[
            {"id_uuid": "serial_weld1", "station": "Welding Cell 1"},
            {"id_uuid": "serial_weld2", "station": "Welding Cell 2"},
            {"id_uuid": "serial_paint", "station": "Painting"},
            {"id_uuid": "serial_assy", "station": "Assembly"},
        ],
        handovers=[
            {
                "uuid": "ho_w1_paint",
                "from_station": "Welding Cell 1",
                "to_station": "Painting",
            },
            {
                "uuid": "ho_w2_paint",
                "from_station": "Welding Cell 2",
                "to_station": "Painting",
            },
            {
                "uuid": "ho_paint_assy",
                "from_station": "Painting",
                "to_station": "Assembly",
            },
        ],
    )


# ============================================================================
# Tests: build_timeline
# ============================================================================


class TestBuildTimeline:

    def test_timeline_row_count(self, tracer):
        timeline = tracer.build_timeline()
        # SN-001: 3 stations, SN-002: 3 stations, SN-003: 3 stations (weld1 + weld2 + paint)
        assert len(timeline) == 9

    def test_timeline_columns(self, tracer):
        timeline = tracer.build_timeline()
        expected = {
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
        }
        assert set(timeline.columns) == expected

    def test_sequential_item_not_parallel(self, tracer):
        timeline = tracer.build_timeline()
        sn001 = timeline[timeline["item_id"] == "SN-001"]
        # SN-001 is fully sequential, no overlap
        assert not sn001["is_parallel"].any()

    def test_parallel_item_detected(self, tracer):
        timeline = tracer.build_timeline()
        sn003 = timeline[timeline["item_id"] == "SN-003"]
        # SN-003 has overlapping welding cells
        parallel_rows = sn003[sn003["is_parallel"]]
        assert len(parallel_rows) >= 2
        parallel_stations = set(parallel_rows["station"])
        assert "Welding Cell 1" in parallel_stations
        assert "Welding Cell 2" in parallel_stations

    def test_step_sequence_per_item(self, tracer):
        timeline = tracer.build_timeline()
        for item_id in ["SN-001", "SN-002", "SN-003"]:
            seqs = timeline[timeline["item_id"] == item_id]["step_sequence"].tolist()
            assert seqs == list(range(1, len(seqs) + 1))

    def test_all_items_present(self, tracer):
        timeline = tracer.build_timeline()
        assert set(timeline["item_id"]) == {"SN-001", "SN-002", "SN-003"}

    def test_empty_data(self):
        tracer = MultiProcessTraceabilityEvents(
            _empty_df(),
            processes=[{"id_uuid": "x", "station": "X"}],
        )
        assert tracer.build_timeline().empty


# ============================================================================
# Tests: lead_time
# ============================================================================


class TestLeadTime:

    def test_lead_time_row_count(self, tracer):
        lead = tracer.lead_time()
        assert len(lead) == 3  # 3 items

    def test_lead_time_columns(self, tracer):
        lead = tracer.lead_time()
        assert "has_parallel" in lead.columns
        assert "station_path" in lead.columns

    def test_sequential_item_lead(self, tracer):
        lead = tracer.lead_time()
        sn001 = lead[lead["item_id"] == "SN-001"].iloc[0]
        assert sn001["stations_visited"] == 3
        assert sn001["has_parallel"] == False
        assert sn001["lead_time_seconds"] > 0

    def test_parallel_item_flagged(self, tracer):
        lead = tracer.lead_time()
        sn003 = lead[lead["item_id"] == "SN-003"].iloc[0]
        assert sn003["has_parallel"] == True

    def test_empty_data(self):
        tracer = MultiProcessTraceabilityEvents(
            _empty_df(),
            processes=[{"id_uuid": "x", "station": "X"}],
        )
        assert tracer.lead_time().empty


# ============================================================================
# Tests: parallel_activity
# ============================================================================


class TestParallelActivity:

    def test_finds_overlap(self, tracer):
        parallel = tracer.parallel_activity()
        # SN-003 has overlap between Welding Cell 1 and Welding Cell 2
        sn003 = parallel[parallel["item_id"] == "SN-003"]
        assert len(sn003) >= 1
        stations = set(sn003["station_a"].tolist() + sn003["station_b"].tolist())
        assert "Welding Cell 1" in stations
        assert "Welding Cell 2" in stations

    def test_overlap_seconds_positive(self, tracer):
        parallel = tracer.parallel_activity()
        if not parallel.empty:
            assert (parallel["overlap_seconds"] > 0).all()

    def test_sequential_items_no_overlap(self, tracer):
        parallel = tracer.parallel_activity()
        sn001 = parallel[parallel["item_id"] == "SN-001"]
        assert sn001.empty

    def test_empty_data(self):
        tracer = MultiProcessTraceabilityEvents(
            _empty_df(),
            processes=[{"id_uuid": "x", "station": "X"}],
        )
        assert tracer.parallel_activity().empty


# ============================================================================
# Tests: handover_log
# ============================================================================


class TestHandoverLog:

    def test_handover_events_detected(self, tracer):
        log = tracer.handover_log()
        assert not log.empty

    def test_handover_columns(self, tracer):
        log = tracer.handover_log()
        expected = {
            "timestamp",
            "item_id",
            "from_station",
            "to_station",
            "handover_uuid",
            "handover_value",
            "uuid",
        }
        assert set(log.columns) == expected

    def test_handover_resolves_item_id(self, tracer):
        log = tracer.handover_log()
        # All handover events should have resolved an item_id
        assert (log["item_id"] != "").all()

    def test_no_handovers_configured(self):
        """Without handover config, returns empty."""
        base = pd.Timestamp("2024-01-15 08:00:00")
        rows = [_row(base + timedelta(minutes=m), "id1", val_str="X") for m in range(5)]
        tracer = MultiProcessTraceabilityEvents(
            pd.DataFrame(rows),
            processes=[{"id_uuid": "id1", "station": "S1"}],
        )
        assert tracer.handover_log().empty

    def test_empty_data(self):
        tracer = MultiProcessTraceabilityEvents(
            _empty_df(),
            processes=[{"id_uuid": "x", "station": "X"}],
            handovers=[{"uuid": "h", "from_station": "X", "to_station": "Y"}],
        )
        assert tracer.handover_log().empty


# ============================================================================
# Tests: station_statistics
# ============================================================================


class TestStationStatistics:

    def test_all_stations_present(self, tracer):
        stats = tracer.station_statistics()
        stations = set(stats["station"])
        assert "Welding Cell 1" in stations
        assert "Welding Cell 2" in stations
        assert "Painting" in stations

    def test_parallel_cells_separate(self, tracer):
        stats = tracer.station_statistics()
        # Welding Cell 1 and Welding Cell 2 should each have their own row
        weld1 = stats[stats["station"] == "Welding Cell 1"]
        weld2 = stats[stats["station"] == "Welding Cell 2"]
        assert len(weld1) == 1
        assert len(weld2) == 1

    def test_item_counts(self, tracer):
        stats = tracer.station_statistics()
        # Welding Cell 1: SN-001 and SN-003
        weld1 = stats[stats["station"] == "Welding Cell 1"].iloc[0]
        assert weld1["item_count"] == 2

    def test_empty_data(self):
        tracer = MultiProcessTraceabilityEvents(
            _empty_df(),
            processes=[{"id_uuid": "x", "station": "X"}],
        )
        assert tracer.station_statistics().empty


# ============================================================================
# Tests: routing_paths
# ============================================================================


class TestRoutingPaths:

    def test_paths_detected(self, tracer):
        paths = tracer.routing_paths()
        assert not paths.empty

    def test_path_columns(self, tracer):
        paths = tracer.routing_paths()
        expected = {
            "station_path",
            "item_count",
            "avg_lead_time_seconds",
            "min_lead_time_seconds",
            "max_lead_time_seconds",
            "has_parallel_steps",
        }
        assert set(paths.columns) == expected

    def test_parallel_path_flagged(self, tracer):
        paths = tracer.routing_paths()
        # At least one path should have parallel steps (SN-003's path)
        assert paths["has_parallel_steps"].any()

    def test_empty_data(self):
        tracer = MultiProcessTraceabilityEvents(
            _empty_df(),
            processes=[{"id_uuid": "x", "station": "X"}],
        )
        assert tracer.routing_paths().empty


# ============================================================================
# Tests: edge cases
# ============================================================================


class TestEdgeCases:

    def test_single_station_single_item(self):
        base = pd.Timestamp("2024-01-15 08:00:00")
        rows = [
            _row(base + timedelta(minutes=m), "id1", val_str="ITEM-A") for m in range(5)
        ]
        tracer = MultiProcessTraceabilityEvents(
            pd.DataFrame(rows),
            processes=[{"id_uuid": "id1", "station": "Only Station"}],
        )
        timeline = tracer.build_timeline()
        assert len(timeline) == 1
        assert timeline.iloc[0]["item_id"] == "ITEM-A"
        assert timeline.iloc[0]["is_parallel"] == False

    def test_many_parallel_cells(self):
        """3 parallel cells of the same station type."""
        base = pd.Timestamp("2024-01-15 08:00:00")
        rows = []
        for m in range(10):
            rows.append(_row(base + timedelta(minutes=m), "cell_1", val_str="ITEM-X"))
            rows.append(_row(base + timedelta(minutes=m), "cell_2", val_str="ITEM-Y"))
            rows.append(_row(base + timedelta(minutes=m), "cell_3", val_str="ITEM-Z"))

        tracer = MultiProcessTraceabilityEvents(
            pd.DataFrame(rows),
            processes=[
                {"id_uuid": "cell_1", "station": "Welding"},
                {"id_uuid": "cell_2", "station": "Welding"},
                {"id_uuid": "cell_3", "station": "Welding"},
            ],
        )
        timeline = tracer.build_timeline()
        assert len(timeline) == 3  # 3 items, each at one cell
        # Different items at parallel cells are NOT "parallel" for the same item
        assert not timeline["is_parallel"].any()

        stats = tracer.station_statistics()
        # 3 separate rows (one per cell UUID)
        assert len(stats) == 3

    def test_same_item_revisits_station(self):
        """Item goes to Station A, then B, then back to A (rework)."""
        base = pd.Timestamp("2024-01-15 08:00:00")
        rows = []
        for m in range(0, 5):
            rows.append(_row(base + timedelta(minutes=m), "sta_a", val_str="RW-001"))
        for m in range(6, 10):
            rows.append(_row(base + timedelta(minutes=m), "sta_b", val_str="RW-001"))
        for m in range(11, 15):
            rows.append(_row(base + timedelta(minutes=m), "sta_a", val_str="RW-001"))

        tracer = MultiProcessTraceabilityEvents(
            pd.DataFrame(rows),
            processes=[
                {"id_uuid": "sta_a", "station": "Station A"},
                {"id_uuid": "sta_b", "station": "Station B"},
            ],
        )
        timeline = tracer.build_timeline()
        rw = timeline[timeline["item_id"] == "RW-001"]
        assert len(rw) == 3  # A, B, A again
        assert rw.iloc[0]["station"] == "Station A"
        assert rw.iloc[1]["station"] == "Station B"
        assert rw.iloc[2]["station"] == "Station A"

        lead = tracer.lead_time()
        assert "Station A -> Station B -> Station A" in lead.iloc[0]["station_path"]
