"""Tests for OEECalculator, AlarmManagementEvents, and BatchTrackingEvents."""

import pandas as pd  # type: ignore
import numpy as np
import pytest
from datetime import datetime, timedelta

from ts_shape.events.production import (
    OEECalculator,
    AlarmManagementEvents,
    BatchTrackingEvents,
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
# OEECalculator fixtures and tests
# ============================================================================


@pytest.fixture
def oee_data():
    """Create synthetic data for OEE calculation.

    Signals:
    - 'machine_state' (value_bool): True=running, with some idle gaps
    - 'part_counter' (value_integer): monotonic counter
    - 'total_counter' (value_integer): total parts counter
    - 'reject_counter' (value_integer): reject parts counter
    """
    base = pd.Timestamp("2024-01-15 06:00:00")
    # One day, one sample per minute for 16 hours (960 minutes)
    timestamps = pd.date_range(base, periods=960, freq="1min")

    rows = []

    # Machine state: running for 50 min, idle for 10 min, repeating
    for i, t in enumerate(timestamps):
        cycle_pos = i % 60
        running = cycle_pos < 50  # 50 min run, 10 min idle
        rows.append(
            {
                "systime": t,
                "uuid": "machine_state",
                "value_bool": running,
                "value_integer": None,
                "value_double": None,
                "value_string": None,
                "is_delta": False,
            }
        )

    # Part counter: increments by 1 every minute when running
    counter = 0
    for i, t in enumerate(timestamps):
        cycle_pos = i % 60
        if cycle_pos < 50:
            counter += 1
        rows.append(
            {
                "systime": t,
                "uuid": "part_counter",
                "value_bool": None,
                "value_integer": counter,
                "value_double": None,
                "value_string": None,
                "is_delta": True,
            }
        )

    # Total counter = same as part counter
    total = 0
    for i, t in enumerate(timestamps):
        cycle_pos = i % 60
        if cycle_pos < 50:
            total += 1
        rows.append(
            {
                "systime": t,
                "uuid": "total_counter",
                "value_bool": None,
                "value_integer": total,
                "value_double": None,
                "value_string": None,
                "is_delta": True,
            }
        )

    # Reject counter: 1 reject every 100 parts
    reject = 0
    parts_since_reject = 0
    for i, t in enumerate(timestamps):
        cycle_pos = i % 60
        if cycle_pos < 50:
            parts_since_reject += 1
            if parts_since_reject >= 100:
                reject += 1
                parts_since_reject = 0
        rows.append(
            {
                "systime": t,
                "uuid": "reject_counter",
                "value_bool": None,
                "value_integer": reject,
                "value_double": None,
                "value_string": None,
                "is_delta": True,
            }
        )

    return pd.DataFrame(rows)


class TestOEECalculator:

    def test_empty_dataframe(self):
        df = _empty_df()
        oee = OEECalculator(df)
        assert oee.calculate_availability("machine_state").empty
        assert oee.calculate_performance("counter", 60.0).empty
        assert oee.calculate_quality("total", "reject").empty
        assert oee.calculate_oee("state", "counter", 60.0).empty

    def test_calculate_availability(self, oee_data):
        oee = OEECalculator(oee_data)
        result = oee.calculate_availability("machine_state")

        assert not result.empty
        assert "date" in result.columns
        assert "availability_pct" in result.columns
        # 50/60 ~ 83.3% availability per cycle
        avail = result["availability_pct"].iloc[0]
        assert 80.0 < avail < 90.0, f"Expected ~83%, got {avail}%"

    def test_calculate_availability_with_planned_time(self, oee_data):
        oee = OEECalculator(oee_data)
        result = oee.calculate_availability("machine_state", planned_time_hours=24.0)

        assert not result.empty
        # With 24h planned, availability should be lower
        avail = result["availability_pct"].iloc[0]
        assert avail < 80.0

    def test_calculate_performance(self, oee_data):
        oee = OEECalculator(oee_data)
        # ideal_cycle_time = 60s means 1 part per minute is 100%
        result = oee.calculate_performance(
            "part_counter", ideal_cycle_time=60.0, run_state_uuid="machine_state"
        )

        assert not result.empty
        assert "performance_pct" in result.columns
        perf = result["performance_pct"].iloc[0]
        # We produce 1 part/min during run time, ideal is 1 part/min -> ~100%
        assert perf > 90.0, f"Expected ~100%, got {perf}%"

    def test_calculate_quality(self, oee_data):
        oee = OEECalculator(oee_data)
        result = oee.calculate_quality("total_counter", "reject_counter")

        assert not result.empty
        assert "quality_pct" in result.columns
        qual = result["quality_pct"].iloc[0]
        # 1 reject per 100 parts -> 99% quality
        assert qual > 95.0, f"Expected ~99%, got {qual}%"

    def test_calculate_oee_full(self, oee_data):
        oee = OEECalculator(oee_data)
        result = oee.calculate_oee(
            run_state_uuid="machine_state",
            counter_uuid="part_counter",
            ideal_cycle_time=60.0,
            total_uuid="total_counter",
            reject_uuid="reject_counter",
        )

        assert not result.empty
        assert set(result.columns) == {
            "date",
            "availability_pct",
            "performance_pct",
            "quality_pct",
            "oee_pct",
        }
        oee_val = result["oee_pct"].iloc[0]
        assert oee_val > 0.0

    def test_calculate_oee_without_quality(self, oee_data):
        oee = OEECalculator(oee_data)
        result = oee.calculate_oee(
            run_state_uuid="machine_state",
            counter_uuid="part_counter",
            ideal_cycle_time=60.0,
        )

        assert not result.empty
        # Quality should default to 100%
        assert result["quality_pct"].iloc[0] == 100.0

    def test_missing_uuid_returns_empty(self, oee_data):
        oee = OEECalculator(oee_data)
        result = oee.calculate_availability("nonexistent_uuid")
        assert result.empty


# ============================================================================
# AlarmManagementEvents fixtures and tests
# ============================================================================


@pytest.fixture
def alarm_data():
    """Create synthetic alarm signal data.

    Produces a signal with:
    - Several normal alarm activations (ON for a few minutes)
    - One chattering period (rapid on/off)
    - One standing alarm (ON for 2 hours)
    """
    base = pd.Timestamp("2024-01-15 06:00:00")
    rows = []

    def add(t, val):
        rows.append(
            {
                "systime": t,
                "uuid": "temp_alarm",
                "value_bool": val,
                "value_integer": None,
                "value_double": None,
                "value_string": None,
                "is_delta": False,
            }
        )

    # Normal alarm 1: ON 5 minutes
    add(base + timedelta(minutes=10), True)
    add(base + timedelta(minutes=15), False)

    # Normal alarm 2: ON 3 minutes
    add(base + timedelta(minutes=30), True)
    add(base + timedelta(minutes=33), False)

    # Chattering period: 10 rapid toggles in 5 minutes
    chatter_start = base + timedelta(hours=1)
    for i in range(10):
        add(chatter_start + timedelta(seconds=30 * i), i % 2 == 0)
    add(chatter_start + timedelta(seconds=300), False)  # end chattering OFF

    # Standing alarm: ON for 2 hours
    add(base + timedelta(hours=3), True)
    add(base + timedelta(hours=5), False)

    # Quiet period at start/end
    add(base, False)
    add(base + timedelta(hours=8), False)

    return pd.DataFrame(rows)


class TestAlarmManagementEvents:

    def test_empty_dataframe(self):
        df = _empty_df()
        alarms = AlarmManagementEvents(df, alarm_uuid="temp_alarm")
        assert alarms.alarm_frequency().empty
        assert alarms.alarm_duration_stats().empty
        assert alarms.chattering_detection().empty
        assert alarms.standing_alarms().empty

    def test_alarm_frequency(self, alarm_data):
        alarms = AlarmManagementEvents(alarm_data, alarm_uuid="temp_alarm")
        result = alarms.alarm_frequency(window="1h")

        assert not result.empty
        assert "alarm_count" in result.columns
        assert "window_start" in result.columns
        # Should have some activations
        total = result["alarm_count"].sum()
        assert total > 0

    def test_alarm_duration_stats(self, alarm_data):
        alarms = AlarmManagementEvents(alarm_data, alarm_uuid="temp_alarm")
        result = alarms.alarm_duration_stats()

        assert not result.empty
        assert len(result) == 1
        assert result["alarm_count"].iloc[0] > 0
        assert result["max_duration_seconds"].iloc[0] > 0
        assert result["min_duration_seconds"].iloc[0] >= 0
        assert result["avg_duration_seconds"].iloc[0] > 0
        assert result["total_duration_seconds"].iloc[0] > 0

    def test_chattering_detection(self, alarm_data):
        alarms = AlarmManagementEvents(alarm_data, alarm_uuid="temp_alarm")
        result = alarms.chattering_detection(min_transitions=5, window="10m")

        assert not result.empty
        assert "transition_count" in result.columns
        # Should detect chattering window
        assert result["transition_count"].max() >= 5

    def test_standing_alarms(self, alarm_data):
        alarms = AlarmManagementEvents(alarm_data, alarm_uuid="temp_alarm")
        result = alarms.standing_alarms(min_duration="1h")

        assert not result.empty
        # The 2-hour alarm should be flagged
        assert result["duration_seconds"].max() >= 3600

    def test_standing_alarms_high_threshold(self, alarm_data):
        alarms = AlarmManagementEvents(alarm_data, alarm_uuid="temp_alarm")
        # No alarm is longer than 3 hours
        result = alarms.standing_alarms(min_duration="3h")
        assert result.empty

    def test_no_alarms_for_wrong_uuid(self, alarm_data):
        alarms = AlarmManagementEvents(alarm_data, alarm_uuid="nonexistent")
        assert alarms.alarm_frequency().empty
        assert alarms.alarm_duration_stats().empty


# ============================================================================
# BatchTrackingEvents fixtures and tests
# ============================================================================


@pytest.fixture
def batch_data():
    """Create synthetic batch production data.

    Signals:
    - 'batch_signal' (value_string): batch IDs that change over time
    - 'batch_counter' (value_integer): monotonic counter for yield
    """
    base = pd.Timestamp("2024-01-15 06:00:00")
    rows = []

    # Batch schedule: BATCH_A (2h) -> BATCH_B (1.5h) -> BATCH_A (2h) -> BATCH_C (1h)
    batches = [
        ("BATCH_A", 120),  # 120 minutes
        ("BATCH_B", 90),
        ("BATCH_A", 120),
        ("BATCH_C", 60),
    ]

    counter = 0
    offset = 0
    for batch_id, duration_min in batches:
        timestamps = pd.date_range(
            base + timedelta(minutes=offset),
            periods=duration_min,
            freq="1min",
        )
        for t in timestamps:
            rows.append(
                {
                    "systime": t,
                    "uuid": "batch_signal",
                    "value_bool": None,
                    "value_integer": None,
                    "value_double": None,
                    "value_string": batch_id,
                    "is_delta": False,
                }
            )
            # Counter increments every minute
            counter += 1
            rows.append(
                {
                    "systime": t,
                    "uuid": "batch_counter",
                    "value_bool": None,
                    "value_integer": counter,
                    "value_double": None,
                    "value_string": None,
                    "is_delta": True,
                }
            )
        offset += duration_min

    return pd.DataFrame(rows)


class TestBatchTrackingEvents:

    def test_empty_dataframe(self):
        df = _empty_df()
        tracker = BatchTrackingEvents(df, batch_uuid="batch_signal")
        assert tracker.detect_batches().empty
        assert tracker.batch_duration_stats().empty
        assert tracker.batch_yield("counter").empty
        assert tracker.batch_transition_matrix().empty

    def test_detect_batches(self, batch_data):
        tracker = BatchTrackingEvents(batch_data, batch_uuid="batch_signal")
        result = tracker.detect_batches()

        assert not result.empty
        assert len(result) == 4  # 4 batch runs
        assert set(result.columns) >= {
            "batch_id",
            "start",
            "end",
            "duration_seconds",
            "sample_count",
            "uuid",
            "source_uuid",
        }
        # Check batch IDs in order
        assert list(result["batch_id"]) == ["BATCH_A", "BATCH_B", "BATCH_A", "BATCH_C"]

    def test_batch_duration_stats(self, batch_data):
        tracker = BatchTrackingEvents(batch_data, batch_uuid="batch_signal")
        result = tracker.batch_duration_stats()

        assert not result.empty
        assert "batch_id" in result.columns
        assert "count" in result.columns
        # BATCH_A appears twice
        batch_a = result[result["batch_id"] == "BATCH_A"]
        assert batch_a["count"].iloc[0] == 2

    def test_batch_yield(self, batch_data):
        tracker = BatchTrackingEvents(batch_data, batch_uuid="batch_signal")
        result = tracker.batch_yield("batch_counter")

        assert not result.empty
        assert "quantity" in result.columns
        assert len(result) == 4
        # Each batch should have produced parts
        assert all(result["quantity"] > 0)
        # First batch (120 min) should produce ~119 parts (diff of 120 samples)
        assert result.iloc[0]["quantity"] > 50

    def test_batch_yield_missing_counter(self, batch_data):
        tracker = BatchTrackingEvents(batch_data, batch_uuid="batch_signal")
        result = tracker.batch_yield("nonexistent_counter")

        assert not result.empty
        # All quantities should be 0
        assert all(result["quantity"] == 0)

    def test_batch_transition_matrix(self, batch_data):
        tracker = BatchTrackingEvents(batch_data, batch_uuid="batch_signal")
        result = tracker.batch_transition_matrix()

        assert not result.empty
        # A->B, B->A, A->C transitions
        assert "total" in result.columns
        # BATCH_A transitions: A->B and A->C
        assert result.loc["BATCH_A", "total"] == 2
        # BATCH_B transitions: B->A
        assert result.loc["BATCH_B", "total"] == 1

    def test_single_batch_no_transitions(self):
        """A single batch should yield an empty transition matrix."""
        base = pd.Timestamp("2024-01-15 06:00:00")
        timestamps = pd.date_range(base, periods=60, freq="1min")
        rows = []
        for t in timestamps:
            rows.append(
                {
                    "systime": t,
                    "uuid": "batch_signal",
                    "value_bool": None,
                    "value_integer": None,
                    "value_double": None,
                    "value_string": "ONLY_BATCH",
                    "is_delta": False,
                }
            )
        df = pd.DataFrame(rows)
        tracker = BatchTrackingEvents(df, batch_uuid="batch_signal")
        result = tracker.batch_transition_matrix()
        assert result.empty
