import pandas as pd  # type: ignore
import pytest
from datetime import datetime, timedelta

from ts_shape.events.production import (
    PartProductionTracking,
    CycleTimeTracking,
    ShiftReporting,
)

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_production_data():
    """Create sample production data with part numbers and counters."""
    t = pd.date_range("2024-01-01 08:00:00", periods=120, freq="1min")

    # Part number signal - switches between PART_A and PART_B
    part_numbers = ["PART_A"] * 60 + ["PART_B"] * 60
    df_parts = pd.DataFrame(
        {
            "uuid": ["part_number"] * len(t),
            "systime": t,
            "value_string": part_numbers,
            "is_delta": [False] * len(t),
        }
    )

    # Production counter - increments by 1-2 per minute
    counter_values = list(range(0, len(t)))
    df_counter = pd.DataFrame(
        {
            "uuid": ["production_counter"] * len(t),
            "systime": t,
            "value_integer": counter_values,
            "is_delta": [True] * len(t),
        }
    )

    return pd.concat([df_parts, df_counter], ignore_index=True)


@pytest.fixture
def sample_cycle_data():
    """Create sample cycle time data with part numbers and cycle triggers."""
    # Create data for cycles with proper rising edge triggers
    timestamps_parts = []
    timestamps_triggers = []
    part_numbers = []
    cycle_triggers = []

    base_time = pd.Timestamp("2024-01-01 08:00:00")
    current_time = base_time

    # 10 cycles for PART_A (45-50 seconds each)
    for i in range(10):
        cycle_duration = 45 + (i % 5)  # 45-49 seconds

        # Part number stays constant
        timestamps_parts.append(current_time)
        part_numbers.append("PART_A")

        # Cycle trigger: False at start, True at end
        timestamps_triggers.append(current_time)
        cycle_triggers.append(False)
        timestamps_triggers.append(current_time + pd.Timedelta(seconds=cycle_duration))
        cycle_triggers.append(True)

        current_time += pd.Timedelta(seconds=cycle_duration)

    # 10 cycles for PART_B (60-65 seconds each)
    for i in range(10):
        cycle_duration = 60 + (i % 5)  # 60-64 seconds

        # Part number stays constant
        timestamps_parts.append(current_time)
        part_numbers.append("PART_B")

        # Cycle trigger: False at start, True at end
        timestamps_triggers.append(current_time)
        cycle_triggers.append(False)
        timestamps_triggers.append(current_time + pd.Timedelta(seconds=cycle_duration))
        cycle_triggers.append(True)

        current_time += pd.Timedelta(seconds=cycle_duration)

    df_parts = pd.DataFrame(
        {
            "uuid": ["part_number"] * len(timestamps_parts),
            "systime": timestamps_parts,
            "value_string": part_numbers,
            "is_delta": [False] * len(timestamps_parts),
        }
    )

    df_cycles = pd.DataFrame(
        {
            "uuid": ["cycle_trigger"] * len(timestamps_triggers),
            "systime": timestamps_triggers,
            "value_bool": cycle_triggers,
            "is_delta": [True] * len(timestamps_triggers),
        }
    )

    return pd.concat([df_parts, df_cycles], ignore_index=True)


@pytest.fixture
def sample_shift_data():
    """Create sample shift data covering 3 shifts over 24 hours."""
    # Create data covering all three shifts
    t = pd.date_range("2024-01-01 06:00:00", periods=288, freq="5min")  # 24 hours

    # Production counter that varies by shift
    counter_values = []
    base_counter = 0
    for i in range(len(t)):
        hour = t[i].hour
        # Day shift (6-14): 10 parts per 5min
        # Afternoon shift (14-22): 9 parts per 5min
        # Night shift (22-6): 7 parts per 5min
        if 6 <= hour < 14:
            increment = 10
        elif 14 <= hour < 22:
            increment = 9
        else:
            increment = 7
        counter_values.append(base_counter + increment)
        base_counter += increment

    df_counter = pd.DataFrame(
        {
            "uuid": ["production_counter"] * len(t),
            "systime": t,
            "value_integer": counter_values,
            "is_delta": [True] * len(t),
        }
    )

    return df_counter


# ============================================================================
# PartProductionTracking Tests
# ============================================================================


def test_production_by_part_basic(sample_production_data):
    """Test basic production tracking by part number."""
    tracker = PartProductionTracking(sample_production_data)
    result = tracker.production_by_part(
        part_id_uuid="part_number", counter_uuid="production_counter", window="1h"
    )

    assert not result.empty
    assert "window_start" in result.columns
    assert "part_number" in result.columns
    assert "quantity" in result.columns
    assert set(result["part_number"].unique()) == {"PART_A", "PART_B"}


def test_production_by_part_custom_window(sample_production_data):
    """Test production tracking with different window sizes."""
    tracker = PartProductionTracking(sample_production_data)

    # 30-minute window
    result_30m = tracker.production_by_part(
        part_id_uuid="part_number", counter_uuid="production_counter", window="30min"
    )

    # 15-minute window
    result_15m = tracker.production_by_part(
        part_id_uuid="part_number", counter_uuid="production_counter", window="15min"
    )

    # More frequent windows should have more rows
    assert len(result_15m) >= len(result_30m)
    assert not result_30m.empty
    assert not result_15m.empty


def test_daily_production_summary(sample_production_data):
    """Test daily production summary."""
    tracker = PartProductionTracking(sample_production_data)
    result = tracker.daily_production_summary(
        part_id_uuid="part_number", counter_uuid="production_counter"
    )

    assert not result.empty
    assert "date" in result.columns
    assert "part_number" in result.columns
    assert "total_quantity" in result.columns
    assert "hours_active" in result.columns
    assert set(result["part_number"].unique()) == {"PART_A", "PART_B"}
    assert all(result["total_quantity"] > 0)


def test_production_totals(sample_production_data):
    """Test production totals over date range."""
    tracker = PartProductionTracking(sample_production_data)
    result = tracker.production_totals(
        part_id_uuid="part_number",
        counter_uuid="production_counter",
        start_date="2024-01-01",
        end_date="2024-01-01",
    )

    assert not result.empty
    assert "part_number" in result.columns
    assert "total_quantity" in result.columns
    assert "days_produced" in result.columns
    assert set(result["part_number"].unique()) == {"PART_A", "PART_B"}


def test_production_tracking_empty_data():
    """Test production tracking with empty data."""
    df_empty = pd.DataFrame(
        columns=["uuid", "systime", "value_string", "value_integer", "is_delta"]
    )
    tracker = PartProductionTracking(df_empty)

    result = tracker.production_by_part(
        part_id_uuid="part_number", counter_uuid="production_counter"
    )

    assert result.empty


# ============================================================================
# CycleTimeTracking Tests
# ============================================================================


def test_cycle_time_by_part_basic(sample_cycle_data):
    """Test basic cycle time tracking by part."""
    tracker = CycleTimeTracking(sample_cycle_data)
    result = tracker.cycle_time_by_part(
        part_id_uuid="part_number", cycle_trigger_uuid="cycle_trigger"
    )

    assert not result.empty
    assert "systime" in result.columns
    assert "part_number" in result.columns
    assert "cycle_time_seconds" in result.columns
    assert set(result["part_number"].unique()) == {"PART_A", "PART_B"}
    # Check cycle times are reasonable (allowing for transitions between parts)
    assert all(result["cycle_time_seconds"] > 0)
    assert all(result["cycle_time_seconds"] < 150)  # Allow for part transitions


def test_cycle_time_statistics(sample_cycle_data):
    """Test cycle time statistics by part."""
    tracker = CycleTimeTracking(sample_cycle_data)
    result = tracker.cycle_time_statistics(
        part_id_uuid="part_number", cycle_trigger_uuid="cycle_trigger"
    )

    assert not result.empty
    assert "part_number" in result.columns
    assert "count" in result.columns
    assert "min_seconds" in result.columns
    assert "avg_seconds" in result.columns
    assert "max_seconds" in result.columns
    assert "std_seconds" in result.columns
    assert "median_seconds" in result.columns

    # Check that statistics are ordered correctly
    for _, row in result.iterrows():
        assert row["min_seconds"] <= row["avg_seconds"] <= row["max_seconds"]
        assert row["count"] > 0


def test_detect_slow_cycles(sample_cycle_data):
    """Test detection of slow cycles."""
    tracker = CycleTimeTracking(sample_cycle_data)
    result = tracker.detect_slow_cycles(
        part_id_uuid="part_number",
        cycle_trigger_uuid="cycle_trigger",
        threshold_factor=1.5,
    )

    # Result should have columns even if no slow cycles detected
    assert "systime" in result.columns
    assert "part_number" in result.columns
    assert "cycle_time_seconds" in result.columns
    assert "median_seconds" in result.columns
    assert "deviation_factor" in result.columns
    assert "is_slow" in result.columns

    # If slow cycles detected, check they exceed threshold
    if not result.empty:
        slow_cycles = result[result["is_slow"]]
        if not slow_cycles.empty:
            assert all(slow_cycles["deviation_factor"] >= 1.5)


def test_cycle_time_trend(sample_cycle_data):
    """Test cycle time trend analysis."""
    tracker = CycleTimeTracking(sample_cycle_data)
    result = tracker.cycle_time_trend(
        part_id_uuid="part_number",
        cycle_trigger_uuid="cycle_trigger",
        part_number="PART_A",
        window_size=5,
    )

    assert not result.empty
    assert "systime" in result.columns
    assert "cycle_time_seconds" in result.columns
    assert "moving_avg" in result.columns
    assert "trend" in result.columns
    # Note: part_number is not in output since we filtered for 'PART_A' in the method call
    # Trend may include NaN for first few rows due to insufficient data for slope calculation
    non_nan_trends = result["trend"].dropna().unique()
    assert set(non_nan_trends).issubset({"improving", "stable", "degrading"})


def test_hourly_cycle_time_summary(sample_cycle_data):
    """Test hourly cycle time summary."""
    tracker = CycleTimeTracking(sample_cycle_data)
    result = tracker.hourly_cycle_time_summary(
        part_id_uuid="part_number", cycle_trigger_uuid="cycle_trigger"
    )

    assert not result.empty
    assert "hour" in result.columns
    assert "part_number" in result.columns
    assert "cycles_completed" in result.columns
    assert "avg_cycle_time" in result.columns
    assert "min_cycle_time" in result.columns
    assert "max_cycle_time" in result.columns

    # Check statistics ordering
    for _, row in result.iterrows():
        assert row["min_cycle_time"] <= row["avg_cycle_time"] <= row["max_cycle_time"]
        assert row["cycles_completed"] > 0


def test_cycle_tracking_with_integer_trigger():
    """Test cycle tracking with integer trigger instead of boolean."""
    t = pd.date_range("2024-01-01 08:00:00", periods=10, freq="1min")

    df_parts = pd.DataFrame(
        {
            "uuid": ["part_number"] * len(t),
            "systime": t,
            "value_string": ["PART_X"] * len(t),
            "is_delta": [False] * len(t),
        }
    )

    # Integer trigger that increments
    df_trigger = pd.DataFrame(
        {
            "uuid": ["cycle_count"] * len(t),
            "systime": t,
            "value_integer": list(range(len(t))),
            "is_delta": [True] * len(t),
        }
    )

    df = pd.concat([df_parts, df_trigger], ignore_index=True)
    tracker = CycleTimeTracking(df)

    result = tracker.cycle_time_by_part(
        part_id_uuid="part_number",
        cycle_trigger_uuid="cycle_count",
        value_column_trigger="value_integer",
    )

    assert not result.empty
    assert all(result["part_number"] == "PART_X")


# ============================================================================
# ShiftReporting Tests
# ============================================================================


def test_shift_production_default_shifts(sample_shift_data):
    """Test shift production with default shift definitions."""
    reporter = ShiftReporting(sample_shift_data)
    result = reporter.shift_production(counter_uuid="production_counter")

    assert not result.empty
    assert "date" in result.columns
    assert "shift" in result.columns
    assert "quantity" in result.columns
    assert set(result["shift"].unique()) == {"shift_1", "shift_2", "shift_3"}
    assert all(result["quantity"] > 0)


def test_shift_production_custom_shifts(sample_shift_data):
    """Test shift production with custom shift definitions."""
    custom_shifts = {
        "day": ("06:00", "14:00"),
        "afternoon": ("14:00", "22:00"),
        "night": ("22:00", "06:00"),
    }
    reporter = ShiftReporting(sample_shift_data, shift_definitions=custom_shifts)
    result = reporter.shift_production(counter_uuid="production_counter")

    assert not result.empty
    assert set(result["shift"].unique()) == {"day", "afternoon", "night"}


def test_shift_production_with_parts():
    """Test shift production with part numbers."""
    t = pd.date_range("2024-01-01 06:00:00", periods=96, freq="15min")  # 24 hours

    # Part numbers change mid-day
    part_numbers = ["PART_A"] * 48 + ["PART_B"] * 48
    df_parts = pd.DataFrame(
        {
            "uuid": ["part_number"] * len(t),
            "systime": t,
            "value_string": part_numbers,
            "is_delta": [False] * len(t),
        }
    )

    # Counter
    df_counter = pd.DataFrame(
        {
            "uuid": ["production_counter"] * len(t),
            "systime": t,
            "value_integer": list(range(len(t))),
            "is_delta": [True] * len(t),
        }
    )

    df = pd.concat([df_parts, df_counter], ignore_index=True)
    reporter = ShiftReporting(df)

    result = reporter.shift_production(
        counter_uuid="production_counter", part_id_uuid="part_number"
    )

    assert not result.empty
    assert "part_number" in result.columns
    assert set(result["part_number"].unique()) == {"PART_A", "PART_B"}


def test_shift_comparison(sample_shift_data):
    """Test shift comparison over multiple days."""
    # Extend data to cover multiple days
    multi_day_data = []
    for day_offset in range(3):
        day_data = sample_shift_data.copy()
        day_data["systime"] = day_data["systime"] + pd.Timedelta(days=day_offset)
        multi_day_data.append(day_data)

    df = pd.concat(multi_day_data, ignore_index=True)
    reporter = ShiftReporting(df)

    result = reporter.shift_comparison(counter_uuid="production_counter", days=3)

    assert not result.empty
    assert "shift" in result.columns
    assert "avg_quantity" in result.columns
    assert "min_quantity" in result.columns
    assert "max_quantity" in result.columns
    assert "std_quantity" in result.columns
    assert "days_count" in result.columns
    assert len(result) == 3  # Three shifts


def test_shift_targets(sample_shift_data):
    """Test shift performance against targets."""
    reporter = ShiftReporting(sample_shift_data)
    targets = {
        "shift_1": 500,
        "shift_2": 450,
        "shift_3": 350,
    }

    result = reporter.shift_targets(counter_uuid="production_counter", targets=targets)

    assert not result.empty
    assert "date" in result.columns
    assert "shift" in result.columns
    assert "actual" in result.columns
    assert "target" in result.columns
    assert "variance" in result.columns
    assert "achievement_pct" in result.columns

    # Check variance calculation
    for _, row in result.iterrows():
        assert row["variance"] == row["actual"] - row["target"]
        assert (
            abs(row["achievement_pct"] - (row["actual"] / row["target"] * 100)) < 0.01
        )


def test_best_and_worst_shifts():
    """Test identification of best and worst performing shifts."""
    # Create data with varying shift performance
    t = pd.date_range("2024-01-01 06:00:00", periods=864, freq="5min")  # 3 days

    counter_values = []
    base_counter = 0
    for i in range(len(t)):
        day_offset = i // 288
        hour = t[i].hour

        # Vary production by shift and day
        if 6 <= hour < 14:
            increment = 10 + day_offset * 2  # Day shift improves
        elif 14 <= hour < 22:
            increment = 9  # Afternoon shift stable
        else:
            increment = 7 - day_offset  # Night shift degrades

        counter_values.append(base_counter + increment)
        base_counter += increment

    df = pd.DataFrame(
        {
            "uuid": ["production_counter"] * len(t),
            "systime": t,
            "value_integer": counter_values,
            "is_delta": [True] * len(t),
        }
    )

    reporter = ShiftReporting(df)
    result = reporter.best_and_worst_shifts(counter_uuid="production_counter", days=3)

    assert "best" in result
    assert "worst" in result
    assert isinstance(result["best"], pd.DataFrame)
    assert isinstance(result["worst"], pd.DataFrame)

    if not result["best"].empty:
        assert "date" in result["best"].columns
        assert "shift" in result["best"].columns
        assert "quantity" in result["best"].columns


def test_shift_reporting_empty_data():
    """Test shift reporting with empty data."""
    df_empty = pd.DataFrame(columns=["uuid", "systime", "value_integer", "is_delta"])
    reporter = ShiftReporting(df_empty)

    result = reporter.shift_production(counter_uuid="production_counter")
    assert result.empty


def test_overnight_shift_boundary():
    """Test that overnight shifts (crossing midnight) are handled correctly."""
    # Create data that spans midnight
    t = pd.date_range("2024-01-01 20:00:00", periods=120, freq="5min")  # 10 hours

    df = pd.DataFrame(
        {
            "uuid": ["production_counter"] * len(t),
            "systime": t,
            "value_integer": list(range(len(t))),
            "is_delta": [True] * len(t),
        }
    )

    reporter = ShiftReporting(df)
    result = reporter.shift_production(counter_uuid="production_counter")

    assert not result.empty
    # Should have data for shift_2 (14-22), shift_3 (22-06), and shift_1 (06-14)
    assert len(result["shift"].unique()) >= 2


# ============================================================================
# Integration Tests
# ============================================================================


def test_complete_production_workflow(sample_production_data, sample_cycle_data):
    """Test a complete production analysis workflow using all three modules."""
    # 1. Production tracking
    prod_tracker = PartProductionTracking(sample_production_data)
    daily_prod = prod_tracker.daily_production_summary(
        part_id_uuid="part_number", counter_uuid="production_counter"
    )
    assert not daily_prod.empty

    # 2. Cycle time analysis
    cycle_tracker = CycleTimeTracking(sample_cycle_data)
    cycle_stats = cycle_tracker.cycle_time_statistics(
        part_id_uuid="part_number", cycle_trigger_uuid="cycle_trigger"
    )
    assert not cycle_stats.empty

    # 3. Verify part numbers match
    prod_parts = set(daily_prod["part_number"].unique())
    cycle_parts = set(cycle_stats["part_number"].unique())
    assert prod_parts == cycle_parts == {"PART_A", "PART_B"}


def test_module_imports():
    """Test that all modules can be imported correctly."""
    from ts_shape.events.production import (
        PartProductionTracking,
        CycleTimeTracking,
        ShiftReporting,
    )

    assert PartProductionTracking is not None
    assert CycleTimeTracking is not None
    assert ShiftReporting is not None
