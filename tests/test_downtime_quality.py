import pandas as pd  # type: ignore
import pytest
from datetime import datetime, timedelta

from ts_shape.events.production import DowntimeTracking, QualityTracking

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_downtime_data():
    """Create sample downtime data with state changes."""
    timestamps = []
    states = []
    reasons = []

    base_time = pd.Timestamp("2024-01-01 06:00:00")

    # Shift 1 (06:00-14:00): 480 minutes
    # Running: 06:00-07:30 (90 min)
    timestamps.extend([base_time, base_time + pd.Timedelta(minutes=90)])
    states.extend(["Running", "Stopped"])
    reasons.extend(["", "Material_Shortage"])

    # Stopped: 07:30-08:00 (30 min)
    timestamps.append(base_time + pd.Timedelta(minutes=120))
    states.append("Running")
    reasons.append("")

    # Running: 08:00-10:00 (120 min)
    timestamps.append(base_time + pd.Timedelta(minutes=240))
    states.append("Stopped")
    reasons.append("Tool_Change")

    # Stopped: 10:00-10:30 (30 min)
    timestamps.append(base_time + pd.Timedelta(minutes=270))
    states.append("Running")
    reasons.append("")

    # Running: 10:30-14:00 (210 min) = 420 min running, 60 min stopped in shift_1

    # Shift 2 (14:00-22:00): 480 minutes
    timestamps.append(base_time + pd.Timedelta(minutes=480))
    states.append("Running")
    reasons.append("")

    # Running: 14:00-16:00 (120 min)
    timestamps.append(base_time + pd.Timedelta(minutes=600))
    states.append("Stopped")
    reasons.append("Quality_Issue")

    # Stopped: 16:00-16:45 (45 min)
    timestamps.append(base_time + pd.Timedelta(minutes=645))
    states.append("Running")
    reasons.append("")

    # Running: 16:45-22:00 (315 min) = 435 min running, 45 min stopped in shift_2

    # Shift 3 (22:00-06:00): 480 minutes
    timestamps.append(base_time + pd.Timedelta(minutes=960))
    states.append("Stopped")
    reasons.append("Maintenance")

    # Stopped: 22:00-23:30 (90 min)
    timestamps.append(base_time + pd.Timedelta(minutes=1050))
    states.append("Running")
    reasons.append("")

    # Running: 23:30-06:00 (390 min) = 390 min running, 90 min stopped in shift_3

    # Create DataFrames
    df_state = pd.DataFrame(
        {
            "uuid": ["machine_state"] * len(timestamps),
            "systime": timestamps,
            "value_string": states,
            "is_delta": [True] * len(timestamps),
        }
    )

    df_reason = pd.DataFrame(
        {
            "uuid": ["downtime_reason"] * len(timestamps),
            "systime": timestamps,
            "value_string": reasons,
            "is_delta": [True] * len(timestamps),
        }
    )

    return pd.concat([df_state, df_reason], ignore_index=True)


@pytest.fixture
def sample_quality_data():
    """Create sample quality data with OK and NOK counters."""
    # 8 hours of data
    t = pd.date_range("2024-01-01 06:00:00", periods=96, freq="5min")

    # OK parts counter - increases steadily
    ok_values = list(range(0, 960, 10))  # 10 parts every 5 minutes
    df_ok = pd.DataFrame(
        {
            "uuid": ["ok_counter"] * len(t),
            "systime": t,
            "value_integer": ok_values,
            "is_delta": [True] * len(t),
        }
    )

    # NOK parts counter - increases occasionally
    nok_values = []
    nok_count = 0
    for i in range(len(t)):
        # Add 1 NOK part every 30 minutes (every 6th reading)
        if i % 6 == 0 and i > 0:
            nok_count += 1
        nok_values.append(nok_count)

    df_nok = pd.DataFrame(
        {
            "uuid": ["nok_counter"] * len(t),
            "systime": t,
            "value_integer": nok_values,
            "is_delta": [True] * len(t),
        }
    )

    # Part numbers - change every 2 hours
    part_numbers = ["PART_A"] * 24 + ["PART_B"] * 24 + ["PART_A"] * 24 + ["PART_B"] * 24
    df_parts = pd.DataFrame(
        {
            "uuid": ["part_number"] * len(t),
            "systime": t,
            "value_string": part_numbers,
            "is_delta": [False] * len(t),
        }
    )

    # Defect reasons
    defect_reasons = []
    for i in range(len(t)):
        if i % 6 == 0 and i > 0:
            reason = "Dimension_Error" if i % 12 == 0 else "Surface_Defect"
            defect_reasons.append(reason)
        else:
            defect_reasons.append("")

    df_reasons = pd.DataFrame(
        {
            "uuid": ["defect_reason"] * len(t),
            "systime": t,
            "value_string": defect_reasons,
            "is_delta": [True] * len(t),
        }
    )

    return pd.concat([df_ok, df_nok, df_parts, df_reasons], ignore_index=True)


# ============================================================================
# DowntimeTracking Tests
# ============================================================================


def test_downtime_by_shift_basic(sample_downtime_data):
    """Test basic downtime calculation by shift."""
    tracker = DowntimeTracking(sample_downtime_data)
    result = tracker.downtime_by_shift(
        state_uuid="machine_state", running_value="Running"
    )

    assert not result.empty
    assert "date" in result.columns
    assert "shift" in result.columns
    assert "total_minutes" in result.columns
    assert "downtime_minutes" in result.columns
    assert "uptime_minutes" in result.columns
    assert "availability_pct" in result.columns

    # Check that all shifts are present
    assert set(result["shift"].unique()) == {"shift_1", "shift_2", "shift_3"}

    # Verify availability is calculated correctly
    for _, row in result.iterrows():
        expected_availability = (row["uptime_minutes"] / row["total_minutes"]) * 100
        assert (
            abs(row["availability_pct"] - expected_availability) < 0.2
        )  # Allow small rounding difference


def test_downtime_by_shift_values(sample_downtime_data):
    """Test that downtime values are reasonable."""
    tracker = DowntimeTracking(sample_downtime_data)
    result = tracker.downtime_by_shift(
        state_uuid="machine_state", running_value="Running"
    )

    # All values should be positive
    assert all(result["total_minutes"] > 0)
    assert all(result["downtime_minutes"] >= 0)
    assert all(result["uptime_minutes"] >= 0)

    # Availability should be between 0 and 100
    assert all(result["availability_pct"] >= 0)
    assert all(result["availability_pct"] <= 100)

    # Total should equal uptime + downtime
    for _, row in result.iterrows():
        total_calc = row["uptime_minutes"] + row["downtime_minutes"]
        assert abs(row["total_minutes"] - total_calc) < 1.0  # Allow 1 minute rounding


def test_downtime_by_reason(sample_downtime_data):
    """Test downtime analysis by reason code."""
    tracker = DowntimeTracking(sample_downtime_data)
    result = tracker.downtime_by_reason(
        state_uuid="machine_state",
        reason_uuid="downtime_reason",
        stopped_value="Stopped",
    )

    assert not result.empty
    assert "reason" in result.columns
    assert "occurrences" in result.columns
    assert "total_minutes" in result.columns
    assert "avg_minutes" in result.columns
    assert "pct_of_total" in result.columns

    # Check that we have the expected reasons
    expected_reasons = {
        "Material_Shortage",
        "Tool_Change",
        "Quality_Issue",
        "Maintenance",
    }
    assert set(result["reason"].unique()).issubset(expected_reasons)

    # Verify percentage sums to approximately 100
    total_pct = result["pct_of_total"].sum()
    assert abs(total_pct - 100.0) < 1.0  # Allow small rounding error

    # Verify avg_minutes calculation
    for _, row in result.iterrows():
        expected_avg = row["total_minutes"] / row["occurrences"]
        assert abs(row["avg_minutes"] - expected_avg) < 0.2


def test_top_downtime_reasons(sample_downtime_data):
    """Test Pareto analysis of top downtime reasons."""
    tracker = DowntimeTracking(sample_downtime_data)
    result = tracker.top_downtime_reasons(
        state_uuid="machine_state",
        reason_uuid="downtime_reason",
        top_n=3,
        stopped_value="Stopped",
    )

    assert not result.empty
    assert "reason" in result.columns
    assert "total_minutes" in result.columns
    assert "pct_of_total" in result.columns
    assert "cumulative_pct" in result.columns

    # Should return at most 3 reasons
    assert len(result) <= 3

    # Cumulative percentage should be monotonically increasing
    cumulative = result["cumulative_pct"].values
    for i in range(len(cumulative) - 1):
        assert cumulative[i] <= cumulative[i + 1]

    # Results should be sorted by total_minutes descending
    total_mins = result["total_minutes"].values
    for i in range(len(total_mins) - 1):
        assert total_mins[i] >= total_mins[i + 1]


def test_availability_trend(sample_downtime_data):
    """Test availability trend calculation."""
    tracker = DowntimeTracking(sample_downtime_data)
    result = tracker.availability_trend(
        state_uuid="machine_state", running_value="Running", window="1D"
    )

    assert not result.empty
    assert "period" in result.columns
    assert "availability_pct" in result.columns
    assert "uptime_minutes" in result.columns
    assert "downtime_minutes" in result.columns

    # Availability should be between 0 and 100
    assert all(result["availability_pct"] >= 0)
    assert all(result["availability_pct"] <= 100)


def test_downtime_empty_data():
    """Test downtime tracking with empty data."""
    df_empty = pd.DataFrame(columns=["uuid", "systime", "value_string", "is_delta"])
    tracker = DowntimeTracking(df_empty)

    result = tracker.downtime_by_shift(state_uuid="machine_state")
    assert result.empty


def test_downtime_custom_shifts():
    """Test downtime tracking with custom shift definitions."""
    t = pd.date_range("2024-01-01 06:00:00", periods=48, freq="30min")

    df = pd.DataFrame(
        {
            "uuid": ["machine_state"] * len(t),
            "systime": t,
            "value_string": ["Running", "Stopped"] * 24,
            "is_delta": [True] * len(t),
        }
    )

    custom_shifts = {
        "day": ("06:00", "14:00"),
        "night": ("14:00", "06:00"),
    }

    tracker = DowntimeTracking(df, shift_definitions=custom_shifts)
    result = tracker.downtime_by_shift(
        state_uuid="machine_state", running_value="Running"
    )

    assert not result.empty
    assert set(result["shift"].unique()) == {"day", "night"}


# ============================================================================
# QualityTracking Tests
# ============================================================================


def test_nok_by_shift_basic(sample_quality_data):
    """Test basic NOK calculation by shift."""
    tracker = QualityTracking(sample_quality_data)
    result = tracker.nok_by_shift(
        ok_counter_uuid="ok_counter", nok_counter_uuid="nok_counter"
    )

    assert not result.empty
    assert "date" in result.columns
    assert "shift" in result.columns
    assert "ok_parts" in result.columns
    assert "nok_parts" in result.columns
    assert "total_parts" in result.columns
    assert "nok_rate_pct" in result.columns
    assert "first_pass_yield_pct" in result.columns
    assert "quality_pct" in result.columns

    # Check that percentages sum to approximately 100
    for _, row in result.iterrows():
        total_pct = row["nok_rate_pct"] + row["first_pass_yield_pct"]
        assert abs(total_pct - 100.0) < 0.5
        # quality_pct should equal first_pass_yield_pct
        assert row["quality_pct"] == row["first_pass_yield_pct"]


def test_nok_by_shift_calculations(sample_quality_data):
    """Test that NOK calculations are correct."""
    tracker = QualityTracking(sample_quality_data)
    result = tracker.nok_by_shift(
        ok_counter_uuid="ok_counter", nok_counter_uuid="nok_counter"
    )

    for _, row in result.iterrows():
        # Verify total parts
        assert row["total_parts"] == row["ok_parts"] + row["nok_parts"]

        # Verify NOK rate
        if row["total_parts"] > 0:
            expected_nok_rate = (row["nok_parts"] / row["total_parts"]) * 100
            assert abs(row["nok_rate_pct"] - expected_nok_rate) < 0.2

            # Verify FPY
            expected_fpy = (row["ok_parts"] / row["total_parts"]) * 100
            assert abs(row["first_pass_yield_pct"] - expected_fpy) < 0.2


def test_quality_by_part(sample_quality_data):
    """Test quality metrics by part number."""
    tracker = QualityTracking(sample_quality_data)
    result = tracker.quality_by_part(
        ok_counter_uuid="ok_counter",
        nok_counter_uuid="nok_counter",
        part_id_uuid="part_number",
    )

    assert not result.empty
    assert "part_number" in result.columns
    assert "ok_parts" in result.columns
    assert "nok_parts" in result.columns
    assert "total_parts" in result.columns
    assert "nok_rate_pct" in result.columns
    assert "first_pass_yield_pct" in result.columns

    # Should have data for PART_A and PART_B
    assert set(result["part_number"].unique()) == {"PART_A", "PART_B"}

    # All parts should have some production
    assert all(result["total_parts"] > 0)


def test_nok_by_reason(sample_quality_data):
    """Test NOK analysis by defect reason."""
    tracker = QualityTracking(sample_quality_data)
    result = tracker.nok_by_reason(
        nok_counter_uuid="nok_counter", defect_reason_uuid="defect_reason"
    )

    assert not result.empty
    assert "reason" in result.columns
    assert "nok_parts" in result.columns
    assert "pct_of_total" in result.columns

    # Should have expected defect reasons
    expected_reasons = {"Dimension_Error", "Surface_Defect"}
    assert set(result["reason"].unique()).issubset(expected_reasons)

    # Percentages should sum to approximately 100
    total_pct = result["pct_of_total"].sum()
    assert abs(total_pct - 100.0) < 1.0


def test_daily_quality_summary(sample_quality_data):
    """Test daily quality summary."""
    tracker = QualityTracking(sample_quality_data)
    result = tracker.daily_quality_summary(
        ok_counter_uuid="ok_counter", nok_counter_uuid="nok_counter"
    )

    assert not result.empty
    assert "date" in result.columns
    assert "ok_parts" in result.columns
    assert "nok_parts" in result.columns
    assert "total_parts" in result.columns
    assert "nok_rate_pct" in result.columns
    assert "first_pass_yield_pct" in result.columns

    # Verify calculations
    for _, row in result.iterrows():
        assert row["total_parts"] == row["ok_parts"] + row["nok_parts"]

        if row["total_parts"] > 0:
            expected_nok_rate = (row["nok_parts"] / row["total_parts"]) * 100
            assert abs(row["nok_rate_pct"] - expected_nok_rate) < 0.2


def test_quality_empty_data():
    """Test quality tracking with empty data."""
    df_empty = pd.DataFrame(columns=["uuid", "systime", "value_integer", "is_delta"])
    tracker = QualityTracking(df_empty)

    result = tracker.nok_by_shift(
        ok_counter_uuid="ok_counter", nok_counter_uuid="nok_counter"
    )
    assert result.empty


def test_quality_custom_shifts():
    """Test quality tracking with custom shift definitions."""
    t = pd.date_range("2024-01-01 06:00:00", periods=96, freq="5min")

    df_ok = pd.DataFrame(
        {
            "uuid": ["ok_counter"] * len(t),
            "systime": t,
            "value_integer": list(range(0, 960, 10)),
            "is_delta": [True] * len(t),
        }
    )

    df_nok = pd.DataFrame(
        {
            "uuid": ["nok_counter"] * len(t),
            "systime": t,
            "value_integer": [i // 6 for i in range(len(t))],
            "is_delta": [True] * len(t),
        }
    )

    df = pd.concat([df_ok, df_nok], ignore_index=True)

    custom_shifts = {
        "morning": ("06:00", "12:00"),
        "afternoon": ("12:00", "18:00"),
        "night": ("18:00", "06:00"),
    }

    tracker = QualityTracking(df, shift_definitions=custom_shifts)
    result = tracker.nok_by_shift(
        ok_counter_uuid="ok_counter", nok_counter_uuid="nok_counter"
    )

    assert not result.empty
    # Should have at least morning and afternoon shifts in 8 hours of data
    assert "morning" in result["shift"].values


def test_quality_only_ok_counter():
    """Test quality tracking with only OK counter (no NOK)."""
    t = pd.date_range("2024-01-01 06:00:00", periods=48, freq="10min")

    df = pd.DataFrame(
        {
            "uuid": ["ok_counter"] * len(t),
            "systime": t,
            "value_integer": list(range(0, 480, 10)),
            "is_delta": [True] * len(t),
        }
    )

    tracker = QualityTracking(df)
    result = tracker.nok_by_shift(
        ok_counter_uuid="ok_counter", nok_counter_uuid="nok_counter"  # Doesn't exist
    )

    # Should still return results with zero NOK parts
    assert not result.empty
    assert all(result["nok_parts"] == 0)
    assert all(result["first_pass_yield_pct"] == 100.0)


def test_quality_only_nok_counter():
    """Test quality tracking with only NOK counter (no OK)."""
    t = pd.date_range("2024-01-01 06:00:00", periods=48, freq="10min")

    df = pd.DataFrame(
        {
            "uuid": ["nok_counter"] * len(t),
            "systime": t,
            "value_integer": list(range(0, 48)),
            "is_delta": [True] * len(t),
        }
    )

    tracker = QualityTracking(df)
    result = tracker.nok_by_shift(
        ok_counter_uuid="ok_counter", nok_counter_uuid="nok_counter"  # Doesn't exist
    )

    # Should still return results with zero OK parts
    assert not result.empty
    assert all(result["ok_parts"] == 0)
    assert all(result["nok_rate_pct"] == 100.0)


# ============================================================================
# Integration Tests
# ============================================================================


def test_downtime_quality_integration(sample_downtime_data, sample_quality_data):
    """Test that downtime and quality modules can work together."""
    # Combine the data
    combined_df = pd.concat(
        [sample_downtime_data, sample_quality_data], ignore_index=True
    )

    # Test downtime tracking
    downtime_tracker = DowntimeTracking(combined_df)
    downtime_result = downtime_tracker.downtime_by_shift(
        state_uuid="machine_state", running_value="Running"
    )
    assert not downtime_result.empty

    # Test quality tracking
    quality_tracker = QualityTracking(combined_df)
    quality_result = quality_tracker.nok_by_shift(
        ok_counter_uuid="ok_counter", nok_counter_uuid="nok_counter"
    )
    assert not quality_result.empty

    # Both should have shift data
    assert "shift" in downtime_result.columns
    assert "shift" in quality_result.columns


def test_module_imports():
    """Test that all modules can be imported correctly."""
    from ts_shape.events.production import DowntimeTracking, QualityTracking

    assert DowntimeTracking is not None
    assert QualityTracking is not None
