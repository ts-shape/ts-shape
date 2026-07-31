"""Reset-aware behaviour of OEECalculator.calculate_performance / calculate_quality."""

import pandas as pd  # type: ignore
import pytest

from ts_shape.events.production import OEECalculator


def _counter_rows(uuid, times, values):
    return [
        {
            "systime": t,
            "uuid": uuid,
            "value_bool": None,
            "value_integer": v,
            "value_double": None,
            "value_string": None,
            "is_delta": True,
        }
        for t, v in zip(times, values)
    ]


@pytest.fixture
def reset_oee_data():
    """One day where the part/total/reject counters each reset mid-day."""
    times = pd.date_range("2024-01-15 06:00:00", periods=7, freq="1h")
    rows = []
    # part / total counter: 0,50,100,150,[reset]0,60,120
    #   naive last-first = 120 ; reset-aware = 270 ; 1 reset
    rows += _counter_rows("part_counter", times, [0, 50, 100, 150, 0, 60, 120])
    rows += _counter_rows("total_counter", times, [0, 50, 100, 150, 0, 60, 120])
    # reject counter: 0,1,2,[reset]0,1 (only 5 readings)
    #   naive last-first = 1 ; reset-aware = 3 ; 1 reset
    rows += _counter_rows(
        "reject_counter", times[:5], [0, 1, 2, 0, 1]
    )
    return pd.DataFrame(rows)


def test_performance_default_under_reports(reset_oee_data):
    oee = OEECalculator(reset_oee_data)
    result = oee.calculate_performance("part_counter", ideal_cycle_time=60.0)
    assert "resets" not in result.columns
    # naive last - first loses the production before the reset
    assert result["actual_parts"].iloc[0] == 120


def test_performance_handles_resets(reset_oee_data):
    oee = OEECalculator(reset_oee_data)
    result = oee.calculate_performance(
        "part_counter", ideal_cycle_time=60.0, handle_resets=True
    )
    assert "resets" in result.columns
    assert result["actual_parts"].iloc[0] == 270
    assert result["resets"].iloc[0] == 1


def test_quality_default_under_reports(reset_oee_data):
    oee = OEECalculator(reset_oee_data)
    result = oee.calculate_quality("total_counter", "reject_counter")
    assert "resets" not in result.columns
    assert result["total_parts"].iloc[0] == 120
    assert result["reject_parts"].iloc[0] == 1
    assert result["good_parts"].iloc[0] == 119


def test_quality_handles_resets(reset_oee_data):
    oee = OEECalculator(reset_oee_data)
    result = oee.calculate_quality(
        "total_counter", "reject_counter", handle_resets=True
    )
    assert "resets" in result.columns
    assert result["total_parts"].iloc[0] == 270
    assert result["reject_parts"].iloc[0] == 3
    assert result["good_parts"].iloc[0] == 267
    # one reset in the total counter + one in the reject counter
    assert result["resets"].iloc[0] == 2
