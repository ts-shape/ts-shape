"""Reset-aware behavior tests for OperatorPerformanceEvents.

Production and quality counters can reset back to a lower value at a shift
change or a controller restart. With ``handle_resets=False`` the affected
window under-reports; by default (``handle_resets=True``), the post-reset production is counted and a
``resets`` column is added.
"""

import pandas as pd
import pytest

from ts_shape.events.production.operator_performance import OperatorPerformanceEvents


def _make_reset_operator_df():
    """Single operator (Alice) in one shift with a resetting production counter.

    Production counter: 10, 30, 55, [reset] 5, 20, 40
        Naive last-first: 30. Reset-aware: 85.
    OK counter (quality): same reset shape -> 30 default / 85 reset-aware.
    NOK counter: monotonic 2..12 (last-first = 10, no reset).
    """
    times = pd.date_range("2024-01-01 06:00", periods=6, freq="1h")
    counter_counts = [10, 30, 55, 5, 20, 40]
    ok_counts = [10, 30, 55, 5, 20, 40]
    nok_counts = [2, 4, 6, 8, 10, 12]
    rows = []
    for t in times:
        rows.append({"systime": t, "uuid": "operator_id", "value_string": "Alice"})
    for t, c in zip(times, counter_counts):
        rows.append({"systime": t, "uuid": "part_counter", "value_integer": c})
    for t, c in zip(times, ok_counts):
        rows.append({"systime": t, "uuid": "ok_parts", "value_integer": c})
    for t, c in zip(times, nok_counts):
        rows.append({"systime": t, "uuid": "nok_parts", "value_integer": c})
    return pd.DataFrame(rows)


def test_production_by_operator_naive_under_reports():
    df = _make_reset_operator_df()
    tracker = OperatorPerformanceEvents(df)
    result = tracker.production_by_operator(
        operator_uuid="operator_id", counter_uuid="part_counter", handle_resets=False
    )

    assert list(result.columns) == [
        "operator",
        "total_produced",
        "shifts_worked",
        "avg_per_shift",
    ]
    assert "resets" not in result.columns
    assert result.loc[result["operator"] == "Alice", "total_produced"].iloc[0] == 30


def test_production_by_operator_handles_resets():
    df = _make_reset_operator_df()
    tracker = OperatorPerformanceEvents(df)
    result = tracker.production_by_operator(
        operator_uuid="operator_id", counter_uuid="part_counter", handle_resets=True
    )

    assert "resets" in result.columns
    row = result[result["operator"] == "Alice"].iloc[0]
    assert row["total_produced"] == 85
    assert row["resets"] == 1


def test_quality_by_operator_naive_under_reports():
    df = _make_reset_operator_df()
    tracker = OperatorPerformanceEvents(df)
    result = tracker.quality_by_operator(
        operator_uuid="operator_id",
        ok_uuid="ok_parts",
        nok_uuid="nok_parts",
        handle_resets=False,
    )

    assert list(result.columns) == [
        "operator",
        "ok_count",
        "nok_count",
        "first_pass_yield_pct",
    ]
    assert "resets" not in result.columns
    row = result[result["operator"] == "Alice"].iloc[0]
    assert row["ok_count"] == 30
    assert row["nok_count"] == 10
    # 30 / 40 * 100 = 75.0
    assert row["first_pass_yield_pct"] == 75.0


def test_quality_by_operator_handles_resets():
    df = _make_reset_operator_df()
    tracker = OperatorPerformanceEvents(df)
    result = tracker.quality_by_operator(
        operator_uuid="operator_id",
        ok_uuid="ok_parts",
        nok_uuid="nok_parts",
        handle_resets=True,
    )

    assert "resets" in result.columns
    row = result[result["operator"] == "Alice"].iloc[0]
    assert row["ok_count"] == 85
    assert row["nok_count"] == 10
    # 85 / 95 * 100 = 89.5
    assert row["first_pass_yield_pct"] == pytest.approx(85 / 95 * 100)
    # One reset in the OK counter, none in the NOK counter.
    assert row["resets"] == 1


def test_production_by_operator_default_is_reset_aware():
    """The default ``handle_resets=True`` counts post-reset production."""
    df = _make_reset_operator_df()
    tracker = OperatorPerformanceEvents(df)

    def call(**kw):
        return tracker.production_by_operator(
            operator_uuid="operator_id", counter_uuid="part_counter", **kw
        )

    default = call()
    pd.testing.assert_frame_equal(default, call(handle_resets=True))
    assert default["resets"].sum() > 0


def test_quality_by_operator_default_is_reset_aware():
    """The default ``handle_resets=True`` counts post-reset production."""
    df = _make_reset_operator_df()
    tracker = OperatorPerformanceEvents(df)

    def call(**kw):
        return tracker.quality_by_operator(
            operator_uuid="operator_id", ok_uuid="ok_parts", nok_uuid="nok_parts", **kw
        )

    default = call()
    pd.testing.assert_frame_equal(default, call(handle_resets=True))
    assert default["resets"].sum() > 0
