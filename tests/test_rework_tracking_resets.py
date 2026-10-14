"""Reset-aware behaviour tests for ReworkEvents.

A rework counter can reset back to a lower value at a shift change or a
controller restart. Without ``handle_resets`` a window containing a reset
under-reports; with it, the post-reset rework is counted and a ``resets``
column is added (except for the rate method, which only threads the flag
through numerator and denominator).
"""

import pandas as pd

from ts_shape.events.production.rework_tracking import ReworkEvents


def _make_reset_rework_df():
    """Rework counter resetting mid-shift, plus reason/part/production signals.

    Single shift (shift_1, 06:00-14:00). Rework counter:
        10, 30, 55, [reset] 5, 20, 40
    Naive last-first: 30. Reset-aware: 85.
    Production counter is monotonic: 100..600 (last-first = 500).
    """
    times = pd.date_range("2024-01-01 06:00", periods=6, freq="1h")
    rework_counts = [10, 30, 55, 5, 20, 40]
    prod_counts = [100, 200, 300, 400, 500, 600]
    rows = []
    for t, c in zip(times, rework_counts):
        rows.append({"systime": t, "uuid": "rework_counter", "value_integer": c})
    for t, c in zip(times, prod_counts):
        rows.append({"systime": t, "uuid": "total_counter", "value_integer": c})
    for t in times:
        rows.append({"systime": t, "uuid": "rework_reason", "value_string": "Tool"})
    for t in times:
        rows.append({"systime": t, "uuid": "part_number", "value_string": "PART_A"})
    return pd.DataFrame(rows)


def test_rework_by_shift_default_underreports():
    df = _make_reset_rework_df()
    tracker = ReworkEvents(df)
    result = tracker.rework_by_shift(rework_uuid="rework_counter")

    assert list(result.columns) == [
        "start",
        "end",
        "duration_seconds",
        "date",
        "shift",
        "rework_count",
    ]
    assert "resets" not in result.columns
    assert result["rework_count"].sum() == 30


def test_rework_by_shift_handles_resets():
    df = _make_reset_rework_df()
    tracker = ReworkEvents(df)
    result = tracker.rework_by_shift(rework_uuid="rework_counter", handle_resets=True)

    assert "resets" in result.columns
    assert result["rework_count"].sum() == 85
    assert result["resets"].sum() == 1


def test_rework_by_reason_handles_resets():
    df = _make_reset_rework_df()
    tracker = ReworkEvents(df)

    default = tracker.rework_by_reason(
        rework_uuid="rework_counter", reason_uuid="rework_reason"
    )
    assert "resets" not in default.columns
    assert default["rework_count"].sum() == 30

    aware = tracker.rework_by_reason(
        rework_uuid="rework_counter", reason_uuid="rework_reason", handle_resets=True
    )
    assert "resets" in aware.columns
    assert aware["rework_count"].sum() == 85
    assert aware["resets"].sum() == 1


def test_rework_rate_threads_handle_resets():
    df = _make_reset_rework_df()
    tracker = ReworkEvents(df)

    default = tracker.rework_rate(
        rework_uuid="rework_counter", total_production_uuid="total_counter"
    )
    # 30 / 500 * 100 = 6.0
    assert default["rework_count"].sum() == 30
    assert default["rework_rate_pct"].iloc[0] == 6.0

    aware = tracker.rework_rate(
        rework_uuid="rework_counter",
        total_production_uuid="total_counter",
        handle_resets=True,
    )
    # 85 / 500 * 100 = 17.0
    assert aware["rework_count"].sum() == 85
    assert aware["total_produced"].iloc[0] == 500
    assert aware["rework_rate_pct"].iloc[0] == 17.0


def test_rework_cost_handles_resets():
    df = _make_reset_rework_df()
    tracker = ReworkEvents(df)

    default = tracker.rework_cost(
        rework_uuid="rework_counter",
        part_id_uuid="part_number",
        rework_costs={"PART_A": 3.0},
    )
    assert "resets" not in default.columns
    assert default["rework_count"].iloc[0] == 30
    assert default["total_cost"].iloc[0] == 90.0

    aware = tracker.rework_cost(
        rework_uuid="rework_counter",
        part_id_uuid="part_number",
        rework_costs={"PART_A": 3.0},
        handle_resets=True,
    )
    assert "resets" in aware.columns
    assert aware["rework_count"].iloc[0] == 85
    assert aware["total_cost"].iloc[0] == 255.0
    assert aware["resets"].iloc[0] == 1


def test_rework_trend_handles_resets():
    df = _make_reset_rework_df()
    tracker = ReworkEvents(df)

    default = tracker.rework_trend(rework_uuid="rework_counter", window="1D")
    assert "resets" not in default.columns
    assert default["rework_count"].sum() == 30

    aware = tracker.rework_trend(
        rework_uuid="rework_counter", window="1D", handle_resets=True
    )
    assert "resets" in aware.columns
    assert aware["rework_count"].sum() == 85
    assert aware["resets"].sum() == 1
