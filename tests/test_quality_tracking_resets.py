"""Reset-aware behaviour tests for QualityEvents.

Quality counters (OK / NOK) can reset back to a lower value at a shift
change or a controller restart. Without ``handle_resets`` a window
containing a reset under-reports; with it, the post-reset parts are counted
and a ``resets`` column is added. This file also covers the interleaving
correctness of ``quality_by_part`` and the zero-total-day division guard in
``daily_quality_summary``.
"""

import pandas as pd

from ts_shape.events.production.quality_tracking import QualityEvents


def _make_reset_quality_df():
    """Single shift with a resetting OK counter and a monotonic NOK counter.

    OK counter:  10, 30, 55, [reset] 5, 20, 40 -> default 30, reset-aware 85.
    NOK counter: 2, 4, 6, 8, 10, 12           -> 10 (no reset).
    One part (PART_A) and one reason for the whole window.
    """
    times = pd.date_range("2024-01-01 06:00", periods=6, freq="1h")
    ok_counts = [10, 30, 55, 5, 20, 40]
    nok_counts = [2, 4, 6, 8, 10, 12]
    rows = []
    for t, c in zip(times, ok_counts):
        rows.append({"systime": t, "uuid": "ok_counter", "value_integer": c})
    for t, c in zip(times, nok_counts):
        rows.append({"systime": t, "uuid": "nok_counter", "value_integer": c})
    for t in times:
        rows.append({"systime": t, "uuid": "part_number", "value_string": "PART_A"})
    for t in times:
        rows.append(
            {"systime": t, "uuid": "defect_reason", "value_string": "Dimension_Error"}
        )
    return pd.DataFrame(rows)


def test_nok_by_shift_default_underreports():
    df = _make_reset_quality_df()
    tracker = QualityEvents(df)
    result = tracker.nok_by_shift(
        ok_counter_uuid="ok_counter", nok_counter_uuid="nok_counter"
    )

    assert "resets" not in result.columns
    row = result.iloc[0]
    assert row["ok_parts"] == 30
    assert row["nok_parts"] == 10
    assert row["total_parts"] == 40
    # 30 / 40 * 100 = 75.0
    assert row["first_pass_yield_pct"] == 75.0


def test_nok_by_shift_handles_resets():
    df = _make_reset_quality_df()
    tracker = QualityEvents(df)
    result = tracker.nok_by_shift(
        ok_counter_uuid="ok_counter",
        nok_counter_uuid="nok_counter",
        handle_resets=True,
    )

    assert "resets" in result.columns
    row = result.iloc[0]
    assert row["ok_parts"] == 85
    assert row["nok_parts"] == 10
    assert row["total_parts"] == 95
    # 85 / 95 * 100 = 89.5
    assert row["first_pass_yield_pct"] == 89.5
    assert row["resets"] == 1


def test_quality_by_part_default_underreports():
    df = _make_reset_quality_df()
    tracker = QualityEvents(df)
    result = tracker.quality_by_part(
        ok_counter_uuid="ok_counter",
        nok_counter_uuid="nok_counter",
        part_id_uuid="part_number",
    )

    assert "resets" not in result.columns
    row = result[result["part_number"] == "PART_A"].iloc[0]
    assert row["ok_parts"] == 30
    assert row["nok_parts"] == 10


def test_quality_by_part_handles_resets():
    df = _make_reset_quality_df()
    tracker = QualityEvents(df)
    result = tracker.quality_by_part(
        ok_counter_uuid="ok_counter",
        nok_counter_uuid="nok_counter",
        part_id_uuid="part_number",
        handle_resets=True,
    )

    assert "resets" in result.columns
    row = result[result["part_number"] == "PART_A"].iloc[0]
    assert row["ok_parts"] == 85
    assert row["nok_parts"] == 10
    assert row["resets"] == 1


def test_quality_by_part_fixes_interleaving_over_count():
    """When parts alternate, ``last - first`` per group over-counts.

    Global OK counter is monotonic (no reset) but part A and part B rows are
    interleaved in time. The reset-aware path sums per-reading increments on
    the full frame, so each part gets only its own increments.
    """
    times = pd.date_range("2024-01-01 06:00", periods=6, freq="1h")
    ok_counts = [100, 110, 130, 145, 170, 190]
    parts = ["PART_A", "PART_B", "PART_A", "PART_B", "PART_A", "PART_B"]
    rows = []
    for t, c in zip(times, ok_counts):
        rows.append({"systime": t, "uuid": "ok_counter", "value_integer": c})
    for t, p in zip(times, parts):
        rows.append({"systime": t, "uuid": "part_number", "value_string": p})
    df = pd.DataFrame(rows)
    tracker = QualityEvents(df)

    default = tracker.quality_by_part(
        ok_counter_uuid="ok_counter",
        nok_counter_uuid="missing_nok",
        part_id_uuid="part_number",
    )
    # last - first per non-contiguous group over-counts: A = 170-100 = 70.
    assert default[default["part_number"] == "PART_A"]["ok_parts"].iloc[0] == 70

    aware = tracker.quality_by_part(
        ok_counter_uuid="ok_counter",
        nok_counter_uuid="missing_nok",
        part_id_uuid="part_number",
        handle_resets=True,
    )
    # Sum of A's own increments: 20 + 25 = 45 (t0 is the baseline, 0).
    assert aware[aware["part_number"] == "PART_A"]["ok_parts"].iloc[0] == 45
    assert aware[aware["part_number"] == "PART_B"]["ok_parts"].iloc[0] == 45
    assert aware["resets"].sum() == 0


def test_nok_by_reason_handles_resets():
    df = _make_reset_quality_df()
    tracker = QualityEvents(df)

    default = tracker.nok_by_reason(
        nok_counter_uuid="ok_counter", defect_reason_uuid="defect_reason"
    )
    assert "resets" not in default.columns
    assert default["nok_parts"].sum() == 30

    aware = tracker.nok_by_reason(
        nok_counter_uuid="ok_counter",
        defect_reason_uuid="defect_reason",
        handle_resets=True,
    )
    assert "resets" in aware.columns
    assert aware["nok_parts"].sum() == 85
    assert aware["resets"].sum() == 1


def test_daily_quality_summary_handles_resets():
    df = _make_reset_quality_df()
    tracker = QualityEvents(df)

    default = tracker.daily_quality_summary(
        ok_counter_uuid="ok_counter", nok_counter_uuid="nok_counter"
    )
    assert "resets" not in default.columns
    assert default["ok_parts"].iloc[0] == 30

    aware = tracker.daily_quality_summary(
        ok_counter_uuid="ok_counter",
        nok_counter_uuid="nok_counter",
        handle_resets=True,
    )
    assert "resets" in aware.columns
    assert aware["ok_parts"].iloc[0] == 85
    assert aware["resets"].iloc[0] == 1


def test_daily_quality_summary_zero_total_day_is_zero_not_nan():
    """A day whose counters do not advance must yield 0.0 rates, not NaN."""
    times = pd.date_range("2024-01-01 06:00", periods=3, freq="1h")
    rows = []
    # Flat OK counter (no production) and flat NOK counter -> total_parts = 0.
    for t in times:
        rows.append({"systime": t, "uuid": "ok_counter", "value_integer": 100})
    for t in times:
        rows.append({"systime": t, "uuid": "nok_counter", "value_integer": 5})
    df = pd.DataFrame(rows)
    tracker = QualityEvents(df)

    result = tracker.daily_quality_summary(
        ok_counter_uuid="ok_counter", nok_counter_uuid="nok_counter"
    )

    assert not result.empty
    row = result.iloc[0]
    assert row["total_parts"] == 0
    assert row["nok_rate_pct"] == 0.0
    assert row["first_pass_yield_pct"] == 0.0
    assert not pd.isna(row["nok_rate_pct"])
    assert not pd.isna(row["first_pass_yield_pct"])
