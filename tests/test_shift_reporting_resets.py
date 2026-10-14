"""Counter-reset handling for ShiftReportEvents.shift_production."""

import pandas as pd

from ts_shape.events.production.shift_reporting import ShiftReportEvents


def _make_reset_shift_df():
    """A single shift_1 (06:00-14:00) whose counter resets mid-shift.

    Readings: 100, 200, 300, [reset] 50, 150.
    Naive quantity: max(0, 150 - 100) = 50 (loses the pre-reset production).
    Reset-aware quantity: 100 + 100 + 50 + 100 = 350, with 1 reset.
    """
    times = pd.to_datetime(
        [
            "2024-01-01 07:00:00",
            "2024-01-01 08:00:00",
            "2024-01-01 09:00:00",
            "2024-01-01 10:00:00",
            "2024-01-01 11:00:00",
        ]
    )
    counts = [100, 200, 300, 50, 150]
    return pd.DataFrame(
        {
            "uuid": ["production_counter"] * len(times),
            "systime": times,
            "value_integer": counts,
            "is_delta": [True] * len(times),
        }
    )


def test_shift_production_under_reports_without_reset_handling():
    reporter = ShiftReportEvents(_make_reset_shift_df())
    result = reporter.shift_production(counter_uuid="production_counter")

    assert list(result.columns) == [
        "start",
        "end",
        "duration_seconds",
        "date",
        "shift",
        "quantity",
    ]
    assert "resets" not in result.columns

    row = result[result["shift"] == "shift_1"].iloc[0]
    assert row["quantity"] == 50  # under-reports: real production is 350


def test_shift_production_handles_resets():
    reporter = ShiftReportEvents(_make_reset_shift_df())
    result = reporter.shift_production(
        counter_uuid="production_counter", handle_resets=True
    )

    assert "resets" in result.columns

    row = result[result["shift"] == "shift_1"].iloc[0]
    assert row["quantity"] == 350
    assert row["resets"] == 1
    assert result["resets"].sum() == 1


def test_shift_production_resets_column_with_parts():
    """The optional resets column is added alongside part_number too."""
    df = _make_reset_shift_df()
    parts = pd.DataFrame(
        {
            "uuid": ["part_number"] * 5,
            "systime": df["systime"],
            "value_string": ["PART_A"] * 5,
            "is_delta": [False] * 5,
        }
    )
    reporter = ShiftReportEvents(pd.concat([df, parts], ignore_index=True))

    result = reporter.shift_production(
        counter_uuid="production_counter",
        part_id_uuid="part_number",
        handle_resets=True,
    )
    assert "part_number" in result.columns
    assert "resets" in result.columns
    row = result.iloc[0]
    assert row["quantity"] == 350
    assert row["resets"] == 1
