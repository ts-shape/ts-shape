"""Counter-reset handling for PeriodSummaryEvents roll-ups."""

import pandas as pd

from ts_shape.events.production.period_summary import PeriodSummaryEvents


def _make_reset_period_df():
    """Two days: day 1 has a mid-day counter reset, day 2 is monotonic.

    Day 1 (2024-01-01): 100, 200, 300, [reset] 50, 150.
        Naive daily quantity: max(0, 150 - 100) = 50.
        Reset-aware daily quantity: 100 + 100 + 50 + 100 = 350, 1 reset.
    Day 2 (2024-01-02): 0, 100, 200, 300, 400 -> 400 either way, 0 resets.
    """
    rows = []
    day1 = pd.to_datetime(
        [
            "2024-01-01 07:00:00",
            "2024-01-01 08:00:00",
            "2024-01-01 09:00:00",
            "2024-01-01 10:00:00",
            "2024-01-01 11:00:00",
        ]
    )
    for t, c in zip(day1, [100, 200, 300, 50, 150]):
        rows.append({"systime": t, "uuid": "prod_counter", "value_integer": c})

    day2 = pd.to_datetime(
        [
            "2024-01-02 07:00:00",
            "2024-01-02 08:00:00",
            "2024-01-02 09:00:00",
            "2024-01-02 10:00:00",
            "2024-01-02 11:00:00",
        ]
    )
    for t, c in zip(day2, [0, 100, 200, 300, 400]):
        rows.append({"systime": t, "uuid": "prod_counter", "value_integer": c})

    return pd.DataFrame(rows)


def test_weekly_summary_under_reports_without_reset_handling():
    summary = PeriodSummaryEvents(_make_reset_period_df())
    result = summary.weekly_summary(counter_uuid="prod_counter")

    assert "resets" not in result.columns
    # Both days fall in the same week: 50 (under-reported) + 400 = 450.
    assert result["total_production"].sum() == 450


def test_weekly_summary_handles_resets():
    summary = PeriodSummaryEvents(_make_reset_period_df())
    result = summary.weekly_summary(counter_uuid="prod_counter", handle_resets=True)

    assert "resets" in result.columns
    # Reset-aware: 350 + 400 = 750.
    assert result["total_production"].sum() == 750
    assert result["resets"].sum() == 1


def test_monthly_summary_handles_resets():
    summary = PeriodSummaryEvents(_make_reset_period_df())
    result = summary.monthly_summary(counter_uuid="prod_counter", handle_resets=True)

    assert "resets" in result.columns
    assert result["total_production"].sum() == 750
    assert result["resets"].sum() == 1

    default = summary.monthly_summary(counter_uuid="prod_counter")
    assert "resets" not in default.columns
    assert default["total_production"].sum() == 450


def test_compare_periods_handles_resets():
    summary = PeriodSummaryEvents(_make_reset_period_df())

    default = summary.compare_periods(
        counter_uuid="prod_counter",
        period1=("2024-01-01", "2024-01-01"),
        period2=("2024-01-02", "2024-01-02"),
    )
    p1_default = default[default["metric"] == "total_production"]["period1_value"].iloc[
        0
    ]
    assert p1_default == 50  # under-reports

    aware = summary.compare_periods(
        counter_uuid="prod_counter",
        period1=("2024-01-01", "2024-01-01"),
        period2=("2024-01-02", "2024-01-02"),
        handle_resets=True,
    )
    p1_aware = aware[aware["metric"] == "total_production"]["period1_value"].iloc[0]
    p2_aware = aware[aware["metric"] == "total_production"]["period2_value"].iloc[0]
    assert p1_aware == 350
    assert p2_aware == 400
