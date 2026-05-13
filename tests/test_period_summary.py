"""Tests for PeriodSummary module."""

import pandas as pd
from datetime import date

from ts_shape.events.production.period_summary import PeriodSummary


def _make_multiday_df(days=14):
    """Create production counter data spanning multiple days."""
    rows = []
    counter = 0
    ok_counter = 0
    nok_counter = 0
    for day in range(1, days + 1):
        day_str = f"2024-01-{day:02d}"
        for hour in range(6, 22):
            t = pd.Timestamp(f"{day_str} {hour:02d}:00:00")
            counter += 80
            ok_counter += 77
            nok_counter += 3
            rows.append(
                {"systime": t, "uuid": "prod_counter", "value_integer": counter}
            )
            rows.append(
                {"systime": t, "uuid": "ok_counter", "value_integer": ok_counter}
            )
            rows.append(
                {"systime": t, "uuid": "nok_counter", "value_integer": nok_counter}
            )
    return pd.DataFrame(rows)


class TestPeriodSummary:

    def test_weekly_summary(self):
        df = _make_multiday_df(14)
        summary = PeriodSummary(df)
        result = summary.weekly_summary(counter_uuid="prod_counter")
        assert not result.empty
        assert "week_start" in result.columns
        assert "total_production" in result.columns
        assert "daily_avg" in result.columns

    def test_weekly_summary_with_quality(self):
        df = _make_multiday_df(14)
        summary = PeriodSummary(df)
        result = summary.weekly_summary(
            counter_uuid="prod_counter",
            ok_counter_uuid="ok_counter",
            nok_counter_uuid="nok_counter",
        )
        assert not result.empty
        assert "quality_pct" in result.columns
        # Quality should be ~96.25% (77/80)
        assert result["quality_pct"].iloc[0] > 90

    def test_monthly_summary(self):
        df = _make_multiday_df(14)
        summary = PeriodSummary(df)
        result = summary.monthly_summary(counter_uuid="prod_counter")
        assert not result.empty
        assert "year" in result.columns
        assert "month" in result.columns
        assert result["year"].iloc[0] == 2024

    def test_compare_periods(self):
        df = _make_multiday_df(14)
        summary = PeriodSummary(df)
        result = summary.compare_periods(
            counter_uuid="prod_counter",
            period1=("2024-01-01", "2024-01-07"),
            period2=("2024-01-08", "2024-01-14"),
        )
        assert not result.empty
        assert "metric" in result.columns
        assert "change_pct" in result.columns
        total_row = result[result["metric"] == "daily_avg"]
        assert not total_row.empty

    def test_from_daily_data_weekly(self):
        """Test pipeline entry-point: from_daily_data with weekly freq."""
        daily = pd.DataFrame(
            [
                {
                    "date": date(2024, 1, 1),
                    "ok_parts": 1000,
                    "nok_parts": 50,
                    "quality_pct": 95.2,
                    "availability_pct": 90.0,
                },
                {
                    "date": date(2024, 1, 2),
                    "ok_parts": 1050,
                    "nok_parts": 40,
                    "quality_pct": 96.3,
                    "availability_pct": 91.5,
                },
                {
                    "date": date(2024, 1, 3),
                    "ok_parts": 980,
                    "nok_parts": 60,
                    "quality_pct": 94.2,
                    "availability_pct": 88.0,
                },
                {
                    "date": date(2024, 1, 4),
                    "ok_parts": 1020,
                    "nok_parts": 45,
                    "quality_pct": 95.8,
                    "availability_pct": 92.0,
                },
                {
                    "date": date(2024, 1, 5),
                    "ok_parts": 1100,
                    "nok_parts": 30,
                    "quality_pct": 97.3,
                    "availability_pct": 93.5,
                },
                {
                    "date": date(2024, 1, 8),
                    "ok_parts": 1010,
                    "nok_parts": 55,
                    "quality_pct": 94.8,
                    "availability_pct": 89.0,
                },
                {
                    "date": date(2024, 1, 9),
                    "ok_parts": 1060,
                    "nok_parts": 35,
                    "quality_pct": 96.8,
                    "availability_pct": 91.0,
                },
            ]
        )

        result = PeriodSummary.from_daily_data(daily, freq="W")
        assert not result.empty
        assert "period_start" in result.columns
        assert "period_end" in result.columns
        assert "days" in result.columns
        # _pct columns should be averaged, not summed
        assert "quality_pct" in result.columns
        assert "availability_pct" in result.columns
        # ok_parts and nok_parts should be summed
        assert "ok_parts" in result.columns

    def test_from_daily_data_monthly(self):
        """Test from_daily_data with monthly freq."""
        daily = pd.DataFrame(
            [
                {
                    "date": date(2024, 1, d),
                    "total_parts": 1000 + d * 10,
                    "quality_pct": 95.0 + d * 0.1,
                }
                for d in range(1, 15)
            ]
        )
        result = PeriodSummary.from_daily_data(daily, freq="MS")
        assert not result.empty
        assert len(result) == 1  # All in January
        assert result["days"].iloc[0] == 14

    def test_from_daily_data_empty(self):
        """Test from_daily_data with empty data."""
        result = PeriodSummary.from_daily_data(pd.DataFrame(columns=["date"]))
        assert result.empty

    def test_from_daily_data_pct_averaged(self):
        """Verify that _pct columns are averaged, not summed."""
        daily = pd.DataFrame(
            [
                {"date": date(2024, 1, 1), "quality_pct": 90.0, "count": 100},
                {"date": date(2024, 1, 2), "quality_pct": 80.0, "count": 200},
            ]
        )
        result = PeriodSummary.from_daily_data(daily, freq="W")
        assert not result.empty
        # quality_pct should be averaged: (90+80)/2 = 85
        assert result["quality_pct"].iloc[0] == 85.0
        # count should be summed: 100+200 = 300
        assert result["count"].iloc[0] == 300

    def test_empty_data(self):
        df = pd.DataFrame(columns=["systime", "uuid", "value_integer"])
        summary = PeriodSummary(df)
        assert summary.weekly_summary(counter_uuid="x").empty
        assert summary.monthly_summary(counter_uuid="x").empty
        assert summary.compare_periods(
            counter_uuid="x",
            period1=("2024-01-01", "2024-01-07"),
            period2=("2024-01-08", "2024-01-14"),
        ).empty
