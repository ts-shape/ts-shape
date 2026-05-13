"""Tests for ShiftHandoverReport module."""

import pandas as pd
from datetime import date

from ts_shape.events.production.shift_handover import ShiftHandoverReport


def _make_handover_df():
    """Create sample data for shift handover testing."""
    rows = []
    # Production counter
    for i, hour in enumerate(range(6, 22)):
        t = pd.Timestamp(f"2024-01-01 {hour:02d}:00:00")
        rows.append({"systime": t, "uuid": "prod_counter", "value_integer": i * 50})
    # OK counter
    for i, hour in enumerate(range(6, 22)):
        t = pd.Timestamp(f"2024-01-01 {hour:02d}:00:00")
        rows.append({"systime": t, "uuid": "ok_counter", "value_integer": i * 48})
    # NOK counter
    for i, hour in enumerate(range(6, 22)):
        t = pd.Timestamp(f"2024-01-01 {hour:02d}:00:00")
        rows.append({"systime": t, "uuid": "nok_counter", "value_integer": i * 2})
    # Machine state
    states = (
        ["Running"] * 3
        + ["Stopped"]
        + ["Running"] * 3
        + ["Stopped"]
        + ["Running"] * 3
        + ["Stopped"]
        + ["Running"] * 3
        + ["Stopped"]
    )
    for hour, state in zip(range(6, 22), states):
        t = pd.Timestamp(f"2024-01-01 {hour:02d}:00:00")
        rows.append({"systime": t, "uuid": "machine_state", "value_string": state})

    return pd.DataFrame(rows)


class TestShiftHandoverReport:

    def test_generate_report(self):
        df = _make_handover_df()
        report = ShiftHandoverReport(df)
        result = report.generate_report(
            counter_uuid="prod_counter",
            ok_counter_uuid="ok_counter",
            nok_counter_uuid="nok_counter",
            state_uuid="machine_state",
            targets={"shift_1": 400, "shift_2": 400, "shift_3": 400},
        )
        assert not result.empty
        assert "production" in result.columns
        assert "quality_pct" in result.columns
        assert "availability_pct" in result.columns

    def test_generate_report_specific_date(self):
        df = _make_handover_df()
        report = ShiftHandoverReport(df)
        result = report.generate_report(
            counter_uuid="prod_counter",
            ok_counter_uuid="ok_counter",
            nok_counter_uuid="nok_counter",
            state_uuid="machine_state",
            report_date="2024-01-01",
        )
        assert not result.empty

    def test_highlight_issues(self):
        df = _make_handover_df()
        report = ShiftHandoverReport(df)
        issues = report.highlight_issues(
            counter_uuid="prod_counter",
            ok_counter_uuid="ok_counter",
            nok_counter_uuid="nok_counter",
            state_uuid="machine_state",
            targets={"shift_1": 400, "shift_2": 400, "shift_3": 400},
            thresholds={
                "production_achievement_pct": 95,
                "quality_pct": 99,
                "availability_pct": 95,
            },
        )
        assert isinstance(issues, list)
        for issue in issues:
            assert "shift" in issue
            assert "metric" in issue
            assert "severity" in issue

    def test_highlight_issues_from_report_df(self):
        """Test highlight_issues with a pre-built report DataFrame."""
        df = _make_handover_df()
        report = ShiftHandoverReport(df)
        report_df = report.generate_report(
            counter_uuid="prod_counter",
            ok_counter_uuid="ok_counter",
            nok_counter_uuid="nok_counter",
            state_uuid="machine_state",
            targets={"shift_1": 400, "shift_2": 400, "shift_3": 400},
        )
        issues = report.highlight_issues(
            report_df=report_df,
            thresholds={"quality_pct": 99, "availability_pct": 95},
        )
        assert isinstance(issues, list)

    def test_from_shift_data_pipeline(self):
        """Test the pipeline entry-point: from_shift_data()."""
        # Simulate upstream module outputs
        production_df = pd.DataFrame(
            [
                {"date": date(2024, 1, 1), "shift": "shift_1", "quantity": 400},
                {"date": date(2024, 1, 1), "shift": "shift_2", "quantity": 380},
            ]
        )
        quality_df = pd.DataFrame(
            [
                {
                    "date": date(2024, 1, 1),
                    "shift": "shift_1",
                    "ok_parts": 390,
                    "nok_parts": 10,
                    "quality_pct": 97.5,
                },
                {
                    "date": date(2024, 1, 1),
                    "shift": "shift_2",
                    "ok_parts": 365,
                    "nok_parts": 15,
                    "quality_pct": 96.1,
                },
            ]
        )
        downtime_df = pd.DataFrame(
            [
                {
                    "date": date(2024, 1, 1),
                    "shift": "shift_1",
                    "availability_pct": 92.0,
                    "downtime_minutes": 38.4,
                },
                {
                    "date": date(2024, 1, 1),
                    "shift": "shift_2",
                    "availability_pct": 85.0,
                    "downtime_minutes": 72.0,
                },
            ]
        )

        result = ShiftHandoverReport.from_shift_data(
            production_df=production_df,
            quality_df=quality_df,
            downtime_df=downtime_df,
            targets={"shift_1": 450, "shift_2": 450},
        )

        assert not result.empty
        assert len(result) == 2
        assert list(result.columns) == ShiftHandoverReport.OUTPUT_COLUMNS
        assert result["production"].tolist() == [400, 380]
        assert result["quality_pct"].tolist() == [97.5, 96.1]
        assert result["availability_pct"].tolist() == [92.0, 85.0]
        assert (result["production_target"] == 450).all()

    def test_from_shift_data_partial(self):
        """Test from_shift_data with only production data (no quality/downtime)."""
        production_df = pd.DataFrame(
            [
                {"date": date(2024, 1, 1), "shift": "shift_1", "quantity": 400},
            ]
        )
        result = ShiftHandoverReport.from_shift_data(production_df=production_df)
        assert not result.empty
        assert result["production"].iloc[0] == 400
        assert result["quality_pct"].iloc[0] == 0.0
        assert result["availability_pct"].iloc[0] == 0.0

    def test_from_shift_data_empty(self):
        """Test from_shift_data with empty data."""
        result = ShiftHandoverReport.from_shift_data(
            production_df=pd.DataFrame(columns=["date", "shift", "quantity"])
        )
        assert result.empty

    def test_empty_data(self):
        df = pd.DataFrame(columns=["systime", "uuid", "value_integer", "value_string"])
        report = ShiftHandoverReport(df)
        result = report.generate_report(
            counter_uuid="x",
            ok_counter_uuid="y",
            nok_counter_uuid="z",
            state_uuid="w",
        )
        assert result.empty

    def test_highlight_issues_empty(self):
        df = pd.DataFrame(columns=["systime", "uuid", "value_integer", "value_string"])
        report = ShiftHandoverReport(df)
        issues = report.highlight_issues(
            counter_uuid="x",
            ok_counter_uuid="y",
            nok_counter_uuid="z",
            state_uuid="w",
        )
        assert issues == []
