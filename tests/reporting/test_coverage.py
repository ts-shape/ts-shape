"""Tests for reporting coverage, gaps and freshness."""

from __future__ import annotations

import pandas as pd
import pytest

from ts_shape.reporting import FacilityCalendar, ReportingCoverage


@pytest.fixture()
def expected(registry) -> pd.DataFrame:
    return FacilityCalendar(registry).period_grid("2026-01-01", "2026-01-03")


@pytest.fixture()
def partial() -> pd.DataFrame:
    """DE-STR complete, US-DTW missing a day and a value, CN-SHA silent."""
    return pd.DataFrame(
        {
            "facility": ["DE-STR"] * 3 + ["US-DTW"] * 2,
            "period": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-03",
                    "2026-01-01",
                    "2026-01-03",
                ]
            ),
            "parts": [10.0, 20.0, 30.0, 5.0, float("nan")],
        }
    )


def test_requires_key_columns(partial):
    with pytest.raises(ValueError, match="not found"):
        ReportingCoverage(partial, period_column="week")


def test_value_columns_are_inferred(partial):
    assert ReportingCoverage(partial).value_columns == ["parts"]


def test_explicit_value_columns_are_validated(partial):
    with pytest.raises(ValueError, match="ghost"):
        ReportingCoverage(partial, value_columns=["ghost"])


def test_missing_lists_every_gap(partial, expected):
    gaps = ReportingCoverage(partial).missing(expected)
    assert len(gaps) == 4  # one US-DTW day + three CN-SHA days
    assert set(gaps["facility"]) == {"US-DTW", "CN-SHA"}


def test_completeness_classifies_sites(partial, expected):
    result = ReportingCoverage(partial).completeness(expected).set_index("facility")
    assert result.loc["DE-STR", "status"] == "complete"
    assert result.loc["US-DTW", "status"] == "partial"
    assert result.loc["CN-SHA", "status"] == "silent"
    assert result.loc["US-DTW", "completeness_pct"] == pytest.approx(200 / 3)
    assert result.loc["US-DTW", "null_pct"] == 50.0


def test_silent_site_is_invisible_without_an_expected_calendar(partial):
    # The default calendar is the observed cross product, which cannot know
    # about a site that never reported at all.
    result = ReportingCoverage(partial).completeness().set_index("facility")
    assert "CN-SHA" not in result.index
    assert result.loc["US-DTW", "status"] == "partial"


def test_unexpected_site_is_kept(partial, expected):
    trimmed = expected[expected["facility"] != "US-DTW"]
    result = ReportingCoverage(partial).completeness(trimmed).set_index("facility")
    assert result.loc["US-DTW", "status"] == "unexpected"


def test_expected_calendar_is_validated(partial):
    with pytest.raises(ValueError, match="missing column"):
        ReportingCoverage(partial).missing(pd.DataFrame({"site": ["A"]}))


def test_summary_is_one_row(partial, expected):
    summary = ReportingCoverage(partial).summary(expected).iloc[0]
    assert summary["facilities_expected"] == 3
    assert summary["facilities_reporting"] == 2
    assert summary["facilities_silent"] == 1
    assert summary["rows_expected"] == 9
    assert summary["rows_reported"] == 5
    assert summary["rows_missing"] == 4
    assert summary["completeness_pct"] == pytest.approx(5 / 9 * 100.0)


def test_null_report_per_metric(partial):
    result = ReportingCoverage(partial).null_report().set_index("facility")
    assert result.loc["DE-STR", "null_pct"] == 0.0
    assert result.loc["US-DTW", "null_pct"] == 50.0


def test_null_report_without_metrics():
    frame = pd.DataFrame({"facility": ["A"], "period": ["p1"]})
    result = ReportingCoverage(frame).null_report()
    assert result.empty
    assert "null_pct" in result.columns


def test_freshness_flags_stale_sites(partial):
    result = (
        ReportingCoverage(partial)
        .freshness(as_of="2026-01-10", max_age_days=3)
        .set_index("facility")
    )
    assert result.loc["DE-STR", "age_days"] == pytest.approx(7.0)
    assert bool(result.loc["DE-STR", "is_stale"]) is True


def test_freshness_defaults_to_the_leading_edge(partial):
    result = ReportingCoverage(partial).freshness().set_index("facility")
    assert result.loc["DE-STR", "age_days"] == 0.0


def test_freshness_with_non_datetime_periods():
    frame = pd.DataFrame({"facility": ["A"], "period": ["week-1"], "parts": [1.0]})
    result = ReportingCoverage(frame).freshness()
    assert pd.isna(result.loc[0, "age_days"])


def test_repr_reports_scale(partial):
    assert "facilities=2" in repr(ReportingCoverage(partial))
