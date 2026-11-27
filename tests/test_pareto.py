"""Tests for ParetoEvents."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import ts_shape
from ts_shape.errors import ColumnNotFoundError
from ts_shape.events.production.pareto import ParetoEvents


def make_alarm_frame(tz: str | None = None) -> pd.DataFrame:
    """Two days of alarm events, uneven distribution."""
    day1 = pd.date_range("2024-01-01", periods=12, freq="1h", tz=tz)
    day2 = pd.date_range("2024-01-02", periods=12, freq="1h", tz=tz)
    return pd.DataFrame(
        {
            "systime": day1.append(day2),
            "uuid": "alarms",
            "value_string": (
                ["A"] * 6
                + ["B"] * 4
                + ["C"] * 2  # day1: 6/4/2
                + ["A"] * 2
                + ["B"] * 2
                + ["C"] * 8  # day2: 2/2/8
            ),
            "value_double": [1.0] * 24,
        }
    )


def test_rank_by_count_pct_sums_to_100():
    out = ParetoEvents(make_alarm_frame()).rank_by_count(window="1D")
    assert list(out.columns) == [
        "start",
        "category",
        "total",
        "share_of_total_pct",
        "cumulative_pct",
        "rank",
    ]
    assert len(out) == 6  # 3 categories x 2 days
    # Each window's share sums to 100.
    for _, group in out.groupby("start"):
        assert group["share_of_total_pct"].sum() == pytest.approx(100.0)
        assert group["cumulative_pct"].iloc[-1] == pytest.approx(100.0)
        assert list(group["rank"]) == [1, 2, 3]
    # Day 1: A=6, B=4, C=2. Day 2: C=8, A=2, B=2 (ordering stable by rank).
    day1 = out[out["start"] == pd.Timestamp("2024-01-01")]
    assert list(day1["category"]) == ["A", "B", "C"]
    assert list(day1["total"]) == [6, 4, 2]
    day2 = out[out["start"] == pd.Timestamp("2024-01-02")]
    assert day2["category"].iloc[0] == "C"  # C is the top on day 2
    assert day2["total"].iloc[0] == 8


def test_rank_by_sum_matches_sum_not_count():
    df = make_alarm_frame().iloc[:12].copy()  # day 1 only
    df["value_double"] = [10.0] * 6 + [100.0] * 4 + [1.0] * 2
    # A contributes 60, B 400, C 2 -> B should be rank 1 even though A has more rows.
    out = ParetoEvents(df).rank_by_sum(
        category_column="value_string", value_column="value_double", window="1D"
    )
    assert out["category"].iloc[0] == "B"
    assert out["total"].iloc[0] == pytest.approx(400.0)
    assert list(out["rank"]) == [1, 2, 3]


def test_top_n_trims_after_share_math():
    out = ParetoEvents(make_alarm_frame()).rank_by_count(window="1D", top_n=1)
    # Only the #1 per day is kept.
    assert len(out) == 2
    assert all(out["rank"] == 1)
    # Cumulative share is still the #1's share, not forced to 100.
    assert out["cumulative_pct"].iloc[0] == pytest.approx(50.0)  # 6/12 on day 1


def test_top_n_zero_raises():
    with pytest.raises(ValueError, match="top_n must be positive"):
        ParetoEvents(make_alarm_frame()).rank_by_count(top_n=0)


def test_missing_category_column_raises():
    df = make_alarm_frame().drop(columns=["value_string"])
    with pytest.raises(ColumnNotFoundError):
        ParetoEvents(df).rank_by_count(category_column="value_string")


def test_empty_frame_returns_typed_empty():
    df = make_alarm_frame().iloc[:0]
    out = ParetoEvents(df).rank_by_count()
    assert len(out) == 0
    assert list(out.columns) == [
        "start",
        "category",
        "total",
        "share_of_total_pct",
        "cumulative_pct",
        "rank",
    ]
    assert pd.api.types.is_datetime64_any_dtype(out["start"])
    assert out["rank"].dtype == np.int64


def test_tz_aware_input_preserves_tz():
    out = ParetoEvents(make_alarm_frame(tz="Europe/Berlin")).rank_by_count(window="1D")
    assert out["start"].dt.tz is not None
    assert str(out["start"].dt.tz) == "Europe/Berlin"


def test_null_categories_are_skipped():
    df = make_alarm_frame()
    df.loc[0:3, "value_string"] = None
    out = ParetoEvents(df).rank_by_count(window="1D")
    # Day-1 totals: A dropped from 6 to 2 (4 nulls removed).
    day1 = out[out["start"] == pd.Timestamp("2024-01-01")]
    # The remaining day-1 rows: A=2, B=4, C=2; total=8.
    assert day1["total"].sum() == 8
    assert day1["category"].iloc[0] == "B"


def test_null_values_are_skipped_in_rank_by_sum():
    df = make_alarm_frame()
    df.loc[0:5, "value_double"] = None
    out = ParetoEvents(df).rank_by_sum(window="1D")
    # Day 1 originally has A=6 (sum 6). Six A-rows NaN -> A drops out entirely.
    day1 = out[out["start"] == pd.Timestamp("2024-01-01")]
    assert "A" not in set(day1["category"])


def test_public_import_works():
    """ParetoEvents is reachable from the top-level lazy namespace."""
    assert ts_shape.ParetoEvents is ParetoEvents


def test_small_frame_single_category_is_100_pct():
    df = pd.DataFrame(
        {
            "systime": pd.date_range("2024-01-01", periods=3, freq="1h"),
            "uuid": "alarms",
            "value_string": ["only"] * 3,
        }
    )
    out = ParetoEvents(df).rank_by_count(window="1D")
    assert len(out) == 1
    assert out["category"].iloc[0] == "only"
    assert out["share_of_total_pct"].iloc[0] == pytest.approx(100.0)
    assert out["cumulative_pct"].iloc[0] == pytest.approx(100.0)
    assert out["rank"].iloc[0] == 1


def _make_state_frame(tz: str | None = None) -> pd.DataFrame:
    """Twelve hours of state changes on one signal: A for 5h, B for 3h,
    then A again for 4h — day totals A=9h, B=3h."""
    start = pd.Timestamp("2024-01-01", tz=tz)
    return pd.DataFrame(
        {
            "systime": [
                start,
                start + pd.Timedelta(hours=5),
                start + pd.Timedelta(hours=8),
                start + pd.Timedelta(hours=12),
            ],
            "uuid": "line1",
            "value_string": ["A", "B", "A", "A"],
        }
    )


def test_rank_by_duration_sums_seconds_per_category():
    out = ParetoEvents(_make_state_frame()).rank_by_duration(window="1D")
    assert list(out.columns) == [
        "start",
        "category",
        "total",
        "share_of_total_pct",
        "cumulative_pct",
        "rank",
    ]
    # A holds 5h (0h→5h) + 4h (8h→12h) = 9h; B holds 3h (5h→8h).
    assert len(out) == 2
    row_a = out[out["category"] == "A"].iloc[0]
    row_b = out[out["category"] == "B"].iloc[0]
    assert row_a["total"] == pytest.approx(9 * 3600)
    assert row_b["total"] == pytest.approx(3 * 3600)
    assert row_a["rank"] == 1
    assert row_b["rank"] == 2
    assert row_a["share_of_total_pct"] == pytest.approx(75.0)


def test_rank_by_duration_last_row_per_uuid_contributes_zero():
    """A single-sample signal has no successor on that uuid; its lone
    row contributes zero duration and is filtered out of the ranking."""
    df = pd.DataFrame(
        {
            "systime": [pd.Timestamp("2024-01-01")],
            "uuid": "single",
            "value_string": ["only"],
        }
    )
    out = ParetoEvents(df).rank_by_duration(window="1D")
    assert out.empty
    # The schema is still the canonical Pareto shape.
    assert list(out.columns) == [
        "start",
        "category",
        "total",
        "share_of_total_pct",
        "cumulative_pct",
        "rank",
    ]


def test_rank_by_duration_is_per_uuid_not_cross_signal():
    """Interleaving a second signal must not create a ghost hold from
    signal A's last row to signal B's first row."""
    t0 = pd.Timestamp("2024-01-01")
    df = pd.DataFrame(
        {
            # Two separate signals, a week apart.
            "systime": [
                t0,
                t0 + pd.Timedelta(hours=2),
                t0 + pd.Timedelta(days=7),
                t0 + pd.Timedelta(days=7, hours=3),
            ],
            "uuid": ["line1", "line1", "line2", "line2"],
            "value_string": ["A", "A", "B", "B"],
        }
    )
    out = ParetoEvents(df).rank_by_duration(window="1D")
    # Day 1: A for 2 hours (line1); Day 8: B for 3 hours (line2).
    # No row spans from line1's last to line2's first.
    assert len(out) == 2
    day1 = out[out["start"] == pd.Timestamp("2024-01-01")]
    day8 = out[out["start"] == pd.Timestamp("2024-01-08")]
    assert day1["category"].iloc[0] == "A"
    assert day1["total"].iloc[0] == pytest.approx(2 * 3600)
    assert day8["category"].iloc[0] == "B"
    assert day8["total"].iloc[0] == pytest.approx(3 * 3600)


def test_rank_by_duration_tz_aware_input_preserves_tz():
    out = ParetoEvents(_make_state_frame(tz="Europe/Berlin")).rank_by_duration(
        window="1D"
    )
    assert out["start"].dt.tz is not None
    assert str(out["start"].dt.tz) == "Europe/Berlin"


def test_rank_by_duration_empty_frame_returns_typed_empty():
    df = _make_state_frame().iloc[:0]
    out = ParetoEvents(df).rank_by_duration(window="1D")
    assert len(out) == 0
    assert pd.api.types.is_datetime64_any_dtype(out["start"])
    assert out["rank"].dtype == np.int64
    assert out["total"].dtype == np.float64


def test_rank_by_duration_top_n_and_validation():
    out = ParetoEvents(_make_state_frame()).rank_by_duration(window="1D", top_n=1)
    assert len(out) == 1
    assert out["category"].iloc[0] == "A"
    with pytest.raises(ValueError, match="top_n must be positive"):
        ParetoEvents(_make_state_frame()).rank_by_duration(top_n=0)


def test_rank_by_duration_missing_category_column_raises():
    df = _make_state_frame().drop(columns=["value_string"])
    with pytest.raises(ColumnNotFoundError):
        ParetoEvents(df).rank_by_duration()


def test_rank_by_duration_window_attribution_by_start():
    """A hold that spans a window boundary is attributed to the window
    of its start — documented behaviour, pinned here."""
    df = pd.DataFrame(
        {
            # Two rows: a 4-hour hold from day 1 23:00 to day 2 03:00.
            "systime": [
                pd.Timestamp("2024-01-01 23:00"),
                pd.Timestamp("2024-01-02 03:00"),
            ],
            "uuid": "line",
            "value_string": ["X", "Y"],
        }
    )
    out = ParetoEvents(df).rank_by_duration(window="1D")
    assert len(out) == 1
    assert out["start"].iloc[0] == pd.Timestamp("2024-01-01")  # start's window
    assert out["total"].iloc[0] == pytest.approx(4 * 3600)
