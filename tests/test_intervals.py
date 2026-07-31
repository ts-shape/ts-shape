"""Tests for the shared state-interval helper."""

import pandas as pd  # type: ignore
import pytest

from ts_shape.utils.intervals import state_intervals


def _frame(pairs):
    return pd.DataFrame(
        [{"systime": pd.Timestamp(t), "state": s} for t, s in pairs]
    )


def test_change_encoded_last_sample_collapses_to_zero():
    # One row per transition: run 600s, idle 120s, run.
    df = _frame(
        [
            ("2024-01-01 08:00:00", True),
            ("2024-01-01 08:10:00", False),
            ("2024-01-01 08:12:00", True),
            ("2024-01-01 08:30:00", False),
        ]
    )
    out = state_intervals(df, time_column="systime", state_column="state")
    # Default "last_sample" -> every single-row interval has zero duration.
    assert (out["duration_seconds"] == 0).all()


def test_change_encoded_next_change_measures_correctly():
    df = _frame(
        [
            ("2024-01-01 08:00:00", True),
            ("2024-01-01 08:10:00", False),
            ("2024-01-01 08:12:00", True),
            ("2024-01-01 08:30:00", False),
        ]
    )
    out = state_intervals(
        df, time_column="systime", state_column="state", interval_end="next_change"
    )
    durations = list(out["duration_seconds"])
    # run 08:00->08:10 = 600, idle 08:10->08:12 = 120, run 08:12->08:30 = 1080,
    # final open interval falls back to its own last sample -> 0.
    assert durations == [600.0, 120.0, 1080.0, 0.0]


def test_next_change_with_data_end_bounds_final_interval():
    df = _frame(
        [
            ("2024-01-01 08:00:00", True),
            ("2024-01-01 08:10:00", False),
        ]
    )
    out = state_intervals(
        df,
        time_column="systime",
        state_column="state",
        interval_end="next_change",
        data_end=pd.Timestamp("2024-01-01 08:15:00"),
    )
    # run 08:00->08:10 = 600, final idle 08:10->data_end 08:15 = 300.
    assert list(out["duration_seconds"]) == [600.0, 300.0]


def test_dense_sampling_next_change_recovers_full_interval():
    # Dense 1s sampling of a 5s idle: last_sample undercounts, next_change is exact.
    times = pd.date_range("2024-01-01 08:00:00", periods=10, freq="1s")
    states = [True] * 3 + [False] * 5 + [True] * 2
    df = pd.DataFrame({"systime": times, "state": states})
    last = state_intervals(df, time_column="systime", state_column="state")
    nxt = state_intervals(
        df, time_column="systime", state_column="state", interval_end="next_change"
    )
    idle_last = last[last["state"] == False].iloc[0]  # noqa: E712
    idle_next = nxt[nxt["state"] == False].iloc[0]  # noqa: E712
    assert idle_last["duration_seconds"] == 4.0  # 5 samples span only 4s
    assert idle_next["duration_seconds"] == 5.0  # to the next transition


def test_empty_returns_columns():
    out = state_intervals(
        pd.DataFrame(columns=["systime", "state"]),
        time_column="systime",
        state_column="state",
    )
    assert list(out.columns) == [
        "start",
        "end",
        "state",
        "duration_seconds",
        "sample_count",
    ]
    assert out.empty


def test_invalid_mode_raises():
    df = _frame([("2024-01-01 08:00:00", True)])
    with pytest.raises(ValueError):
        state_intervals(
            df, time_column="systime", state_column="state", interval_end="bogus"
        )
