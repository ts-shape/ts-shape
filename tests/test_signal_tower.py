"""Tests for SignalTowerDecoder (transform pack)."""

from __future__ import annotations

import pandas as pd
import pytest

import ts_shape
from ts_shape.transform.signal_tower import (
    DEFAULT_OFF_STATE,
    DEFAULT_PRIORITY,
    SignalTowerDecoder,
)


T0 = pd.Timestamp("2025-01-01 00:00")


def _lamp_rows(uuid: str, pattern: list[tuple[int, bool]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "systime": [T0 + pd.Timedelta(minutes=mins) for mins, _ in pattern],
            "uuid": uuid,
            "value_bool": [val for _, val in pattern],
        }
    )


def _three_lamp_frame() -> pd.DataFrame:
    return pd.concat(
        [
            _lamp_rows("green", [(0, True), (30, False), (60, True), (90, True)]),
            _lamp_rows("yellow", [(0, False), (40, True), (70, False)]),
            _lamp_rows("red", [(0, False), (90, True)]),
        ],
        ignore_index=True,
    )


def test_decode_resolves_by_default_priority():
    df = _three_lamp_frame()
    out = SignalTowerDecoder(
        df, {"green": "running", "yellow": "warning", "red": "alarm"}
    ).decode()
    assert list(out.columns) == ["systime", "uuid", "value_string", "is_delta"]
    assert set(out["is_delta"]) == {True}
    # First state at t0: only green is on -> running.
    assert out.iloc[0]["value_string"] == "running"
    # 00:30 green turns off, yellow still off -> stopped.
    assert out.iloc[1]["value_string"] == "stopped"
    # 00:40 yellow turns on -> warning (alarm not yet on).
    assert out.iloc[2]["value_string"] == "warning"
    # 01:30: both green and red are on -> alarm wins by priority.
    final = out.iloc[-1]
    assert final["value_string"] == "alarm"


def test_decode_custom_priority_wins():
    df = _three_lamp_frame()
    out = SignalTowerDecoder(
        df,
        {"green": "running", "yellow": "warning", "red": "alarm"},
        priority=("running", "warning", "alarm"),  # inverted: running wins
    ).decode()
    # 01:30: green and red both on -> running under inverted priority.
    assert out.iloc[-1]["value_string"] == "running"


def test_decode_requires_lamp_map():
    with pytest.raises(ValueError, match="lamp_map"):
        SignalTowerDecoder(_three_lamp_frame(), {})


def test_decode_feeds_state_detector():
    """The decoded signal is accepted by MachineStateEvents / StateTransitionEvents."""
    df = _three_lamp_frame()
    decoded = SignalTowerDecoder(
        df, {"green": "running", "yellow": "warning", "red": "alarm"}
    ).decode()
    # Hand it to StateTransitionEvents and check it reads cleanly.
    tr = ts_shape.StateTransitionEvents(decoded, "transform:signal_tower")
    transitions = tr.transitions()
    assert not transitions.empty
    assert set(transitions["from_state"]) <= {"running", "stopped", "warning", "alarm"}


def test_decode_two_lamps_same_state_are_ored():
    df = pd.concat(
        [
            _lamp_rows("yellow_1", [(0, False), (20, True), (40, False)]),
            _lamp_rows("yellow_2", [(0, False), (10, False), (30, True), (50, False)]),
        ],
        ignore_index=True,
    )
    out = SignalTowerDecoder(
        df, {"yellow_1": "warning", "yellow_2": "warning"}
    ).decode()
    # The decoded signal is "warning" whenever either lamp is on.
    states = list(out["value_string"])
    assert "warning" in states
    assert DEFAULT_OFF_STATE in states


def test_decode_empty_frame():
    empty = pd.DataFrame(columns=["systime", "uuid", "value_bool"])
    out = SignalTowerDecoder(empty, {"g": "running"}).decode()
    assert out.empty
    assert list(out.columns) == ["systime", "uuid", "value_string", "is_delta"]
