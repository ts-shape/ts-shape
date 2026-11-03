"""Tests for StateTransitionEvents (engineering pack)."""

from __future__ import annotations

import pandas as pd
import pytest

import ts_shape
from ts_shape.events.engineering.state_transition import StateTransitionEvents


def _state_signal(states: list[str]) -> pd.DataFrame:
    return ts_shape.make_id_signal("state", states, hold=10, freq="1min")


def test_transitions_emits_canonical_point_events():
    df = _state_signal(["Idle", "Run", "Idle", "Alarm", "Idle"])
    events = StateTransitionEvents(df, "state").transitions()
    assert list(events.columns)[:3] == ["systime", "uuid", "source_uuid"]
    assert set(events.columns) >= {"from_state", "to_state", "dwell_seconds"}
    assert len(events) == 4  # four changes between five intervals
    assert list(events["from_state"]) == ["Idle", "Run", "Idle", "Alarm"]
    assert list(events["to_state"]) == ["Run", "Idle", "Alarm", "Idle"]
    # Each dwell equals the hold time of the state being left (10 minutes).
    assert set(events["dwell_seconds"].astype(int)) == {600}


def test_transition_matrix_counts_and_shares():
    df = _state_signal(["Idle", "Run", "Idle", "Run", "Idle", "Alarm", "Idle"])
    m = StateTransitionEvents(df, "state").transition_matrix()
    assert set(m.columns) == {
        "from_state",
        "to_state",
        "sample_count",
        "total_dwell_seconds",
        "mean_dwell_seconds",
        "share_of_transitions_pct",
    }
    # Idle -> Run happens twice, Run -> Idle twice, Idle -> Alarm once, Alarm -> Idle once.
    counts = m.set_index(["from_state", "to_state"])["sample_count"]
    assert counts.loc[("Idle", "Run")] == 2
    assert counts.loc[("Run", "Idle")] == 2
    assert counts.loc[("Idle", "Alarm")] == 1
    assert counts.loc[("Alarm", "Idle")] == 1
    assert pytest.approx(m["share_of_transitions_pct"].sum()) == 100.0


def test_transitions_empty_when_signal_never_changes():
    df = _state_signal(["Run", "Run", "Run"])
    events = StateTransitionEvents(df, "state").transitions()
    assert events.empty
    matrix = StateTransitionEvents(df, "state").transition_matrix()
    assert matrix.empty
    assert set(matrix.columns) >= {"from_state", "to_state", "sample_count"}


def test_transitions_empty_frame():
    empty = pd.DataFrame(columns=["uuid", "systime", "value_string"])
    events = StateTransitionEvents(empty, "state").transitions()
    assert events.empty
