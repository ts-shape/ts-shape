"""FlowConstraintEvents interval_end: intervals end at the next change by default."""

from __future__ import annotations

import pandas as pd
import pytest

from ts_shape.events.production.flow_constraints import FlowConstraintEvents

ROLES = {"upstream_run": "up", "downstream_run": "dn"}


def _flow(up: list[bool], dn: list[bool]) -> FlowConstraintEvents:
    t = pd.date_range("2025-01-01", periods=len(up), freq="10s")
    df = pd.concat(
        [
            pd.DataFrame({"systime": t, "uuid": "up", "value_bool": up}),
            pd.DataFrame({"systime": t, "uuid": "dn", "value_bool": dn}),
        ],
        ignore_index=True,
    )
    return FlowConstraintEvents(df)


def test_blocked_ends_at_release() -> None:
    # Blocked at 10 s and 20 s, released at 30 s: 20 s blocked, not 10 s.
    flow = _flow([True] * 5, [True, False, False, True, True])
    out = flow.blocked_events(roles=ROLES)
    assert out["duration"].dt.total_seconds().tolist() == [20.0]
    assert out["end"].iloc[0] == pd.Timestamp("2025-01-01 00:00:30")
    old = flow.blocked_events(roles=ROLES, interval_end="last_sample")
    assert old["duration"].dt.total_seconds().tolist() == [10.0]
    assert old["end"].iloc[0] == pd.Timestamp("2025-01-01 00:00:20")


def test_single_sample_blockage_has_duration() -> None:
    flow = _flow([True] * 4, [True, False, True, True])
    assert flow.blocked_events(roles=ROLES)["duration"].dt.total_seconds().tolist() == [
        10.0
    ]
    old = flow.blocked_events(roles=ROLES, interval_end="last_sample")
    assert old["duration"].dt.total_seconds().tolist() == [0.0]
    # min_duration now filters on the real duration.
    assert flow.blocked_events(roles=ROLES, min_duration="5s").shape[0] == 1
    assert (
        flow.blocked_events(
            roles=ROLES, min_duration="5s", interval_end="last_sample"
        ).shape[0]
        == 0
    )


def test_starved_ends_at_next_change_and_open_interval_at_last_sample() -> None:
    # Starved from 10 s to the end of the data (still open at 30 s).
    flow = _flow([True, False, False, False], [True, True, True, True])
    out = flow.starved_events(roles=ROLES)
    assert out["start"].iloc[0] == pd.Timestamp("2025-01-01 00:00:10")
    assert out["end"].iloc[0] == pd.Timestamp("2025-01-01 00:00:30")
    old = flow.starved_events(roles=ROLES, interval_end="last_sample")
    pd.testing.assert_frame_equal(out, old)


def test_analytics_forwards_interval_end() -> None:
    flow = _flow([True] * 5, [True, False, False, True, True])
    new = flow.flow_constraint_analytics(roles=ROLES)
    old = flow.flow_constraint_analytics(roles=ROLES, interval_end="last_sample")
    assert new["total_duration_seconds"].tolist() == [20.0, 0.0]
    assert old["total_duration_seconds"].tolist() == [10.0, 0.0]


def test_interval_end_is_keyword_only_and_validated() -> None:
    flow = _flow([True] * 3, [True, False, True])
    with pytest.raises(ValueError, match="interval_end"):
        flow.blocked_events(roles=ROLES, interval_end="bogus")
    with pytest.raises(ValueError, match="interval_end"):
        flow.starved_events(roles=ROLES, interval_end="bogus")
