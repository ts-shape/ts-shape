"""Mapping-free control-loop profiler against the loop simulator's truth."""

from __future__ import annotations

import pandas as pd
import pytest

import ts_shape
from ts_shape.events.engineering.control_loop_profiling import ControlLoopProfileEvents

EXPECTED = {"good": "ok", "oscillating": "oscillating", "sluggish": "sluggish"}


@pytest.mark.parametrize("seed", [1, 2, 5])
def test_loops_paired_and_diagnosed(seed: int) -> None:
    frame, truth = ts_shape.make_control_loops(seed=seed)
    det = ControlLoopProfileEvents(frame)
    pairs = det.loop_pairs().set_index("setpoint_uuid")
    assert set(pairs.index) == set(truth["setpoint_uuid"])  # no distractor
    register = det.loop_register().set_index("setpoint_uuid")
    for row in truth.itertuples(index=False):
        assert pairs.loc[row.setpoint_uuid, "actual_uuid"] == row.actual_uuid
        assert pairs.loc[row.setpoint_uuid, "output_uuid"] == row.output_uuid
        assert register.loc[row.setpoint_uuid, "diagnosis"] == EXPECTED[row.quality]
    good = truth.loc[truth["quality"] == "good", "setpoint_uuid"].iloc[0]
    assert register.loc[good, "health_rank"] == len(truth)  # the best one
    steps = det.loop_steps()
    assert set(steps["loop"]) == set(truth["setpoint_uuid"])
    assert det.summary()["evaluable"]


def test_too_few_steps_and_no_loops() -> None:
    frame, _ = ts_shape.make_control_loops(
        seed=1, loops=("good",), duration="40min", mean_step_minutes=30.0
    )
    det = ControlLoopProfileEvents(frame, min_steps=50)
    window = det.evaluation_window()
    assert not window["evaluable"].any()
    assert set(window["reason"]) == {"too_few_units"}
    assert set(det.loop_register()["diagnosis"]) == {"unknown"}
    frame, _ = ts_shape.make_control_loops(seed=1, loops=(), duration="1h")
    det = ControlLoopProfileEvents(frame)
    assert det.loop_pairs().empty and det.loop_register().empty
    assert det.evaluation_window()["reason"].iloc[0].startswith("no_candidate")
    with pytest.raises(ValueError, match="unknown loop quality"):
        ts_shape.make_control_loops(loops=("bad",))


def _on_change(frame, deadband: float = 0.3):
    """Keep a reading only when it moved by more than ``deadband`` (setpoints
    and held settings: on every change)."""
    parts = []
    for uuid, part in frame.groupby("uuid", sort=False):
        band = 0.0 if uuid.endswith(("_sp", "_set")) else deadband
        values = part["value_double"].to_numpy()
        keep, last = [], None
        for v in values:
            take = last is None or abs(v - last) > band
            keep.append(take)
            if take:
                last = v
        parts.append(part[keep])
    return pd.concat(parts).sort_values("systime").reset_index(drop=True)


@pytest.mark.parametrize("seed", [1, 9])
def test_loops_logged_on_change(seed: int) -> None:
    frame, truth = ts_shape.make_control_loops(seed=seed)
    frame = _on_change(frame)
    register = (
        ControlLoopProfileEvents(frame).loop_register().set_index("setpoint_uuid")
    )
    for row in truth.itertuples(index=False):
        assert register.loc[row.setpoint_uuid, "actual_uuid"] == row.actual_uuid
        assert register.loc[row.setpoint_uuid, "output_uuid"] == row.output_uuid
        assert register.loc[row.setpoint_uuid, "diagnosis"] == EXPECTED[row.quality]
