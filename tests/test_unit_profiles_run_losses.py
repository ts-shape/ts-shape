"""Mapping-free unit profilers, phase 1: stops, changeovers, start-ups.

Each profiler is checked against the ground truth of the simulators
(:func:`ts_shape.make_plc_cycles` with its stop / changeover / ramp-up
options, :func:`ts_shape.make_line_stops`), plus degenerate inputs.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import ts_shape
from ts_shape._profile_result import ProfileResult, UnitProfilesResult
from ts_shape.events.production._unit_profile import (
    effect_size,
    enrichment,
    potential_register,
    window_labels,
)
from ts_shape.events.production.changeover_profiling import (
    ChangeoverProfileEvents,
    StartupProfileEvents,
)
from ts_shape.events.production.cycle_profiling import CycleProfileEvents
from ts_shape.events.production.stop_profiling import StopProfileEvents
from ts_shape.unit_profiles import profile_units

pytestmark = pytest.mark.filterwarnings("ignore::ts_shape.errors.DataQualityWarning")


@pytest.fixture(scope="module")
def stop_plant() -> tuple[pd.DataFrame, pd.DataFrame, CycleProfileEvents]:
    frame, truth = ts_shape.make_plc_cycles(
        n_cycles=400, seed=3, stop_causes=4, stop_probability=0.08, rampup_cycles=4
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        profile = CycleProfileEvents(frame)
        profile.units()
    return frame, truth, profile


@pytest.fixture(scope="module")
def changeover_plant() -> tuple[pd.DataFrame, pd.DataFrame, CycleProfileEvents]:
    frame, truth = ts_shape.make_plc_cycles(
        n_cycles=600,
        seed=2,
        article_block=40,
        changeover_steps=True,
        rampup_cycles=4,
        stop_causes=3,
        stop_probability=0.05,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        profile = CycleProfileEvents(frame)
        profile.units()
    return frame, truth, profile


def _match(records: pd.DataFrame, truth: pd.DataFrame, column: str) -> pd.DataFrame:
    """Each record with the truth row of the cycle it ends."""
    left = records.sort_values("systime")
    right = truth.assign(systime=truth["end"]).sort_values("systime")
    return pd.merge_asof(
        left,
        right[["systime", column]],
        on="systime",
        direction="nearest",
        tolerance=pd.Timedelta("3s"),
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def test_potential_register() -> None:
    table = pd.DataFrame(
        {"step": ["x"] * 10 + ["y"] * 10, "seconds": [*range(10), *[5] * 10]}
    )
    reg = potential_register(table, keys=["step"])
    assert reg["step"].iloc[0] == "x" and reg["potential_rank"].tolist() == [1, 2]
    assert reg.set_index("step").loc["y", "total_potential_seconds"] == 0


def test_enrichment_and_effect_size() -> None:
    units = [{"alarm", "x"}] * 8 + [{"x"}] * 2
    base = [{"x"}] * 50 + [{"alarm"}] * 1
    out = enrichment(units, base).set_index("label")
    assert bool(out.loc["alarm", "is_significant"])
    assert not bool(out.loc["x", "is_significant"])
    eff = effect_size([10, 11, 12, 10, 11], [14, 15, 13, 14, 16])
    assert (
        eff["shift"] > 2 and eff["cliffs_delta_ratio"] > 0.8 and eff["p_value"] < 0.05
    )


def test_window_labels() -> None:
    events = pd.DataFrame({"ns": [1, 5, 9], "event": ["a", "b", "c"]})
    assert window_labels(events, np.array([0, 4]), np.array([6, 10])) == [
        {"a", "b"},
        {"b", "c"},
    ]


# ---------------------------------------------------------------------------
# Stops
# ---------------------------------------------------------------------------


def test_stop_causes_are_recovered(stop_plant) -> None:
    frame, truth, profile = stop_plant
    stops = StopProfileEvents(frame, cycle_profile=profile)
    window = stops.evaluation_window().iloc[0]
    assert bool(window["evaluable"]) and window["reason"] == "ok"
    matched = _match(stops.stop_records(), truth, "stop_cause")
    expected = {
        "jam": "plc:alarm_1↑",
        "material_low": "plc:alarm_2↑",
        "manual_fix": "plc:mode=2",
        "unknown": "unexplained",
    }
    for cause, signature in expected.items():
        part = matched[matched["stop_cause"] == cause]
        assert len(part) >= 3
        assert (part["signature"] == signature).mean() >= 0.8, cause
    register = stops.stop_signatures().set_index("signature")
    assert {"plc:alarm_1↑", "plc:alarm_2↑", "plc:mode=2"} <= set(register.index)
    alarm2 = register.loc["plc:alarm_2↑"]
    assert 200 < alarm2["mttr_median_seconds"] < 450  # about 5 min stops
    summary = stops.summary()
    assert summary["evaluable"] and 0.5 < summary["explained_share_ratio"] <= 1.0


def test_consequences_are_not_causes(stop_plant) -> None:
    frame, _, profile = stop_plant
    signatures = set(
        StopProfileEvents(frame, cycle_profile=profile).stop_signatures()["signature"]
    )
    assert not any("mold_temp" in s for s in signatures)


def test_line_stops_without_machine_cycle() -> None:
    frame, truth = ts_shape.make_line_stops(duration="48h", seed=0)
    stops = StopProfileEvents(frame)
    assert bool(stops.evaluation_window()["evaluable"].iloc[0])
    records = stops.stop_records()
    records = records[records["stop_class"] == "stop"]
    left = records.sort_values("systime")
    right = truth.assign(systime=truth["end"]).sort_values("systime")
    matched = pd.merge_asof(
        left,
        right[["systime", "cause"]],
        on="systime",
        direction="nearest",
        tolerance=pd.Timedelta("30s"),
    )
    for cause, signature in {
        "web_break": "line:alarm_break↑",
        "die_pressure": "line:alarm_pressure↑",
    }.items():
        part = matched[matched["cause"] == cause]
        assert (part["signature"] == signature).mean() >= 0.8
    assert "line:running↓" not in set(stops.stop_signatures()["signature"])


def test_stops_on_empty_and_quiet_data() -> None:
    empty = StopProfileEvents(pd.DataFrame(columns=["systime", "uuid", "value_double"]))
    assert empty.stop_records().empty
    assert not bool(empty.evaluation_window()["evaluable"].iloc[0])
    frame, _ = ts_shape.make_plc_cycles(n_cycles=120, seed=1, pause_every=0)
    quiet = StopProfileEvents(frame, include_micro_stops=False)
    window = quiet.evaluation_window().iloc[0]
    assert not bool(window["evaluable"]) and window["reason"].startswith(
        "too_few_units"
    )


# ---------------------------------------------------------------------------
# Changeovers and start-ups
# ---------------------------------------------------------------------------


def test_changeover_steps_and_order(changeover_plant) -> None:
    frame, truth, profile = changeover_plant
    co = ChangeoverProfileEvents(frame, cycle_profile=profile)
    window = co.evaluation_window().iloc[0]
    assert bool(window["evaluable"])
    assert window["unit_count"] >= 0.8 * (truth["changeover_seconds"] > 0).sum()
    view = co.changeover_view()
    steps = view["step"].tolist()
    order = ["plc:setup_mode↑", "plc:unlock_tool↑", "plc:lock_tool↑", "plc:setup_mode↓"]
    assert [s for s in steps if s in order] == order
    fixture = view.set_index("step").loc["plc:fixture_idΔ"]
    assert fixture["step_hint"] == "external_possible"
    assert not any("mold_temp" in s for s in steps)
    register = co.changeover_register()
    assert register["total_potential_seconds"].is_monotonic_decreasing
    assert not co.changeover_matrix().empty


def test_changeover_without_variant_is_not_evaluable() -> None:
    frame, _ = ts_shape.make_plc_cycles(n_cycles=150, seed=1, articles=("A",))
    co = ChangeoverProfileEvents(frame)
    window = co.evaluation_window().iloc[0]
    assert not bool(window["evaluable"])
    assert window["reason"].startswith("no_candidate_signal")
    assert co.changeover_records().empty and co.changeover_view().empty


def test_startups_find_the_slow_signal(changeover_plant) -> None:
    frame, _, profile = changeover_plant
    su = StartupProfileEvents(frame, cycle_profile=profile)
    assert bool(su.evaluation_window()["evaluable"].iloc[0])
    signals = su.startup_signals()
    assert signals["signal"].iloc[0] == "plc:mold_temp"
    assert signals.set_index("signal").loc["plc:mold_temp", "last_to_band_ratio"] > 0.5
    records = su.startup_records()
    assert set(records["trigger"]) <= {"pause", "changeover"}
    assert (records["lost_seconds"] >= 0).all()
    assert set(su.startup_register()["trigger"]) == set(records["trigger"])


# ---------------------------------------------------------------------------
# Orchestrator and result types
# ---------------------------------------------------------------------------


def test_profile_units_and_save_load(tmp_path, changeover_plant) -> None:
    frame, _, _ = changeover_plant
    result = profile_units(frame)
    assert {"cycles", "stops", "changeovers", "startups"} <= set(result.results)
    assert result.summary["stops"]["evaluable"]
    result.save(tmp_path)
    loaded = UnitProfilesResult.load(tmp_path)
    assert set(loaded.results) == set(result.results)
    assert isinstance(loaded["stops"], ProfileResult)
    with pytest.raises(ValueError, match="unknown profiler"):
        profile_units(frame, profilers=["stopz"])


def test_cycle_profile_result_is_a_profile_result() -> None:
    from ts_shape.cycle_profile import CycleProfileResult, profile_cycles

    frame, _ = ts_shape.make_plc_cycles(n_cycles=120, seed=1)
    result = profile_cycles(frame)
    assert isinstance(result, CycleProfileResult) and isinstance(result, ProfileResult)
    assert result.kind == "cycles"
