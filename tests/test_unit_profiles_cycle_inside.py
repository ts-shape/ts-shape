"""Mapping-free unit profilers, phases 3 and 4: handshakes, curves, settings."""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

import ts_shape
from ts_shape.events.production.curve_signature import CurveSignatureEvents
from ts_shape.events.production.handshake_profiling import HandshakeProfileEvents
from ts_shape.events.production.setting_impact import SettingImpactEvents

pytestmark = pytest.mark.filterwarnings("ignore::ts_shape.errors.DataQualityWarning")


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


def test_handshake_pair_found() -> None:
    frame, truth = ts_shape.make_plc_cycles(
        n_cycles=300, seed=2, handshake_partner=True
    )
    det = _quiet(HandshakeProfileEvents, frame)
    pairs = det.handshake_pairs()
    hit = pairs[
        (pairs["request_uuid"] == "plc:req_conveyor")
        & (pairs["answer_uuid"] == "plc:ack_conveyor")
    ]
    assert len(hit) >= 1
    row = hit.iloc[0]
    assert row["pair_kind"] == "four_phase"
    assert row["waits_on"] == "partner"
    assert row["median_delay_seconds"] == pytest.approx(
        truth["handshake_wait_seconds"].median(), rel=0.2
    )
    register = det.handshake_register()
    assert set(register["pair"]) == set(pairs["pair"])
    assert det.summary()["evaluable"]


def test_no_handshake_partner() -> None:
    frame, _ = ts_shape.make_plc_cycles(n_cycles=150, seed=1)
    pairs = _quiet(HandshakeProfileEvents, frame).handshake_pairs()
    assert not pairs["request_uuid"].astype(str).str.contains("conveyor").any()


@pytest.mark.parametrize("seed", [5, 6])
def test_curve_patterns_match_faults(seed: int) -> None:
    frame, truth = ts_shape.make_plc_cycles(
        n_cycles=400,
        seed=seed,
        articles=("A",),
        curve_faults={"low_plateau": 0.04, "late_rise": 0.04},
    )
    det = _quiet(CurveSignatureEvents, frame)
    scores = det.curve_cycle_scores()
    scores = scores[scores["source_uuid"] == "plc:pressure"]
    merged = pd.merge_asof(
        scores.sort_values("systime"),
        truth.assign(systime=truth["end"])[["systime", "curve_fault"]].sort_values(
            "systime"
        ),
        on="systime",
        direction="nearest",
        tolerance=pd.Timedelta("2s"),
    )
    fault = merged["curve_fault"].fillna("").astype(str) != ""
    flagged = merged["is_anomaly"].astype(bool)
    assert (flagged & ~fault).sum() <= 1  # (almost) no false alarms
    assert (flagged & fault).sum() >= 0.8 * fault.sum()
    patterns = det.curve_patterns()
    assert len(patterns) >= 2
    # every pattern is dominated by one fault type
    for pid in patterns["pattern_id"]:
        kinds = merged.loc[merged["pattern_id"] == pid, "curve_fault"].fillna("")
        assert kinds.value_counts(normalize=True).iloc[0] >= 0.8


@pytest.mark.parametrize("seed", [1, 3])
def test_setting_impact_sign_and_confidence(seed: int) -> None:
    frame, _ = ts_shape.make_plc_cycles(
        n_cycles=400,
        seed=seed,
        setting_changes=[(150, 120.0), (280, 90.0)],
        energy=True,
    )
    det = _quiet(SettingImpactEvents, frame)
    changes = det.setting_changes()
    assert list(changes["source_uuid"]) == ["plc:speed_override"] * 2
    impact = det.setting_impact()
    cycle = impact[impact["metric"] == "cycle_seconds"].sort_values("change_ns")
    faster, slower = cycle.iloc[0], cycle.iloc[1]
    assert faster["shift"] < 0 and slower["shift"] > 0
    assert set(cycle["confidence"]) == {"high"}
    energy = impact[impact["metric"] == "energy plc:energy_kwh"]
    assert len(energy) == 2
    register = det.setting_register().iloc[0]
    assert register["best_value_observed"] == 120.0
    assert register["sensitivity_seconds_per_unit"] < 0
    assert det.summary()["best_cycle_time_change"].endswith("100 -> 120")


def test_setting_impact_without_settings() -> None:
    frame, _ = ts_shape.make_plc_cycles(n_cycles=150, seed=1)
    det = _quiet(SettingImpactEvents, frame)
    window = det.evaluation_window().iloc[0]
    assert not bool(window["evaluable"])
    assert window["reason"].startswith("no_candidate_signal")
    assert det.setting_impact().empty and det.setting_register().empty


def test_profile_units_runs_new_profilers(tmp_path) -> None:
    frame, _ = ts_shape.make_plc_cycles(
        n_cycles=200, seed=2, handshake_partner=True, setting_changes=[(100, 120.0)]
    )
    result = _quiet(
        ts_shape.profile_units,
        frame,
        profilers=("handshakes", "curves", "settings"),
    )
    for name in ("handshakes", "curves", "settings"):
        assert name in result.results
    assert "setting_impact" in result["settings"].tables
    result.save(tmp_path)
    loaded = type(result).load(tmp_path)
    assert set(loaded.results) == set(result.results)


def test_curve_signal_that_starts_mid_recording() -> None:
    frame, truth = ts_shape.make_plc_cycles(
        n_cycles=300, seed=5, articles=("A",), curve_faults={"low_plateau": 0.05}
    )
    late = truth["start"].iloc[60]
    frame = frame[~((frame["uuid"] == "plc:pressure") & (frame["systime"] < late))]
    summary = _quiet(CurveSignatureEvents, frame).summary()
    assert summary["evaluable"] and summary["pattern_count"] >= 1


def test_setting_changed_with_another_setting() -> None:
    frame, truth = ts_shape.make_plc_cycles(
        n_cycles=400, seed=1, setting_changes=[(150, 120.0)]
    )
    change = frame.loc[
        (frame["uuid"] == "plc:speed_override") & (frame["value_double"] == 120.0),
        "systime",
    ].iloc[0]
    other = pd.DataFrame(
        {
            "systime": [frame["systime"].min(), change],
            "uuid": "plc:temperature_set",
            "value_double": [200.0, 210.0],
        }
    )
    frame = pd.concat([frame, other], ignore_index=True)
    impact = _quiet(SettingImpactEvents, frame).setting_impact()
    cycle = impact[impact["metric"] == "cycle_seconds"]
    assert set(cycle["setting"]) == {"plc:speed_override", "plc:temperature_set"}
    assert cycle["confounders"].str.contains("other setting").all()
    assert (cycle["confidence"] != "high").all()


def test_setting_recipe_parameter_is_not_tested() -> None:
    frame, truth = ts_shape.make_plc_cycles(n_cycles=400, seed=2, article_block=50)
    # A setting that changes exactly with the article: a recipe parameter.
    starts = truth.loc[
        truth["article"] != truth["article"].shift(), ["start", "article"]
    ]
    values = starts["article"].map(
        {a: 100.0 + 10 * i for i, a in enumerate(sorted(starts["article"].unique()))}
    )
    recipe = pd.DataFrame(
        {"systime": starts["start"], "uuid": "plc:recipe_speed", "value_double": values}
    )
    impact = _quiet(
        SettingImpactEvents, pd.concat([frame, recipe], ignore_index=True)
    ).setting_impact()
    rows = impact[impact["setting"] == "plc:recipe_speed"]
    assert len(rows) and rows["recipe_parameter"].all()
    assert rows["q_value"].isna().all() and (rows["confidence"] == "low").all()


def test_handshake_wait_resolution_rule(monkeypatch) -> None:
    frame, _ = ts_shape.make_plc_cycles(n_cycles=200, seed=2, handshake_partner=True)
    det = _quiet(HandshakeProfileEvents, frame)
    pair = "plc:req_conveyor↑ → plc:ack_conveyor↑"
    wait = float(
        det.handshake_pairs().set_index("pair").at[pair, "median_delay_seconds"]
    )
    for resolution, ok in ((0.45 * wait, True), (0.55 * wait, False)):
        monkeypatch.setattr(det, "_resolution", lambda r=resolution: r)
        window = det.evaluation_window().set_index("scope")
        assert bool(window.at[pair, "evaluable"]) is ok, resolution
