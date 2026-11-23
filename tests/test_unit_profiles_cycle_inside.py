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
