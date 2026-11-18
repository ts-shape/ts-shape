"""Quality of the cycle profile: benchmark against ground truth and the
statistics, structure, shape and self-check additions.

The benchmark scenarios of :data:`ts_shape.cycle_profile.BENCHMARK_SCENARIOS`
cover the cases the profile must handle without configuration: polled
signals, several parts per cycle, an alternative route, curve-shape faults,
a second independent machine on the same controller and a controller that
reports its own cycle time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import ts_shape
from ts_shape import CycleProfileEvents
from ts_shape.cycle_profile import (
    BENCHMARK_SCENARIOS,
    benchmark_profiles,
    case_log,
    profile_cycles,
)
from ts_shape.events.production.cycle_profiling import (
    _analog_levels,
    _bh_qvalues,
    _phase_locking,
)
from ts_shape.utils.frames import to_ns


@pytest.fixture(scope="module")
def benchmark() -> pd.DataFrame:
    return benchmark_profiles(seeds=(11, 23), n_cycles=250)


def test_benchmark_covers_every_scenario(benchmark):
    assert set(benchmark["scenario"]) == set(BENCHMARK_SCENARIOS)


@pytest.mark.parametrize(
    "column",
    [
        "anchor_ok",
        "route_ok",
        "parts_ok",
        "lane_ok",
        "cycle_time_signal_ok",
    ],
)
def test_benchmark_checks_pass_everywhere(benchmark, column):
    failed = benchmark.loc[~benchmark[column].astype(bool), ["scenario", "seed"]]
    assert failed.empty, failed.to_dict("records")


def test_benchmark_cycles_and_classes_match_the_truth(benchmark):
    assert (benchmark["cycle_match_ratio"] == 1.0).all()
    for cls in ("long", "short", "pause"):
        for kind in ("precision", "recall"):
            values = benchmark[f"{cls}_{kind}_ratio"].dropna()
            assert (values == 1.0).all(), (cls, kind, values.tolist())


def test_benchmark_durations_are_exact_or_within_one_poll(benchmark):
    polled = benchmark["scenario"] == "polled"
    assert (benchmark.loc[~polled, "duration_mae_seconds"] < 1e-6).all()
    assert (benchmark.loc[polled, "duration_mae_seconds"] < 2.0).all()


def test_benchmark_finds_shape_faults(benchmark):
    shape = benchmark[benchmark["scenario"] == "shape_faults"]
    assert (shape["shape_recall_ratio"] >= 0.8).all()
    assert (shape["shape_precision_ratio"] >= 0.8).all()


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def plc() -> pd.DataFrame:
    frame, _ = ts_shape.make_plc_cycles(n_cycles=300, seed=7)
    return frame


def test_register_intervals_bracket_the_estimates(plc):
    reg = CycleProfileEvents(plc).contribution_register()
    pot = reg.dropna(subset=["potential_ci_low_seconds"])
    assert len(pot) > 0
    assert (pot["potential_ci_low_seconds"] <= pot["potential_ci_high_seconds"]).all()
    tol = 1e-9
    share = reg.dropna(subset=["variance_contribution_ci_low_factor"])
    assert (
        share["variance_contribution_ci_low_factor"]
        <= share["variance_contribution_ci_high_factor"] + tol
    ).all()
    # the camera re-inspection step carries most of the variance, surely
    top = (
        reg[reg["variant"] == "A"].sort_values("variance_contribution_factor").iloc[-1]
    )
    assert top["variance_contribution_ci_low_factor"] > 0.5


def test_register_intervals_are_repeatable(plc):
    a = CycleProfileEvents(plc).contribution_register()
    b = CycleProfileEvents(plc).contribution_register()
    pd.testing.assert_frame_equal(a, b)


def test_influencers_carry_fdr_corrected_significance(plc):
    inf = CycleProfileEvents(plc).detect_cycle_influencers()
    assert inf["p_value"].between(0, 1).all()
    assert (inf["q_value"] >= inf["p_value"] - 1e-12).all()
    long = inf[inf["comparison"] == "long_vs_normal"]
    nok = long[long["feature"] == "event_count:plc:camera_nok↑"].iloc[0]
    assert nok["is_significant"] and nok["q_value"] < 1e-6


def test_bh_qvalues_match_the_textbook_example():
    p = np.array([0.01, 0.04, 0.03, 0.005, np.nan])
    q = _bh_qvalues(p)
    np.testing.assert_allclose(q[:4], [0.02, 0.04, 0.04, 0.02])
    assert np.isnan(q[4])


def test_analog_features_use_a_comparable_window(plc):
    """Analog means cover the variant's best cycle time from the cycle start."""
    det = CycleProfileEvents(plc)
    cycles = det._cycles()
    features, meta = det._features(cycles)
    col = next(
        c
        for c, (uuid, _) in meta.items()
        if uuid == "plc:pressure" and c.startswith("mean")
    )
    body = cycles[cycles["cycle_class"] != "pause"]
    p10 = body.groupby("variant")["duration_seconds"].quantile(0.10)
    readings = plc[plc["uuid"] == "plc:pressure"]
    ns = to_ns(readings["systime"])
    values = readings["value_double"].to_numpy()
    for idx in body.index[5:10]:
        row = cycles.loc[idx]
        window = min(p10[row["variant"]], row["duration_seconds"])
        end = row["start_ns"] + int(window * 1e9)
        inside = (ns >= row["start_ns"]) & (ns < end)
        assert features.loc[idx, col] == pytest.approx(values[inside].mean())


# ----------------------------------------------------------------------------
# Event abstraction and cycle discovery
# ----------------------------------------------------------------------------


def test_analog_levels_are_learned():
    rng = np.random.default_rng(0)
    values = np.r_[
        rng.normal(5, 1, 400), rng.normal(90, 1, 200), rng.normal(180, 1, 400)
    ]
    centers = _analog_levels(values, "auto")
    assert len(centers) == 3
    np.testing.assert_allclose(centers, [5, 90, 180], atol=1.0)
    assert len(_analog_levels(values, 2)) == 2


def test_analog_levels_auto_gives_level_events(plc):
    ev = CycleProfileEvents(plc, analog_levels="auto").signal_events()
    levels = sorted(ev.loc[ev["source_uuid"] == "plc:pressure", "event"].unique())
    assert levels == ["plc:pressure≈180", "plc:pressure≈4.97"]
    with pytest.raises(ValueError):
        CycleProfileEvents(plc, analog_levels=1)


def test_phase_locking_separates_locked_and_drifting_events():
    anchor = np.arange(0, 100) * 1_000
    locked = anchor[:-1] + 300
    drifting = np.arange(0, 88) * 1_130 + 50
    assert _phase_locking(anchor, locked) > 0.99
    assert _phase_locking(anchor, drifting) < 0.3


def test_anchor_candidates_report_coverage(plc):
    table = CycleProfileEvents(plc).cycle_anchor_candidates()
    assert table["coverage_ratio"].between(0, 1).all()
    assert table.loc[table["is_anchor"], "coverage_ratio"].item() > 0.8


# ----------------------------------------------------------------------------
# Structure, shapes, self-check, export
# ----------------------------------------------------------------------------


def test_result_has_the_structure_tables(plc):
    t = profile_cycles(plc).tables
    for name in (
        "recurring_events",
        "sequence_variants",
        "cycle_sequences",
        "process_graph",
        "parts_per_cycle",
        "parallel_lanes",
        "curve_bands",
        "curve_scores",
        "curve_flagged",
        "cycle_time_check",
    ):
        assert name in t, name
    graph = t["process_graph"]
    assert (graph["source"] == "▶ cycle start").any()
    assert (graph["target"] == "■ cycle end").any()
    assert graph["share_ratio"].between(0, 1).all()


def test_confidence_summary(plc):
    conf = profile_cycles(plc).summary["confidence"]
    assert conf["anchor_support_count"] >= 4
    assert conf["main_sequence_share_ratio"] > 0.9
    assert conf["parts_per_cycle"] == 1.0
    assert conf["independent_lanes"] == []
    assert conf["cycle_time_signal"] is None


def test_case_log_is_pm4py_ready(plc):
    t = profile_cycles(plc).tables
    log = case_log(t["cycle_events"], t["cycle_records"])
    assert list(log.columns[:3]) == [
        "case:concept:name",
        "concept:name",
        "time:timestamp",
    ]
    assert log["time:timestamp"].is_monotonic_increasing
    assert log["case:concept:name"].nunique() == len(t["cycle_records"])
