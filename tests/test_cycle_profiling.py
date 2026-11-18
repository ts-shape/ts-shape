"""Acceptance tests of mapping-free cycle time profiling, one per spec ID.

The spec IDs (US-1 ... OUT-1) are those of the cycle time profiling
requirements, listed with their sources in
``docs/modules/production/cycle-profiling.md``. Every test runs on the
synthetic PLC of :func:`ts_shape.make_plc_cycles`, whose ground truth says
which cycle is long (camera NOK), short (aborted), a pause (break) and which
article it ran.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import ts_shape
from ts_shape import CycleProfileEvents, CycleTimeEvents
from ts_shape.utils.frames import to_ns


@pytest.fixture(scope="module")
def plc() -> tuple[pd.DataFrame, pd.DataFrame]:
    return ts_shape.make_plc_cycles(n_cycles=300, seed=7)


@pytest.fixture(scope="module")
def profile(plc) -> CycleProfileEvents:
    frame, _ = plc
    return CycleProfileEvents(frame)


@pytest.fixture(scope="module")
def matched(plc, profile) -> pd.DataFrame:
    """Cycle records joined with the ground truth by start time."""
    _, truth = plc
    records = profile.cycle_records()
    truth = truth.assign(start_ns=to_ns(truth["start"]))
    return records.merge(truth, on="start_ns", how="left", suffixes=("", "_truth"))


# ----------------------------------------------------------------------------
# US-1: find the unknown cycle signal from raw data only
# ----------------------------------------------------------------------------


def test_us1_anchor_found_without_any_uuid(profile):
    table = profile.cycle_anchor_candidates()
    anchor = table[table["is_anchor"]]
    assert len(anchor) == 1
    assert anchor["event"].item() == "plc:part_counter+1"
    assert anchor["rank"].item() == 1


def test_us1_heartbeat_never_becomes_the_anchor(profile):
    table = profile.cycle_anchor_candidates().set_index("event")
    # The alive bit is the most regular signal of all, but nothing repeats in
    # step with it.
    assert table.loc["plc:heartbeat↑", "support_count"] == 0
    assert (
        table.loc["plc:heartbeat↑", "interval_cv_factor"]
        <= table.loc["plc:part_counter+1", "interval_cv_factor"]
    )
    assert not table.loc["plc:heartbeat↑", "is_anchor"]


def test_us1_anchor_uuid_overrides_the_ranking(plc):
    frame, _ = plc
    table = CycleProfileEvents(
        frame, anchor_uuid="plc:clamp_closed"
    ).cycle_anchor_candidates()
    assert table.loc[table["is_anchor"], "source_uuid"].item() == "plc:clamp_closed"


def test_us1_every_cycle_matches_a_true_cycle(matched, plc):
    _, truth = plc
    assert matched["kind"].notna().all()
    # The first counter sample is no increment, so one cycle is not seen.
    assert len(matched) == len(truth) - 1


# ----------------------------------------------------------------------------
# US-2: short and long cycles detected statistically; transferable KPIs
# ----------------------------------------------------------------------------


def test_us2_short_cycles_detected(matched):
    short = matched["cycle_class"] == "short"
    assert short.sum() > 0
    assert (matched.loc[short, "kind"] == "short").all()
    assert (matched.loc[matched["kind"] == "short", "cycle_class"] == "short").all()


def test_us2_long_cycles_detected(matched):
    truth_long = matched["kind"] == "long"
    found = matched["cycle_class"] == "long"
    assert truth_long.sum() > 0
    # No false alarm; robust 3-sigma bounds may miss a borderline re-inspection.
    assert (matched.loc[found, "kind"] == "long").all()
    assert (found & truth_long).sum() >= 0.9 * truth_long.sum()


def test_us2_unusual_cycles_are_the_long_and_short_records(profile):
    unusual = profile.detect_unusual_cycles()
    assert set(unusual["cycle_class"]) == {"long", "short"}
    assert set(unusual["direction"]) == {"above", "below"}
    records = profile.cycle_records()
    assert len(unusual) == records["cycle_class"].isin(["long", "short"]).sum()


def test_us2_classes_do_not_depend_on_the_time_scale(plc):
    """Bounds are relative: the same machine 6x slower gives the same classes."""
    frame, _ = plc
    t0 = frame["systime"].min()
    slow = frame.assign(systime=t0 + (frame["systime"] - t0) * 6)
    fast_classes = CycleProfileEvents(frame).cycle_records()["cycle_class"]
    slow_classes = CycleProfileEvents(slow).cycle_records()["cycle_class"]
    pd.testing.assert_series_equal(fast_classes, slow_classes)


def test_us2_cycle_time_trend_band_is_relative():
    """A +-1 % band labels a 5 s and a 300 s cycle with the same pattern alike."""
    pattern = [1.0, 1.0, 1.003, 0.99, 1.05, 1.0, 0.95, 1.0, 1.0, 1.0]

    def trend(scale: float) -> list[str]:
        times = np.cumsum([0.0] + [p * scale for p in pattern])
        t = pd.Timestamp("2025-01-01") + pd.to_timedelta(times, unit="s")
        df = pd.concat(
            [
                pd.DataFrame(
                    {"systime": t, "uuid": "done", "value_integer": range(len(t))}
                ),
                pd.DataFrame({"systime": t[:1], "uuid": "part", "value_string": ["A"]}),
            ]
        )
        out = CycleTimeEvents(df).cycle_time_trend(
            "part", "done", "A", window_size=1, value_column_trigger="value_integer"
        )
        return out["trend"].astype(str).tolist()

    assert trend(5.0) == trend(300.0)
    # 1.003 is inside the 1 % band (stable), 0.99 -> 1.05 is not.
    assert trend(5.0)[2] == "stable" and trend(5.0)[4] == "degrading"


def test_us2_cycle_time_statistics_has_robust_columns():
    df = pd.concat(
        [
            ts_shape.make_id_signal("part", ["A", "B"], hold=30),
            ts_shape.make_timeseries(["cycle"], n_points=60, value_column="value_bool"),
        ]
    )
    stats = CycleTimeEvents(df).cycle_time_statistics("part", "cycle")
    for column in ("p10_seconds", "p90_seconds", "mad_seconds"):
        assert column in stats.columns
    assert (stats["p10_seconds"] <= stats["median_seconds"]).all()
    assert (stats["median_seconds"] <= stats["p90_seconds"]).all()


def test_us2_cycle_time_events_detects_both_directions():
    seconds = [60, 61, 59, 60, 30, 60, 61, 150, 60, 59, 60]
    t = pd.Timestamp("2025-01-01") + pd.to_timedelta(np.cumsum([0] + seconds), unit="s")
    df = pd.concat(
        [
            pd.DataFrame(
                {"systime": t, "uuid": "done", "value_integer": range(len(t))}
            ),
            pd.DataFrame({"systime": t[:1], "uuid": "part", "value_string": ["A"]}),
        ]
    )
    out = CycleTimeEvents(df).detect_unusual_cycles(
        "part", "done", value_column_trigger="value_integer"
    )
    assert out["cycle_time_seconds"].tolist() == [30.0, 150.0]
    assert out["cycle_class"].tolist() == ["short", "long"]


# ----------------------------------------------------------------------------
# US-3: state information that explains why
# ----------------------------------------------------------------------------


def test_us3_camera_nok_explains_long_cycles(profile):
    inf = profile.detect_cycle_influencers()
    long = inf[inf["comparison"] == "long_vs_normal"]
    assert len(long) <= 10
    top = long.head(4)
    assert "plc:camera_nok" in set(top["source_uuid"])
    nok = long[long["feature"] == "event_count:plc:camera_nok↑"].iloc[0]
    assert nok["direction"] == "higher"
    assert nok["importance_ratio"] > 0.9
    assert long["importance_ratio"].between(0, 1).all()


def test_us3_aborted_steps_explain_short_cycles(profile):
    inf = profile.detect_cycle_influencers()
    short = inf[inf["comparison"] == "short_vs_normal"]
    assert "event_count:plc:step=40" in set(short["feature"])
    assert (
        short.loc[short["feature"] == "event_count:plc:step=40", "direction"] == "lower"
    ).all()


def test_us3_heartbeat_count_is_no_influencer(profile):
    inf = profile.detect_cycle_influencers(top_n=50)
    assert not inf["feature"].str.startswith("event_count:plc:heartbeat").any()


# ----------------------------------------------------------------------------
# DQ-1: one start signal; its recurrence is the stop
# ----------------------------------------------------------------------------


def test_dq1_cycles_are_back_to_back(profile):
    rec = profile.cycle_records()
    assert (
        rec["start_ns"].iloc[1:].to_numpy() == rec["end_ns"].iloc[:-1].to_numpy()
    ).all()
    assert (rec["duration_seconds"] > 0).all()


# ----------------------------------------------------------------------------
# DQ-2: pauses handled separately (kept, not dropped)
# ----------------------------------------------------------------------------


def test_dq2_pauses_classified_not_dropped(matched):
    pauses = matched["kind"] == "pause"
    assert pauses.sum() >= 2
    assert (matched.loc[pauses, "cycle_class"] == "pause").all()
    assert (matched["cycle_class"] == "pause").sum() == pauses.sum()


def test_dq2_max_duration_marks_a_pause(plc):
    frame, _ = plc
    rec = CycleProfileEvents(frame, max_duration="55s").cycle_records()
    assert (rec.loc[rec["duration_seconds"] > 55, "cycle_class"] == "pause").all()


def test_dq2_pauses_are_left_out_of_the_register(profile):
    reg = profile.contribution_register()
    st = profile.cycle_time_statistics().set_index("variant")
    for variant, part in reg.groupby("variant"):
        non_pause = st.loc[variant, "cycle_count"] - st.loc[variant, "pause_count"]
        assert (part["cycle_count"] + part["deviation_cycle_count"] == non_pause).all()


# ----------------------------------------------------------------------------
# DQ-3: 50-100 coherent cycles, otherwise optimize organizationally first
# ----------------------------------------------------------------------------


def test_dq3_too_few_cycles_is_not_evaluable():
    frame, _ = ts_shape.make_plc_cycles(n_cycles=40, seed=2, pause_every=0)
    window = CycleProfileEvents(frame).evaluation_window()
    row = window[window["variant"] == "all"].iloc[0]
    assert not row["evaluable"]
    assert row["reason"] == "too_few_cycles: optimize organizationally first"


def test_dq3_enough_cycles_is_evaluable(profile):
    window = profile.evaluation_window().set_index("variant")
    assert window.loc["all", "evaluable"]
    assert window.loc["all", "reason"] == "ok"
    # Pauses break coherent runs; stitched cycles still count.
    assert (
        window.loc["all", "longest_coherent_cycle_count"]
        < window.loc["all", "cycle_count"]
    )
    assert window.loc["all", "pause_count"] >= 2


def test_dq3_no_anchor_says_so():
    t = pd.date_range("2025-01-01", periods=50, freq="1min")
    df = pd.DataFrame({"systime": t, "uuid": "sensor:flat", "value_double": 1.0})
    window = CycleProfileEvents(df).evaluation_window()
    assert window["reason"].tolist() == ["no_cycle_anchor"]
    assert not window["evaluable"].item()
    assert window["start"].item() == t[0] and window["end"].item() == t[-1]


# ----------------------------------------------------------------------------
# DQ-4: cycle limits derived; user bounds optional (decision: derive)
# ----------------------------------------------------------------------------


def test_dq4_limits_are_derived_and_bounds_only_add(plc):
    frame, _ = plc
    derived = CycleProfileEvents(frame).cycle_time_statistics()
    assert derived["lower_limit_seconds"].notna().all()
    assert (derived["lower_limit_seconds"] < derived["median_seconds"]).all()
    assert (derived["median_seconds"] < derived["upper_limit_seconds"]).all()
    bounded = CycleProfileEvents(frame, min_duration=44.0).cycle_records()
    assert (
        bounded.loc[bounded["duration_seconds"] < 44.0, "cycle_class"] != "normal"
    ).all()


# ----------------------------------------------------------------------------
# LL-1: signal kinds classified; enough data per variant
# ----------------------------------------------------------------------------


def test_ll1_signal_kinds_are_inferred(profile):
    kinds = profile.signal_events().groupby("source_uuid")["signal_kind"].first()
    assert kinds["plc:part_counter"] == "counter"
    assert kinds["plc:step"] == "state"
    assert kinds["plc:article"] == "text"
    assert kinds["plc:pressure"] == "continuous"
    assert kinds["plc:camera_nok"] == "bool"


def test_ll1_results_per_variant(profile, plc):
    _, truth = plc
    window = profile.evaluation_window()
    assert window["variant_uuid"].iloc[0] == "plc:article"
    assert set(window["variant"]) == {"all", "A", "B"}
    st = profile.cycle_time_statistics().set_index("variant")
    assert st.loc["B", "median_seconds"] > st.loc["A", "median_seconds"] + 4
    reg = profile.contribution_register()
    step40 = reg[reg["to_event"] == "plc:step=50"].set_index("variant")[
        "median_seconds"
    ]
    # Article B is 6 s (x noise) slower in step 40, which ends at step 50.
    assert 5.0 < step40["B"] - step40["A"] < 7.0


def test_ll1_variant_matches_truth(matched):
    assert (matched["variant"] == matched["article"]).all()


# ----------------------------------------------------------------------------
# Standard view, segments and the contribution register
# ----------------------------------------------------------------------------


def test_standard_view_is_the_step_chain_in_order(profile):
    view = profile.standard_view()
    for _variant, part in view.groupby("variant"):
        steps = [e for e in part["event"] if e.startswith("plc:step=")]
        assert steps == [f"plc:step={s}" for s in range(10, 90, 10)]
        assert part["median_phase_ratio"].is_monotonic_increasing
        assert not part["event"].str.contains("heartbeat").any()
        assert (part["presence_ratio"] >= 0.9).all()


def test_segments_add_up_to_the_cycle(profile):
    seg = profile.segment_durations()
    rec = profile.cycle_records().set_index("cycle_uuid")
    total = seg.groupby("cycle_uuid")["duration_seconds"].sum()
    np.testing.assert_allclose(
        total.to_numpy(), rec.loc[total.index, "duration_seconds"].to_numpy(), atol=1e-6
    )
    in_sequence = ~rec["sequence_deviation"]
    assert set(total.index) == set(rec.index[in_sequence])


def test_register_variance_contributions_add_up_to_one(profile):
    reg = profile.contribution_register()
    sums = reg.groupby("variant")["variance_contribution_factor"].sum()
    np.testing.assert_allclose(sums.to_numpy(), 1.0, atol=1e-9)


def test_register_potential_and_spread(profile):
    reg = profile.contribution_register()
    np.testing.assert_allclose(
        reg["potential_seconds"], reg["median_seconds"] - reg["p10_seconds"]
    )
    np.testing.assert_allclose(
        reg["spread_seconds"], reg["p90_seconds"] - reg["p10_seconds"]
    )
    shares = reg.groupby("variant")["median_share_ratio"].sum()
    np.testing.assert_allclose(shares.to_numpy(), 1.0)
    assert (
        reg.groupby("variant")["potential_rank"]
        .apply(lambda r: sorted(r) == list(range(1, len(r) + 1)))
        .all()
    )


def test_register_finds_the_disturbed_segments(profile):
    reg = profile.contribution_register()
    a = reg[reg["variant"] == "A"].set_index("segment")
    # The camera re-inspection lands between the first camera pulse and step
    # 60; the conveyor wait in step 60 is bimodal.
    top2 = a["variance_contribution_factor"].nlargest(2).index
    assert "plc:camera_done↓ → plc:step=60" in top2
    assert a.loc["plc:step=60 → plc:step=70", "is_bimodal"]
    assert a.loc["plc:step=60 → plc:step=70", "potential_rank"] <= 2


# ----------------------------------------------------------------------------
# VIZ-1: target cycle time from the data
# ----------------------------------------------------------------------------


def test_viz1_target_is_the_best_demonstrated_cycle(profile):
    st = profile.cycle_time_statistics().set_index("variant")
    rec = profile.cycle_records()
    for variant, row in st.iterrows():
        d = rec.loc[
            (rec["variant"] == variant) & (rec["cycle_class"] != "pause"),
            "duration_seconds",
        ]
        assert row["target_seconds"] == pytest.approx(np.quantile(d, 0.1))
        assert row["potential_seconds"] == pytest.approx(
            row["median_seconds"] - row["target_seconds"]
        )


# ----------------------------------------------------------------------------
# OUT-1: all outputs of a cycle carry the same timestamp
# ----------------------------------------------------------------------------


def test_out1_cycle_record_fields_share_systime(profile):
    rec = profile.cycle_records()
    assert (to_ns(rec["systime"]) == rec["end_ns"].to_numpy()).all()
    assert rec["cycle_uuid"].is_unique
    for column in ("duration_seconds", "cycle_class", "variant", "dominant_segment"):
        assert column in rec.columns
    in_sequence = ~rec["sequence_deviation"]
    assert rec.loc[in_sequence, "dominant_segment"].notna().all()


def test_out1_long_cycles_lose_time_in_the_camera_segment(matched):
    long = matched[matched["kind"] == "long"]
    # The re-inspection lies after the camera started (step 50) and before
    # step 60; where the variant's standard view has the camera pulse, the
    # segment starts there.
    dominant = long["dominant_segment"]
    assert dominant.str.endswith("→ plc:step=60").all()
    assert dominant.str.contains("camera_done|step=50").all()


# ----------------------------------------------------------------------------
# Robustness
# ----------------------------------------------------------------------------


def test_tz_aware_input_gives_the_same_profile(plc):
    frame, _ = plc
    aware = frame.assign(systime=frame["systime"].dt.tz_localize("Europe/Berlin"))
    naive = CycleProfileEvents(frame).cycle_records()
    tz = CycleProfileEvents(aware).cycle_records()
    assert str(tz["systime"].dt.tz) == "Europe/Berlin"
    assert tz["cycle_class"].tolist() == naive["cycle_class"].tolist()
    seg = CycleProfileEvents(aware).segment_durations()
    assert str(seg["start"].dt.tz) == "Europe/Berlin"


def test_empty_input_keeps_columns():
    empty = ts_shape.make_plc_cycles(n_cycles=5, seed=0)[0].iloc[0:0]
    profile = CycleProfileEvents(empty)
    for method in (
        "signal_events",
        "cycle_anchor_candidates",
        "standard_view",
        "segment_durations",
        "contribution_register",
        "cycle_time_statistics",
        "detect_cycle_influencers",
        "cycle_records",
        "detect_unusual_cycles",
    ):
        out = getattr(profile, method)()
        assert out.empty and len(out.columns) > 0, method


def test_invalid_parameters_raise(plc):
    frame, _ = plc
    with pytest.raises(ValueError):
        CycleProfileEvents(frame, pause_factor=1.0)
    with pytest.raises(ValueError):
        CycleProfileEvents(frame, presence_ratio=0.0)
    with pytest.raises(ValueError):
        CycleProfileEvents(frame).detect_cycle_influencers(top_n=0)
