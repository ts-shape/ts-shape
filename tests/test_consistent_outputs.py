"""Consistent outputs: KPI basis columns, seconds, DataFrame returns, run state.

* KPI percentages carry the basis they were computed from (OEE, line
  throughput OEE, downtime, performance loss).
* Every ``*_minutes`` / ``*_s`` duration column has a ``*_seconds`` twin.
* Summary methods return a DataFrame; ``as_frame=False`` returns the legacy
  dict / list with a ``DeprecationWarning``.
* Downtime and shift handover accept a string or a boolean run state.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from ts_shape import (
    BottleneckDetectionEvents,
    DowntimeEvents,
    FlowConstraintEvents,
    LineThroughputEvents,
    MachineStateEvents,
    OEEEvents,
    PerformanceLossEvents,
    SetupTimeEvents,
    ShiftHandoverEvents,
    SteadyStateDetectionEvents,
    TargetEvents,
    ToleranceDeviationEvents,
)
from ts_shape.events.development.recipe_phase_adherence import (
    RecipePhaseAdherenceEvents,
)

T0 = pd.Timestamp("2024-01-01 06:00")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def _state_rows(states: list[bool], freq: str = "1min", start=T0, *, as_bool: bool):
    t = pd.date_range(start, periods=len(states), freq=freq)
    if as_bool:
        return pd.DataFrame({"systime": t, "uuid": "state", "value_bool": states})
    return pd.DataFrame(
        {
            "systime": t,
            "uuid": "state",
            "value_string": ["Running" if s else "Stopped" for s in states],
        }
    )


_PATTERN = ([True] * 40 + [False] * 20) * 6  # six hours, 2/3 running


def _plant(as_bool: bool) -> pd.DataFrame:
    """A state signal (string or bool), a reason code and part counters."""
    state = _state_rows(_PATTERN, as_bool=as_bool)
    t = state["systime"]
    reason = pd.DataFrame(
        {"systime": t, "uuid": "reason", "value_string": "Jam"},
    )
    counter = pd.DataFrame(
        {"systime": t, "uuid": "counter", "value_integer": np.arange(len(t)) * 2}
    )
    ok = counter.assign(uuid="ok", value_integer=np.arange(len(t)) * 2)
    nok = counter.assign(uuid="nok", value_integer=np.arange(len(t)) // 10)
    return pd.concat([state, reason, counter, ok, nok], ignore_index=True)


# ---------------------------------------------------------------------------
# 1. KPI basis
# ---------------------------------------------------------------------------


def _oee_frame() -> pd.DataFrame:
    t = pd.date_range("2024-01-01 06:00", periods=120, freq="1min")
    # A 3 h logging gap after the last reading of the first block.
    t = t.append(pd.date_range("2024-01-01 11:00", periods=60, freq="1min"))
    run = [True] * 90 + [False] * 30 + [True] * 60
    counter = np.arange(len(t)) * 2
    counter[150:] -= counter[149]  # one counter reset
    total = np.arange(len(t)) * 2
    reject = np.arange(len(t)) // 30
    return pd.concat(
        [
            pd.DataFrame({"systime": t, "uuid": "run", "value_bool": run}),
            pd.DataFrame({"systime": t, "uuid": "cnt", "value_integer": counter}),
            pd.DataFrame({"systime": t, "uuid": "tot", "value_integer": total}),
            pd.DataFrame({"systime": t, "uuid": "rej", "value_integer": reject}),
        ],
        ignore_index=True,
    )


def test_oee_carries_the_basis_of_its_percentages():
    oee = OEEEvents(_oee_frame())
    daily = oee.oee(
        "run", "cnt", 30.0, total_uuid="tot", reject_uuid="rej", max_gap="15min"
    )
    avail = oee.availability("run", max_gap="15min")
    perf = oee.performance("cnt", 30.0, run_state_uuid="run", max_gap="15min")
    qual = oee.quality("tot", "rej")
    row = daily.iloc[0]

    for col in ("run_seconds", "planned_seconds", "excluded_seconds"):
        assert row[col] == avail[col].iloc[0]
    for col in ("actual_parts", "ideal_parts"):
        assert row[col] == perf[col].iloc[0]
    for col in ("total_parts", "reject_parts", "good_parts"):
        assert row[col] == qual[col].iloc[0]
    assert row["resets"] == perf["resets"].iloc[0] + qual["resets"].iloc[0]

    # The gap 07:59 -> 11:00 is excluded, not run time.
    assert row["excluded_seconds"] == pytest.approx(3 * 3600 + 60)
    assert row["planned_seconds"] + row["excluded_seconds"] == pytest.approx(
        (pd.Timestamp("2024-01-01 11:59") - T0).total_seconds()
    )
    # The percentages are reproducible from the basis.
    assert row["availability_pct"] == pytest.approx(
        row["run_seconds"] / row["planned_seconds"] * 100
    )
    assert row["performance_pct"] == pytest.approx(
        min(row["actual_parts"] / row["ideal_parts"] * 100, 100.0)
    )
    assert row["quality_pct"] == pytest.approx(
        row["good_parts"] / row["total_parts"] * 100
    )


def test_oee_without_quality_counters_has_no_quality_basis():
    daily = OEEEvents(_oee_frame()).oee("run", "cnt", 30.0, handle_resets=False)
    assert (daily["quality_pct"] == 100.0).all()
    assert daily[["total_parts", "reject_parts", "good_parts"]].isna().all().all()
    assert "resets" not in daily.columns
    assert (daily["excluded_seconds"] == 0).all()  # no max_gap


def test_oee_empty_keeps_basis_columns():
    empty = OEEEvents(_oee_frame().iloc[:0]).oee("run", "cnt", 30.0)
    assert {"run_seconds", "actual_parts", "good_parts", "resets"} <= set(empty.columns)


def test_throughput_oee_carries_run_and_planned_seconds():
    lt = LineThroughputEvents(_oee_frame())
    out = lt.throughput_oee("cnt", window="1h")
    assert (out["planned_seconds"] == out["duration_seconds"]).all()
    assert (out["run_seconds"] == out["duration_seconds"] * out["availability"]).all()
    assert (out["resets"] == lt.count_parts("cnt", window="1h")["resets"]).all()
    plain = lt.throughput_oee("cnt", window="1h", handle_resets=False)
    assert "resets" not in plain.columns
    empty = LineThroughputEvents(_oee_frame().iloc[:0]).throughput_oee("cnt")
    assert list(empty.columns) == list(out.columns)


# ---------------------------------------------------------------------------
# 2. Seconds next to minutes
# ---------------------------------------------------------------------------


def _assert_seconds(df: pd.DataFrame, pairs: dict[str, tuple[str, float]]) -> None:
    cols = list(df.columns)
    for seconds, (legacy, factor) in pairs.items():
        assert cols.index(seconds) == cols.index(legacy) + 1, (seconds, cols)
        np.testing.assert_allclose(
            df[seconds].astype(float), df[legacy].astype(float) * factor
        )


def test_downtime_reports_seconds_and_excluded_time():
    df = _plant(as_bool=False)
    df = df.drop(index=df.index[(df["uuid"] == "state")][100:160])  # 1 h gap
    dt = DowntimeEvents(df)
    by_shift = dt.downtime_by_shift("state", max_gap="10min")
    _assert_seconds(
        by_shift,
        {
            "total_seconds": ("total_minutes", 60),
            "downtime_seconds": ("downtime_minutes", 60),
            "uptime_seconds": ("uptime_minutes", 60),
        },
    )
    assert by_shift["excluded_seconds"].sum() == pytest.approx(61 * 60)
    assert (dt.downtime_by_shift("state")["excluded_seconds"] == 0).all()

    trend = dt.availability_trend("state", window="1h", max_gap="10min")
    _assert_seconds(
        trend,
        {
            "uptime_seconds": ("uptime_minutes", 60),
            "downtime_seconds": ("downtime_minutes", 60),
        },
    )
    assert trend["excluded_seconds"].sum() == pytest.approx(61 * 60)

    reasons = dt.downtime_by_reason("state", "reason")
    _assert_seconds(
        reasons,
        {"total_seconds": ("total_minutes", 60), "avg_seconds": ("avg_minutes", 60)},
    )
    top = dt.top_downtime_reasons("state", "reason")
    _assert_seconds(top, {"total_seconds": ("total_minutes", 60)})


def test_downtime_empty_results_keep_new_columns():
    dt = DowntimeEvents(_plant(as_bool=False).iloc[:0])
    assert "excluded_seconds" in dt.downtime_by_shift("state").columns
    assert "uptime_seconds" in dt.availability_trend("state").columns
    assert "avg_seconds" in dt.downtime_by_reason("state", "reason").columns
    assert "total_seconds" in dt.top_downtime_reasons("state", "reason").columns


def test_performance_loss_reports_seconds_and_run_seconds():
    df = _plant(as_bool=False)
    perf = PerformanceLossEvents(df)
    for out in (
        perf.performance_by_shift("counter", 20.0),
        perf.performance_trend("counter", 20.0, window="1h"),
        perf.slow_periods("counter", 20.0, threshold_pct=100.0),
    ):
        assert not out.empty
        _assert_seconds(
            out,
            {
                "avg_cycle_time_seconds": ("avg_cycle_time_s", 1),
                "loss_seconds": ("loss_minutes", 60),
            },
        )
        np.testing.assert_allclose(
            out["run_seconds"], out["actual_parts"] * out["avg_cycle_time_seconds"]
        )
        assert out["performance_pct"].iloc[0] == pytest.approx(
            min(out["actual_parts"].iloc[0] * 20.0 / out["run_seconds"].iloc[0], 1)
            * 100
        )
    by_shift = perf.performance_by_shift("counter", 20.0)
    _assert_seconds(by_shift, {"target_cycle_time_seconds": ("target_cycle_time_s", 1)})


def test_setup_time_reports_seconds():
    times = pd.date_range("2024-01-01 06:00", periods=12, freq="30min")
    states = (
        ["Running"] * 3 + ["Setup"] * 2 + ["Running"] * 4 + ["Setup"] + ["Running"] * 2
    )
    parts = ["A"] * 4 + ["B"] * 6 + ["C"] * 2
    df = pd.concat(
        [
            pd.DataFrame({"systime": times, "uuid": "st", "value_string": states}),
            pd.DataFrame({"systime": times, "uuid": "part", "value_string": parts}),
        ],
        ignore_index=True,
    )
    setup = SetupTimeEvents(df)
    _assert_seconds(
        setup.setup_by_product("st", "part"),
        {
            "avg_seconds": ("avg_minutes", 60),
            "min_seconds": ("min_minutes", 60),
            "max_seconds": ("max_minutes", 60),
        },
    )
    stats = setup.setup_statistics("st")
    _assert_seconds(
        stats,
        {
            "total_seconds": ("total_minutes", 60),
            "avg_seconds": ("avg_minutes", 60),
            "median_seconds": ("median_minutes", 60),
            "std_seconds": ("std_minutes", 60),
        },
    )
    _assert_seconds(
        setup.setup_trend("st", window="1D"),
        {
            "avg_setup_seconds": ("avg_setup_minutes", 60),
            "total_setup_seconds": ("total_setup_minutes", 60),
        },
    )
    durations = setup.setup_durations("st")
    assert "_dur_s" not in durations.columns
    np.testing.assert_allclose(
        durations["duration_seconds"], durations["duration_minutes"] * 60
    )
    assert "total_seconds" in SetupTimeEvents(df.iloc[:0]).setup_statistics("st")


def test_shift_handover_reports_downtime_seconds():
    report = ShiftHandoverEvents(_plant(as_bool=False)).generate_report(
        "counter", "ok", "nok", "state"
    )
    _assert_seconds(report, {"downtime_seconds": ("downtime_minutes", 60)})


def test_from_shift_data_derives_the_missing_downtime_unit():
    day = pd.Timestamp("2024-01-01").date()
    prod = pd.DataFrame({"date": [day], "shift": ["s1"], "quantity": [10]})
    seconds_only = pd.DataFrame(
        {
            "date": [day],
            "shift": ["s1"],
            "availability_pct": [90.0],
            "downtime_seconds": [600.0],
        }
    )
    out = ShiftHandoverEvents.from_shift_data(prod, downtime_df=seconds_only)
    assert out["downtime_minutes"].iloc[0] == 10.0
    minutes_only = seconds_only.rename(columns={"downtime_seconds": "downtime_minutes"})
    minutes_only["downtime_minutes"] = 10.0
    with pytest.warns(DeprecationWarning, match="downtime_minutes"):
        out = ShiftHandoverEvents.from_shift_data(prod, downtime_df=minutes_only)
    assert out["downtime_seconds"].iloc[0] == 600.0
    assert list(out.columns) == ShiftHandoverEvents.OUTPUT_COLUMNS


def _recipe_df() -> pd.DataFrame:
    t = pd.date_range("2024-01-01", periods=20, freq="1min")
    phases = ["heat"] * 10 + ["hold"] * 10
    return pd.concat(
        [
            pd.DataFrame({"systime": t, "uuid": "phase", "value_string": phases}),
            pd.DataFrame(
                {"systime": t, "uuid": "temp", "value_double": np.linspace(20, 80, 20)}
            ),
        ],
        ignore_index=True,
    )


def test_recipe_spec_accepts_duration_seconds():
    df = _recipe_df()
    with pytest.warns(DeprecationWarning, match="duration_s"):
        legacy = RecipePhaseAdherenceEvents(
            df, "phase", "temp", {"heat": {"duration_s": (0, 60)}}
        ).evaluate()
    new = RecipePhaseAdherenceEvents(
        df, "phase", "temp", {"heat": {"duration_seconds": (0, 60)}}
    ).evaluate()
    assert legacy["criteria_failed"].iloc[0] == ["duration_s"]
    assert new["criteria_failed"].iloc[0] == ["duration_seconds"]
    assert list(legacy["pass"]) == list(new["pass"])
    with pytest.raises(ValueError, match="duration_s"):
        RecipePhaseAdherenceEvents(
            df,
            "phase",
            "temp",
            {"heat": {"duration_s": (0, 60), "duration_seconds": (0, 60)}},
        )


# ---------------------------------------------------------------------------
# 3. DataFrame returns; as_frame=False deprecated
# ---------------------------------------------------------------------------


def _steady():
    t = pd.date_range("2024-01-01", periods=120, freq="10s", tz="Europe/Berlin")
    df = pd.DataFrame({"systime": t, "uuid": "temp", "value_double": 50.0})
    return SteadyStateDetectionEvents(df, "temp")


def _stations():
    t = pd.date_range("2024-01-01", periods=240, freq="1min")
    return pd.concat(
        [
            pd.DataFrame(
                {"systime": t, "uuid": "a", "value_bool": np.arange(240) % 10 != 0}
            ),
            pd.DataFrame(
                {"systime": t, "uuid": "b", "value_bool": np.arange(240) % 2 == 0}
            ),
        ],
        ignore_index=True,
    )


def _tolerance():
    t = pd.date_range("2024-01-01", periods=12, freq="1min")
    df = pd.concat(
        [
            pd.DataFrame({"systime": t[:1], "uuid": "usl", "value_double": [12.0]}),
            pd.DataFrame({"systime": t[:1], "uuid": "lsl", "value_double": [8.0]}),
            pd.DataFrame(
                {
                    "systime": t[1:],
                    "uuid": "x",
                    "value_double": 10 + np.sin(np.arange(11)),
                }
            ),
        ],
        ignore_index=True,
    )
    return ToleranceDeviationEvents(
        df,
        "value_double",
        "value_double",
        actual_uuid="x",
        upper_tolerance_uuid="usl",
        lower_tolerance_uuid="lsl",
        event_uuid="q:cap",
    )


def _flow_df():
    t = pd.date_range("2024-01-01", periods=120, freq="1s")
    up = np.ones(120, dtype=bool)
    dn = np.ones(120, dtype=bool)
    dn[10:14] = False  # 3 s blocked
    dn[40:60] = False  # 19 s blocked
    up[80:90] = False  # 9 s starved
    return pd.concat(
        [
            pd.DataFrame({"systime": t, "uuid": "up", "value_bool": up}),
            pd.DataFrame({"systime": t, "uuid": "dn", "value_bool": dn}),
        ],
        ignore_index=True,
    )


_ROLES = {"upstream_run": "up", "downstream_run": "dn"}

CASES = {
    "steady_state_statistics": (
        _steady,
        lambda o, **kw: o.steady_state_statistics(window="1min", **kw),
    ),
    "throughput_constraint_summary": (
        lambda: BottleneckDetectionEvents(_stations()),
        lambda o, **kw: o.throughput_constraint_summary(["a", "b"], **kw),
    ),
    "flow_constraint_analytics": (
        lambda: FlowConstraintEvents(_flow_df()),
        lambda o, **kw: o.flow_constraint_analytics(roles=_ROLES, **kw),
    ),
    "state_quality_metrics": (
        lambda: MachineStateEvents(_stations(), "a"),
        lambda o, **kw: o.state_quality_metrics(**kw),
    ),
    "target_hit_rate": (
        lambda: TargetEvents(_plant(as_bool=False)),
        lambda o, **kw: o.target_hit_rate("counter", 500, **kw),
    ),
    "capability_indices": (
        _tolerance,
        lambda o, **kw: o.capability_indices(**kw),
    ),
    "highlight_issues": (
        lambda: ShiftHandoverEvents(_plant(as_bool=False)),
        lambda o, **kw: o.highlight_issues(
            counter_uuid="counter",
            ok_counter_uuid="ok",
            nok_counter_uuid="nok",
            state_uuid="state",
            **kw,
        ),
    ),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_summary_methods_return_a_frame(name):
    make, call = CASES[name]
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        frame = call(make())
    assert isinstance(frame, pd.DataFrame)
    assert {"start", "end", "duration_seconds"} <= set(frame.columns)
    assert len(frame) >= 1


@pytest.mark.parametrize("name", sorted(CASES))
def test_as_frame_false_warns_and_returns_the_legacy_object(name):
    make, call = CASES[name]
    with pytest.warns(DeprecationWarning, match="next major release") as record:
        legacy = call(make(), as_frame=False)
    # The warning points at the caller, not into ts_shape's wrappers.
    assert record[0].filename == __file__
    assert isinstance(legacy, (dict, list))


@pytest.mark.parametrize(
    "name",
    [
        "steady_state_statistics",
        "state_quality_metrics",
        "target_hit_rate",
        "capability_indices",
    ],
)
def test_one_row_frame_has_the_dict_keys_as_columns(name):
    make, call = CASES[name]
    obj = make()
    frame = call(obj)
    with pytest.warns(DeprecationWarning):
        legacy = call(obj, as_frame=False)
    assert len(frame) == 1
    row = frame.iloc[0]
    for key, value in legacy.items():
        if isinstance(value, float) and np.isnan(value):
            assert np.isnan(row[key])
        else:
            assert row[key] == value, key


def test_summary_frame_bounds_keep_the_timezone():
    frame = _steady().steady_state_statistics(window="1min")
    assert str(frame["start"].dt.tz) == "Europe/Berlin"
    assert frame["duration_seconds"].iloc[0] == 119 * 10


def test_throughput_constraint_summary_frame_matches_dict():
    obj = BottleneckDetectionEvents(_stations())
    frame = obj.throughput_constraint_summary(["a", "b"])
    with pytest.warns(DeprecationWarning):
        legacy = obj.throughput_constraint_summary(["a", "b"], as_frame=False)
    assert list(frame["uuid"]) == ["a", "b"]
    for _, row in frame.iterrows():
        uid = row["uuid"]
        assert row["bottleneck_count"] == legacy["bottleneck_counts"].get(uid, 0)
        assert row["bottleneck_pct"] == pytest.approx(
            legacy["bottleneck_percentages"].get(uid, 0.0)
        )
        assert row["avg_utilization_pct"] == pytest.approx(
            legacy["avg_utilization_per_station"][uid]
        )
        assert row["most_frequent_bottleneck"] == (
            uid == legacy["most_frequent_bottleneck"]
        )
    empty = BottleneckDetectionEvents(_stations().iloc[:0])
    assert list(empty.throughput_constraint_summary(["a"]).columns) == list(
        frame.columns
    )


def test_flow_constraint_analytics_frame_matches_dict():
    obj = FlowConstraintEvents(_flow_df())
    frame = obj.flow_constraint_analytics(roles=_ROLES).set_index("constraint_type")
    with pytest.warns(DeprecationWarning):
        summary = obj.flow_constraint_analytics(roles=_ROLES, as_frame=False)["summary"]
    assert list(frame.index) == ["blocked", "starved"]
    for kind in ("blocked", "starved"):
        row = frame.loc[kind]
        assert row["event_count"] == summary[f"{kind}_count"]
        assert row["total_duration_seconds"] == pytest.approx(
            summary[f"{kind}_total_duration"].total_seconds()
        )
        breakdown = summary[f"{kind}_severity_breakdown"]
        for level in ("minor", "moderate", "severe"):
            assert row[f"{level}_count"] == breakdown.get(level, 0)
    assert frame["event_count"].sum() == summary["total_constraint_events"]
    assert frame["time_alignment_quality"].mean() == pytest.approx(
        summary["overall_alignment_quality"]
    )


def test_flow_constraint_analytics_applies_its_thresholds():
    """Regression: warning/critical thresholds were accepted but ignored."""
    obj = FlowConstraintEvents(_flow_df())
    default = obj.flow_constraint_analytics(roles=_ROLES).set_index("constraint_type")
    # 3 s and 19 s blocked: minor and moderate under the 5 s / 30 s defaults.
    assert default.loc["blocked", ["minor_count", "moderate_count"]].tolist() == [1, 1]
    strict = obj.flow_constraint_analytics(
        roles=_ROLES, warning_threshold="1s", critical_threshold="10s"
    ).set_index("constraint_type")
    assert strict.loc["blocked", ["moderate_count", "severe_count"]].tolist() == [1, 1]
    with pytest.warns(DeprecationWarning):
        legacy = obj.flow_constraint_analytics(
            roles=_ROLES,
            warning_threshold="1s",
            critical_threshold="10s",
            as_frame=False,
        )
    assert sorted(legacy["blocked_events"]["severity"]) == ["moderate", "severe"]


def test_highlight_issues_frame_matches_list():
    obj = ShiftHandoverEvents(_plant(as_bool=False))
    uuids = dict(
        counter_uuid="counter",
        ok_counter_uuid="ok",
        nok_counter_uuid="nok",
        state_uuid="state",
    )
    frame = obj.highlight_issues(**uuids)
    with pytest.warns(DeprecationWarning):
        legacy = obj.highlight_issues(**uuids, as_frame=False)
    assert (
        frame[["shift", "metric", "value", "threshold", "severity"]].to_dict("records")
        == legacy
    )
    assert frame["start"].notna().all()


def test_empty_summary_frames_keep_their_columns():
    full = _steady().steady_state_statistics(window="1min")
    t = pd.DataFrame(columns=["systime", "uuid", "value_double"])
    empty = SteadyStateDetectionEvents(t, "temp").steady_state_statistics()
    assert list(empty.columns) == list(full.columns)
    assert empty["num_steady_periods"].iloc[0] == 0
    hit = TargetEvents(_plant(as_bool=False).iloc[:0]).target_hit_rate("x", 1)
    assert hit["total_days"].iloc[0] == 0 and pd.isna(hit["start"].iloc[0])


# ---------------------------------------------------------------------------
# 4. Run state as string or boolean
# ---------------------------------------------------------------------------


def _numeric(df: pd.DataFrame) -> pd.DataFrame:
    return df.select_dtypes("number").reset_index(drop=True)


def test_downtime_bool_run_state_matches_string_run_state():
    as_str, as_bool = DowntimeEvents(_plant(False)), DowntimeEvents(_plant(True))
    pd.testing.assert_frame_equal(
        _numeric(as_str.downtime_by_shift("state")),
        _numeric(as_bool.downtime_by_shift("state")),
    )
    pd.testing.assert_frame_equal(
        _numeric(as_str.availability_trend("state", window="1h")),
        _numeric(as_bool.availability_trend("state", window="1h")),
    )
    pd.testing.assert_frame_equal(
        as_str.downtime_by_reason("state", "reason"),
        as_bool.downtime_by_reason("state", "reason"),
    )
    by_shift = as_bool.downtime_by_shift("state")
    assert by_shift["downtime_seconds"].sum() == pytest.approx(
        _PATTERN[:-1].count(False) * 60
    )


def test_downtime_explicit_bool_running_value():
    dt = DowntimeEvents(_plant(True))
    implicit = dt.downtime_by_shift("state")
    explicit = dt.downtime_by_shift("state", running_value=True)
    pd.testing.assert_frame_equal(implicit, explicit)
    # running_value=False: a signal that is True while the machine is down.
    inverted = dt.downtime_by_shift("state", running_value=False)
    assert inverted["uptime_seconds"].iloc[0] == implicit["downtime_seconds"].iloc[0]
    stopped = dt.downtime_by_reason("state", "reason", stopped_value=False)
    assert stopped["total_seconds"].iloc[0] == implicit["downtime_seconds"].iloc[0]


def test_shift_handover_bool_run_state_matches_string_run_state():
    kwargs = dict(
        counter_uuid="counter",
        ok_counter_uuid="ok",
        nok_counter_uuid="nok",
        state_uuid="state",
    )
    as_str = ShiftHandoverEvents(_plant(False)).generate_report(**kwargs)
    as_bool = ShiftHandoverEvents(_plant(True)).generate_report(**kwargs)
    pd.testing.assert_frame_equal(as_str, as_bool)
    assert as_bool["availability_pct"].iloc[0] > 0
