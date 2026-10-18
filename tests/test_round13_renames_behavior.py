"""Round 13: interval ends, shift grouping and canonical output twins.

Hand-computed expectations for the behavior changes that came with the
round-14 renames (the renamed keywords themselves are tested in
``test_deprecated_kwargs_all.py``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ts_shape.events.development.golden_batch import GoldenBatchDeviationEvents
from ts_shape.events.production.micro_stop_detection import MicroStopEvents
from ts_shape.events.production.operating_state import OperatingStateEvents
from ts_shape.events.quality.tolerance_deviation import ToleranceDeviationEvents
from ts_shape.loader.context.context_enricher import ContextEnricher

# ---------------------------------------------------------------------------
# OperatingStateEvents: interval_end
# ---------------------------------------------------------------------------


def _states_frame():
    # low at 0/10/20 s, high at 30/40/50/60 s, low at 70 s (4 low, 4 high)
    t = pd.Timestamp("2024-01-01 06:00", tz="UTC") + pd.to_timedelta(
        np.arange(0, 80, 10), unit="s"
    )
    values = [1.0, 1.1, 0.9, 9.0, 9.2, 8.8, 9.1, 1.2]
    return pd.DataFrame({"systime": t, "uuid": "sig", "value_double": values})


def test_intervals_end_at_the_next_state_by_default():
    ops = OperatingStateEvents(_states_frame(), "sig")
    states = ops.infer_states(n_states=2)
    # [0, 30) low, [30, 70) high, [70, 70] low: the intervals tile the signal.
    assert states["state_label"].tolist() == ["low", "high", "low"]
    assert states["duration_seconds"].tolist() == [30.0, 40.0, 0.0]
    assert (states["end"].iloc[:-1].to_numpy() == states["start"].iloc[1:]).all()
    assert states["duration_seconds"].sum() == 70.0  # first to last sample


def test_last_sample_reproduces_the_old_intervals():
    ops = OperatingStateEvents(_states_frame(), "sig")
    states = ops.infer_states(n_states=2, interval_end="last_sample")
    # Each interval ends at its own last sample: 10 s per gap is in no state.
    assert states["duration_seconds"].tolist() == [20.0, 30.0, 0.0]


def test_state_duration_statistics_with_interval_end():
    ops = OperatingStateEvents(_states_frame(), "sig")
    new = ops.state_duration_statistics(n_states=2).set_index("state_label")
    old = ops.state_duration_statistics(
        n_states=2, interval_end="last_sample"
    ).set_index("state_label")
    assert new.loc["low", "total_duration_seconds"] == 30.0  # 30 + 0
    assert new.loc["high", "total_duration_seconds"] == 40.0
    assert old.loc["low", "total_duration"] == 20.0  # 20 + 0
    assert old.loc["high", "total_duration"] == 30.0
    assert new.loc["low", "mean_duration_seconds"] == 15.0
    assert new.loc["low", "max_duration_seconds"] == 30.0
    assert new.loc["low", "min_duration_seconds"] == 0.0


def test_state_duration_statistics_twins_sit_next_to_the_old_columns():
    stats = OperatingStateEvents(_states_frame(), "sig").state_duration_statistics()
    cols = list(stats.columns)
    for old in ("mean", "std", "min", "max", "total"):
        legacy, canonical = f"{old}_duration", f"{old}_duration_seconds"
        assert cols.index(canonical) == cols.index(legacy) + 1
        pd.testing.assert_series_equal(
            stats[legacy], stats[canonical], check_names=False
        )


def test_dwell_time_violation_uses_the_full_interval():
    ops = OperatingStateEvents(_states_frame(), "sig")
    hit = ops.dwell_time_violation(1, "35s")
    assert hit["duration_seconds"].tolist() == [40.0]
    assert hit["excess_seconds"].tolist() == [5.0]
    assert ops.dwell_time_violation(1, "35s", interval_end="last_sample").empty


def test_unknown_interval_end_raises():
    with pytest.raises(ValueError, match="interval_end"):
        OperatingStateEvents(_states_frame(), "sig").infer_states(interval_end="x")


# ---------------------------------------------------------------------------
# MicroStopEvents: shifts and *_seconds twins
# ---------------------------------------------------------------------------


def _run_signal(stops):
    """Run signal (1 s resolution around each stop) with ``(start, seconds)`` stops."""
    rows = []
    for start, seconds in stops:
        t0 = pd.Timestamp(start)
        rows += [
            (t0 - pd.Timedelta("1min"), True),
            (t0, False),
            (t0 + pd.Timedelta(seconds=seconds), True),
        ]
    df = pd.DataFrame(rows, columns=["systime", "value_bool"])
    return df.assign(uuid="run").sort_values("systime", ignore_index=True)


def test_night_shift_across_midnight_is_one_group():
    # One night shift 22:00-06:00: stops at 23:00 (Jan 1) and 02:00 (Jan 2).
    df = _run_signal([("2024-01-01 23:00", 10), ("2024-01-02 02:00", 20)])
    out = MicroStopEvents(df, "run").micro_stop_patterns(hour_grouping=False)
    assert out["shift"].tolist() == ["shift_3"]
    row = out.iloc[0]
    # One shift with two stops (by calendar date it was two "days" of one).
    assert row["avg_count"] == 2.0
    assert row["avg_lost_time_seconds"] == 30.0


def test_shift_names_default_to_shift_n_and_can_be_passed():
    df = _run_signal([("2024-01-01 07:00", 10), ("2024-01-01 15:00", 5)])
    ms = MicroStopEvents(df, "run")
    default = ms.micro_stop_patterns(hour_grouping=False)
    assert default["shift"].tolist() == ["shift_1", "shift_2"]
    named = ms.micro_stop_patterns(
        hour_grouping=False,
        shift_definitions={
            "morning": ("06:00", "14:00"),
            "afternoon": ("14:00", "22:00"),
            "night": ("22:00", "06:00"),
        },
    )
    assert sorted(named["shift"]) == ["afternoon", "morning"]
    assert named.set_index("shift").loc["morning", "avg_lost_time_seconds"] == 10.0


def test_micro_stop_seconds_twins():
    df = _run_signal([("2024-01-01 07:00", 10), ("2024-01-01 07:30", 5)])
    ms = MicroStopEvents(df, "run")
    for frame, twins in (
        (
            ms.micro_stop_impact(window="1h"),
            {
                "total_run_time": "total_run_time_seconds",
                "total_micro_stop_time": "total_micro_stop_time_seconds",
            },
        ),
        (
            ms.micro_stop_frequency(window="1h"),
            {"total_lost_time": "total_lost_time_seconds"},
        ),
        (ms.micro_stop_patterns(), {"avg_lost_time": "avg_lost_time_seconds"}),
    ):
        cols = list(frame.columns)
        for old, new in twins.items():
            assert cols.index(new) == cols.index(old) + 1
            pd.testing.assert_series_equal(frame[old], frame[new], check_names=False)
    impact = ms.micro_stop_impact(window="1h")
    assert impact["total_micro_stop_time_seconds"].sum() == 15.0


def test_empty_micro_stop_patterns_keep_the_twin():
    df = _run_signal([("2024-01-01 07:00", 600)])  # a 10 min stop: no micro-stop
    out = MicroStopEvents(df, "run").micro_stop_patterns(hour_grouping=False)
    assert list(out.columns) == [
        "shift",
        "avg_count",
        "avg_lost_time",
        "avg_lost_time_seconds",
    ]


# ---------------------------------------------------------------------------
# ContextEnricher: lower_limit / upper_limit
# ---------------------------------------------------------------------------


def _ts():
    return pd.DataFrame(
        {
            "systime": pd.date_range("2024-01-01", periods=2, freq="1min"),
            "uuid": ["a", "b"],
            "value_double": [1.0, 2.0],
        }
    )


def test_enrich_with_tolerances_canonical_table():
    tol = pd.DataFrame(
        {"uuid": ["a", "b"], "lower_limit": [0, 1], "upper_limit": [5, 6]}
    )
    out = ContextEnricher(_ts()).enrich_with_tolerances(tol)
    assert list(out.columns)[-2:] == ["lower_limit", "upper_limit"]
    assert out["upper_limit"].tolist() == [5, 6]


def test_enrich_with_tolerances_former_layout_keeps_its_twins():
    tol = pd.DataFrame({"uuid": ["a", "b"], "low_limit": [0, 1], "high_limit": [5, 6]})
    out = ContextEnricher(_ts()).enrich_with_tolerances(tol)
    assert list(out.columns)[-4:] == [
        "low_limit",
        "lower_limit",
        "high_limit",
        "upper_limit",
    ]
    assert out["lower_limit"].tolist() == out["low_limit"].tolist() == [0, 1]
    assert out["upper_limit"].tolist() == out["high_limit"].tolist() == [5, 6]


def test_enrich_with_tolerances_named_columns():
    tol = pd.DataFrame({"uuid": ["a", "b"], "lo": [0, 1], "hi": [5, 6]})
    out = ContextEnricher(_ts()).enrich_with_tolerances(
        tol, lower_limit_column="lo", upper_limit_column="hi"
    )
    assert list(out.columns)[-4:] == ["lo", "lower_limit", "hi", "upper_limit"]


def test_enrich_with_mapping_named_columns():
    df = pd.DataFrame({"uuid": ["s", "s"], "value_string": ["0", "1"]})
    mapping = pd.DataFrame(
        {"uuid": ["s", "s"], "code": ["0", "1"], "text": ["off", "on"]}
    )
    out = ContextEnricher(df).enrich_with_mapping(
        mapping, value_column_raw="code", value_column_mapped="text"
    )
    assert out["mapped_value"].tolist() == ["off", "on"]


# ---------------------------------------------------------------------------
# ToleranceDeviationEvents.capability_indices: cp / cpk
# ---------------------------------------------------------------------------


def test_capability_indices_lower_case_twins():
    t = pd.date_range("2024-01-01", periods=20, freq="1min")
    rng = np.random.default_rng(1)
    actual = pd.DataFrame(
        {"systime": t, "uuid": "act", "value_double": rng.normal(10, 1, 20)}
    )
    limits = pd.DataFrame(
        {"systime": [t[0], t[0]], "uuid": ["usl", "lsl"], "value_double": [14.0, 6.0]}
    )
    res = ToleranceDeviationEvents(
        pd.concat([limits, actual], ignore_index=True),
        value_column_tolerance="value_double",
        value_column_actual="value_double",
        actual_uuid="act",
        upper_tolerance_uuid="usl",
        lower_tolerance_uuid="lsl",
    ).capability_indices()
    cols = list(res.columns)
    assert cols.index("cp") == cols.index("Cp") + 1
    assert cols.index("cpk") == cols.index("Cpk") + 1
    assert res["cp"].iloc[0] == res["Cp"].iloc[0]
    assert res["cpk"].iloc[0] == res["Cpk"].iloc[0]


# ---------------------------------------------------------------------------
# GoldenBatchDeviationEvents: tz-aware batches
# ---------------------------------------------------------------------------


def test_golden_batch_compares_tz_aware_batches():
    t = pd.date_range("2024-01-01", periods=50, freq="1min", tz="Europe/Berlin")
    golden = pd.DataFrame(
        {"systime": t, "uuid": "temp", "value_double": np.linspace(0, 10, 50)}
    )
    batch = golden.assign(value_double=golden["value_double"] + 1.0)
    res = GoldenBatchDeviationEvents(golden, "temp").compare(batch)
    assert res["start"].iloc[0].tz is not None
    assert res.filter(like="max").iloc[0].iloc[0] == pytest.approx(1.0)
