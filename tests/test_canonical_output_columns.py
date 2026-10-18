"""Canonical output column names next to their deprecated twins.

The twins are plain copies: reading either name does not warn. The
cycle-quality tolerance is a percent; the former fraction still works with a
``DeprecationWarning``.
"""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

from ts_shape import (
    DutyCycleEvents,
    LineThroughputEvents,
    StatisticalProcessControlEvents,
)

T0 = pd.Timestamp("2024-01-01 06:00")


def _cycles(seconds: list[float]) -> pd.DataFrame:
    """Rising edges of a boolean trigger with the given gaps between them."""
    edges = [T0]
    for s in seconds:
        edges.append(edges[-1] + pd.Timedelta(seconds=s))
    rows = []
    for t in edges:
        rows.append({"systime": t, "uuid": "c", "value_bool": True})
        rows.append(
            {"systime": t + pd.Timedelta(seconds=1), "uuid": "c", "value_bool": False}
        )
    return pd.DataFrame(rows)


#: 100 s expected: deviations 0, 5 %, 20 %, 50 %.
CYCLES = _cycles([100.0, 105.0, 120.0, 150.0])


def test_deviation_pct_is_percent_and_ratio_is_fraction():
    out = LineThroughputEvents(CYCLES).cycle_quality_check(
        "c", expected_cycle_time=100.0
    )
    assert out["deviation_pct"].tolist() == pytest.approx([0.0, 5.0, 20.0, 50.0])
    assert out["deviation_ratio"].tolist() == pytest.approx([0.0, 0.05, 0.2, 0.5])
    assert out["expected_time_seconds"].tolist() == [100.0] * 4
    cols = list(out.columns)
    assert cols.index("expected_time_seconds") == cols.index("expected_time") + 1


def test_default_tolerance_keeps_the_former_bins():
    # 10 % tolerance: good up to 10 %, warning up to 25 %, critical above.
    out = LineThroughputEvents(CYCLES).cycle_quality_check(
        "c", expected_cycle_time=100.0
    )
    assert out["quality_flag"].tolist() == ["good", "good", "warning", "critical"]
    assert out["is_valid"].tolist() == [True, True, False, False]


def test_quality_flag_bins_follow_the_tolerance():
    # 4 % tolerance: good up to 4 %, warning up to 10 %, critical above.
    out = LineThroughputEvents(CYCLES).cycle_quality_check(
        "c", expected_cycle_time=100.0, tolerance_pct=4.0
    )
    assert out["quality_flag"].tolist() == ["good", "warning", "critical", "critical"]
    assert out["is_valid"].tolist() == [True, False, False, False]


def test_tolerance_ratio_is_the_same_tolerance_as_a_fraction():
    lt = LineThroughputEvents(CYCLES)
    by_pct = lt.cycle_quality_check("c", expected_cycle_time=100.0, tolerance_pct=25.0)
    by_ratio = lt.cycle_quality_check(
        "c", expected_cycle_time=100.0, tolerance_ratio=0.25
    )
    pd.testing.assert_frame_equal(by_pct, by_ratio)
    with pytest.raises(ValueError, match="either"):
        lt.cycle_quality_check("c", tolerance_pct=25.0, tolerance_ratio=0.25)


def test_fractional_tolerance_pct_still_works_with_a_deprecation_warning():
    lt = LineThroughputEvents(CYCLES)
    with pytest.warns(DeprecationWarning, match="tolerance_pct=25"):
        legacy = lt.cycle_quality_check(
            "c", expected_cycle_time=100.0, tolerance_pct=0.25
        )
    current = lt.cycle_quality_check("c", expected_cycle_time=100.0, tolerance_pct=25.0)
    pd.testing.assert_frame_equal(legacy, current)


def test_reading_a_twin_does_not_warn():
    t = pd.date_range(T0, periods=4, freq="15min")
    df = pd.DataFrame({"systime": t, "uuid": "m", "value_bool": [True, False] * 2})
    out = DutyCycleEvents(df, "m").duty_cycle_by_window(window="1h")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert out["on_time"].tolist() == out["on_time_seconds"].tolist()
        assert out["off_time"].tolist() == out["off_time_seconds"].tolist()


def test_spc_detect_violations_uses_the_rule_violation_names():
    t = pd.date_range(T0, periods=41, freq="1min")
    ref = pd.DataFrame(
        {"systime": t[:20], "uuid": "ref", "value_double": [10.0, 11.0] * 10}
    )
    act = pd.DataFrame(
        {"systime": t[20:], "uuid": "act", "value_double": [10.5] * 20 + [40.0]}
    )
    spc = StatisticalProcessControlEvents(
        pd.concat([ref, act]), "value_double", "ref", "act", "quality:spc"
    )
    out = spc.detect_violations(selected_rules=["rule_1"])
    by_rule = spc.detect_rule_violations(selected_rules=["rule_1"])
    assert {"value", "rule"} <= set(out.columns)
    assert out["rule"].tolist() == out["triggered_rule"].tolist() == ["rule_1"]
    assert out["value"].tolist() == by_rule["value"].tolist() == [40.0]
    cols = list(out.columns)
    assert cols.index("value") == cols.index("value_double") + 1
    assert cols.index("rule") == cols.index("triggered_rule") + 1
    plain = spc.detect_violations(selected_rules=["rule_1"], include_severity=False)
    assert "value" in plain.columns and "rule" not in plain.columns
