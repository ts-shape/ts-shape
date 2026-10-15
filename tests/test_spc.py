import pandas as pd

from ts_shape.events.quality.statistical_process_control import (
    StatisticalProcessControlEvents,
)

T0 = pd.Timestamp("2024-01-01")


def test_spc_rule_1_violation_and_process():
    # Tolerance rows 9, 10, 11 repeated: center 10. Moving ranges
    # 1, 1, 2, 1, 1, 2, ... (29 of them: 10 ones-pairs and 9 twos -> sum
    # 2 * 10 + 9 * 2 = 38), so sigma = (38 / 29) / 1.128 (Montgomery 6.4).
    tol_uuid = "tol"
    act_uuid = "act"
    event_uuid = "evt"

    df = pd.DataFrame(
        {
            "uuid": [tol_uuid] * 30 + [act_uuid] * 5,
            "systime": pd.date_range("2024-01-01", periods=35, freq="min"),
            "value_double": [9.0, 10.0, 11.0] * 10 + [9, 10, 20, 10, 9],
            "is_delta": [True] * 35,
        }
    )

    spc = StatisticalProcessControlEvents(
        df,
        value_column="value_double",
        tolerance_uuid=tol_uuid,
        actual_uuid=act_uuid,
        event_uuid=event_uuid,
    )
    limits = spc.control_limits().iloc[0]
    sigma = (38 / 29) / 1.128
    assert limits["mean"] == 10.0
    assert abs(limits["3sigma_upper"] - (10 + 3 * sigma)) < 1e-12
    assert abs(limits["1sigma_lower"] - (10 - sigma)) < 1e-12

    processed = spc.detect_violations(selected_rules=["rule_1"])
    # Only the 20 is beyond 3 sigma (13.48); 9 and 10 are within.
    assert list(processed["value_double"]) == [20.0]
    assert list(processed["systime"]) == [pd.Timestamp("2024-01-01 00:32")]
    assert list(processed["uuid"]) == [event_uuid]
    assert list(processed["triggered_rule"]) == ["rule_1"]
    assert list(processed["severity"]) == ["critical"]


def test_spc_rule_1_method_flags_points_beyond_3_sigma():
    spc = _spc([10.0, 13.5, 12.9, 6.5, 7.0, 10.0])
    actual = _actual(spc)
    limits = spc.control_limits()  # mean 10, sigma 1
    flagged = spc.rule_1(actual, limits)
    # 13.5 and 6.5 are beyond 13 / 7; 12.9 is inside and 7.0 is on the limit.
    assert list(flagged["value_double"]) == [13.5, 6.5]
    assert flagged["rule_1"].all()
    assert len(actual) == 6  # input not modified


def _spc(values, *, tol_uuid="tol", act_uuid="act", event_uuid="evt"):
    """Build an SPC monitor over ``values``; tolerance 9, 10, 11 = mean 10, sigma 1.

    ``sigma_method="std"``: the sample std of 9, 10, 11 is exactly 1, so the
    rule tests have exact zone edges (the default MR-bar / 1.128 is covered
    in tests/reference/test_kpi_references_quality.py).
    """
    n = len(values)
    tol = pd.DataFrame(
        {
            "uuid": tol_uuid,
            "systime": [T0 - pd.Timedelta(minutes=m) for m in (3, 2, 1)],
            "value_double": [9.0, 10.0, 11.0],
        }
    )
    act = pd.DataFrame(
        {
            "uuid": act_uuid,
            "systime": pd.date_range(T0, periods=n, freq="min"),
            "value_double": [float(v) for v in values],
        }
    )
    df = pd.concat([tol, act], ignore_index=True)
    df["is_delta"] = True
    return StatisticalProcessControlEvents(
        df,
        value_column="value_double",
        tolerance_uuid=tol_uuid,
        actual_uuid=act_uuid,
        event_uuid=event_uuid,
        sigma_method="std",
    )


def _actual(spc):
    df = spc.dataframe
    return df[df["uuid"] == "act"].sort_values("systime").reset_index(drop=True)


def test_spc_rule_4_flags_alternating_series():
    # 24 strictly alternating points: 23 alternating differences. Fourteen
    # points (13 differences) first complete at index 13, then at every
    # later point.
    spc = _spc([0.0, 1.0] * 12)
    flagged = spc.rule_4(_actual(spc))
    assert flagged.index.tolist() == list(range(13, 24))


def test_spc_rule_4_ignores_monotonic_series():
    # Monotonic increasing -> sign(diff) is constant -> no alternation.
    spc = _spc([float(i) for i in range(24)])
    flagged = spc.rule_4(_actual(spc))
    assert flagged.empty


def test_spc_rule_4_process_does_not_crash():
    # Regression: rule_4 used to call Series.shift() on a NumPy array
    # inside rolling.apply(raw=True), raising AttributeError.
    spc = _spc([0.0, 1.0] * 12)
    processed = spc.detect_violations(selected_rules=["rule_4"])
    expected = list(pd.date_range(T0, periods=24, freq="min")[13:])
    assert list(processed["systime"]) == expected
    assert set(processed["triggered_rule"]) == {"rule_4"}
