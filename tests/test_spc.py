import pandas as pd  # type: ignore
from ts_shape.events.quality.statistical_process_control import (
    StatisticalProcessControlRuleBased,
)


def test_spc_rule_1_violation_and_process():
    # Tolerance rows define mean and sigma based on value_column
    tol_uuid = "tol"
    act_uuid = "act"
    event_uuid = "evt"

    df = pd.DataFrame(
        {
            "uuid": [tol_uuid] * 10 + [act_uuid] * 5,
            "systime": pd.date_range("2024-01-01", periods=15, freq="min"),
            "value_double": [10] * 10
            + [9, 10, 20, 10, 9],  # 20 should be a 3σ outlier if sigma small
            "is_delta": [True] * 15,
        }
    )

    spc = StatisticalProcessControlRuleBased(
        df,
        value_column="value_double",
        tolerance_uuid=tol_uuid,
        actual_uuid=act_uuid,
        event_uuid=event_uuid,
    )
    limits = spc.calculate_control_limits()
    assert {"mean", "1sigma_upper", "3sigma_upper"}.issubset(limits.columns)

    processed = spc.process(selected_rules=["rule_1"])
    # Should flag the clear outlier at value 20
    assert event_uuid in processed["uuid"].unique()


def _spc(values, *, tol_uuid="tol", act_uuid="act", event_uuid="evt"):
    """Build an SPC monitor over ``values`` as the actual signal."""
    n = len(values)
    df = pd.DataFrame(
        {
            "uuid": [tol_uuid] * n + [act_uuid] * n,
            "systime": list(pd.date_range("2024-01-01", periods=n, freq="min")) * 2,
            "value_double": [10.0] * n + list(values),
            "is_delta": [True] * (2 * n),
        }
    )
    return StatisticalProcessControlRuleBased(
        df,
        value_column="value_double",
        tolerance_uuid=tol_uuid,
        actual_uuid=act_uuid,
        event_uuid=event_uuid,
    )


def test_spc_rule_4_flags_alternating_series():
    # 24 strictly alternating points -> sign(diff) alternates -> rule 4 fires.
    spc = _spc([0.0, 1.0] * 12)
    actual = spc.dataframe[spc.dataframe["uuid"] == "act"].copy()
    flagged = spc.rule_4(actual)
    assert not flagged.empty


def test_spc_rule_4_ignores_monotonic_series():
    # Monotonic increasing -> sign(diff) is constant -> no alternation.
    spc = _spc([float(i) for i in range(24)])
    actual = spc.dataframe[spc.dataframe["uuid"] == "act"].copy()
    flagged = spc.rule_4(actual)
    assert flagged.empty


def test_spc_rule_4_process_does_not_crash():
    # Regression: rule_4 used to call Series.shift() on a NumPy array
    # inside rolling.apply(raw=True), raising AttributeError.
    spc = _spc([0.0, 1.0] * 12)
    processed = spc.process(selected_rules=["rule_4"])
    assert isinstance(processed, pd.DataFrame)
