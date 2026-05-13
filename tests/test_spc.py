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
