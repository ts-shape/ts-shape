import operator
import pandas as pd  # type: ignore
from ts_shape.events.quality.tolerance_deviation import ToleranceDeviationEvents


def test_tolerance_deviation_events_grouping():
    tol_uuid = "tol"
    act_uuid = "act"
    evt_uuid = "evt"

    df = pd.DataFrame(
        {
            "uuid": [tol_uuid, act_uuid, act_uuid, tol_uuid, act_uuid],
            "systime": pd.to_datetime(
                [
                    "2024-01-01 00:00:00",
                    "2024-01-01 00:01:00",
                    "2024-01-01 00:02:00",
                    "2024-01-01 00:10:00",
                    "2024-01-01 00:11:00",
                ]
            ),
            "value_double": [5.0, 6.0, 4.0, 3.5, 4.0],
            "is_delta": [True, True, True, True, True],
        }
    )

    tde = ToleranceDeviationEvents(
        dataframe=df.copy(),
        tolerance_column="value_double",
        actual_column="value_double",
        tolerance_uuid=tol_uuid,
        actual_uuid=act_uuid,
        event_uuid=evt_uuid,
        compare_func=operator.ge,  # actual >= tolerance
        time_threshold="5min",
    )

    out = tde.process_and_group_data_with_events()
    assert "uuid" in out.columns
    assert (out["uuid"] == evt_uuid).all()
    assert out["is_delta"].all()
