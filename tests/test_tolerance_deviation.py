import operator
import warnings

import numpy as np
import pandas as pd
import pytest

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

    with pytest.warns(DeprecationWarning, match="tolerance_uuid is deprecated"):
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

    out = tde.detect_deviations()
    assert "uuid" in out.columns
    assert (out["uuid"] == evt_uuid).all()
    assert out["is_delta"].all()


@pytest.mark.parametrize("sides", ["two", "upper", "single"])
def test_other_signals_do_not_change_the_deviations(sides):
    """detect_deviations reads only the actual and tolerance signals: rows of
    other signals (interleaved, at the same instants, with their own values)
    leave every event, value, dtype and index label unchanged."""
    rng = np.random.default_rng(4)
    t = pd.date_range("2024-01-01", periods=300, freq="10s")
    act = pd.DataFrame(
        {"systime": t, "uuid": "act", "value_double": rng.normal(50, 8, 300)}
    )
    limits = pd.DataFrame(
        {
            "systime": [t[0], t[0], t[120], t[120]],
            "uuid": ["usl", "lsl", "usl", "lsl"],
            "value_double": [60.0, 40.0, 55.0, 45.0],
        }
    )
    stamps = np.unique(np.r_[t[::3], t[5::7]])
    other = pd.DataFrame(
        {
            "systime": stamps,
            "uuid": "other",
            "value_double": rng.normal(0, 100, len(stamps)),
        }
    )
    base = pd.concat([limits, act], ignore_index=True).assign(is_delta=True)
    noisy = pd.concat([limits, other, act], ignore_index=True).assign(is_delta=True)
    noisy.index = noisy.index * 2  # distinct labels; events keep their own
    base.index = noisy.index[(noisy["uuid"] != "other").to_numpy()]
    kwargs = {
        "two": {"upper_tolerance_uuid": "usl", "lower_tolerance_uuid": "lsl"},
        "upper": {"upper_tolerance_uuid": "usl", "tolerance_lag": "15s"},
        "single": {"tolerance_uuid": "usl"},
    }[sides]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        events = [
            ToleranceDeviationEvents(
                frame, "value_double", "value_double", actual_uuid="act", **kwargs
            ).detect_deviations()
            for frame in (base, noisy)
        ]
    assert not events[0].empty
    pd.testing.assert_frame_equal(events[0], events[1], check_exact=True)
