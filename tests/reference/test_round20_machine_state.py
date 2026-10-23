"""Round 20: text run states in MachineStateEvents."""

import warnings

import pandas as pd
import pytest

from ts_shape import MachineStateEvents
from ts_shape.errors import DataQualityWarning


def _text_frame() -> pd.DataFrame:
    t = pd.date_range("2025-01-01 08:00", periods=4, freq="10min")
    return pd.DataFrame(
        {
            "systime": t,
            "uuid": "state",
            "value_string": ["RUN", "IDLE", "RUN", "IDLE"],
        }
    )


def test_text_states_without_mapping_warn() -> None:
    with pytest.warns(DataQualityWarning, match="run_states"):
        MachineStateEvents(_text_frame(), "state", value_column="value_string")


def test_run_states_maps_text_values() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", DataQualityWarning)
        ev = MachineStateEvents(
            _text_frame(), "state", value_column="value_string", run_states={"RUN"}
        )
    out = ev.detect_run_idle()
    assert out["state"].tolist() == ["run", "idle", "run", "idle"]
    assert out["duration_seconds"].tolist() == [600.0, 600.0, 600.0, 0.0]


def test_run_states_keeps_missing_unknown() -> None:
    df = _text_frame()
    df.loc[1, "value_string"] = None
    ev = MachineStateEvents(
        df, "state", value_column="value_string", run_states={"RUN"}
    )
    out = ev.detect_run_idle()
    # 08:10-08:20 has no known state: in neither a run nor an idle interval.
    assert out["duration_seconds"].sum() == 1200.0


def test_bool_states_do_not_warn() -> None:
    t = pd.date_range("2025-01-01", periods=3, freq="1min")
    df = pd.DataFrame({"systime": t, "uuid": "r", "value_bool": [True, False, True]})
    with warnings.catch_warnings():
        warnings.simplefilter("error", DataQualityWarning)
        MachineStateEvents(df, "r")
