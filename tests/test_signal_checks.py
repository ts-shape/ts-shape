"""Every detector checks its signal arguments the same way."""

import warnings

import pandas as pd
import pytest

from ts_shape.errors import ColumnNotFoundError, DataQualityWarning
from ts_shape.events.quality.change_point import ChangePointEvents


def _frame():
    t = pd.date_range("2024-01-01", periods=20, freq="1min")
    return pd.DataFrame({"systime": t, "uuid": "temp", "value_double": range(20)})


def test_unknown_uuid_in_constructor_warns_and_names_available():
    with pytest.warns(
        DataQualityWarning, match=r"signal_uuid='tmep' not found.*'temp'"
    ):
        ev = ChangePointEvents(_frame(), "tmep")
    assert ev.detect_change_points().empty


def test_unknown_uuid_in_method_warns_once_even_with_nested_calls():
    from ts_shape.events.production.oee_calculator import OEEEvents

    # oee() calls availability() and performance() internally; the user
    # should see one warning for the call they made, not one per sub-call.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        df = _frame().assign(value_integer=range(20), value_bool=True)
        OEEEvents(df).oee("line1:run", "temp", 30.0)
    dq = [w for w in caught if issubclass(w.category, DataQualityWarning)]
    assert len(dq) == 1
    assert "run_state_uuid='line1:run'" in str(dq[0].message)


def test_known_uuid_is_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DataQualityWarning)
        ChangePointEvents(_frame(), "temp").segment_statistics()


def test_missing_uuid_column_raises_column_not_found():
    df = _frame().rename(columns={"uuid": "tag"})
    with pytest.raises(ColumnNotFoundError, match="'uuid'"):
        ChangePointEvents(df, "temp")
    assert issubclass(ColumnNotFoundError, ValueError)


def test_missing_time_column_raises_column_not_found():
    df = _frame().rename(columns={"systime": "ts"})
    with pytest.raises(ColumnNotFoundError, match="time column 'systime'"):
        ChangePointEvents(df, "temp")
    ev = ChangePointEvents(df.rename(columns={"ts": "t"}), "temp", time_column="t")
    assert len(ev.signal) == 20


def test_empty_frame_is_silent():
    empty = _frame().iloc[0:0]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert ChangePointEvents(empty, "anything").detect_change_points().empty
