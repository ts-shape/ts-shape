"""A missing value column is reported as ColumnNotFoundError, not KeyError."""

import pandas as pd
import pytest

from ts_shape.errors import ColumnNotFoundError
from ts_shape.events.production.machine_state import MachineStateEvents
from ts_shape.events.quality.multivariate_spc import MultivariateSPCEvents
from ts_shape.events.quality.outlier_detection import OutlierDetectionEvents

T = pd.date_range("2024-01-01", periods=5, freq="1min")
DF = pd.DataFrame({"systime": T, "uuid": "a", "value_double": range(5)})


@pytest.mark.parametrize(
    "call,column",
    [
        (lambda: MultivariateSPCEvents(DF, ["a", "b"], value_column="nope"), "nope"),
        (lambda: OutlierDetectionEvents(DF, "nope").detect_outliers_iqr(), "nope"),
        (
            lambda: MachineStateEvents(
                DF, "a", value_column="value_bool"
            ).detect_run_idle(),
            "value_bool",
        ),
    ],
)
def test_missing_value_column_names_the_column(call, column):
    with pytest.raises(ColumnNotFoundError, match=repr(column)) as info:
        call()
    assert "value_double" in str(info.value)  # lists what is there
    assert isinstance(info.value, ValueError)
