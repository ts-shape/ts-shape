import sys
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

# Ensure `src` is on the Python path so tests can import `ts_shape`
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ---------------------------------------------------------------------------
# Shared edge-case fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def empty_df():
    """DataFrame with the standard schema but zero rows."""
    return pd.DataFrame(
        {
            "systime": pd.Series(dtype="datetime64[ns]"),
            "uuid": pd.Series(dtype="str"),
            "value_double": pd.Series(dtype="float64"),
            "value_integer": pd.Series(dtype="int64"),
            "value_bool": pd.Series(dtype="bool"),
            "value_string": pd.Series(dtype="str"),
        }
    )


@pytest.fixture()
def single_row_df():
    """DataFrame with exactly one data row."""
    return pd.DataFrame(
        {
            "systime": pd.to_datetime(["2024-01-01 00:00:00"]),
            "uuid": ["sensor:temp"],
            "value_double": [42.0],
            "value_integer": [1],
            "value_bool": [True],
            "value_string": ["ok"],
        }
    )


@pytest.fixture()
def all_nan_df():
    """DataFrame where every value column is entirely NaN."""
    return pd.DataFrame(
        {
            "systime": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "uuid": ["s1", "s1", "s1"],
            "value_double": [float("nan")] * 3,
            "value_integer": pd.array([pd.NA, pd.NA, pd.NA], dtype="Int64"),
            "value_bool": pd.array([pd.NA, pd.NA, pd.NA], dtype="boolean"),
            "value_string": [None, None, None],
        }
    )


@pytest.fixture()
def duplicate_timestamps_df():
    """DataFrame with intentional duplicate (systime, uuid) pairs."""
    ts = pd.to_datetime(["2024-01-01 00:00:00"] * 3 + ["2024-01-01 00:01:00"])
    return pd.DataFrame(
        {
            "systime": ts,
            "uuid": ["s1", "s1", "s1", "s1"],
            "value_double": [1.0, 2.0, 3.0, 4.0],
        }
    )
