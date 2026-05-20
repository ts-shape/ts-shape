import pandas as pd  # type: ignore
import pytest

from ts_shape.errors import ColumnNotFoundError
from ts_shape.utils.base import Base


def test_base_sorts_by_given_time_column():
    df = pd.DataFrame(
        {
            "systime": pd.to_datetime(["2023-01-03", "2023-01-01", "2023-01-02"]),
            "value_integer": [3, 1, 2],
        }
    )
    base = Base(df, column_name="systime")
    out = base.get_dataframe()
    assert list(out["systime"]) == sorted(df["systime"].tolist())


def test_base_detects_time_column_when_not_provided():
    df = pd.DataFrame(
        {
            "created_time": ["2023-01-02", "2023-01-01", "2023-01-03"],
            "value_integer": [2, 1, 3],
        }
    )
    base = Base(df)
    out = base.get_dataframe()
    # created_time should be converted to datetime and sorted asc
    assert pd.api.types.is_datetime64_any_dtype(out["created_time"])
    assert list(out["created_time"]) == sorted(
        pd.to_datetime(df["created_time"]).tolist()
    )


def test_validate_uuid_raises_with_available_list():
    df = pd.DataFrame(
        {"systime": pd.to_datetime(["2024-01-01", "2024-01-02"]), "uuid": ["a", "b"]}
    )
    with pytest.raises(ValueError, match="missing"):
        Base._validate_uuid(df, "missing")


def test_validate_uuid_is_noop_on_empty_or_missing_uuid_column():
    # Empty frame and uuid-less frame must not raise.
    Base._validate_uuid(pd.DataFrame(columns=["uuid"]), "anything")
    Base._validate_uuid(pd.DataFrame({"x": [1]}), "anything")


def test_validate_column_raises_column_not_found_error():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ColumnNotFoundError):
        Base._validate_column(df, "b")
    # Backwards compatible: ColumnNotFoundError subclasses ValueError.
    with pytest.raises(ValueError):
        Base._validate_column(df, "b")


def test_repr_reports_rows_and_class():
    df = pd.DataFrame({"systime": pd.to_datetime(["2024-01-01"]), "value_integer": [1]})
    r = repr(Base(df))
    assert r.startswith("Base(") and "rows=1" in r
