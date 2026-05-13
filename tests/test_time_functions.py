import pytest
import pandas as pd  # type: ignore

from ts_shape.transform.time_functions.timestamp_converter import TimestampConverter
from ts_shape.transform.time_functions.timezone_shift import TimezoneShift


def test_timestamp_converter_ms_to_tz():
    ts_ms = [1_700_000_000_000, 1_700_000_360_000]  # base timestamps
    t2_ms = [x + 3_600_000 for x in ts_ms]  # +1 hour each
    df = pd.DataFrame({"t1": ts_ms, "t2": t2_ms})
    out = TimestampConverter.convert_to_datetime(
        df, columns=["t1", "t2"], unit="ms", timezone="UTC"
    )
    assert "UTC" in str(out["t1"].dtype)
    assert (out["t2"] - out["t1"]).dt.total_seconds().iloc[1] == 3600


def test_timestamp_converter_invalid_unit():
    df = pd.DataFrame({"t": [0, 1]})
    with pytest.raises(ValueError):
        TimestampConverter.convert_to_datetime(df, ["t"], unit="bad", timezone="UTC")


def test_timezone_shift_end_to_end_and_helpers():
    df = pd.DataFrame(
        {"systime": pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 01:00:00"])}
    )
    # Convert naive UTC times to Europe/Berlin and add new column
    df2 = TimezoneShift.add_timezone_column(df, "systime", "UTC", "Europe/Berlin")
    new_col = "systime_Europe_Berlin"
    assert new_col in df2.columns
    assert TimezoneShift.detect_timezone_awareness(df2, new_col) is True

    # Shift in-place column
    df3 = df.copy()
    df3["systime"] = pd.to_datetime(df3["systime"]).dt.tz_localize("UTC")
    df_shifted = TimezoneShift.shift_timezone(df3, "systime", "UTC", "Europe/Berlin")
    assert str(df_shifted["systime"].dtype).endswith("Europe/Berlin]")

    # Calculate difference between two tz-aware columns
    df_shifted["end"] = df_shifted["systime"] + pd.Timedelta(hours=2)
    diffs = TimezoneShift.calculate_time_difference(df_shifted, "systime", "end")
    assert diffs.iloc[0] == 7200

    # Revert timezone
    reverted = TimezoneShift.revert_to_original_timezone(
        df_shifted.copy(), "systime", "UTC"
    )
    assert str(reverted["systime"].dtype).endswith("UTC]")

    # Mismatch awareness raises
    df_mismatch = pd.DataFrame(
        {
            "a": pd.to_datetime(["2024-01-01 00:00:00"]),
            "b": pd.to_datetime(["2024-01-01 01:00:00"]).tz_localize("UTC"),
        }
    )
    with pytest.raises(ValueError):
        TimezoneShift.calculate_time_difference(df_mismatch, "a", "b")
