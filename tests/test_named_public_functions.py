"""Public functions and methods that no other test called by name."""

import numpy as np
import pandas as pd
import pytest

from ts_shape.eventlog import adapters
from ts_shape.eventlog.schema import OCEL_OID, OCEL_TYPE, empty_objects
from ts_shape.eventlog.taxonomy import template_fields
from ts_shape.features.stats.numeric_stats import NumericStatistics
from ts_shape.features.stats.string_stats import StringStatistics
from ts_shape.features.stats.timestamp_stats import TimestampStatistics
from ts_shape.signals.spec import SignalSpec
from ts_shape.utils.frames import window_offset
from ts_shape.utils.signal_checks import remember_uuids

VALUES = pd.DataFrame({"value_double": [1.0, 2.0, 2.0, 3.0, 10.0]})


def test_numeric_spread_statistics():
    v = VALUES["value_double"]
    assert NumericStatistics.column_std(VALUES, "value_double") == pytest.approx(
        v.std()
    )
    assert NumericStatistics.column_variance(VALUES, "value_double") == pytest.approx(
        v.var()
    )
    assert NumericStatistics.column_skewness(VALUES, "value_double") == pytest.approx(
        v.skew()
    )
    # mean absolute deviation around the mean 3.6: (2.6+1.6+1.6+0.6+6.4) / 5
    assert NumericStatistics.column_mad(VALUES, "value_double") == pytest.approx(2.56)
    assert NumericStatistics.column_mode(VALUES, "value_double") == 2.0


def test_numeric_mode_of_an_empty_column():
    empty = VALUES.iloc[:0]
    assert NumericStatistics.column_mode(empty, "value_double") is None


def test_string_length_summary():
    df = pd.DataFrame({"value_string": ["a", "abc", "ab", None]})
    out = StringStatistics.string_length_summary(df)
    row = out.iloc[0]
    assert row["Min Length"] == 1
    assert row["Max Length"] == 3
    assert row["Average Length"] == pytest.approx(2.0)


def test_standard_deviation_of_timestamp_gaps():
    df = pd.DataFrame(
        {
            "systime": pd.to_datetime(
                ["2024-01-01 00:00", "2024-01-01 00:01", "2024-01-01 00:04"]
            )
        }
    )
    std = TimestampStatistics.standard_deviation_timestamps(df)
    # gaps of 60 s and 180 s
    assert std.total_seconds() == pytest.approx(np.std([60, 180], ddof=1), abs=1e-6)


def test_signal_spec_encoding_properties():
    state = SignalSpec("press:state", kind="state", encoding="change_encoded")
    assert state.is_change_encoded
    assert state.holds_value
    event = SignalSpec("press:alarm", kind="event")
    assert not event.is_change_encoded
    assert not event.holds_value


def test_window_offset_daily_is_a_calendar_day():
    assert window_offset("1h") == pd.tseries.frequencies.to_offset("1h")
    berlin = pd.Timestamp("2024-03-31", tz="Europe/Berlin")
    assert (berlin + window_offset("1D")) - berlin == pd.Timedelta("23h")


def test_window_offset_return_type_per_pandas_version():
    # Pins what the docstring says the day offset is on each pandas major.
    got = window_offset("3D")
    if isinstance(pd.tseries.frequencies.to_offset("1D"), pd.tseries.offsets.Tick):
        assert type(got) is pd.DateOffset and got.kwds == {"days": 3}
    else:
        assert got == pd.tseries.frequencies.to_offset("3D")
        assert type(got).__name__ == "Day" and got.n == 3
    assert window_offset("24h") == pd.tseries.frequencies.to_offset("24h")


def test_template_fields():
    assert template_fields("production.{shift}.{state}") == ["shift", "state"]
    assert template_fields("no.placeholders") == []


def test_get_override_resolves_registered_adapters():
    assert adapters.get_override("NoSuchDetector", "nothing") is None

    def custom(frame, **kwargs):  # pragma: no cover - never called
        raise AssertionError

    try:
        adapters.register_adapter("NamedFnProbe", "probe")(custom)
        assert adapters.get_override("NamedFnProbe", "probe") is custom
    finally:
        adapters._OVERRIDES.pop(("NamedFnProbe", "probe"), None)


def test_empty_objects_schema():
    out = empty_objects()
    assert out.empty
    assert list(out.columns) == [OCEL_OID, OCEL_TYPE]
    assert all(str(t) == "string" for t in out.dtypes)


def test_remember_uuids_ignores_objects_that_cannot_hold_it():
    frame = pd.DataFrame({"uuid": ["a"]})
    remember_uuids(object(), frame, "uuid", {"a"})  # no __dict__: silently skipped

    class Holder:
        pass

    holder = Holder()
    remember_uuids(holder, frame, "uuid", {"a"})
    assert any(isinstance(v, tuple) for v in vars(holder).values())
