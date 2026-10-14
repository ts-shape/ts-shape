"""Regression tests: tz-aware systime survives (no naive-UTC .values)."""

import pandas as pd
import pytest

from ts_shape.errors import DataQualityWarning
from ts_shape.events.engineering.rate_of_change import RateOfChangeEvents
from ts_shape.events.engineering.threshold_monitoring import ThresholdMonitoringEvents
from ts_shape.events.engineering.warmup_analysis import WarmUpCoolDownEvents
from ts_shape.events.maintenance.degradation_detection import (
    DegradationDetectionEvents,
)
from ts_shape.events.quality.data_gap_analysis import DataGapAnalysisEvents
from ts_shape.events.quality.signal_quality import SignalQualityEvents
from ts_shape.events.supplychain.inventory_monitoring import (
    InventoryMonitoringEvents,
)
from ts_shape.transform.filter.datetime_filter import DateTimeFilter
from ts_shape.transform.time_functions.timezone_shift import TimezoneShift
from ts_shape.utils.base import Base

BERLIN = "Europe/Berlin"


def _assert_berlin(series):
    assert str(series.dt.tz) == BERLIN


def test_find_gaps_keeps_local_time(berlin_signal):
    df = berlin_signal([1.0, 2.0, 3.0], start="2024-01-01 00:00", freq="1min")
    df.loc[2, "systime"] = df.loc[1, "systime"] + pd.Timedelta(hours=1)
    out = DataGapAnalysisEvents(df, "sig").detect_gaps(min_gap="10min")
    _assert_berlin(out["start"])
    assert out["start"].iloc[0] == pd.Timestamp("2024-01-01 00:01", tz=BERLIN)


def test_coverage_by_period_no_type_error(berlin_signal):
    df = berlin_signal([1.0] * 12, freq="10min")
    out = DataGapAnalysisEvents(df, "sig").coverage_by_period(
        window="1h", min_gap="10min"
    )
    assert not out.empty


def test_interpolation_candidates_keep_local_time(berlin_signal):
    df = berlin_signal([1.0, 1.1, 1.2], start="2024-01-01 00:00", freq="5min")
    out = DataGapAnalysisEvents(df, "sig").interpolation_candidates(
        max_gap="1h", min_gap="1min"
    )
    _assert_berlin(out["start"])


def test_signal_quality_missing_data(berlin_signal):
    df = berlin_signal([1.0, 2.0], start="2024-01-01 00:00", freq="1h")
    out = SignalQualityEvents(df, "sig").detect_missing_data(expected_freq="1min")
    _assert_berlin(out["start"])


def test_threshold_with_hysteresis(berlin_signal):
    df = berlin_signal([0.0, 100.0, 0.0], start="2024-01-01 00:00", freq="10min")
    out = ThresholdMonitoringEvents(df, "sig").threshold_with_hysteresis(90, 10)
    assert out["start"].iloc[0] == pd.Timestamp("2024-01-01 00:10", tz=BERLIN)


def test_rate_of_change_step(berlin_signal):
    df = berlin_signal([0.0, 50.0], start="2024-01-01 00:00", freq="1s")
    out = RateOfChangeEvents(df, "sig").detect_step_changes(min_delta=10)
    _assert_berlin(out["time"])


def test_warmup_across_spring_forward(berlin_signal):
    df = berlin_signal([20.0, 30.0, 40.0, 50.0, 60.0, 60.0], start="2024-03-31 01:40")
    out = WarmUpCoolDownEvents(df, "sig").detect_warmup(
        min_rise=20, min_duration="1min"
    )
    _assert_berlin(out["start"])


def test_level_shift_no_type_error(berlin_signal):
    df = berlin_signal([0.0] * 10 + [10.0] * 10, freq="1min")
    out = DegradationDetectionEvents(df, "sig").detect_level_shift(
        min_shift=5, hold="5min"
    )
    assert len(out) >= 1
    _assert_berlin(out["systime"])


@pytest.mark.parametrize("start", ["2024-01-01 00:00", "2024-10-27 01:30"])
def test_stockout_prediction_keeps_tz(start):
    t = pd.date_range(start, periods=5, freq="30min", tz=BERLIN)
    df = pd.DataFrame(
        {"systime": t, "uuid": "lvl", "value_double": [100.0, 90, 80, 70, 60]}
    )
    out = InventoryMonitoringEvents(df, "lvl").stockout_prediction(
        consumption_rate_window="2h"
    )
    _assert_berlin(out["systime"])


# --- transform: datetime filters, timezone shift, Base ----------------------


def test_datetime_filter_on_tz_aware_column(berlin_signal):
    df = berlin_signal([1.0, 2.0, 3.0], start="2024-01-01 23:00", freq="1h")
    out = DateTimeFilter.filter_after_date(df, "systime", "2024-01-02")
    assert len(out) == 1  # 01:00 local; 00:00 is not strictly after
    between = DateTimeFilter.filter_between_datetimes(
        df, "systime", "2024-01-01 23:00", "2024-01-02 01:00", inclusive="both"
    )
    assert len(between) == 3


def test_datetime_filter_default_bounds_exclusive():
    df = pd.DataFrame({"systime": pd.date_range("2024-01-01", periods=3, freq="D")})
    out = DateTimeFilter.filter_between_dates(df, "systime", "2024-01-01", "2024-01-03")
    assert len(out) == 1


def test_shift_timezone_handles_dst_and_does_not_mutate():
    naive = pd.to_datetime(["2024-03-31 01:30", "2024-03-31 02:30"])
    df = pd.DataFrame({"systime": naive})
    with pytest.raises(Exception):
        TimezoneShift.shift_timezone(df, "systime", BERLIN, "UTC")
    out = TimezoneShift.shift_timezone(
        df, "systime", BERLIN, "UTC", nonexistent="shift_forward"
    )
    assert str(out["systime"].dt.tz) == "UTC"
    assert df["systime"].dt.tz is None  # caller's frame untouched


def test_base_warns_on_nat_timestamps():
    df = pd.DataFrame({"systime": ["2024-01-01 00:00", "not-a-time"], "uuid": "a"})
    with pytest.warns(DataQualityWarning, match="NaT"):
        Base(df)
