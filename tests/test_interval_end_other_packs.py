"""Regression tests (Theme B): interval_end="next_change" on change-encoded data.

With one row per change, closing an interval at its last flagged sample gives
zero-length intervals. The opt-in mode closes it at the next change.
"""

import numpy as np
import pandas as pd
import pytest

from ts_shape.events.correlation.signal_correlation import SignalCorrelationEvents
from ts_shape.events.engineering.steady_state_detection import (
    SteadyStateDetectionEvents,
)
from ts_shape.events.engineering.threshold_monitoring import ThresholdMonitoringEvents
from ts_shape.events.maintenance.vibration_analysis import VibrationAnalysisEvents
from ts_shape.events.quality.signal_quality import SignalQualityEvents
from ts_shape.events.supplychain.inventory_monitoring import (
    InventoryMonitoringEvents,
)
from ts_shape.features.segment_analysis.segment_extractor import SegmentExtractor
from ts_shape.utils.intervals import interval_end_times

T0 = pd.Timestamp("2024-01-01 10:00")


def _sig(points, uuid="sig", col="value_double"):
    t = [T0 + pd.Timedelta(minutes=m) for m, _ in points]
    return pd.DataFrame({"systime": t, "uuid": uuid, col: [v for _, v in points]})


# 0 @10:00, 100 @10:05, 0 @10:55 -> above 90 for 50 minutes
ALARM = [(0, 0.0), (5, 100.0), (55, 0.0)]


def test_interval_end_times_modes():
    t = pd.Series(pd.date_range(T0, periods=3, freq="5min"))
    assert interval_end_times(t, "last_sample").equals(t)
    nxt = interval_end_times(t)  # "next_change" is the default
    assert list(nxt) == [t[1], t[2], t[2]]
    with pytest.raises(ValueError):
        interval_end_times(t, "bogus")


def test_multi_level_threshold():
    ev = ThresholdMonitoringEvents(_sig(ALARM), "sig")
    # The default closes the interval at the next change.
    out = ev.multi_level_threshold({"alarm": 90})
    assert out["duration"].iloc[0] == pd.Timedelta(minutes=50)
    last = ev.multi_level_threshold({"alarm": 90}, interval_end="last_sample")
    assert last["duration"].iloc[0] == pd.Timedelta(0)


def test_time_above_threshold_uses_real_time():
    ev = ThresholdMonitoringEvents(_sig(ALARM), "sig")
    # Default: real time above the threshold (each value held to the next).
    out = ev.time_above_threshold(90, window="1h")
    assert out["time_above"].iloc[0] == pytest.approx(3000.0)
    by_sample = ev.time_above_threshold(90, window="1h", interval_end="last_sample")
    assert by_sample["time_above"].iloc[0] == pytest.approx(1200.0)


def test_detect_out_of_range():
    ev = SignalQualityEvents(_sig(ALARM), "sig")
    # The default closes the interval at the next change.
    out = ev.detect_out_of_range(0, 90)
    assert out["duration"].iloc[0] == pd.Timedelta(minutes=50)
    last = ev.detect_out_of_range(0, 90, interval_end="last_sample")
    assert last["duration"].iloc[0] == pd.Timedelta(0)


def test_detect_low_stock():
    df = _sig([(0, 100.0), (10, 5.0), (70, 100.0)], uuid="lvl")
    ev = InventoryMonitoringEvents(df, "lvl")
    out = ev.detect_low_stock(10, hold="30min")
    assert out["duration_seconds"].iloc[0] == 3600.0
    assert ev.detect_low_stock(10, hold="30min", interval_end="last_sample").empty


def test_segment_extractor():
    df = _sig([(0, "A"), (10, "B"), (20, "A")], col="value_string")
    out = SegmentExtractor.extract_time_ranges(df, "sig", interval_end="next_change")
    assert list(out["segment_duration"][:2]) == [pd.Timedelta(minutes=10)] * 2


def test_rms_exceedance():
    pts = [(m, 0.0) for m in range(10)] + [(10, 5.0), (40, 0.0)]
    ev = VibrationAnalysisEvents(_sig(pts), "sig")
    out = ev.detect_rms_exceedance(1.0, window="1min")
    assert out["duration_seconds"].iloc[0] == 30 * 60
    last = ev.detect_rms_exceedance(1.0, window="1min", interval_end="last_sample")
    assert last["duration_seconds"].iloc[0] < 30 * 60


def test_steady_state_next_change():
    pts = [(m, 50.0) for m in range(30)] + [(30, 90.0), (31, 10.0), (32, 70.0)]
    ev = SteadyStateDetectionEvents(_sig(pts), "sig")
    last = ev.detect_steady_state(
        window="5min",
        std_threshold=1.0,
        min_duration="1min",
        interval_end="last_sample",
    )
    default = ev.detect_steady_state(
        window="5min", std_threshold=1.0, min_duration="1min"
    )
    assert (default["end"].iloc[0] - last["end"].iloc[0]) == pd.Timedelta(minutes=1)


def test_correlation_breakdown_next_change():
    t = pd.date_range(T0, periods=40, freq="1min")
    a = np.sin(np.arange(40) / 3.0)
    idx = np.arange(40)
    b = np.where((idx >= 15) & (idx < 25), -a, a)
    df = pd.concat(
        [
            pd.DataFrame({"systime": t, "uuid": "a", "value_double": a}),
            pd.DataFrame({"systime": t, "uuid": "b", "value_double": b}),
        ]
    )
    ev = SignalCorrelationEvents(df)
    last = ev.correlation_breakdown(
        "a", "b", window_size=5, threshold=0.0, interval_end="last_sample"
    )
    default = ev.correlation_breakdown("a", "b", window_size=5, threshold=0.0)
    assert len(default) == len(last) > 0
    assert (default["duration_seconds"] >= last["duration_seconds"]).all()
    assert default["duration_seconds"].sum() > last["duration_seconds"].sum()
