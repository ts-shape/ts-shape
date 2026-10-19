"""Definitions checked on hand-computed probes (round 14, part b).

Each test states the definition it checks in its docstring; the expected
values follow from that definition, not from the implementation.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from ts_shape.errors import DataQualityWarning
from ts_shape.events.engineering.control_loop_health import ControlLoopHealthEvents
from ts_shape.events.maintenance.asset_health_index import AssetHealthIndexEvents
from ts_shape.events.maintenance.degradation_detection import (
    DegradationDetectionEvents,
)
from ts_shape.events.quality.anomaly_classification import (
    AnomalyClassificationEvents,
)
from ts_shape.events.quality.data_gap_analysis import DataGapAnalysisEvents
from ts_shape.events.quality.sampling_resolution import SamplingResolutionEvents
from ts_shape.events.quality.signal_quality import SignalQualityEvents
from ts_shape.events.quality.tolerance_deviation import ToleranceDeviationEvents
from ts_shape.signals.primitives import windows

T0 = pd.Timestamp("2024-01-01")


def _series(values, freq: str = "1s", uuid: str = "s", start=T0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "systime": pd.date_range(start, periods=len(values), freq=freq),
            "uuid": uuid,
            "value_double": [float(v) for v in values],
            "is_delta": True,
        }
    )


def _data_quality(records) -> list[str]:
    return [
        str(r.message) for r in records if issubclass(r.category, DataQualityWarning)
    ]


# ---------------------------------------------------------------------------
# 1. coverage_by_period: edge windows count the time outside the data.
# ---------------------------------------------------------------------------


def test_coverage_edge_windows_count_time_outside_the_data():
    """1 s data 10:30:00-11:29:59, hourly windows.

    The data covers [10:30:00, 11:29:59 + 1 s) = [10:30, 11:30): the 10:00
    window misses [10:00, 10:30) = 1800 s, the 11:00 window [11:30, 12:00)
    = 1800 s, so both read 50 % -- as data_completeness does (1800 of 3600
    expected samples). (Main read 100 % in both.)
    """
    t = pd.date_range("2024-01-01 10:30:00", "2024-01-01 11:29:59", freq="1s")
    df = _series(np.ones(len(t)), start=t[0])
    cov = DataGapAnalysisEvents(df, "s").coverage_by_period(window="1h")
    assert list(cov["coverage_pct"]) == [50.0, 50.0]
    assert list(cov["gap_seconds"]) == [1800.0, 1800.0]
    assert list(cov["gap_count"]) == [0, 0]
    comp = SignalQualityEvents(df, "s").data_completeness(window="1h")
    assert list(comp["completeness_pct"]) == [50.0, 50.0]


def test_coverage_edge_and_internal_gap_add_up():
    """1 min data from 00:10 to 00:49 with 00:20-00:30 missing, one 1 h window.

    Samples 00:10..00:19 and 00:30..00:49; the median step is 1 min. Missing:
    [00:00, 00:10) = 600 s, the gap 00:19 -> 00:30 = 660 s, and
    [00:50, 01:00) = 600 s: 1860 s, coverage (1 - 1860/3600) = 48.33 %.
    """
    t = list(pd.date_range("2024-01-01 00:10", "2024-01-01 00:19", freq="1min"))
    t += list(pd.date_range("2024-01-01 00:30", "2024-01-01 00:49", freq="1min"))
    df = pd.DataFrame({"systime": t, "uuid": "s", "value_double": 1.0})
    cov = DataGapAnalysisEvents(df, "s").coverage_by_period(window="1h")
    assert cov["gap_seconds"].iloc[0] == pytest.approx(1860.0)
    assert cov["gap_count"].iloc[0] == 1
    assert cov["coverage_pct"].iloc[0] == pytest.approx((1 - 1860 / 3600) * 100)


# ---------------------------------------------------------------------------
# 2. detect_aliasing: the sampling rate of the non-NaN values.
# ---------------------------------------------------------------------------


def test_aliasing_sampling_rate_ignores_nan_rows():
    """1 s grid, every other value NaN, 0.2 Hz sine.

    The values are sampled every 2 s: 0.5 Hz, Nyquist 0.25 Hz. The FFT of
    the 100 values at 2 s spacing peaks at 0.2 Hz; 0.2 > 0.7 * 0.25 = 0.175
    and <= 0.9 * 0.25 = 0.225: risk "medium". (Main: 1.0 Hz, dominant 0.4,
    risk "none".)
    """
    n = 200
    v = np.sin(2 * np.pi * 0.2 * np.arange(n))
    v[1::2] = np.nan
    df = _series(v)
    out = SamplingResolutionEvents(df, "s").detect_aliasing(
        window="1h", expected_frequency_hz=0.2
    )
    row = out.iloc[0]
    assert row["sampling_rate_hz"] == pytest.approx(0.5)
    assert row["nyquist_hz"] == pytest.approx(0.25)
    assert row["dominant_frequency"] == pytest.approx(0.2)
    assert row["aliasing_risk"] == "medium"


# ---------------------------------------------------------------------------
# 3. Degradation baseline [first, first + window) like the rolling windows.
# ---------------------------------------------------------------------------


def test_variance_increase_stationary_alternation_is_not_flagged():
    """0/1 alternating at 1 s, window 4 s.

    Baseline [t0, t0 + 4 s) = 0, 1, 0, 1: variance 1/3. Every full rolling
    window (t - 4 s, t] also holds two 0 and two 1: ratio 1 < 1.05. (Main's
    5-sample baseline 0, 1, 0, 1, 0 had variance 0.3: ratio 1.11 everywhere.)
    """
    df = _series([i % 2 for i in range(40)])
    det = DegradationDetectionEvents(df, "s")
    assert det.detect_variance_increase(window="4s", threshold_factor=1.05).empty
    hs = det.health_score(window="4s", baseline_window="4s")
    # From t0 + 3 s on, every rolling window holds two 0 and two 1.
    assert hs["variance_factor"].iloc[3:].to_numpy() == pytest.approx(1.0)


def test_variance_increase_after_baseline_is_flagged():
    """Baseline 0/1 (variance 1/3), then 0/3 alternation from t0 + 20 s.

    The window ending at t0 + 23 s holds 0, 3, 0, 3: variance 3, ratio 9.
    """
    vals = [i % 2 for i in range(20)] + [3 * (i % 2) for i in range(20)]
    out = DegradationDetectionEvents(_series(vals), "s").detect_variance_increase(
        window="4s", threshold_factor=5.0
    )
    assert len(out) == 1
    assert out["start"].iloc[0] >= T0 + pd.Timedelta(seconds=20)


# ---------------------------------------------------------------------------
# 4. Zero-crossings carry the last non-zero sign.
# ---------------------------------------------------------------------------


def _loop(errors) -> ControlLoopHealthEvents:
    sp = pd.DataFrame({"systime": [T0], "uuid": "sp", "value_double": [0.0]})
    pv = _series(errors, uuid="pv")
    return ControlLoopHealthEvents(pd.concat([sp, pv], ignore_index=True), "sp", "pv")


def test_loop_oscillation_zero_between_signs_is_one_crossing():
    """Error 1,0,-1,0,1,0,-1,0,1 at 1 s.

    Carried signs + + - - + + - - +: changes at 2, 4, 6, 8 s -- 4 crossings,
    2 s apart, period 4 s. (Main: 8 crossings, period 2 s.)
    """
    out = _loop([1, 0, -1, 0, 1, 0, -1, 0, 1]).detect_oscillation(
        window="5min", min_crossings=4
    )
    assert out["crossing_count"].tolist() == [4]
    assert out["estimated_period_seconds"].iloc[0] == pytest.approx(4.0)


def test_loop_oscillation_touching_zero_is_no_crossing():
    """Error 1,0,1,0,1: the sign never changes, 0 crossings, no event."""
    out = _loop([1, 0, 1, 0, 1]).detect_oscillation(window="5min", min_crossings=1)
    assert out.empty


def test_anomaly_oscillation_zero_on_the_line_is_one_crossing():
    """Values 0,1,0,-1,0,1,0,-1,0 at 1 s in one 1 min window.

    The line through the first and last value (both 0) is 0. Carried signs
    (the leading 0 takes the first non-zero sign): + + + - - + + - -; changes
    at 3, 5, 7 s -- 3 crossings. (Main read 0 as +: 4.)
    """
    det = AnomalyClassificationEvents(_series([0, 1, 0, -1, 0, 1, 0, -1, 0]), "s")
    out = det.detect_oscillation(window="1min", min_crossings=1)
    assert out["crossing_count"].tolist() == [3]


def test_anomaly_oscillation_touching_the_line_from_below_is_no_crossing():
    """Values 0,-1,0,-1,0,-1,0: below or on the line, 0 crossings.

    (Main read each 0 as + and counted 6.)
    """
    det = AnomalyClassificationEvents(_series([0, -1, 0, -1, 0, -1, 0]), "s")
    assert det.detect_oscillation(window="1min", min_crossings=1).empty
    assert det.classify_anomalies(window="1min").empty


# ---------------------------------------------------------------------------
# 5. Value-column warnings for the signals= mapping.
# ---------------------------------------------------------------------------


def _vib_temp() -> pd.DataFrame:
    t = pd.date_range(T0, periods=120, freq="1min")
    vib = pd.DataFrame(
        {
            "systime": t,
            "uuid": "vib",
            "value_double": np.linspace(1.0, 2.0, 120),
            "value_integer": np.nan,
        }
    )
    temp = pd.DataFrame(
        {
            "systime": t,
            "uuid": "temp",
            "value_double": np.nan,
            "value_integer": np.arange(120.0),
        }
    )
    return pd.concat([vib, temp], ignore_index=True)


def test_asset_health_partial_misconfiguration_warns_once():
    """vib in value_double, temp in value_integer: the index is computed from
    vib alone, and exactly one warning names temp and value_integer."""
    det = AssetHealthIndexEvents(
        _vib_temp(), "press", {"vib": "bearing", "temp": "thermal"}
    )
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = det.health_index(window="10min", baseline_window="30min")
        det.health_index(window="10min", baseline_window="30min")
    assert not out.empty
    messages = _data_quality(rec)
    assert len(messages) == 1
    assert "'temp'" in messages[0] and "value_integer" in messages[0]


def test_asset_health_all_misconfigured_warns_once_per_signal():
    """Both signals in value_integer: an empty index, one warning each."""
    df = _vib_temp()
    df["value_integer"] = df["value_integer"].fillna(df["value_double"])
    df["value_double"] = np.nan
    det = AssetHealthIndexEvents(df, "press", {"vib": "bearing", "temp": "thermal"})
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = det.health_index(window="10min", baseline_window="30min")
    assert out.empty
    messages = _data_quality(rec)
    assert len(messages) == 2
    assert any("'vib'" in m for m in messages)
    assert any("'temp'" in m for m in messages)


def test_asset_health_mistyped_signal_uuid_warns():
    """A signals= key that is not in the data gets the uuid-not-found warning."""
    with pytest.warns(DataQualityWarning, match=r"'tmep' not found in the data"):
        AssetHealthIndexEvents(_vib_temp(), "press", {"vib": "bearing", "tmep": "x"})


def test_tolerance_deviation_warns_when_actual_values_are_integers():
    """Actual values only in value_integer, read from value_double: warned."""
    t = pd.date_range(T0, periods=6, freq="1min")
    tol = pd.DataFrame(
        {
            "systime": t[:1],
            "uuid": "tol",
            "value_double": [5.0],
            "value_integer": np.nan,
        }
    )
    act = pd.DataFrame(
        {
            "systime": t,
            "uuid": "act",
            "value_double": np.nan,
            "value_integer": [1.0, 2.0, 9.0, 9.0, 2.0, 1.0],
        }
    )
    det = ToleranceDeviationEvents(
        pd.concat([tol, act], ignore_index=True),
        value_column_tolerance="value_double",
        value_column_actual="value_double",
        upper_tolerance_uuid="tol",
        actual_uuid="act",
        event_uuid="ev",
    )
    with pytest.warns(DataQualityWarning, match=r"'act'.*value_integer"):
        det.detect_deviations()


# ---------------------------------------------------------------------------
# 6. windows(): sub-daily windows are fixed lengths, daily ones calendar days.
# ---------------------------------------------------------------------------


def test_windows_sub_daily_fixed_length_daily_calendar():
    """Berlin, spring DST change on 2024-03-31 (02:00 -> 03:00).

    "12h" windows are 12 h each from 2024-03-30 00:00+01: after the change
    their wall-clock start is 13:00+02. "1D" windows are calendar days:
    24 h, then 23 h.
    """
    w = windows("2024-03-30", "2024-04-01", "12h", tz="Europe/Berlin")
    assert (w["duration_seconds"] == 43200.0).all()
    assert w["start"].iloc[3] == pd.Timestamp("2024-03-31 13:00", tz="Europe/Berlin")
    d = windows("2024-03-30", "2024-04-01", "1D", tz="Europe/Berlin")
    assert d["duration_seconds"].tolist() == [86400.0, 82800.0]
