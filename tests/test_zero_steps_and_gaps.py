"""Regression tests: zero-length steps and max_gap on interval detectors."""

import numpy as np
import pandas as pd

from ts_shape.events.engineering.threshold_monitoring import ThresholdMonitoringEvents
from ts_shape.events.production.duty_cycle import DutyCycleEvents
from ts_shape.events.quality.anomaly_classification import (
    AnomalyClassificationEvents,
)
from ts_shape.events.quality.data_gap_analysis import DataGapAnalysisEvents


def _signal(times, values, uuid="sig", column="value_double"):
    return pd.DataFrame(
        {"systime": pd.to_datetime(times), "uuid": uuid, column: values}
    ).assign(is_delta=True)


def _minutes(n, start="2024-01-01 00:00"):
    return pd.date_range(start, periods=n, freq="1min")


# --- DataGapAnalysisEvents ----------------------------------------------------


def _with_duplicate_step():
    t = list(_minutes(5))
    t.insert(2, t[2])  # duplicate timestamp: a zero-length step
    return _signal(t, np.arange(6.0))


def test_zero_length_step_is_never_a_gap():
    ev = DataGapAnalysisEvents(_with_duplicate_step(), "sig")
    gaps = ev.detect_gaps(min_gap="0s")
    assert (gaps["duration_seconds"] > 0).all()
    assert len(gaps) == 4
    cands = ev.interpolation_candidates(max_gap="10min")
    assert (cands["gap_duration_seconds"] > 0).all()
    assert len(cands) == 4


# --- max_gap: flatline, hysteresis, duty cycle ---------------------------------


def _gap_frame():
    # Two bursts of three samples, two hours apart.
    t = list(_minutes(3)) + list(_minutes(3, start="2024-01-01 02:00"))
    return t


def test_flatline_duplicate_sample_is_not_a_flatline():
    t = list(_minutes(6))
    t.insert(3, t[3])
    values = [1.0, 2.0, 3.0, 4.0, 4.0, 5.0, 6.0]
    out = AnomalyClassificationEvents(_signal(t, values), "sig").detect_flatline()
    assert out.empty


def test_flatline_max_gap_splits_at_data_gap():
    df = _signal(_gap_frame(), 5.0)
    ev = AnomalyClassificationEvents(df, "sig")
    assert len(ev.detect_flatline()) == 1  # default: spans the gap
    out = ev.detect_flatline(max_gap="10min")
    assert len(out) == 2
    assert (out["duration_seconds"] == 120.0).all()


def test_hysteresis_max_gap_closes_alarm_at_gap():
    df = _signal(_gap_frame(), [0.0, 10.0, 10.0, 4.0, 4.0, 0.0])
    ev = ThresholdMonitoringEvents(df, "sig")
    assert len(ev.threshold_with_hysteresis(5.0, 2.0)) == 1
    out = ev.threshold_with_hysteresis(5.0, 2.0, max_gap="10min")
    # The alarm ends at its last sample before the gap; after the gap the
    # value (4) is below the entry limit, so no new alarm starts.
    assert len(out) == 1
    assert out["end"].iloc[0] == pd.Timestamp("2024-01-01 00:02")


def test_duty_cycle_max_gap_splits_interval():
    df = _signal(_gap_frame(), True, column="value_bool")
    ev = DutyCycleEvents(df, "sig")
    assert len(ev.on_off_intervals()) == 1
    out = ev.on_off_intervals(max_gap="10min")
    assert out["state"].tolist() == ["on", "on"]
    assert out["duration_seconds"].tolist() == [120.0, 120.0]
