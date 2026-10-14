"""Regression tests for the second round of lined-out screening items."""

import numpy as np
import pandas as pd
import pytest

from ts_shape.errors import DataQualityWarning
from ts_shape.events.energy.consumption_analysis import EnergyConsumptionEvents
from ts_shape.events.engineering.setpoint_events import SetpointChangeEvents
from ts_shape.events.engineering.startup_events import StartupDetectionEvents
from ts_shape.events.engineering.warmup_analysis import WarmUpCoolDownEvents
from ts_shape.events.quality.anomaly_classification import AnomalyClassificationEvents
from ts_shape.events.quality.data_gap_analysis import DataGapAnalysisEvents
from ts_shape.events.quality.gauge_repeatability import GaugeRepeatabilityEvents
from ts_shape.events.quality.signal_quality import SignalQualityEvents
from ts_shape.utils.counters import counter_increments

BERLIN = "Europe/Berlin"


def _sig(times, values, uuid="s"):
    return pd.DataFrame({"systime": times, "uuid": uuid, "value_double": values})


# --- DST: real window length --------------------------------------------------


@pytest.fixture()
def dst_minutely():
    t = pd.date_range("2024-03-30", "2024-04-01 23:59", freq="1min", tz=BERLIN)
    return _sig(t, 1.0)


def test_data_completeness_23h_day_is_complete(dst_minutely):
    out = SignalQualityEvents(dst_minutely, "s").data_completeness(
        window="1D", expected_freq="1min"
    )
    assert list(out["expected_count"]) == [1440, 1380, 1440]
    assert (out["completeness_pct"] == 100.0).all()


def test_coverage_by_period_23h_day(dst_minutely):
    out = DataGapAnalysisEvents(dst_minutely, "s").coverage_by_period(window="1D")
    assert (out["coverage_pct"] == 100.0).all()


def test_coverage_auto_min_gap_ignores_duplicates():
    t = pd.date_range("2024-01-01", periods=20, freq="1min")
    df = pd.concat([_sig(t, 1.0)] * 3)  # two thirds of the intervals are 0
    with pytest.warns(DataQualityWarning):
        out = DataGapAnalysisEvents(df, "s").coverage_by_period(window="1h")
    assert out["gap_count"].sum() == 0


# --- flatline -----------------------------------------------------------------


def test_flatline_starts_at_first_stuck_sample():
    t = pd.date_range("2024-01-01", periods=6, freq="1min")
    df = _sig(t, [1.0, 5.0, 5.0, 5.0, 5.0, 9.0])
    # Dense sampling: end the run at its own last sample.
    out = AnomalyClassificationEvents(df, "s").detect_flatline(
        min_duration="1min", interval_end="last_sample"
    )
    assert out["start"].iloc[0] == t[1]
    assert out["duration"].iloc[0] == pd.Timedelta(minutes=3)


def test_flatline_deadband_single_row():
    t = pd.to_datetime(["2024-01-01 00:00", "2024-01-01 00:01", "2024-01-01 03:00"])
    df = _sig(t, [1.0, 5.0, 9.0])  # stuck at 5 for ~3 h, logged once
    ev = AnomalyClassificationEvents(df, "s")
    assert ev.detect_flatline(min_duration="1h", interval_end="last_sample").empty
    out = ev.detect_flatline(min_duration="1h")
    assert out["stuck_value"].tolist() == [5.0]


# --- gauge R&R ----------------------------------------------------------------


def test_repeatability_excludes_operator_bias():
    # Two operators, offset by 1.0, each perfectly repeatable within +-0.1.
    rows = []
    for part, base in [("P1", 10.0), ("P2", 20.0)]:
        for op, bias in [("A", 0.0), ("B", 1.0)]:
            for v in (-0.1, 0.0, 0.1):
                rows.append((part, op, base + bias + v))
    df = pd.DataFrame(rows, columns=["value_string", "operator", "value_double"])
    df["uuid"] = "g"
    df["systime"] = pd.date_range("2024-01-01", periods=len(df), freq="1min")
    pooled = GaugeRepeatabilityEvents(df, "g").repeatability()
    per_op = GaugeRepeatabilityEvents(
        df, "g", operator_column="operator"
    ).repeatability()
    assert pooled["range"].iloc[0] == pytest.approx(1.2)
    assert per_op["range"].tolist() == pytest.approx([0.2, 0.2])
    assert per_op["EV"].iloc[0] == pytest.approx(0.2 / 1.693, abs=1e-6)


# --- warm-up ramps --------------------------------------------------------------


def test_warmup_max_pause_bridges_a_flat_sample():
    t = pd.date_range("2024-01-01", periods=9, freq="1min")
    df = _sig(t, [20.0, 30, 40, 40, 50, 60, 70, 70, 70])
    ev = WarmUpCoolDownEvents(df, "s")
    assert ev.detect_warmup(min_rise=40, min_duration="1min").empty
    out = ev.detect_warmup(min_rise=40, min_duration="1min", max_pause=1)
    assert len(out) == 1
    assert out.iloc[0]["start_value"] == 20.0 and out.iloc[0]["end_value"] == 70.0


# --- adaptive startup baseline_window -----------------------------------------


def test_adaptive_baseline_window_is_used():
    t = pd.date_range("2024-01-01", periods=200, freq="1min")
    rng = np.random.default_rng(0)
    v = np.concatenate([rng.normal(10, 1, 150), np.full(50, 40.0)])
    ev = StartupDetectionEvents(_sig(t, v, uuid="speed"), "speed")
    out = ev.detect_startup_adaptive(baseline_window="30min", min_above="5min")
    assert out["start"].iloc[0] == t[150]


# --- undershoot ----------------------------------------------------------------


def test_undershoot_is_measured_after_reaching_setpoint():
    t0 = pd.Timestamp("2024-01-01")
    sp = pd.DataFrame(
        {
            "systime": [t0, t0 + pd.Timedelta("60s")],
            "uuid": "sp",
            "value_double": [0.0, 10.0],
        }
    )
    ts = np.arange(0, 400)
    # Rises, overshoots to 12, swings back to 9, settles at 10.
    pv_v = np.interp(ts, [0, 60, 100, 140, 180, 400], [0, 0, 12, 9, 10, 10])
    pv = _sig(t0 + pd.to_timedelta(ts, unit="s"), pv_v, uuid="pv")
    out = SetpointChangeEvents(pd.concat([sp, pv]), "sp").overshoot_metrics("pv")
    row = out.iloc[0]
    assert row["overshoot_abs"] == pytest.approx(2.0)
    assert row["undershoot_abs"] == pytest.approx(1.0)  # was 10.0 (the step itself)


# --- cumulative kWh register ----------------------------------------------------


def test_cumulative_register_via_counter_increments():
    t = pd.date_range("2024-01-01", periods=5, freq="30min")
    register = pd.Series([1000.0, 1010.0, 1025.0, 5.0, 20.0])  # meter reset
    df = _sig(t, counter_increments(register).to_numpy(), uuid="m")
    out = EnergyConsumptionEvents(df).consumption_by_window("m", window="1h")
    assert out["consumption"].sum() == pytest.approx(10 + 15 + 5 + 15)
