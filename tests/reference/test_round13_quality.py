"""Quality-pack definitions checked on hand-computed probes (round 13).

Each test states the definition it checks in its docstring; the expected
values follow from that definition, not from the implementation.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from ts_shape.errors import DataQualityWarning
from ts_shape.eventlog import to_event_log
from ts_shape.eventlog.schema import TS_SEVERITY
from ts_shape.events.maintenance.asset_health_index import AssetHealthIndexEvents
from ts_shape.events.production.flow_constraints import FlowConstraintEvents
from ts_shape.events.quality.anomaly_classification import (
    AnomalyClassificationEvents,
)
from ts_shape.events.quality.calibration_validation import (
    CalibrationValidationEvents,
)
from ts_shape.events.quality.data_gap_analysis import DataGapAnalysisEvents
from ts_shape.events.quality.environmental_compensation import (
    EnvironmentalCompensationEvents,
)
from ts_shape.events.quality.sampling_resolution import SamplingResolutionEvents
from ts_shape.events.quality.statistical_process_control import (
    StatisticalProcessControlEvents,
)
from ts_shape.events.quality.tolerance_deviation import ToleranceDeviationEvents
from ts_shape.transform.calculator.numeric_calc import IntegerCalc

T0 = pd.Timestamp("2024-01-01")


def _minutes(m: float) -> pd.Timestamp:
    return T0 + pd.Timedelta(minutes=m)


def _signal(times, values, uuid="s") -> pd.DataFrame:
    return pd.DataFrame(
        {"systime": times, "uuid": uuid, "value_double": values, "is_delta": True}
    )


# ---------------------------------------------------------------------------
# 1-2. ToleranceDeviationEvents
# ---------------------------------------------------------------------------


def _tolerance_frame(rows, *, is_delta=True) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["systime", "uuid", "value_double"])
    if is_delta is not None:
        df["is_delta"] = is_delta
    return df


_LAG_ROWS = [
    (_minutes(0), "usl", 10.0),
    (_minutes(10), "usl", 20.0),
    (_minutes(0), "lsl", 0.0),
    (_minutes(1), "act", 5.0),
    (_minutes(12), "act", 15.0),
    (_minutes(30), "act", 25.0),
]


def _tde(df: pd.DataFrame, **kwargs) -> ToleranceDeviationEvents:
    return ToleranceDeviationEvents(
        df,
        "value_double",
        "value_double",
        actual_uuid="act",
        upper_tolerance_uuid="usl",
        lower_tolerance_uuid="lsl",
        **kwargs,
    )


def test_tolerance_lag_delays_the_limits_not_the_events():
    """A limit set at ``t`` takes effect at ``t + tolerance_lag``.

    Upper limit 10 from min 0 and 20 from min 10 (lower 0). With a 5 min lag
    the limit 10 is in force until min 15, so 15 at min 12 is outside; 25 at
    min 30 is outside 20. Events keep the measured times (00:12, 00:30).
    Without the lag, 15 at min 12 is inside the new limit 20.
    """
    lagged = _tde(_tolerance_frame(_LAG_ROWS), tolerance_lag="5min")
    out = lagged.detect_deviations()
    assert out["systime"].tolist() == [_minutes(12), _minutes(30)]
    assert out["value_double"].tolist() == [15.0, 25.0]
    assert out["deviation_abs"].tolist() == [5.0, 5.0]

    plain = _tde(_tolerance_frame(_LAG_ROWS)).detect_deviations()
    assert plain["systime"].tolist() == [_minutes(30)]


def test_tolerance_snapshot_limits_and_missing_is_delta_are_used():
    """Every row of a limit signal sets the limit, whatever ``is_delta``.

    Limit rows flagged ``is_delta=False`` (snapshots) and frames without an
    ``is_delta`` column give the same events as ``is_delta=True``.
    """
    expected = _tde(_tolerance_frame(_LAG_ROWS)).detect_deviations()
    assert expected["systime"].tolist() == [_minutes(30)]
    snapshot = _tolerance_frame(_LAG_ROWS)
    snapshot.loc[snapshot["uuid"] != "act", "is_delta"] = False
    no_flag = _tolerance_frame(_LAG_ROWS, is_delta=None)
    for frame in (snapshot, no_flag):
        out = _tde(frame).detect_deviations()
        assert out["systime"].tolist() == [_minutes(30)]
        assert out["value_double"].tolist() == [25.0]


def test_tolerance_severity_is_by_distance_beyond_the_limit():
    """Severity from the distance ``d`` beyond the violated limit.

    Limits [0, 10], range R = 10, warning_threshold 0.8: ``minor`` for
    ``d <= 8``, ``major`` for ``8 < d <= 20``, ``critical`` for ``d > 20``.
    11 -> d=1 minor, 19 -> d=9 major, 31 -> d=21 critical, -9 -> d=9 major.
    ``severity_level``: minor -> warn (the detector's override), major and
    critical -> critical.
    """
    rows = [
        (_minutes(0), "usl", 10.0),
        (_minutes(0), "lsl", 0.0),
        (_minutes(1), "act", 11.0),
        (_minutes(11), "act", 19.0),
        (_minutes(21), "act", 31.0),
        (_minutes(31), "act", -9.0),
    ]
    out = _tde(_tolerance_frame(rows)).detect_deviations()
    assert out["deviation_abs"].tolist() == [1.0, 9.0, 21.0, 9.0]
    assert out["severity"].tolist() == ["minor", "major", "critical", "major"]
    assert out["severity_level"].tolist() == [
        "warn",
        "critical",
        "critical",
        "critical",
    ]


def test_capability_indices_without_actual_signal_is_empty():
    """No rows of the actual signal: an empty frame and a DataQualityWarning."""
    rows = [(_minutes(0), "usl", 10.0), (_minutes(0), "lsl", 0.0)]
    with pytest.warns(DataQualityWarning):
        out = _tde(_tolerance_frame(rows)).capability_indices()
    assert out.empty
    assert "Cpk" in out.columns


# ---------------------------------------------------------------------------
# 3-4. AnomalyClassificationEvents
# ---------------------------------------------------------------------------


def _brute_force_run_starts(v: np.ndarray, tol: float) -> np.ndarray:
    """The definition: a run continues while |v - first value of run| <= tol."""
    starts = np.zeros(len(v), dtype=bool)
    anchor = np.nan
    for i, x in enumerate(v):
        if not abs(x - anchor) <= tol:
            starts[i] = True
            anchor = x
    return starts


def test_flatline_run_is_anchored_at_its_first_value():
    """A slow ramp is no flatline.

    ``v = 0.05 * i`` at 1 s for 121 s, tolerance 0.1: each run holds at most
    three samples (|v - anchor| <= 0.1), 2 s long, so nothing reaches
    ``min_duration="1min"``. (Consecutive steps of 0.05 would have made the
    whole ramp one 2-minute flatline.) A constant 5.0 is one 120 s flatline,
    and values wobbling within 0.08 of the first value stay one run.
    """
    t = pd.date_range(T0, periods=121, freq="1s")
    ramp = AnomalyClassificationEvents(_signal(t, np.arange(121) * 0.05), "s")
    assert ramp.detect_flatline(min_duration="1min", tolerance=0.1).empty

    stuck = AnomalyClassificationEvents(_signal(t, np.full(121, 5.0)), "s")
    out = stuck.detect_flatline(min_duration="1min", tolerance=0.1)
    assert out["duration_seconds"].tolist() == [120.0]
    assert out["stuck_value"].tolist() == [5.0]

    wobble = 5.0 + np.where(np.arange(121) % 2 == 0, 0.0, 0.08)
    out = AnomalyClassificationEvents(_signal(t, wobble), "s").detect_flatline(
        min_duration="1min", tolerance=0.1
    )
    assert out["duration_seconds"].tolist() == [120.0]


def test_flatline_run_starts_match_the_definition():
    """The vectorized run split equals the sample-by-sample definition."""
    rng = np.random.default_rng(3)
    v = np.cumsum(rng.normal(0, 0.03, 3000))
    v[1000:1400] = v[1000]  # a long stuck stretch (crosses search chunks)
    v[2000] = np.nan
    ramp = np.arange(3000) * 0.01 + rng.normal(0, 0.002, 3000)
    for values in (v, ramp):
        for tol in (0.0, 0.02, 0.1, 0.5):
            got = AnomalyClassificationEvents._anchor_run_starts(
                values, tol, np.zeros(len(values), dtype=bool)
            )
            np.testing.assert_array_equal(got, _brute_force_run_starts(values, tol))


def test_drift_evaluates_the_trailing_partial_window():
    """Windows are ``[t0 + k*w, t0 + (k+1)*w)`` and cover all samples.

    91 one-minute samples: flat 0 for 00:00-00:59, then +1 per minute (1/60
    per second) to 00:90. ``window="1h"``: [00:00, 01:00) is flat, the
    trailing partial window [01:00, 02:00) holds the ramp, slope 1/60 >= 0.01.
    """
    t = pd.date_range(T0, periods=91, freq="1min")
    v = np.where(np.arange(91) >= 60, np.arange(91) - 60.0, 0.0)
    det = AnomalyClassificationEvents(_signal(t, v), "s")
    out = det.detect_drift(window="1h")
    assert out["start"].tolist() == [_minutes(60)]
    assert out["end"].tolist() == [_minutes(120)]
    assert out["slope"].tolist() == pytest.approx([1 / 60])
    kinds = det.classify_anomalies(window="1h")
    assert kinds["anomaly_type"].tolist() == ["flatline", "drift"]


def test_window_edge_sample_belongs_to_the_later_window_only():
    """A sample at ``t0 + w`` is in the second window, not in both.

    0 for 00:00-00:59 and 1000 from 01:00 to 01:29: both half-open windows are
    flat, so no drift (a closed first window would end on the step to 1000,
    a fitted slope of 1000 * 30 / 18910 = 1.59 per minute = 0.026 / s >= 0.01).
    """
    t = pd.date_range(T0, periods=90, freq="1min")
    v = np.where(np.arange(90) >= 60, 1000.0, 0.0)
    assert (
        AnomalyClassificationEvents(_signal(t, v), "s").detect_drift(window="1h").empty
    )


# ---------------------------------------------------------------------------
# 5. SamplingResolutionEvents
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "step_us,rate_hz",
    [(1500, 1e6 / 1500), (500, 2000.0)],
)
def test_sub_millisecond_sampling_intervals(step_us, rate_hz):
    """Intervals in float seconds from the nanosecond times.

    40 samples of a 50 Hz sine every 1.5 ms: mean interval 1.5 ms, rate
    1 / 0.0015 = 666.7 Hz, FFT bins of 1 / (40 * 0.0015) = 16.67 Hz, so the
    3 full cycles put the peak exactly on 50 Hz. Every 0.5 ms: 2000 Hz
    (bins of 50 Hz, one full cycle -> 50 Hz).
    """
    n = 40
    t = T0 + pd.to_timedelta(np.arange(n) * step_us, unit="us")
    v = np.sin(2 * np.pi * 50 * np.arange(n) * step_us / 1e6)
    det = SamplingResolutionEvents(_signal(t, v), "s")
    alias = det.detect_aliasing(window="1h")
    assert alias["sampling_rate_hz"].tolist() == pytest.approx([rate_hz])
    assert alias["dominant_frequency"].tolist() == pytest.approx([50.0])
    jitter = det.sampling_jitter(window="1h")
    assert jitter["mean_interval_seconds"].tolist() == pytest.approx([step_us / 1e6])
    assert jitter["std_interval_seconds"].tolist() == pytest.approx([0.0], abs=1e-12)


# ---------------------------------------------------------------------------
# 6. DataGapAnalysisEvents
# ---------------------------------------------------------------------------


def test_gap_is_strictly_longer_than_min_gap():
    """Regular 10-minute sampling with ``min_gap="10min"`` has no gaps."""
    t = pd.date_range(T0, periods=12, freq="10min")
    det = DataGapAnalysisEvents(_signal(t, np.arange(12.0)), "s")
    assert det.detect_gaps(min_gap="10min").empty
    assert len(det.detect_gaps(min_gap="9min")) == 11


def test_default_min_gap_is_twice_the_median_interval():
    """``min_gap=None``: 2 x the median positive sampling interval.

    Regular 1-minute samples 00:00..00:10: no gaps, gap_fraction 0, span
    600 s, shortest / mean gap NaN. Dropping 00:04..00:07 leaves one 5-minute
    step (00:03 -> 00:08) > 2 min: one gap of 300 s over a 600 s span.
    """
    t = pd.date_range(T0, periods=11, freq="1min")
    det = DataGapAnalysisEvents(_signal(t, np.arange(11.0)), "s")
    assert det.detect_gaps().empty
    row = det.gap_summary().iloc[0]
    assert row["total_gaps"] == 0
    assert row["total_missing_seconds"] == 0.0
    assert row["gap_fraction"] == 0.0
    assert row["data_span_seconds"] == 600.0
    assert np.isnan(row["shortest_gap_seconds"])
    assert np.isnan(row["mean_gap_seconds"])

    holed = t.delete([4, 5, 6, 7])
    det = DataGapAnalysisEvents(_signal(holed, np.arange(7.0)), "s")
    gaps = det.detect_gaps()
    assert gaps["start"].tolist() == [_minutes(3)]
    assert gaps["end"].tolist() == [_minutes(8)]
    row = det.gap_summary().iloc[0]
    assert row["total_gaps"] == 1
    assert row["data_span_seconds"] == 600.0
    assert row["gap_fraction"] == 0.5
    assert row["shortest_gap_seconds"] == row["mean_gap_seconds"] == 300.0


# ---------------------------------------------------------------------------
# 7. Calendar windows
# ---------------------------------------------------------------------------


def _months_frame() -> pd.DataFrame:
    """Hourly signal 10.5 in January, 11.0 in February 2024; reference
    10.0 in January, 10.2 in February."""
    t = pd.date_range("2024-01-01", "2024-02-29 23:00", freq="1h")
    jan = t.month == 1
    sig = _signal(t, np.where(jan, 10.5, 11.0), "sig")
    ref = _signal(t, np.where(jan, 10.0, 10.2), "ref")
    return pd.concat([sig, ref], ignore_index=True)


def test_calibration_month_start_windows():
    """``window="MS"``: calendar months [Jan 1, Feb 1), [Feb 1, Mar 1).

    The reference is averaged over the same month: bias 10.5 - 10.0 = 0.5,
    then 11.0 - 10.2 = 0.8; drift 0.3. Stability over the lookback: bias
    mean 0.5, then (0.5 + 0.8) / 2 = 0.65.
    """
    det = CalibrationValidationEvents(_months_frame(), "sig", reference_uuid="ref")
    out = det.detect_recalibration_need(window="MS", max_bias=0.6)
    assert out["start"].tolist() == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-02-01"),
    ]
    assert out["end"].tolist() == [
        pd.Timestamp("2024-02-01"),
        pd.Timestamp("2024-03-01"),
    ]
    assert out["bias"].tolist() == pytest.approx([0.5, 0.8])
    assert out["drift_rate"].tolist() == pytest.approx([0.0, 0.3])
    assert out["recalibration_needed"].tolist() == [False, True]
    stab = det.calibration_stability(window="MS")
    assert stab["bias_mean"].tolist() == pytest.approx([0.5, 0.65])


def test_env_exceedance_calendar_windows():
    """``window="MS"`` and a calendar ``"1D"`` give the true window end.

    Environment 20 in January and 40 (above the valid 0..30) in February:
    one February row ending Mar 1. In Europe/Berlin the day of the
    spring-forward change (2024-03-31) is 23 h: its window ends at
    2024-04-01 00:00+02:00.
    """
    t = pd.date_range("2024-01-01", "2024-02-29 23:00", freq="1h")
    env = _signal(t, np.where(t.month == 1, 20.0, 40.0), "env")
    det = EnvironmentalCompensationEvents(
        pd.concat([env, _signal(t, np.ones(len(t)), "sig")]), "sig", env_uuid="env"
    )
    out = det.detect_env_exceedance((0.0, 30.0), window="MS")
    assert out["start"].tolist() == [pd.Timestamp("2024-02-01")]
    assert out["window_end"].tolist() == [pd.Timestamp("2024-03-01")]
    assert out["end"].tolist() == [pd.Timestamp("2024-03-01")]

    berlin = pd.date_range("2024-03-31", periods=23, freq="1h", tz="Europe/Berlin")
    env = _signal(berlin, np.full(23, 40.0), "env")
    det = EnvironmentalCompensationEvents(
        pd.concat([env, _signal(berlin, np.ones(23), "sig")]), "sig", env_uuid="env"
    )
    out = det.detect_env_exceedance((0.0, 30.0), window="1D")
    end = pd.Timestamp("2024-04-01", tz="Europe/Berlin")
    assert out["window_end"].tolist() == [end]
    assert out["duration_seconds"].tolist() == [23 * 3600.0]


# ---------------------------------------------------------------------------
# 8. AssetHealthIndexEvents.health_events direction
# ---------------------------------------------------------------------------


def test_health_event_direction_follows_the_band_change():
    """``below`` when the band worsens, ``above`` on a recovery.

    Scores 90, 40, 60, 60, 80, 65 with thresholds warning 70 / critical 50:
    bands ok, critical, warning, warning, ok, warning. Crossings: 40 falls
    below 50 (critical), 60 rises above 50 (warning, a recovery), 65 falls
    below 70 (warning). Every-window mode adds the second 60, sitting below
    70.
    """
    t = pd.date_range(T0, periods=40, freq="1h")
    det = AssetHealthIndexEvents(
        _signal(t, np.full(40, 50.0), "temp"),
        asset_id="press-01",
        signals={"temp": "thermal"},
    )
    starts = pd.date_range(T0, periods=6, freq="1h")
    index = pd.DataFrame(
        {
            "start": starts,
            "end": starts + pd.Timedelta(hours=1),
            "asset_health_score": [90.0, 40.0, 60.0, 60.0, 80.0, 65.0],
            "dominant_failure_mode": "thermal",
            "confidence_level": 1.0,
        }
    )
    det.health_index = lambda **_: index  # type: ignore[method-assign]
    out = det.health_events(warning_threshold=70.0, critical_threshold=50.0)
    assert out["asset_health_score"].tolist() == [40.0, 60.0, 65.0]
    assert out["severity"].tolist() == ["critical", "warning", "warning"]
    assert out["direction"].tolist() == ["below", "above", "below"]
    assert out["threshold"].tolist() == [50.0, 50.0, 70.0]
    every = det.health_events(
        warning_threshold=70.0, critical_threshold=50.0, only_crossings=False
    )
    assert every["asset_health_score"].tolist() == [40.0, 60.0, 60.0, 65.0]
    assert every["direction"].tolist() == ["below", "above", "below", "below"]


# ---------------------------------------------------------------------------
# 9. IntegerCalc.power_column
# ---------------------------------------------------------------------------


def test_integer_power_falls_back_to_float_when_integers_cannot_hold_it():
    """Exact int64 for a whole non-negative exponent that fits; else float64.

    [2, 3, 10] ** 64 = 1.8e19, 3.4e30, 1e64 (beyond int64); ** -1 = 0.5,
    0.333, 0.1; ** 2 = 4, 9, 100 (int64).
    """
    df = pd.DataFrame({"value_integer": [2, 3, 10]})
    big = IntegerCalc.power_column(df, power_value=64)["value_integer"]
    assert big.dtype == np.float64
    assert big.tolist() == pytest.approx([2.0**64, 3.0**64, 1e64])
    inv = IntegerCalc.power_column(df, power_value=-1)["value_integer"]
    assert inv.tolist() == pytest.approx([0.5, 1 / 3, 0.1])
    sq = IntegerCalc.power_column(df, power_value=2)["value_integer"]
    assert sq.dtype == np.int64
    assert sq.tolist() == [4, 9, 100]
    nullable = pd.DataFrame({"value_integer": pd.array([2, None, 10], dtype="Int64")})
    out = IntegerCalc.power_column(nullable, power_value=-1)["value_integer"]
    assert str(out.dtype) == "Float64"
    assert out[0] == 0.5 and pd.isna(out[1]) and out[2] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# 10. One severity table: event log severity == severity_level
# ---------------------------------------------------------------------------


def _log_matches_frame(frame: pd.DataFrame, detector: str) -> list:
    assert "severity_level" in frame.columns
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DataQualityWarning)
        log = to_event_log(frame, detector=detector)
    got = (
        log.events[TS_SEVERITY]
        .astype(object)
        .where(log.events[TS_SEVERITY].notna(), None)
    )
    want = (
        frame["severity_level"]
        .astype(object)
        .where(frame["severity_level"].notna(), None)
    )
    assert sorted(got.tolist(), key=str) == sorted(want.tolist(), key=str)
    return sorted({v for v in want if v is not None})


def test_event_log_severity_of_tolerance_deviations():
    """minor -> warn (detector override), major / critical -> critical."""
    rows = [
        (_minutes(0), "usl", 10.0),
        (_minutes(0), "lsl", 0.0),
        (_minutes(1), "act", 11.0),
        (_minutes(11), "act", 19.0),
        (_minutes(21), "act", 31.0),
    ]
    out = _tde(_tolerance_frame(rows)).detect_deviations()
    levels = _log_matches_frame(out, "ToleranceDeviationEvents.detect_deviations")
    assert levels == ["critical", "warn"]


def test_event_log_severity_of_cusum_shifts():
    """CUSUM ``high`` -> warn (override), ``critical`` -> critical.

    Reference values alternate 9 / 11 (MR-bar 2, sigma 2 / 1.128); actual
    values of 14 accumulate past H = 5 sigma and then 2 H.
    """
    ref = _signal(
        pd.date_range(T0, periods=20, freq="1min"),
        np.where(np.arange(20) % 2 == 0, 9.0, 11.0),
        "ref",
    )
    act = _signal(
        pd.date_range(_minutes(30), periods=20, freq="1min"), np.full(20, 14.0), "act"
    )
    det = StatisticalProcessControlEvents(
        pd.concat([ref, act], ignore_index=True), "value_double", "ref", "act", "spc"
    )
    out = det.detect_cusum_shifts()
    assert {"high", "critical"} <= set(out["severity"])
    levels = _log_matches_frame(
        out, "StatisticalProcessControlEvents.detect_cusum_shifts"
    )
    assert levels == ["critical", "warn"]


def test_event_log_severity_of_flow_constraints():
    """minor -> info, moderate -> warn, severe -> critical.

    Upstream runs throughout; downstream is stopped for 3 s, 10 s and 60 s:
    blocked events of 3 s (minor), 10 s (moderate) and 60 s (severe).
    """
    t = pd.date_range(T0, periods=200, freq="1s")
    sec = np.arange(200)
    down = ~(
        ((sec >= 10) & (sec < 13))
        | ((sec >= 30) & (sec < 40))
        | ((sec >= 100) & (sec < 160))
    )
    frame = pd.concat(
        [
            pd.DataFrame({"systime": t, "uuid": "up", "value_bool": True}),
            pd.DataFrame({"systime": t, "uuid": "down", "value_bool": down}),
        ],
        ignore_index=True,
    ).assign(is_delta=True)
    out = FlowConstraintEvents(frame).blocked_events(
        roles={"upstream_run": "up", "downstream_run": "down"}
    )
    assert out["severity"].tolist() == ["minor", "moderate", "severe"]
    assert out["severity_level"].tolist() == ["info", "warn", "critical"]
    levels = _log_matches_frame(out, "FlowConstraintEvents.blocked_events")
    assert levels == ["critical", "info", "warn"]
