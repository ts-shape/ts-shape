"""Engineering pack on hand-computed examples (time-weighted definitions).

The signals are sampled, but the quantities are defined on time: a sample
holds its value until the next one (zero-order hold, the library's
``interval_end="next_change"`` convention), the time after the last sample
is not observed, and a step that crosses a window boundary belongs to both
windows in proportion to the time in each. Rates describe the step from the
previous sample to the current one. Every expected value below follows from
those definitions by hand.
"""

import math

import pandas as pd
import pytest

from ts_shape.events.engineering.control_loop_health import ControlLoopHealthEvents
from ts_shape.events.engineering.rate_of_change import RateOfChangeEvents
from ts_shape.events.engineering.startup_events import StartupDetectionEvents
from ts_shape.events.engineering.steady_state_detection import (
    SteadyStateDetectionEvents,
)
from ts_shape.events.engineering.threshold_monitoring import (
    ThresholdMonitoringEvents,
)
from ts_shape.events.engineering.warmup_analysis import WarmUpCoolDownEvents

T0 = pd.Timestamp("2025-01-01")


def _at(minutes: float) -> pd.Timestamp:
    return T0 + pd.Timedelta(minutes=minutes)


def _rows(uuid: str, points: list[tuple[float, float]]) -> pd.DataFrame:
    """One signal from ``(minute, value)`` points."""
    return pd.DataFrame(
        {
            "systime": [_at(m) for m, _ in points],
            "uuid": uuid,
            "value_double": [float(v) for _, v in points],
        }
    )


def _series(values: list[float], freq: str = "1min") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "systime": pd.date_range(T0, periods=len(values), freq=freq),
            "uuid": "s",
            "value_double": [float(v) for v in values],
        }
    )


def _loop(pv: list[tuple[float, float]], out=None) -> ControlLoopHealthEvents:
    parts = [_rows("sp", [(0, 0.0)]), _rows("pv", pv)]
    if out is not None:
        parts.append(_rows("out", out))
    df = pd.concat(parts, ignore_index=True)
    return ControlLoopHealthEvents(
        df, "sp", "pv", output_uuid="out" if out is not None else None
    )


# ---------------------------------------------------------------------------
# 1. Error integrals: IAE = int |e| dt, ISE = int e^2 dt,
#    ITAE = int (t - window_start) |e| dt, steps split at window boundaries.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("max_gap", [None, "1h"])
def test_error_integrals_no_time_lost_at_window_edges(max_gap):
    """e = 1 at minutes 0, 30, 60, 90; 1 h windows.

    Window 00:00 holds e = 1 for the full hour: IAE = ISE = 3600 s,
    ITAE = int_0^3600 t dt = 3600^2 / 2. Window 01:00 is observed from 60 to
    90 min (no time after the last sample): IAE = ISE = 1800 s,
    ITAE = 1800^2 / 2. (Per-window sums used to drop the step from 30 to 60
    min that crosses into the next window: IAE 1800 for window 00:00.)
    """
    loop = _loop([(0, 1.0), (30, 1.0), (60, 1.0), (90, 1.0)])
    out = loop.error_integrals(window="1h", max_gap=max_gap)
    assert out["start"].tolist() == [_at(0), _at(60)]
    assert out["iae"].tolist() == [3600.0, 1800.0]
    assert out["ise"].tolist() == [3600.0, 1800.0]
    assert out["itae"].tolist() == [3600.0**2 / 2, 1800.0**2 / 2]
    assert out["sample_count"].tolist() == [2, 2]
    assert out["bias"].tolist() == [1.0, 1.0]


def test_error_integrals_fully_observed_windows_are_3600():
    """e = 1 at minutes 0, 30, ..., 120: both hours fully observed, IAE 3600 each."""
    loop = _loop([(m, 1.0) for m in (0, 30, 60, 90, 120)])
    out = loop.error_integrals(window="1h")
    assert out["iae"].tolist()[:2] == [3600.0, 3600.0]


def test_error_integrals_step_assignment_per_mode():
    """e = 1 @ 0 min, 2 @ 50 min, 3 @ 70 min; 1 h windows.

    Default (``max_gap=None``): the step (t[i-1], t[i]] carries e[i].
    (0, 50] -> 2 for 3000 s; (50, 70] -> 3, 600 s in each window.
    IAE: 6000 + 1800 = 7800 and 1800. ITAE of window 01:00:
    3 * 600^2 / 2 = 540000.

    ``max_gap`` set: the step [t[i], t[i+1]) carries e[i] (zero-order hold).
    [0, 50) -> 1 for 3000 s; [50, 70) -> 2, 600 s in each window.
    IAE: 3000 + 1200 = 4200 and 1200. ISE of window 01:00: 4 * 600 = 2400.
    """
    loop = _loop([(0, 1.0), (50, 2.0), (70, 3.0)])
    default = loop.error_integrals(window="1h")
    assert default["iae"].tolist() == [7800.0, 1800.0]
    assert default["ise"].tolist() == [4 * 3000.0 + 9 * 600.0, 9 * 600.0]
    assert default["itae"].iloc[1] == pytest.approx(3 * 600.0**2 / 2)
    held = loop.error_integrals(window="1h", max_gap="1h")
    assert held["iae"].tolist() == [4200.0, 1200.0]
    assert held["ise"].tolist() == [3000.0 + 4 * 600.0, 4 * 600.0]
    # ITAE window 00:00: int_0^3000 t dt + 2 * int_3000^3600 t dt.
    expected = 3000.0**2 / 2 + 2 * (3600.0**2 - 3000.0**2) / 2
    assert held["itae"].iloc[0] == pytest.approx(expected)


def test_error_integrals_max_gap_drops_gap_and_empty_window():
    """e = 1 @ 0, 10, 100 min with ``max_gap='30min'``.

    [0, 10) is integrated (600 s); [10, 100) is a gap and contributes
    nothing, so window 01:00 receives no time and is omitted.
    """
    loop = _loop([(0, 1.0), (10, 1.0), (100, 1.0)])
    out = loop.error_integrals(window="1h", max_gap="30min")
    assert out["start"].tolist() == [_at(0)]
    assert out["iae"].tolist() == [600.0]


def test_error_integrals_window_without_samples_gets_held_time():
    """e = 2 @ 0 min, 2 @ 150 min (default mode): (0, 150] carries 2.

    Windows 00:00 and 01:00 hold 2 for 3600 s each (IAE 7200); window 02:00
    gets 30 min (IAE 3600). Window 01:00 has no sample of its own: it is
    still reported, with sample_count 0 and bias NaN.
    """
    loop = _loop([(0, 2.0), (150, 2.0)])
    out = loop.error_integrals(window="1h")
    assert out["iae"].tolist() == [7200.0, 7200.0, 3600.0]
    assert out["sample_count"].tolist() == [1, 0, 1]
    assert math.isnan(out["bias"].iloc[1])


# ---------------------------------------------------------------------------
# 2. Output saturation: shares of observed time, runs measured in time.
# ---------------------------------------------------------------------------


def test_output_saturation_is_time_weighted():
    """Output 100 @ 0 min, 50 @ 50 min, 50 @ 55 min; 1 h window, limits 0-100.

    Held values: 100 over [0, 50) (saturated high), 50 over [50, 55). The
    window is observed for 55 min (none after the last sample), so
    pct_time_at_high = 50 / 55 = 90.909 % and the longest saturated run is
    3000 s. (By sample counts it used to read 1 / 3 = 33.3 % and 0 s: the run
    of a single sample has no length between its own first and last sample.)
    """
    loop = _loop([(0, 0.0), (60, 0.0)], out=[(0, 100.0), (50, 50.0), (55, 50.0)])
    out = loop.output_saturation(window="1h")
    assert len(out) == 1
    assert out["pct_time_at_high"].iloc[0] == pytest.approx(100 * 50 / 55)
    assert out["pct_time_at_low"].iloc[0] == 0.0
    assert out["pct_time_saturated"].iloc[0] == pytest.approx(100 * 50 / 55)
    assert out["longest_saturation_seconds"].iloc[0] == 3000.0


def test_output_saturation_run_cut_at_window_boundary():
    """Output 0 @ 30 min, 50 @ 90 min: saturated low over [30, 90).

    Window 00:00: 30 min observed, all low -> 100 %, longest 1800 s.
    Window 01:00: [60, 90) observed, all low -> 100 %, longest 1800 s.
    """
    loop = _loop([(0, 0.0), (90, 0.0)], out=[(30, 0.0), (90, 50.0)])
    out = loop.output_saturation(window="1h")
    assert out["pct_time_at_low"].tolist() == [100.0, 100.0]
    assert out["longest_saturation_seconds"].tolist() == [1800.0, 1800.0]


# ---------------------------------------------------------------------------
# 3. Oscillation period: sub-millisecond spacing is not truncated.
# ---------------------------------------------------------------------------


def test_oscillation_period_keeps_sub_millisecond_precision():
    """Error +1 / -1 alternating every 1.5 ms: half-period 1.5 ms, period 3 ms.

    (Converting the crossing intervals to whole milliseconds read 1 ms
    half-periods, a 2 ms period.)
    """
    times = T0 + pd.to_timedelta([1500 * i for i in range(10)], unit="us")
    df = pd.concat(
        [
            pd.DataFrame({"systime": [T0], "uuid": "sp", "value_double": [0.0]}),
            pd.DataFrame(
                {
                    "systime": times,
                    "uuid": "pv",
                    "value_double": [1.0 if i % 2 == 0 else -1.0 for i in range(10)],
                }
            ),
        ],
        ignore_index=True,
    )
    out = ControlLoopHealthEvents(df, "sp", "pv").detect_oscillation(window="5min")
    assert out["estimated_period_seconds"].iloc[0] == pytest.approx(0.003)


# ---------------------------------------------------------------------------
# 4. Time above threshold: held values cover windows without samples.
# ---------------------------------------------------------------------------


def test_time_above_threshold_keeps_held_windows_and_first_exceedance():
    """100 @ 0 min, 0 @ 180 min, 0 @ 200 min; threshold 50; 1 h windows.

    100 is held over [0, 180): windows 00, 01, 02 are 3600 s above (100 %);
    window 03 holds 0 over [180, 200): 0 s. The signal starts above the
    threshold, which starts one exceedance at the first sample. (Windows
    01 and 02 have no samples and used to be dropped; the first sample did
    not count as an exceedance.)
    """
    df = _rows("s", [(0, 100.0), (180, 0.0), (200, 0.0)])
    out = ThresholdMonitoringEvents(df, "s").time_above_threshold(50, window="1h")
    assert out["start"].tolist() == [_at(0), _at(60), _at(120), _at(180)]
    assert out["time_above_seconds"].tolist() == [3600.0, 3600.0, 3600.0, 0.0]
    assert out["pct_above"].tolist() == [100.0, 100.0, 100.0, 0.0]
    assert out["exceedance_count"].tolist() == [1, 0, 0, 0]


# ---------------------------------------------------------------------------
# 5. Steady state: statistics over the steady samples only.
# ---------------------------------------------------------------------------


def test_steady_state_statistics_exclude_the_closing_sample():
    """20 zeros then 100 at 1 min spacing; window 5 min, std_threshold 1.

    Samples 1..19 have rolling std 0 (steady); sample 20 (the jump) does
    not. The interval runs from 00:01 to the transition at 00:20, but its
    mean and std are those of the steady samples: 0 and 0. (The closing
    sample 100 was included: mean 5.0, std 22.36.)
    """
    det = SteadyStateDetectionEvents(_series([0] * 20 + [100]), "s")
    kw = {"window": "5min", "std_threshold": 1, "min_duration": "10min"}
    out = det.detect_steady_state(**kw)
    assert out["start"].tolist() == [_at(1)]
    assert out["end"].tolist() == [_at(20)]
    assert out["mean_value"].tolist() == [0.0]
    assert out["std_value"].tolist() == [0.0]
    bands = det.steady_state_value_bands(**kw)
    assert bands[["mean_value", "lower_band", "upper_band"]].iloc[0].tolist() == [
        0.0,
        0.0,
        0.0,
    ]


@pytest.mark.parametrize("values", [[], [1.0]])
def test_steady_state_statistics_undefined_ratios_are_nan(values):
    """No data or one sample: no span, so steady_pct and the average are NaN."""
    df = _series(values) if values else _series([1.0]).iloc[:0]
    stats = SteadyStateDetectionEvents(df, "s").steady_state_statistics()
    assert math.isnan(stats["steady_pct"].iloc[0])
    assert math.isnan(stats["avg_steady_duration_seconds"].iloc[0])
    assert stats["num_steady_periods"].iloc[0] == 0


# ---------------------------------------------------------------------------
# 6. Time to target: overshoot within the ramp's own span.
# ---------------------------------------------------------------------------


def test_time_to_target_overshoot_stays_in_its_ramp():
    """[20, 60, 100, 100, 60, 20, 20, 80, 150, 150] at 1 min; target 100.

    Ramp 1 starts at 00:00 and reaches 100 at 00:02; until ramp 2 starts
    (00:06) the peak is 100: overshoot 0. Ramp 2 reaches 150 at 00:08:
    overshoot 50. (Ramp 1 used to take ramp 2's peak: overshoot 50.)
    """
    df = _series([20, 60, 100, 100, 60, 20, 20, 80, 150, 150])
    out = WarmUpCoolDownEvents(df, "s").time_to_target(100)
    assert out["start"].tolist() == [_at(0), _at(6)]
    assert out["time_to_target_seconds"].tolist() == [120.0, 120.0]
    assert out["overshoot"].tolist() == [0.0, 50.0]


# ---------------------------------------------------------------------------
# 7. Rapid change / slope startup: the rate at sample i is the step i-1 -> i.
# ---------------------------------------------------------------------------


def test_rapid_change_starts_at_sample_before_first_fast_step():
    """[0, 0, 10, 20, 30, 30, 30] at 1 s; threshold 5 units/s.

    Fast steps 1->2, 2->3, 3->4: the interval is 00:00:01 - 00:00:04 (3 s).
    (It used to start at 00:00:02: 2 s.)
    """
    df = _series([0, 0, 10, 20, 30, 30, 30], freq="1s")
    out = RateOfChangeEvents(df, "s").detect_rapid_change(5)
    assert out["start"].tolist() == [T0 + pd.Timedelta("1s")]
    assert out["end"].tolist() == [T0 + pd.Timedelta("4s")]
    assert out["duration_seconds"].tolist() == [3.0]


def test_rapid_change_single_step_lasts_one_interval():
    """[0, 0, 10, 10, 0] at 1 s; threshold 5: jumps 1->2 and 3->4, 1 s each."""
    df = _series([0, 0, 10, 10, 0], freq="1s")
    out = RateOfChangeEvents(df, "s").detect_rapid_change(5)
    assert out["duration_seconds"].tolist() == [1.0, 1.0]
    assert out["direction"].tolist() == ["increasing", "decreasing"]


def test_slope_startup_starts_at_sample_before_first_fast_step():
    """Same signals as the rapid-change probes, ``min_slope=5``.

    [0, 0, 10, 20, 30, 30, 30]: 00:00:01 - 00:00:04 (3 s; was 2 s).
    [0, 0, 10, 10, 0]: the single rising step lasts 1 s (was 0 s).
    """
    ramp = StartupDetectionEvents(_series([0, 0, 10, 20, 30, 30, 30], "1s"), "s")
    out = ramp.detect_startup_by_slope(min_slope=5)
    assert out["start"].tolist() == [T0 + pd.Timedelta("1s")]
    assert out["duration_seconds"].tolist() == [3.0]
    jump = StartupDetectionEvents(_series([0, 0, 10, 10, 0], "1s"), "s")
    out = jump.detect_startup_by_slope(min_slope=5)
    assert out["duration_seconds"].tolist() == [1.0]


def test_slope_startup_tz_aware_with_slope_window():
    """A positive ``slope_window`` on tz-aware data (used to raise).

    [0, 10, 20, 30] at 10 s, window 10 s: each slope is 1 unit/s over the
    step before it, so the run is 00:00:00 - 00:00:30.
    """
    df = _series([0, 10, 20, 30], freq="10s")
    df["systime"] = df["systime"].dt.tz_localize("Europe/Berlin")
    out = StartupDetectionEvents(df, "s").detect_startup_by_slope(
        min_slope=0.5, slope_window="10s"
    )
    assert out["duration_seconds"].tolist() == [30.0]


# ---------------------------------------------------------------------------
# 8. Startup quality: stability = 1 / (1 + CV) of the final >= 2 samples.
# ---------------------------------------------------------------------------


def test_startup_quality_stability_uses_at_least_two_samples():
    """Startup over [0, 10, 20, 30, 30] (1 min spacing).

    20 % of 5 samples is one sample, which has no std; the tail is at least
    two samples: [30, 30], CV 0, stability 1 / (1 + 0) = 1.0 (was 0.0).
    """
    sd = StartupDetectionEvents(_series([0, 10, 20, 30, 30]), "s")
    events = pd.DataFrame({"start": [_at(0)], "end": [_at(4)]})
    out = sd.assess_startup_quality(events)
    assert out["stability_score"].tolist() == [1.0]


def test_startup_quality_single_sample_is_nan():
    """A startup holding one sample has no stability: NaN, not 0.0."""
    sd = StartupDetectionEvents(_series([0, 10, 20, 30, 30]), "s")
    events = pd.DataFrame({"start": [_at(2)], "end": [_at(2)]})
    out = sd.assess_startup_quality(events)
    assert math.isnan(out["stability_score"].iloc[0])
