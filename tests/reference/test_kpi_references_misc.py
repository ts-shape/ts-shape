"""KPI definitions checked against their textbook sources.

Each test is a small probe whose expected value follows from the cited
definition, not from the implementation:

* ISO 22400-2 (KPIs for manufacturing operations management): performance,
  quality, availability, OEE, MTTR / MTBF; a KPI over a longer period is
  computed from the summed elements, not by summing or averaging ratios.
* K. Ogata, *Modern Control Engineering*: settling time ``t_s`` (the time
  after which the response stays within the band) and maximum percent
  overshoot ``M_p = exp(-zeta*pi / sqrt(1 - zeta^2))``.
* ISO 50006 (energy baselines and EnPIs): the EnPI of a period is the
  period's energy over its output.
* S. Nakajima, *Introduction to TPM*: minor stoppages are one of the six big
  losses and count as a *performance* (speed) loss.
"""

import warnings
from datetime import date

import numpy as np
import pandas as pd
import pytest

from ts_shape.errors import DataQualityWarning
from ts_shape.events.correlation.signal_correlation import SignalCorrelationEvents
from ts_shape.events.energy.efficiency_tracking import EnergyEfficiencyEvents
from ts_shape.events.energy.energy_performance_indicator import (
    EnergyPerformanceIndicatorEvents,
)
from ts_shape.events.engineering.setpoint_events import SetpointChangeEvents
from ts_shape.events.production.micro_stop_detection import MicroStopEvents
from ts_shape.events.production.performance_loss import PerformanceLossEvents
from ts_shape.events.production.period_summary import PeriodSummaryEvents
from ts_shape.events.supplychain.inventory_monitoring import (
    InventoryMonitoringEvents,
)

T0 = pd.Timestamp("2024-01-01 08:00", tz="UTC")

# ---------------------------------------------------------------------------
# A3. PerformanceLossEvents -- ISO 22400-2 performance
# ---------------------------------------------------------------------------


def _counter_plus_two_per_minute() -> pd.DataFrame:
    """Counter +2 every minute for 60 min: 120 parts at a 30 s cycle."""
    return pd.DataFrame(
        {
            "systime": [T0 + pd.Timedelta(minutes=i) for i in range(61)],
            "uuid": "cnt",
            "value_integer": [2 * i for i in range(61)],
        }
    )


def test_performance_loss_counts_parts_not_rows():
    """ISO 22400-2: performance = ideal cycle time x produced qty / run time.

    120 parts in 3600 s at an ideal 30 s cycle is 100 % with no speed loss
    (formerly 59 parts, 60 s cycle, 50 %, 1770 s loss: counter rows were
    counted and the first increment dropped).
    """
    perf = PerformanceLossEvents(_counter_plus_two_per_minute())
    row = perf.performance_by_shift("cnt", target_cycle_time=30.0).iloc[0]
    assert row["actual_parts"] == 120
    assert row["run_seconds"] == pytest.approx(3600.0)
    assert row["avg_cycle_time_seconds"] == pytest.approx(30.0)
    assert row["performance_pct"] == pytest.approx(100.0)
    assert row["loss_seconds"] == pytest.approx(0.0)

    trend = perf.performance_trend("cnt", target_cycle_time=30.0).iloc[0]
    assert trend["actual_parts"] == 120
    assert trend["performance_pct"] == pytest.approx(100.0)


def test_performance_loss_half_speed_and_reset():
    """A 60 s actual cycle against a 30 s ideal is 50 %; a reset loses nothing."""
    values = [0, 1, 2, 3, 0, 1, 2]  # reset after 3: +1 per minute throughout
    df = pd.DataFrame(
        {
            "systime": [T0 + pd.Timedelta(minutes=i) for i in range(len(values))],
            "uuid": "cnt",
            "value_integer": values,
        }
    )
    row = (
        PerformanceLossEvents(df)
        .performance_by_shift("cnt", target_cycle_time=30.0)
        .iloc[0]
    )
    # Reset-aware increments: 1, 1, 1, 0 (reset to 0 = 0 parts), 1, 1.
    assert row["actual_parts"] == 5
    assert row["run_seconds"] == pytest.approx(360.0)  # 0 -> 6 min
    assert row["performance_pct"] == pytest.approx(5 * 30 / 360 * 100)
    assert row["loss_seconds"] == pytest.approx(360 - 5 * 30)


# ---------------------------------------------------------------------------
# A4. SetpointChangeEvents -- Ogata settling time and percent overshoot
# ---------------------------------------------------------------------------

ZETA, WN = 0.2, 0.1  # damping ratio, natural frequency (rad/s)


def _second_order_step() -> pd.DataFrame:
    """Unit step response of an underdamped 2nd-order system, 1 s samples."""
    wd = WN * np.sqrt(1 - ZETA**2)
    phi = np.arccos(ZETA)
    t = np.arange(0, 601.0)
    y = 1 - np.exp(-ZETA * WN * t) / np.sqrt(1 - ZETA**2) * np.sin(wd * t + phi)
    sp = pd.DataFrame(
        {
            "systime": [T0 - pd.Timedelta("10s"), T0],
            "uuid": "sp",
            "value_double": [0.0, 1.0],
        }
    )
    pv = pd.DataFrame(
        {"systime": T0 + pd.to_timedelta(t, unit="s"), "uuid": "pv", "value_double": y}
    )
    return pd.concat([sp, pv], ignore_index=True)


def test_settling_time_is_last_entry_into_band():
    """Ogata: t_s is when the response enters the 2 % band and stays there.

    Theory (envelope approximation) 4 / (zeta * wn) = 200 s; the exact
    sampled response settles at 197 s. The first entry into the band, at
    18 s, was formerly reported.
    """
    out = SetpointChangeEvents(_second_order_step(), "sp").time_to_settle(
        "pv", settle_pct=0.02, lookahead="600s"
    )
    row = out.iloc[0]
    assert row["t_settle_seconds"] == pytest.approx(197.0)
    assert row["t_settle_seconds"] == pytest.approx(4 / (ZETA * WN), rel=0.03)
    assert bool(row["settled_within_window"])
    assert bool(row["settled"])


def test_not_settled_within_window():
    """A window ending at the 5th peak (t = 5 pi / wd = 160 s, 4 % off)."""
    out = SetpointChangeEvents(_second_order_step(), "sp").time_to_settle(
        "pv", settle_pct=0.02, lookahead="160s"
    )
    row = out.iloc[0]
    assert not bool(row["settled_within_window"])
    assert not bool(row["settled"])
    assert pd.isna(row["t_settle_seconds"])


def test_percent_overshoot_is_percent():
    """Ogata: M_p = exp(-zeta pi / sqrt(1 - zeta^2)) = 52.66 % for zeta = 0.2.

    ``overshoot_pct`` was formerly the fraction 0.5266.
    """
    mp = np.exp(-ZETA * np.pi / np.sqrt(1 - ZETA**2))
    out = SetpointChangeEvents(_second_order_step(), "sp").overshoot_metrics(
        "pv", window="600s"
    )
    row = out.iloc[0]
    assert row["overshoot_pct"] == pytest.approx(100 * mp, abs=0.01)
    assert row["overshoot_pct"] == pytest.approx(52.66, abs=0.01)
    assert row["overshoot_ratio"] == pytest.approx(mp, abs=1e-4)
    # The first undershoot (swing back past the setpoint) is M_p^2.
    assert row["undershoot_ratio"] == pytest.approx(mp**2, abs=1e-3)
    assert row["undershoot_pct"] == pytest.approx(100 * row["undershoot_ratio"])


def _swings_then_noise() -> pd.DataFrame:
    """A step 0 -> 10 whose error swings -10, +3, -2, +1 and then sits in
    +/-0.1 noise that flips sign on every sample (100 s at 1 Hz)."""
    err = np.r_[[-10.0] * 5, [3.0] * 5, [-2.0] * 5, [1.0] * 5, np.tile([0.1, -0.1], 40)]
    sp = pd.DataFrame(
        {
            "systime": [T0 - pd.Timedelta("10s"), T0],
            "uuid": "sp",
            "value_double": [0.0, 10.0],
        }
    )
    pv = pd.DataFrame(
        {
            "systime": T0 + pd.to_timedelta(np.arange(len(err)), unit="s"),
            "uuid": "pv",
            "value_double": 10.0 + err,
        }
    )
    return pd.concat([sp, pv], ignore_index=True)


def test_oscillation_count_ignores_noise_inside_settle_band():
    """Three swings through the setpoint (-10 -> +3 -> -2 -> +1); the +/-0.1
    noise afterwards stays inside the 0.5 settle band and is no oscillation.

    The former count of every sign change of the error was 3 + 79 = 82
    (80 noise samples alternate sign 79 times).
    """
    ev = SetpointChangeEvents(_swings_then_noise(), "sp")
    row = ev.control_quality_metrics("pv", tol=0.5, lookahead="100s").iloc[0]
    assert row["oscillation_count"] == 3
    # The settle band of settle_pct is 5 % of the 10-unit step = 0.5 as well.
    pct = ev.control_quality_metrics("pv", settle_pct=0.05, lookahead="100s")
    assert pct.iloc[0]["oscillation_count"] == 3
    # deadband=0 restores counting every sign change.
    raw = ev.control_quality_metrics("pv", tol=0.5, lookahead="100s", deadband=0)
    assert raw.iloc[0]["oscillation_count"] == 82
    assert ev.overshoot_metrics("pv", window="100s").iloc[0]["oscillation_count"] == 82
    banded = ev.overshoot_metrics("pv", window="100s", deadband=0.5)
    assert banded.iloc[0]["oscillation_count"] == 3


def test_oscillation_frequency_from_swings_outside_deadband():
    """Crossings at 5, 10 and 15 s: two half periods of 5 s, period 10 s."""
    ev = SetpointChangeEvents(_swings_then_noise(), "sp")
    out = ev.oscillation_frequency(
        "pv", window="100s", min_oscillations=1, deadband=0.5
    ).iloc[0]
    assert out["period_seconds"] == pytest.approx(10.0)
    assert out["oscillation_freq_hz"] == pytest.approx(0.1)
    cqm = ev.control_quality_metrics("pv", tol=0.5, lookahead="100s").iloc[0]
    # control_quality_metrics needs 2 oscillations (4 crossings): only 3 here.
    assert pd.isna(cqm["oscillation_freq_hz"])


def test_negative_deadband_rejected():
    ev = SetpointChangeEvents(_swings_then_noise(), "sp")
    with pytest.raises(ValueError, match="deadband"):
        ev.overshoot_metrics("pv", deadband=-1.0)


# ---------------------------------------------------------------------------
# A5. PeriodSummaryEvents.from_daily_data -- ratios from summed bases
# ---------------------------------------------------------------------------

D1, D2 = date(2024, 1, 1), date(2024, 1, 2)


def test_period_quality_from_summed_parts():
    """ISO 22400-2: quality rate = good quantity / produced quantity.

    900 good of 1000 and 0 of 10 is 900 / 1010 = 89.1 % (formerly the
    unweighted mean of 90 % and 0 %: 45.0 %).
    """
    daily = pd.DataFrame(
        {
            "date": [D1, D2],
            "ok_parts": [900, 0],
            "nok_parts": [100, 10],
            "total_parts": [1000, 10],
            "quality_pct": [90.0, 0.0],
            "nok_rate_pct": [10.0, 100.0],
        }
    )
    row = PeriodSummaryEvents.from_daily_data(daily).iloc[0]
    assert row["quality_pct"] == pytest.approx(100 * 900 / 1010)
    assert round(row["quality_pct"], 1) == 89.1
    assert row["nok_rate_pct"] == pytest.approx(100 * 110 / 1010)
    assert row["total_parts"] == 1010


def test_period_mttr_mtbf_from_failures():
    """ISO 22400-2: MTTR = repair time / failures, MTBF = uptime / failures.

    MTTR 1 h and 3 h with one failure each is 2 h (formerly summed: 4 h).
    """
    daily = pd.DataFrame(
        {"date": [D1, D2], "mttr_hours": [1.0, 3.0], "failures": [1, 1]}
    )
    row = PeriodSummaryEvents.from_daily_data(daily).iloc[0]
    assert row["mttr_hours"] == pytest.approx(2.0)
    assert row["failures"] == 2

    with_basis = pd.DataFrame(
        {
            "date": [D1, D2],
            "operating_hours": [20.0, 10.0],
            "repair_hours": [1.0, 6.0],
            "failures": [1, 2],
            "mtbf_hours": [20.0, 5.0],
            "mttr_hours": [1.0, 3.0],
            "availability_pct": [100 * 20 / 21, 100 * 10 / 16],
        }
    )
    row = PeriodSummaryEvents.from_daily_data(with_basis).iloc[0]
    assert row["mtbf_hours"] == pytest.approx(30 / 3)
    assert row["mttr_hours"] == pytest.approx(7 / 3)
    assert row["availability_pct"] == pytest.approx(100 * 30 / 37)


def test_period_enpi_not_summed():
    """ISO 50006: EnPI of a period = its energy / its output.

    EnPI 2 and 2 is 2 (formerly summed: 4). With the basis present the
    ratio of sums is used.
    """
    daily = pd.DataFrame({"date": [D1, D2], "enpi": [2.0, 2.0]})
    with pytest.warns(DataQualityWarning, match="enpi"):
        row = PeriodSummaryEvents.from_daily_data(daily).iloc[0]
    assert row["enpi"] == pytest.approx(2.0)

    based = pd.DataFrame(
        {
            "date": [D1, D2],
            "energy_kwh": [100.0, 30.0],
            "units_produced": [100.0, 10.0],
            "enpi": [1.0, 3.0],
        }
    )
    row = PeriodSummaryEvents.from_daily_data(based).iloc[0]
    assert row["enpi"] == pytest.approx(130 / 110)


def test_period_oee_from_summed_bases():
    """ISO 22400-2: OEE = availability x performance x quality of the period."""
    daily = pd.DataFrame(
        {
            "date": [D1, D2],
            "run_seconds": [3600.0, 7200.0],
            "planned_seconds": [7200.0, 7200.0],
            "actual_parts": [100, 200],
            "ideal_parts": [120.0, 240.0],
            "total_parts": [100, 200],
            "reject_parts": [10, 0],
            "good_parts": [90, 200],
            "availability_pct": [50.0, 100.0],
            "performance_pct": [100 * 100 / 120, 100 * 200 / 240],
            "quality_pct": [90.0, 100.0],
            "oee_pct": [37.5, 83.333],
        }
    )
    row = PeriodSummaryEvents.from_daily_data(daily).iloc[0]
    a, p, q = 10800 / 14400, 300 / 360, 290 / 300
    assert row["availability_pct"] == pytest.approx(100 * a)
    assert row["performance_pct"] == pytest.approx(100 * p)
    assert row["quality_pct"] == pytest.approx(100 * q)
    assert row["oee_pct"] == pytest.approx(100 * a * p * q)
    assert row["run_seconds"] == pytest.approx(10800.0)


def test_period_unknown_numeric_column_dropped_with_warning():
    daily = pd.DataFrame({"date": [D1, D2], "parts": [1, 2], "speed_setting": [5, 7]})
    with pytest.warns(DataQualityWarning, match="speed_setting"):
        out = PeriodSummaryEvents.from_daily_data(daily)
    assert "speed_setting" not in out.columns
    assert out["parts"].iloc[0] == 3
    kept = PeriodSummaryEvents.from_daily_data(
        daily, aggregations={"speed_setting": "max"}
    )
    assert kept["speed_setting"].iloc[0] == 7


# ---------------------------------------------------------------------------
# A7a. EnPI / SEC rolling baselines -- ISO 50006
# ---------------------------------------------------------------------------


def _energy_frame() -> pd.DataFrame:
    """Hour 1: 100 kWh for 100 units (EnPI 1); hour 2: 30 kWh for 10 (EnPI 3)."""
    h0 = pd.Timestamp("2024-01-01 00:00")
    return pd.DataFrame(
        [
            {"systime": h0, "uuid": "m", "value_double": 100.0},
            {"systime": h0 + pd.Timedelta("1h"), "uuid": "m", "value_double": 30.0},
            {"systime": h0 - pd.Timedelta("1min"), "uuid": "c", "value_integer": 0},
            {"systime": h0 + pd.Timedelta("30min"), "uuid": "c", "value_integer": 100},
            {"systime": h0 + pd.Timedelta("90min"), "uuid": "c", "value_integer": 110},
        ]
    )


def test_enpi_rolling_baseline_is_ratio_of_sums():
    """Baseline over both hours = 130 kWh / 110 units (formerly mean 2.0)."""
    df = _energy_frame()
    # The counter's baseline reading sits in the hour before (no energy there).
    out = EnergyPerformanceIndicatorEvents(df).enpi_vs_baseline(
        "m", "c", window="1h", baseline_window=2
    )
    out = out[out["start"] >= pd.Timestamp("2024-01-01 00:00")]
    assert out["enpi"].tolist() == pytest.approx([1.0, 3.0])
    assert out["baseline_enpi"].iloc[-1] == pytest.approx(130 / 110)


def test_sec_trend_uses_ratio_of_sums():
    """SEC trend compares with the trailing period's summed energy / output."""
    out = EnergyEfficiencyEvents(_energy_frame()).specific_energy_consumption(
        "m", "c", window="1h"
    )
    out = out[out["start"] >= pd.Timestamp("2024-01-01 00:00")]
    assert out["sec"].tolist() == pytest.approx([1.0, 3.0])
    # 3.0 vs 130/110 = 1.18 -> worsening (the mean 2.0 would also say so);
    # the first hour's 1.0 vs its own window's ratio 1.0 -> stable.
    assert out["sec_trend"].tolist() == ["stable", "worsening"]


# ---------------------------------------------------------------------------
# A7b. Inventory stockout prediction -- refill-aware consumption
# ---------------------------------------------------------------------------


def test_stockout_prediction_refill_does_not_cancel_consumption():
    """Consume 10/h, refill +10, consume 10/h twice: 30 units used in 4 h.

    Rate 7.5/h and 80 / 7.5 = 10.67 h to stockout (formerly first - last =
    20 / 4 h = 5/h and 16 h).
    """
    tt = pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {"systime": tt, "uuid": "lvl", "value_double": [100.0, 90.0, 100.0, 90.0, 80.0]}
    )
    inv = InventoryMonitoringEvents(df, "lvl")
    out = inv.stockout_prediction(consumption_rate_window="4h")
    last = out.iloc[-1]
    assert last["consumption_rate"] == pytest.approx(7.5)
    assert last["estimated_stockout_time_hours"] == pytest.approx(80 / 7.5)
    # Right after the refill: 10 units used in 2 h (formerly 0 -> no estimate).
    assert out["consumption_rate"].iloc[2] == pytest.approx(5.0)
    old = inv.stockout_prediction(consumption_rate_window="4h", handle_refills=False)
    assert old["consumption_rate"].iloc[-1] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# A7c. Micro-stops -- Nakajima: a performance loss
# ---------------------------------------------------------------------------


def test_micro_stop_loss_is_performance_loss():
    """Minor stoppages are a performance loss (six big losses)."""
    states = [True, False, True, False, True]
    tt = [T0 + pd.Timedelta(seconds=s) for s in (0, 100, 110, 210, 220)]
    df = pd.DataFrame({"systime": tt, "uuid": "run", "value_bool": states})
    out = MicroStopEvents(df, "run").micro_stop_impact(window="1h", max_duration="30s")
    row = out.iloc[0]
    assert "performance_loss_pct" in out.columns
    assert row["performance_loss_pct"] == row["availability_loss_pct"]
    assert row["performance_loss_pct"] > 0


# ---------------------------------------------------------------------------
# A7d. Signal correlation lag sign convention
# ---------------------------------------------------------------------------


def test_lag_sign_convention():
    """``lag_periods = k`` pairs a[t] with b[t - k]: b(t) = a(t - 5) gives -5.

    A negative best lag means b lags (follows) a; positive means b leads a.
    """
    rng = np.random.default_rng(0)
    n = 300
    t = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    x = rng.normal(size=n + 5)
    df = pd.concat(
        [
            pd.DataFrame({"systime": t, "uuid": "a", "value_double": x[5:]}),
            pd.DataFrame({"systime": t, "uuid": "b", "value_double": x[:-5]}),
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DataQualityWarning)
        out = SignalCorrelationEvents(df).lag_correlation(
            "a", "b", window="1min", max_lag=10
        )
    best = out.loc[out["is_best_lag"]].iloc[0]
    assert best["lag_periods"] == -5
    assert best["correlation"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# EnPI regression baseline -- ISO 50006: fit only windows with output
# ---------------------------------------------------------------------------


def _regression_frame() -> pd.DataFrame:
    """Days 1-9 on the line energy = 200 + 2 * units; day 0 used 600 kWh
    with no output (the counter's stop/warm-up day)."""
    units = [0, 50, 120, 80, 200, 150, 60, 250, 90, 170]
    rows, count = [], 0
    day0 = pd.Timestamp("2024-01-01", tz="UTC")
    for d, u in enumerate(units):
        day = day0 + pd.Timedelta(days=d)
        energy = 600.0 if u == 0 else 200.0 + 2.0 * u
        rows.append((day + pd.Timedelta(hours=12), "meter", energy, None))
        rows.append((day, "counter", None, count))
        count += u
        rows.append((day + pd.Timedelta(hours=23, minutes=59), "counter", None, count))
    return pd.DataFrame(
        rows, columns=["systime", "uuid", "value_double", "value_integer"]
    )


def test_enpi_regression_baseline_skips_windows_without_output():
    """The fit recovers a = 200 kWh, b = 2 kWh/unit exactly. With the
    zero-output day in the fit (formerly) the intercept rose to ~330."""
    ev = EnergyPerformanceIndicatorEvents(_regression_frame())
    with pytest.warns(DataQualityWarning, match=r"1 baseline window.*2024-01-01"):
        out = ev.enpi_vs_baseline(
            "meter", "counter", window="1D", baseline="regression", baseline_window=10
        )
    on_line = out.iloc[1:]
    assert on_line["deviation_pct"].abs().max() < 1e-9
    assert out["expected_energy_kwh"].iloc[0] == pytest.approx(200.0)
    assert on_line["baseline_enpi"].iloc[0] == pytest.approx(200.0 / 50 + 2.0)
    # The former fit, including the idle day:
    units = np.array([0, 50, 120, 80, 200, 150, 60, 250, 90, 170], dtype=float)
    energy = np.where(units == 0, 600.0, 200.0 + 2.0 * units)
    old_slope, old_intercept = np.polyfit(units, energy, 1)
    assert old_intercept > 300 and old_slope < 1.5
