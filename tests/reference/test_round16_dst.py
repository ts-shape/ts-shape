"""Round 16: per-window time shares on DST days, hand-computed.

On tz-aware data a daily window is a calendar day: in Europe/Berlin
2024-03-31 (spring forward) lasts 23 h = 82800 s and 2024-10-27 (fall back)
25 h = 90000 s. A time share, rate or percentage per window is measured
against that real ``[start, end)``, not a fixed 24 h. A summer day
(2024-06-01, 86400 s) checks that ordinary days are unchanged. The second
part checks window bounds and membership: a daily row ends at the next local
midnight and counts what falls inside that real day.

Each test states the definition it checks in its docstring; the expected
values follow from that definition, not from the implementation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ts_shape.events.engineering.control_loop_health import (
    ControlLoopHealthEvents,
)
from ts_shape.events.engineering.disturbance_recovery import (
    DisturbanceRecoveryEvents,
)
from ts_shape.events.engineering.material_balance import MaterialBalanceEvents
from ts_shape.events.engineering.threshold_monitoring import (
    ThresholdMonitoringEvents,
    _time_weighted_above_fraction_loop,
)
from ts_shape.events.maintenance.asset_health_index import AssetHealthIndexEvents
from ts_shape.events.production.bottleneck_detection import (
    BottleneckDetectionEvents,
)
from ts_shape.events.production.duty_cycle import DutyCycleEvents
from ts_shape.events.production.flow_metrics import FlowMetricsEvents
from ts_shape.events.production.line_balancing import LineBalancingEvents
from ts_shape.events.production.micro_stop_detection import MicroStopEvents
from ts_shape.events.production.performance_loss import PerformanceLossEvents
from ts_shape.events.quality.multi_sensor_validation import (
    MultiSensorValidationEvents,
)
from ts_shape.events.quality.sensor_drift import SensorDriftEvents
from ts_shape.events.supplychain.inventory_monitoring import (
    InventoryMonitoringEvents,
)

TZ = "Europe/Berlin"

#: (local day, its length in seconds)
SPRING = ("2024-03-31", 82800.0)
FALL = ("2024-10-27", 90000.0)
SUMMER = ("2024-06-01", 86400.0)
DAYS = [SPRING, FALL, SUMMER]
DAY_IDS = ["spring-23h", "fall-25h", "summer-24h"]


def _midnight(day: str) -> pd.Timestamp:
    return pd.Timestamp(day, tz=TZ)


def _next_midnight(day: str) -> pd.Timestamp:
    return _midnight(day) + pd.DateOffset(days=1)


def _grid(day: str, freq: str, *, closed: bool = False) -> pd.DatetimeIndex:
    """Instants every ``freq`` (in real time) over the local day ``day``."""
    t = pd.date_range(
        _midnight(day).tz_convert("UTC"),
        _next_midnight(day).tz_convert("UTC"),
        freq=freq,
        inclusive="both" if closed else "left",
    )
    return t.tz_convert(TZ)


def _frame(times, uuid: str, column: str, values) -> pd.DataFrame:
    df = pd.DataFrame({"systime": times, "uuid": uuid, column: values})
    for col in ("value_double", "value_integer", "value_bool", "value_string"):
        if col not in df.columns:
            df[col] = np.nan if col != "value_bool" else None
    return df


def _first_day(out: pd.DataFrame, day: str) -> pd.Series:
    rows = out[out["start"] == _midnight(day)]
    assert len(rows) == 1, out
    return rows.iloc[0]


# ---------------------------------------------------------------------------
# ThresholdMonitoringEvents.time_above_threshold
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("day", "seconds"), DAYS, ids=DAY_IDS)
def test_time_above_always_above_is_the_whole_day(day, seconds):
    """Always above (samples every 15 min through to next midnight): the time
    above is the whole calendar day and the share 100 %."""
    t = _grid(day, "15min", closed=True)
    df = _frame(t, "s", "value_double", 10.0)
    out = ThresholdMonitoringEvents(df, "s").time_above_threshold(5.0, window="1D")
    row = _first_day(out, day)
    assert row["end"] == _next_midnight(day)
    assert row["duration_seconds"] == seconds
    assert row["time_above"] == seconds
    assert row["time_above_seconds"] == seconds
    assert row["pct_above"] == 100.0
    assert row["time_above_pct"] == 100.0


@pytest.mark.parametrize(("day", "seconds"), DAYS, ids=DAY_IDS)
def test_time_above_from_noon_is_twelve_hours_of_the_real_day(day, seconds):
    """Below until 12:00 local, above from 12:00 to midnight: 12 h = 43200 s
    above, as a share of the real day (52.17 % on 23 h, 48 % on 25 h)."""
    t = _grid(day, "15min", closed=True)
    noon = pd.Timestamp(f"{day} 12:00", tz=TZ)
    df = _frame(t, "s", "value_double", np.where(t >= noon, 10.0, 0.0))
    out = ThresholdMonitoringEvents(df, "s").time_above_threshold(5.0, window="1D")
    row = _first_day(out, day)
    assert row["time_above"] == 43200.0
    assert row["pct_above"] == pytest.approx(43200.0 / seconds * 100, rel=1e-12)


@pytest.mark.parametrize(("day", "seconds"), DAYS, ids=DAY_IDS)
def test_time_above_last_sample_scales_the_sample_share_by_the_real_day(day, seconds):
    """``interval_end="last_sample"``: share of samples above times the real
    window length. Uniform 15-min samples, above from 12:00: 48 of the day's
    samples (92 / 100 / 96) are above, i.e. 43200 s."""
    t = _grid(day, "15min")
    noon = pd.Timestamp(f"{day} 12:00", tz=TZ)
    df = _frame(t, "s", "value_double", np.where(t >= noon, 10.0, 0.0))
    out = ThresholdMonitoringEvents(df, "s").time_above_threshold(
        5.0, window="1D", interval_end="last_sample"
    )
    row = _first_day(out, day)
    assert row["duration_seconds"] == seconds
    assert row["time_above"] == pytest.approx(43200.0, rel=1e-12)
    assert row["pct_above"] == pytest.approx(48 / (seconds / 900) * 100, rel=1e-12)


@pytest.mark.parametrize(("day", "seconds"), DAYS, ids=DAY_IDS)
def test_time_above_reference_loop_uses_the_real_window(day, seconds):
    """The reference loop (used when a timestamp is missing) divides by the
    real window length too: always above over the day is a fraction of 1."""
    t = _grid(day, "15min", closed=True)
    above = pd.Series(1.0, index=t)
    starts = pd.DatetimeIndex([_midnight(day)])
    ends = pd.DatetimeIndex([_next_midnight(day)])
    assert _time_weighted_above_fraction_loop(above, starts, ends) == [1.0]
    fast = ThresholdMonitoringEvents._time_weighted_above_fraction(above, starts, ends)
    assert fast == [1.0]


def test_threshold_exceedance_trend_total_is_the_real_day():
    """``total_duration_seconds`` is the time above per window: always above
    on 2024-03-31 and 2024-10-27 gives 82800 s and 90000 s."""
    for day, seconds in (SPRING, FALL):
        t = _grid(day, "15min").append(_grid(_next_midnight(day).strftime("%F"), "1h"))
        df = _frame(t, "s", "value_double", 10.0)
        out = ThresholdMonitoringEvents(df, "s").threshold_exceedance_trend(
            5.0, window="1D"
        )
        row = _first_day(out, day)
        assert row["duration_seconds"] == seconds
        assert row["total_duration_seconds"] == seconds
        assert row["total_duration"] == seconds


# ---------------------------------------------------------------------------
# MicroStopEvents.micro_stop_frequency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("day", "seconds"), DAYS, ids=DAY_IDS)
def test_micro_stop_share_is_of_the_real_day(day, seconds):
    """One 20-s stop at half past every hour of the day: 23 / 25 / 24 stops,
    460 / 500 / 480 s lost, i.e. 5/9 % of the real day on each of them."""
    hours = _grid(day, "1h")
    n = len(hours)
    stop = hours + pd.Timedelta("30min")
    back = stop + pd.Timedelta("20s")
    times = (
        hours.append(stop).append(back).append(pd.DatetimeIndex([_next_midnight(day)]))
    )
    values = [True] * n + [False] * n + [True] * n + [True]
    df = _frame(times, "run", "value_bool", values).sort_values("systime")
    out = MicroStopEvents(df, "run").micro_stop_frequency(
        window="1D", max_duration="30s"
    )
    row = _first_day(out, day)
    assert row["duration_seconds"] == seconds
    assert row["count"] == n == seconds / 3600
    assert row["total_lost_time"] == 20.0 * n
    assert row["pct_of_window"] == pytest.approx(5 / 9, rel=1e-12)
    assert row["share_of_window_pct"] == row["pct_of_window"]


# ---------------------------------------------------------------------------
# DutyCycleEvents.duty_cycle_by_window (sample mean)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("day", "seconds"), DAYS, ids=DAY_IDS)
def test_duty_cycle_sample_mean_splits_the_real_day(day, seconds):
    """``time_weighted=False``: alternating on/off samples give a 50 % duty
    cycle; on and off time are each half of the real day."""
    t = _grid(day, "15min")
    df = _frame(t, "m", "value_bool", np.arange(len(t)) % 2 == 0)
    out = DutyCycleEvents(df, "m").duty_cycle_by_window(
        window="1D", time_weighted=False
    )
    row = _first_day(out, day)
    assert row["duration_seconds"] == seconds
    assert row["duty_cycle_pct"] == 50.0
    assert row["on_time"] == seconds / 2
    assert row["off_time"] == seconds / 2
    assert row["on_time"] + row["off_time"] == row["duration_seconds"]


# ---------------------------------------------------------------------------
# InventoryMonitoringEvents.consumption_rate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("day", "seconds"), DAYS, ids=DAY_IDS)
def test_consumption_rate_is_per_hour_of_the_real_day(day, seconds):
    """One unit consumed per real hour: a reading every hour, from the hour
    before midnight through the day. The day's readings (23 / 25 / 24) each
    report a drop of 1, so 1.0 unit per hour; ``first - last`` is one unit
    less (the first reading's drop happened before it)."""
    hours = _grid(day, "1h")
    t = pd.DatetimeIndex([hours[0] - pd.Timedelta("1h")]).append(hours)
    level = 1000.0 - np.arange(len(t), dtype=float)
    df = _frame(t, "lvl", "value_double", level)
    inv = InventoryMonitoringEvents(df, "lvl")
    row = _first_day(inv.consumption_rate(window="1D"), day)
    n = seconds / 3600
    assert row["duration_seconds"] == seconds
    assert row["consumption_rate"] == pytest.approx(1.0, rel=1e-12)
    row = _first_day(inv.consumption_rate(window="1D", handle_refills=False), day)
    assert row["consumption_rate"] == pytest.approx((n - 1) / n, rel=1e-12)


# ---------------------------------------------------------------------------
# FlowMetricsEvents.throughput / flow_summary
# ---------------------------------------------------------------------------


def _flow_frame(day: str) -> pd.DataFrame:
    """Entry at :00, exit at :30 of every real hour of ``day`` (pulses)."""
    hours = _grid(day, "1h")
    parts = []
    for uuid, shift in (("in", "0min"), ("out", "30min")):
        on = hours + pd.Timedelta(shift)
        off = on + pd.Timedelta("1min")
        times = on.append(off)
        parts.append(
            _frame(times, uuid, "value_bool", [True] * len(on) + [False] * len(on))
        )
    return pd.concat(parts, ignore_index=True).sort_values("systime")


@pytest.mark.parametrize(("day", "seconds"), DAYS, ids=DAY_IDS)
def test_throughput_is_per_hour_of_the_real_day(day, seconds):
    """One unit out per real hour: 23 / 25 / 24 units, 1.0 per hour, and the
    window ends at the next local midnight."""
    flow = FlowMetricsEvents(_flow_frame(day), "in", "out")
    row = _first_day(flow.throughput(window="1D"), day)
    assert row["end"] == _next_midnight(day)
    assert row["duration_seconds"] == seconds
    assert row["units_out"] == seconds / 3600
    assert row["throughput_per_hour"] == 1.0
    summary = _first_day(flow.flow_summary(window="1D"), day)
    assert summary["end"] == _next_midnight(day)
    assert summary["duration_seconds"] == seconds
    assert summary["throughput_per_hour"] == 1.0


# ---------------------------------------------------------------------------
# DisturbanceRecoveryEvents.disturbance_frequency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("day", "seconds"), DAYS, ids=DAY_IDS)
def test_disturbance_share_is_of_the_real_day(day, seconds, monkeypatch):
    """A disturbance lasting a tenth of the real day reads 10 %.

    The detector is fed one known disturbance (``detect_disturbances`` is
    stubbed), so only the per-window share is under test.
    """
    t = _grid(day, "15min")
    df = _frame(t, "s", "value_double", 1.0)
    det = DisturbanceRecoveryEvents(df, "s")
    start = pd.Timestamp(f"{day} 10:00", tz=TZ)
    disturbances = pd.DataFrame(
        {"start": [start], "end": [start], "duration_seconds": [seconds / 10]}
    )
    recovery = pd.DataFrame(
        {
            "start": pd.Series([], dtype=f"datetime64[ns, {TZ}]"),
            "recovered": pd.Series([], dtype=bool),
            "recovery_time_seconds": pd.Series([], dtype=float),
        }
    )
    monkeypatch.setattr(det, "detect_disturbances", lambda **_: disturbances)
    monkeypatch.setattr(det, "recovery_time", lambda **_: recovery)
    row = _first_day(det.disturbance_frequency(window="1D"), day)
    assert row["end"] == _next_midnight(day)
    assert row["duration_seconds"] == seconds
    assert row["disturbance_count"] == 1
    assert row["total_disturbance_seconds"] == seconds / 10
    assert row["pct_time_disturbed"] == pytest.approx(10.0, rel=1e-12)


# ===========================================================================
# Round 16 follow-up: window bounds and membership on DST days
#
# Methods that label rows with a resample / floor window must report the
# window's real ``end`` (the next local midnight) and ``duration_seconds``,
# and must count what falls inside that real window -- not inside
# ``start + 24 h``.
# ===========================================================================


def _assert_day_bounds(row: pd.Series, day: str, seconds: float) -> None:
    assert row["start"] == _midnight(day)
    assert row["end"] == _next_midnight(day)
    assert row["duration_seconds"] == seconds


def _two_days(day: str, freq: str) -> pd.DatetimeIndex:
    """``_grid`` over ``day`` and the following local day."""
    nxt = _next_midnight(day).strftime("%F")
    return _grid(day, freq).append(_grid(nxt, freq))


@pytest.mark.parametrize(("day", "seconds"), DAYS, ids=DAY_IDS)
def test_detect_bottleneck_window_ends_at_next_midnight(day, seconds):
    """The bottleneck row of a day spans that calendar day."""
    t = _grid(day, "15min", closed=True)
    df = pd.concat(
        [
            _frame(t, "s1", "value_bool", True),
            _frame(t, "s2", "value_bool", np.arange(len(t)) % 2 == 0),
        ],
        ignore_index=True,
    )
    out = BottleneckDetectionEvents(df).detect_bottleneck(["s1", "s2"], window="1D")
    row = _first_day(out, day)
    _assert_day_bounds(row, day, seconds)
    assert row["bottleneck_uuid"] == "s1"


@pytest.mark.parametrize(("day", "seconds"), DAYS, ids=DAY_IDS)
def test_asset_health_index_window_is_the_calendar_day(day, seconds):
    """A daily health score covers the calendar day: 23 / 25 / 24 h."""
    t = _two_days(day, "15min")
    df = _frame(t, "s", "value_double", 1.0)
    out = AssetHealthIndexEvents(df, "a", {"s": "x"}).health_index(
        window="1D", baseline_window="1h"
    )
    _assert_day_bounds(_first_day(out, day), day, seconds)


@pytest.mark.parametrize(("day", "seconds"), DAYS, ids=DAY_IDS)
def test_detect_disagreement_window_is_the_calendar_day(day, seconds):
    """Two sensors 10 apart all day: one daily disagreement, the whole day
    long (``duration`` = 23 / 25 / 24 h)."""
    t = _grid(day, "15min")
    df = pd.concat(
        [_frame(t, "a", "value_double", 0.0), _frame(t, "b", "value_double", 10.0)],
        ignore_index=True,
    )
    out = MultiSensorValidationEvents(df, ["a", "b"]).detect_disagreement(
        1.0, window="1D"
    )
    row = _first_day(out, day)
    _assert_day_bounds(row, day, seconds)
    assert row["duration"] == pd.Timedelta(seconds=seconds)
    assert row["max_spread"] == 10.0


def _drift_frame(day: str) -> pd.DataFrame:
    """Signal 15 over two days; reference 10 inside ``day`` and 1000 on the
    following day, so a reference window that leaks past midnight shows."""
    t = _two_days(day, "15min")
    inside = t < _next_midnight(day)
    return pd.concat(
        [
            _frame(t, "s", "value_double", 15.0),
            _frame(t, "ref", "value_double", np.where(inside, 10.0, 1000.0)),
        ],
        ignore_index=True,
    )


@pytest.mark.parametrize(("day", "seconds"), DAYS, ids=DAY_IDS)
def test_sensor_drift_reference_window_is_the_calendar_day(day, seconds):
    """The reference of a daily window is the reference mean over that day:
    10. Zero offset 15 - 10 = 5, sensitivity 15 / 10 = 1.5, bias 5."""
    sd = SensorDriftEvents(_drift_frame(day), "s", reference_uuid="ref")
    zero = _first_day(sd.detect_zero_drift(window="1D"), day)
    _assert_day_bounds(zero, day, seconds)
    assert zero["mean_offset"] == 5.0
    span = _first_day(sd.detect_span_drift(window="1D"), day)
    _assert_day_bounds(span, day, seconds)
    assert span["sensitivity"] == 1.5
    health = _first_day(sd.calibration_health(window="1D", tolerance=10.0), day)
    _assert_day_bounds(health, day, seconds)
    assert health["bias"] == 5.0


@pytest.mark.parametrize(("day", "seconds"), DAYS, ids=DAY_IDS)
def test_balance_exceedance_ends_at_the_last_window_end(day, seconds):
    """Input 100, output 50 all day: one daily window out of balance, so one
    exceedance from midnight to the next midnight."""
    t = _grid(day, "15min", closed=True)
    df = pd.concat(
        [
            _frame(t, "in", "value_double", 100.0),
            _frame(t, "out", "value_double", 50.0),
        ],
        ignore_index=True,
    )
    out = MaterialBalanceEvents(df, ["in"], ["out"]).detect_balance_exceedance(
        window="1D", tolerance_pct=5.0, min_duration="0s"
    )
    _assert_day_bounds(_first_day(out, day), day, seconds)


def _pulses(day: str, uuid: str, shift: str = "0min") -> pd.DataFrame:
    """A 1-min True pulse at ``shift`` past every real hour of ``day``."""
    on = _grid(day, "1h") + pd.Timedelta(shift)
    off = on + pd.Timedelta("1min")
    times = on.append(off)
    return _frame(times, uuid, "value_bool", [True] * len(on) + [False] * len(on))


@pytest.mark.parametrize(("day", "seconds"), DAYS, ids=DAY_IDS)
def test_line_balancing_windows_are_the_calendar_day(day, seconds):
    """Station cycle times and balance metrics per calendar day."""
    df = pd.concat(
        [_pulses(day, "s1"), _pulses(day, "s2", "20min")], ignore_index=True
    ).sort_values("systime")
    lb = LineBalancingEvents(df, {"s1": "S1", "s2": "S2"})
    for out in (
        lb.station_cycle_times(window="1D"),
        lb.balance_metrics(window="1D", takt_time=3600),
    ):
        assert len(out)
        for _, row in out.iterrows():
            _assert_day_bounds(row, day, seconds)


@pytest.mark.parametrize(("day", "seconds"), DAYS, ids=DAY_IDS)
def test_slow_periods_window_is_the_calendar_day(day, seconds):
    """One part every 15 min against a 1-s target is slow all day; the slow
    window is the calendar day."""
    t = _grid(day, "15min")
    df = _frame(t, "n", "value_integer", np.arange(len(t)))
    out = PerformanceLossEvents(df).slow_periods("n", 1.0, window="1D")
    _assert_day_bounds(_first_day(out, day), day, seconds)


@pytest.mark.parametrize(("day", "seconds"), DAYS, ids=DAY_IDS)
def test_wip_over_time_window_is_the_calendar_day(day, seconds):
    """Work in process per calendar day."""
    flow = FlowMetricsEvents(_flow_frame(day), "in", "out")
    _assert_day_bounds(_first_day(flow.wip_over_time(window="1D"), day), day, seconds)


def _edge_hour(day: str) -> pd.Timestamp:
    """The hour a fixed 24 h window gets wrong: on the 23 h day the first
    hour of the next day (inside ``start + 24 h``), on the 25 h day the last
    hour of the day itself (outside ``start + 24 h``)."""
    if day == FALL[0]:
        return _next_midnight(day) - pd.Timedelta("1h")
    return _next_midnight(day)


def _loop_frame(day: str) -> pd.DataFrame:
    """Setpoint 0; actual 0.5 (no zero crossing) except during the edge hour,
    where it alternates +1 / -1 every minute."""
    t = _two_days(day, "1min")
    edge = _edge_hour(day)
    in_edge = (t >= edge) & (t < edge + pd.Timedelta("1h"))
    sign = np.where(np.arange(len(t)) % 2 == 0, 1.0, -1.0)
    actual = np.where(in_edge, sign, 0.5)
    return pd.concat(
        [
            _frame(t, "sp", "value_double", 0.0),
            _frame(t, "pv", "value_double", actual),
        ],
        ignore_index=True,
    )


@pytest.mark.parametrize(("day", "seconds"), DAYS, ids=DAY_IDS)
def test_detect_oscillation_window_ends_at_next_midnight(day, seconds):
    """Oscillation on daily analysis windows: one episode, the calendar day
    that holds the edge hour (the next day for the 23 h and 24 h days)."""
    out = ControlLoopHealthEvents(_loop_frame(day), "sp", "pv").detect_oscillation(
        window="1D", min_crossings=4
    )
    assert len(out) == 1
    edge_day = _edge_hour(day).strftime("%F")
    row = out.iloc[0]
    assert row["start"] == _midnight(edge_day)
    assert row["end"] == _next_midnight(edge_day)


@pytest.mark.parametrize(("day", "seconds"), DAYS, ids=DAY_IDS)
def test_loop_health_counts_oscillations_inside_the_calendar_day(day, seconds):
    """``oscillation_count`` counts episodes starting inside the real day.
    The only episode (hourly analysis windows) is the edge hour: on the 23 h
    day it belongs to the next day, on the 25 h day to the day itself."""
    out = ControlLoopHealthEvents(_loop_frame(day), "sp", "pv").loop_health_summary(
        window="1D", oscillation_window="1h", min_crossings=4
    )
    row = _first_day(out, day)
    _assert_day_bounds(row, day, seconds)
    assert row["oscillation_count"] == (1 if day == FALL[0] else 0)


@pytest.mark.parametrize(("day", "seconds"), DAYS, ids=DAY_IDS)
def test_excessive_cycling_averages_intervals_inside_the_calendar_day(day, seconds):
    """On for 30 min at the top of every real hour; in the edge hour on for
    only 5 min. The day's mean on-duration uses the on intervals that start
    inside the real day: 1800 s on the 23 h and 24 h days (the edge hour is
    the next day's), (24 * 1800 + 300) / 25 = 1740 s on the 25 h day."""
    t = _two_days(day, "1h")
    edge = _edge_hour(day)
    on_for = [pd.Timedelta("5min") if h == edge else pd.Timedelta("30min") for h in t]
    off = pd.DatetimeIndex([h + d for h, d in zip(t, on_for, strict=True)])
    times = t.append(off)
    df = _frame(times, "m", "value_bool", [True] * len(t) + [False] * len(t))
    out = DutyCycleEvents(df.sort_values("systime"), "m").excessive_cycling(
        max_transitions=1, window="1D"
    )
    row = _first_day(out, day)
    _assert_day_bounds(row, day, seconds)
    expected = 1740.0 if day == FALL[0] else 1800.0
    assert row["avg_on_duration"] == pytest.approx(expected, rel=1e-12)
