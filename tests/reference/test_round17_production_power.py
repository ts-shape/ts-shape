"""Round 17: performance-loss period booking, power integration windows and
grid sampling in the harmonizer, hand-computed.

Covers cycles split at window and shift boundaries in
``PerformanceLossEvents``, calendar windows (``'W'``, ``'MS'``, ``'ME'``,
DST days) and the total coverage of ``PowerIntegrator.energy_by_window``,
and the ``sampling`` modes and edge handling of ``DataHarmonizer``.

Each test states the definition it checks in its docstring; the expected
values follow from that definition, not from the implementation.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ts_shape.events.production.performance_loss import PerformanceLossEvents
from ts_shape.transform.calculator.power_integration import PowerIntegrator
from ts_shape.transform.harmonization import DataHarmonizer


def _counter(times: list[pd.Timestamp], values: list[int]) -> pd.DataFrame:
    return pd.DataFrame({"systime": times, "uuid": "c", "value_integer": values})


# -- PerformanceLossEvents: a cycle is booked to the windows it ran in ----


def test_trend_day_run_seconds_never_exceed_the_day():
    """A cycle ``(t_prev, t]`` is cut at midnight.

    Readings every 10 min from 2024-03-04 23:40; the 23:50 reading does not
    count up (idle), so the first cycle is 23:40 -> 00:00 (1200 s, 1 part)
    and lies wholly in 03-04. 03-05 then holds the 143 cycles 00:00 -> 23:50:
    85 800 s. The old booking (whole cycle to the window of its last reading)
    gave 03-05 87 000 s > 86 400 s.
    """
    t = list(pd.date_range("2024-03-04 23:40", "2024-03-05 23:50", freq="10min"))
    df = _counter(t, [0, 0, *range(1, len(t) - 1)])
    out = PerformanceLossEvents(df).performance_trend("c", 600.0, window="1D")
    assert list(out["start"]) == [
        pd.Timestamp("2024-03-04"),
        pd.Timestamp("2024-03-05"),
    ]
    assert list(out["actual_parts"]) == [1, 143]
    assert list(out["run_seconds"]) == [1200.0, 85800.0]
    # 1 part x 600 s / 1200 s; 143 x 600 / 85 800.
    assert list(out["performance_pct"]) == pytest.approx([50.0, 100.0])
    assert (out["run_seconds"] <= out["duration_seconds"]).all()


def _long_cycle_frame() -> pd.DataFrame:
    """1 part/min in hour 00, one 50-min cycle to 01:50, 1/min from 02:00."""
    times = (
        list(pd.date_range("2024-01-01 00:00", "2024-01-01 01:00", freq="1min"))
        + [pd.Timestamp("2024-01-01 01:50")]
        + list(pd.date_range("2024-01-01 02:00", "2024-01-01 03:00", freq="1min"))
    )
    return _counter(times, list(range(len(times))))


def test_slow_periods_no_false_slow_hour_after_a_long_cycle():
    """Hour 02 is running at target; only hour 01 is slow.

    The cycle (00:59, 01:00] is hour 00's. Hour 01 holds (01:00, 01:50]
    (3000 s, 1 part) and (01:50, 02:00]
    (600 s, 1 part): 3600 s, 2 parts, 2 x 60 / 3600 = 3.33 %. Hour 02 holds
    the 60 one-minute cycles ending 02:01 .. 03:00: 3600 s, 100 %. The old
    booking put the 01:50->02:00 cycle in hour 02 (4140 s, 86.96 %, flagged).
    """
    out = PerformanceLossEvents(_long_cycle_frame()).slow_periods(
        "c", 60.0, window="1h"
    )
    assert list(out["start"]) == [pd.Timestamp("2024-01-01 01:00")]
    row = out.iloc[0]
    assert row["end"] == pd.Timestamp("2024-01-01 02:00")
    assert row["duration_seconds"] == 3600.0
    assert row["run_seconds"] == 3600.0
    assert row["actual_parts"] == 2
    assert row["performance_pct"] == pytest.approx(200 / 60)
    assert row["loss_seconds"] == pytest.approx(3600.0 - 120.0)

    trend = PerformanceLossEvents(_long_cycle_frame()).performance_trend(
        "c", 60.0, window="1h"
    )
    assert list(trend["run_seconds"]) == [3600.0, 3600.0, 3600.0]
    assert list(trend["actual_parts"]) == [60, 2, 60]


def test_slow_periods_keeps_windows_holding_one_long_cycle():
    """A 150-min cycle spreads over three hours (no ``len(grp) < 2`` drop).

    Counter 0 at 00:00, 1 at 02:30: the one part is shared by time,
    3600 / 9000 = 0.4 to hours 00 and 01 and 0.2 to hour 02. Target 60 s:
    performance = 0.4 x 60 / 3600 = 0.667 % (hours 00, 01) and
    0.2 x 60 / 1800 = 0.667 % (hour 02).
    """
    df = _counter(
        [pd.Timestamp("2024-01-01 00:00"), pd.Timestamp("2024-01-01 02:30")], [0, 1]
    )
    out = PerformanceLossEvents(df).slow_periods("c", 60.0, window="1h")
    assert list(out["start"]) == list(
        pd.date_range("2024-01-01 00:00", periods=3, freq="1h")
    )
    assert list(out["run_seconds"]) == [3600.0, 3600.0, 1800.0]
    assert list(out["actual_parts"]) == pytest.approx([0.4, 0.4, 0.2])
    assert list(out["performance_pct"]) == pytest.approx([2 / 3] * 3)


def test_performance_by_shift_splits_a_cycle_at_the_shift_change():
    """The 13:50 -> 14:10 cycle is half in shift_1 and half in shift_2.

    Counter 0, 1, 2, 3 at 13:00, 13:50, 14:10, 15:00. shift_1 (06-14):
    (13:00, 13:50] 3000 s, 1 part + (13:50, 14:00] 600 s, 0.5 part = 3600 s,
    1.5 parts. shift_2: (14:00, 14:10] 600 s, 0.5 part + (14:10, 15:00]
    3000 s, 1 part = 3600 s, 1.5 parts. Target 600 s: 1.5 x 600 / 3600 =
    25 %, loss 3600 - 900 = 2700 s.
    """
    times = [
        pd.Timestamp(f"2024-01-02 {hm}") for hm in ("13:00", "13:50", "14:10", "15:00")
    ]
    out = PerformanceLossEvents(_counter(times, [0, 1, 2, 3])).performance_by_shift(
        "c", 600.0
    )
    assert list(out["shift"]) == ["shift_1", "shift_2"]
    assert list(out["run_seconds"]) == [3600.0, 3600.0]
    assert list(out["actual_parts"]) == pytest.approx([1.5, 1.5])
    assert list(out["performance_pct"]) == pytest.approx([25.0, 25.0])
    assert list(out["loss_seconds"]) == pytest.approx([2700.0, 2700.0])


# -- PowerIntegrator: calendar windows and total coverage ------------------


def _flat_kw(start: str, end: str, tz: str | None = None) -> pd.DataFrame:
    t = pd.date_range(start, end, freq="1h", tz=tz)
    return pd.DataFrame({"systime": t, "uuid": "m", "value_double": 1.0})


@pytest.mark.parametrize("window", ["ME", "MS"])
def test_month_windows_are_the_real_months(window):
    """1 kW from 01-10 00:00 to 02-20 00:00: Jan 22 d x 24 h = 528 kWh,
    Feb 19 d x 24 h = 456 kWh. 'ME' is labeled by the month's last day; the
    window is still [month start, next month start). The old 'ME' grid
    started at the label and billed February -477 540 kWh.
    """
    out = PowerIntegrator.energy_by_window(
        _flat_kw("2024-01-10", "2024-02-20"), window=window
    )
    assert list(out["start"]) == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-02-01"),
    ]
    assert list(out["end"]) == [pd.Timestamp("2024-02-01"), pd.Timestamp("2024-03-01")]
    assert list(out["kwh"]) == pytest.approx([528.0, 456.0])
    assert list(out["duration_seconds"]) == [31 * 86400.0, 29 * 86400.0]
    assert list(out["coverage"]) == pytest.approx([528 / 744, 456 / 696])


def test_week_windows_run_monday_to_monday():
    """'W' (W-SUN) bins are Mon 00:00 .. next Mon 00:00. 2024-01-10 is a
    Wednesday: the first week (01-08 .. 01-15) holds 5 days = 120 kWh;
    the last (02-19 .. 02-26) holds 02-19 only = 24 kWh.
    """
    out = PowerIntegrator.energy_by_window(
        _flat_kw("2024-01-10", "2024-02-20"), window="W"
    )
    assert out["start"].iloc[0] == pd.Timestamp("2024-01-08")
    assert out["end"].iloc[0] == pd.Timestamp("2024-01-15")
    assert out["kwh"].iloc[0] == pytest.approx(120.0)
    assert out["start"].iloc[-1] == pd.Timestamp("2024-02-19")
    assert out["kwh"].iloc[-1] == pytest.approx(24.0)
    assert out["kwh"].sum() == pytest.approx(41 * 24.0)
    assert (out["kwh"] >= 0).all()


def test_daily_window_is_the_local_day():
    """Europe/Berlin 2024-03-31 lasts 23 h: 1 kW gives 23 kWh, 82 800 s,
    and the window ends at the next local midnight."""
    out = PowerIntegrator.energy_by_window(
        _flat_kw("2024-03-30", "2024-04-02", tz="Europe/Berlin"), window="1D"
    )
    day = out[out["start"] == pd.Timestamp("2024-03-31", tz="Europe/Berlin")].iloc[0]
    assert day["kwh"] == pytest.approx(23.0)
    assert day["duration_seconds"] == 82800.0
    assert day["end"] == pd.Timestamp("2024-04-01", tz="Europe/Berlin")
    assert day["coverage"] == pytest.approx(1.0)


def test_total_coverage_counts_a_missing_meter_as_zero():
    """m1 100 kW 00:00-02:00, m2 50 kW 00:00-01:00. Hour 00: both meters
    cover it fully (coverage 1.0, 150 kWh). Hour 01: m2 has no data, so the
    total is under-reported: coverage min(1.0, 0.0) = 0.0, 100 kWh."""
    df = pd.DataFrame(
        {
            "systime": pd.to_datetime(
                ["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 02:00"]
                + ["2024-01-01 00:00", "2024-01-01 01:00"]
            ),
            "uuid": ["m1"] * 3 + ["m2"] * 2,
            "value_double": [100.0, 100.0, 100.0, 50.0, 50.0],
        }
    )
    out = PowerIntegrator.energy_by_window(df, window="1h")
    assert list(out["kwh"]) == pytest.approx([150.0, 100.0])
    assert list(out["coverage"]) == [1.0, 0.0]


# -- DataHarmonizer: grid sampling ------------------------------------------

T0 = pd.Timestamp("2024-01-01")


def _ramp() -> DataHarmonizer:
    """A 10/s ramp (value = 10 x seconds) sampled at 0.5, 1.5, 2.5 s."""
    return DataHarmonizer(
        pd.DataFrame(
            {
                "systime": [T0 + pd.Timedelta(seconds=x) for x in (0.5, 1.5, 2.5)],
                "uuid": "r",
                "value_double": [5.0, 15.0, 25.0],
            }
        )
    )


def test_sampling_first_is_the_first_sample_of_the_bin():
    """``"first"``: the bin [t, t+1 s) reports its first sample, at t."""
    out = _ramp().resample_to_uniform("1s")
    assert list(out.index) == [T0 + pd.Timedelta(seconds=s) for s in range(3)]
    assert list(out["r"]) == [5.0, 15.0, 25.0]


@pytest.mark.parametrize("call", ["resample", "merge"])
def test_sampling_interpolate_is_the_value_at_the_grid_time(call):
    """``"interpolate"``: the ramp at 1 s is 10 and at 2 s is 20; 0 s lies
    before the first sample and stays NaN."""
    h = _ramp()
    if call == "resample":
        out = h.resample_to_uniform("1s", sampling="interpolate")
    else:
        out = h.merge_multi_signals(window="1s", sampling="interpolate")
    assert pd.isna(out["r"].iloc[0])
    assert list(out["r"].iloc[1:]) == pytest.approx([10.0, 20.0])


@pytest.mark.parametrize("sampling", ["first", "interpolate"])
def test_signal_is_not_extended_past_its_last_sample(sampling):
    """a at 0, 1, 2 s; b at 0 and 5 s. On the 1 s grid b is interpolated
    inside its span (20, 30, 40, 50) while a, which ended at 2 s, is NaN at
    3, 4, 5 s (it used to repeat 3.0)."""
    df = pd.DataFrame(
        {
            "systime": [T0 + pd.Timedelta(seconds=x) for x in (0, 1, 2, 0, 5)],
            "uuid": ["a"] * 3 + ["b"] * 2,
            "value_double": [1.0, 2.0, 3.0, 10.0, 60.0],
        }
    )
    out = DataHarmonizer(df).merge_multi_signals(window="1s", sampling=sampling)
    assert list(out["a"].iloc[:3]) == [1.0, 2.0, 3.0]
    assert out["a"].iloc[3:].isna().all()
    assert list(out["b"]) == pytest.approx([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])


def test_sampling_rejects_unknown_mode():
    with pytest.raises(ValueError, match="sampling"):
        _ramp().resample_to_uniform("1s", sampling="mean")  # type: ignore[arg-type]
