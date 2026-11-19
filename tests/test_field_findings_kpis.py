"""Field test 0.0.88 (real process data): core KPI correctness.

Each test rebuilds the shape of a finding on synthetic data:

* a value column left at its default follows the data (Desma's part counter
  in ``value_double``; a mode signal in ``value_integer``), with a warning;
  a column the caller passes is never replaced;
* run signals that are not a bool (Desma's mode 1 manual / 2 automatic /
  5 setup) are mapped with ``run_states`` in every run-state detector, and
  warn without a mapping instead of counting every mode as running;
* a numeric part number (TPV's recipe in ``value_integer``) works in part and
  quality tracking and comes back as text;
* OEE: ``parts_per_cycle``; no production is NaN, not 0 %;
* a polled cycle trigger that misses pulses warns;
* duty-cycle intervals are clipped to their window; holes are unknown time.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import ts_shape
from ts_shape.errors import DataQualityWarning
from ts_shape.events.energy.idle_energy_detection import IdleEnergyDetectionEvents
from ts_shape.events.production.cycle_time_tracking import CycleTimeEvents
from ts_shape.events.production.downtime_tracking import DowntimeEvents
from ts_shape.events.production.duty_cycle import DutyCycleEvents
from ts_shape.events.production.machine_state import MachineStateEvents
from ts_shape.events.production.micro_stop_detection import MicroStopEvents
from ts_shape.events.production.oee_calculator import OEEEvents
from ts_shape.events.production.part_tracking import PartProductionEvents
from ts_shape.events.production.quality_tracking import QualityEvents
from ts_shape.events.production.runtime_accounting import RuntimeAccountingEvents
from ts_shape.utils.coerce import run_mask
from ts_shape.utils.normalize import id_text

VALUE_COLUMNS = ("value_double", "value_integer", "value_bool", "value_string")


def _long(uuid: str, times: pd.DatetimeIndex, column: str, values) -> pd.DataFrame:
    n = len(times)
    df = pd.DataFrame(
        {
            "systime": times,
            "uuid": uuid,
            "value_double": np.full(n, np.nan),
            "value_integer": pd.array([pd.NA] * n, dtype="Int64"),
            "value_bool": pd.array([pd.NA] * n, dtype="boolean"),
            "value_string": pd.Series([None] * n, dtype=object),
        }
    )
    df[column] = values
    return df


def _messages(record) -> list[str]:
    return [str(w.message) for w in record if issubclass(w.category, Warning)]


# ---------------------------------------------------------------------------
# Desma-like press: mode 1/2/5 in value_integer, counter in value_double
# ---------------------------------------------------------------------------

# Two days, one sample a minute; automatic (2) 06:00-18:00, setup (5)
# 18:00-20:00, manual (1) otherwise. The counter counts 6 parts per shot,
# one shot every 2 minutes while automatic.
T = pd.date_range("2026-10-05", periods=2 * 24 * 60, freq="1min")
_hour = T.hour
MODE = np.where((_hour >= 6) & (_hour < 18), 2, np.where(_hour < 20, 5, 1))
MODE = np.where(_hour < 6, 1, MODE)
_auto = MODE == 2
SHOTS = np.cumsum(_auto & (np.arange(len(T)) % 2 == 0))
PARTS = SHOTS * 6.0


def _press() -> pd.DataFrame:
    return pd.concat(
        [
            _long("mode", T, "value_integer", pd.array(MODE, dtype="Int64")),
            _long("parts", T, "value_double", PARTS),
            _long("part", T[:1], "value_string", ["P-1"]),
        ],
        ignore_index=True,
    )


def test_counter_in_value_double_is_counted_with_a_warning() -> None:
    with pytest.warns(DataQualityWarning, match="value_column_counter") as record:
        out = PartProductionEvents(_press()).production_totals("part", "parts")
    assert out["total_quantity"].sum() == pytest.approx(PARTS[-1] - PARTS[0])
    assert any("value_double, which is used" in m for m in _messages(record))


def test_explicit_column_is_never_replaced() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DataQualityWarning)
        out = PartProductionEvents(_press()).production_totals(
            "part", "parts", value_column_counter="value_integer"
        )
    assert out.empty or out["total_quantity"].sum() == 0


def test_run_mask_maps_a_mode_signal() -> None:
    mode = pd.Series([1, 2, 5, None], dtype="Int64")
    assert run_mask(mode, run_states={2}).tolist() == [False, True, False, pd.NA]
    assert run_mask(mode.astype("float64"), run_states={2}).tolist()[:3] == [
        False,
        True,
        False,
    ]
    with pytest.warns(DataQualityWarning, match=r"run_states=\{"):
        run_mask(pd.Series([1, 2, 5]), what="mode")
    with warnings.catch_warnings():
        warnings.simplefilter("error", DataQualityWarning)
        run_mask(pd.Series([0, 1, 1]))  # a 0/1 signal reads as a bool


def test_runtime_with_mode_counts_only_automatic() -> None:
    truth_hours = _auto.sum() / 60.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DataQualityWarning)
        rt = RuntimeAccountingEvents(_press(), "mode", run_states={2})
        hours = rt.runtime_summary()["run_hours"].sum()
    assert hours == pytest.approx(truth_hours, abs=1 / 60)


def test_runtime_without_mapping_warns() -> None:
    with pytest.warns(DataQualityWarning) as record:
        RuntimeAccountingEvents(_press(), "mode").runtime_summary()
    assert any("[1, 2, 5]" in m for m in _messages(record))


def test_every_run_detector_takes_run_states() -> None:
    df = _press()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DataQualityWarning)
        states = MachineStateEvents(df, "mode", run_states={2}).detect_run_idle()
        assert set(states["state"]) == {"run", "idle"}
        MicroStopEvents(df, "mode", run_states={2}).detect_micro_stops()
        duty = DutyCycleEvents(df, "mode", run_states={2}).duty_cycle_by_window(
            window="1D"
        )
        assert duty["duty_cycle_pct"].round(1).tolist() == [50.0, 50.0]
        down = DowntimeEvents(df).downtime_by_shift("mode", running_value={2})
        assert down["uptime_minutes"].sum() == pytest.approx(_auto.sum(), abs=2)


def test_downtime_with_text_value_on_a_mode_signal_warns() -> None:
    with pytest.warns(DataQualityWarning, match="never occurs"):
        DowntimeEvents(_press()).downtime_by_shift("mode")


def test_idle_energy_with_mode_signal_does_not_crash() -> None:
    df = pd.concat(
        [_press(), _long("kwh", T, "value_double", np.arange(len(T), dtype=float))],
        ignore_index=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DataQualityWarning)
        out = IdleEnergyDetectionEvents(df).idle_energy_by_window(
            "kwh", "mode", run_states={2}, window="1D"
        )
    assert out["machine_running_pct"].round(1).tolist() == [50.0, 50.0]


def test_oee_parts_per_cycle_and_undefined_days() -> None:
    df = _press()
    ideal_cycle = 120.0  # one shot every 2 minutes
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DataQualityWarning)
        oee = OEEEvents(df)
        perf = oee.performance(
            "parts",
            ideal_cycle,
            run_state_uuid="mode",
            run_states={2},
            parts_per_cycle=6,
        )
    assert perf["performance_pct"].round(0).tolist() == [100.0, 100.0]
    with pytest.warns(DataQualityWarning, match="parts_per_cycle"):
        OEEEvents(df).performance(
            "parts", ideal_cycle, run_state_uuid="mode", run_states={2}
        )


def test_quality_without_parts_is_nan() -> None:
    t = pd.date_range("2026-10-05", periods=3 * 24, freq="1h")
    # No production from day 2 00:00 until day 3 12:00.
    idle = (t >= "2026-10-06") & (t < "2026-10-07 12:00")
    total = np.cumsum(~idle)
    df = pd.concat(
        [
            _long("total", t, "value_integer", pd.array(total, dtype="Int64")),
            _long("reject", t, "value_integer", pd.array([0] * len(t), dtype="Int64")),
        ],
        ignore_index=True,
    )
    out = OEEEvents(df).quality("total", "reject")
    assert out["quality_pct"].isna().sum() == 1
    assert out["quality_pct"].dropna().eq(100.0).all()


def test_oee_day_without_production_is_not_zero_quality() -> None:
    df = _press()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DataQualityWarning)
        out = OEEEvents(df).oee(
            "mode", "parts", 120.0, run_states={2}, parts_per_cycle=6
        )
    assert (out["quality_pct"] == 100.0).all()  # not measured: assumed
    assert out["oee_pct"].notna().all()


# ---------------------------------------------------------------------------
# TPV-like line: numeric recipe number in value_integer
# ---------------------------------------------------------------------------


def _tpv() -> pd.DataFrame:
    t = pd.date_range("2026-10-02", periods=200, freq="1min")
    recipe = pd.array(np.where(np.arange(200) < 100, 4711, 815), dtype="Int64")
    return pd.concat(
        [
            _long("recipe", t, "value_integer", recipe),
            _long("ok", t, "value_integer", pd.array(np.arange(200), dtype="Int64")),
            _long(
                "nok", t, "value_integer", pd.array(np.arange(200) // 10, dtype="Int64")
            ),
        ],
        ignore_index=True,
    )


@pytest.mark.parametrize("explicit", [False, True])
def test_numeric_part_number_in_part_and_quality_tracking(explicit: bool) -> None:
    kw = {"value_column_part": "value_integer"} if explicit else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DataQualityWarning)
        part = PartProductionEvents(_tpv())
        by_part = part.production_by_part("recipe", "ok", **kw)
        assert set(by_part["part_number"]) == {"4711", "815"}
        assert by_part["quantity"].sum() == 199
        part.detect_resets("recipe", "ok", **kw)
        part.daily_production_summary("recipe", "ok", **kw)
        quality = QualityEvents(_tpv()).quality_by_part("ok", "nok", "recipe", **kw)
    assert quality["part_number"].tolist() == ["4711", "815"]
    assert quality["ok_parts"].sum() == 199


def test_id_text() -> None:
    values = pd.Series([4711, 4711.0, "A-1", None, 1.5], dtype=object)
    assert id_text(values).tolist() == ["4711", "4711", "A-1", None, "1.5"]


# ---------------------------------------------------------------------------
# Cycle trigger polled every 13 s, pulses of 2 s, cycles of 664 s
# ---------------------------------------------------------------------------


def _polled_pulses(poll: str = "13s") -> pd.DataFrame:
    t = pd.date_range("2026-10-02", periods=6000, freq=poll)
    sec = (t - t[0]).total_seconds().to_numpy()
    on = np.zeros(len(t), dtype=bool)
    for start in np.arange(0, sec[-1], 664.0):
        on |= (sec >= start) & (sec < start + 2)
    return pd.concat(
        [
            _long("cycle", t, "value_bool", pd.array(on, dtype="boolean")),
            _long("part", t[:1], "value_string", ["A"]),
        ],
        ignore_index=True,
    )


def test_polled_trigger_missing_pulses_warns() -> None:
    with pytest.warns(DataQualityWarning, match="polled every 13 s"):
        CycleTimeEvents(_polled_pulses()).cycle_time_by_part("part", "cycle")


def test_well_sampled_trigger_does_not_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", DataQualityWarning)
        out = CycleTimeEvents(_polled_pulses("1s")).cycle_time_by_part("part", "cycle")
    assert out["cycle_time_seconds"].median() == 664.0


def test_integer_pulse_counts_rising_edges_only() -> None:
    t = pd.date_range("2026-10-02", periods=40, freq="10s")
    pulse = pd.array((np.arange(40) % 10 == 0).astype(int), dtype="Int64")
    df = pd.concat(
        [
            _long("cycle", t, "value_integer", pulse),
            _long("part", t[:1], "value_string", ["A"]),
        ],
        ignore_index=True,
    )
    out = CycleTimeEvents(df).cycle_time_by_part(
        "part", "cycle", value_column_trigger="value_integer"
    )
    assert out["cycle_time_seconds"].tolist() == [100.0, 100.0, 100.0]


# ---------------------------------------------------------------------------
# Duty cycle: intervals clipped to windows, holes are unknown
# ---------------------------------------------------------------------------


def test_duty_cycle_clips_a_long_off_interval_to_its_windows() -> None:
    # Change-encoded: off at day 1 12:00, on again 57.3 h later.
    times = pd.to_datetime(
        ["2026-10-01 00:00", "2026-10-01 12:00", "2026-10-03 21:18", "2026-10-04 00:00"]
    )
    df = _long("knit", times, "value_bool", pd.array([True, False, True, True]))
    out = DutyCycleEvents(df, "knit").duty_cycle_by_window(window="1D", max_gap=None)
    total = out["on_time"] + out["off_time"] + out["unknown_seconds"]
    assert (total <= out["duration_seconds"] + 1e-6).all()
    assert out["off_time"].sum() == pytest.approx(57.3 * 3600)


def test_duty_cycle_hole_in_sampled_signal_is_unknown() -> None:
    t1 = pd.date_range("2026-10-01 00:00", "2026-10-01 10:00", freq="1min")
    t2 = pd.date_range("2026-10-03 08:00", "2026-10-03 12:00", freq="1min")
    times = t1.append(t2)
    on = pd.array(np.arange(len(times)) % 2 == 0, dtype="boolean")
    df = _long("knit", times, "value_bool", on)
    out = DutyCycleEvents(df, "knit").duty_cycle_by_window(window="1h")
    hole = out[(out["start"] > "2026-10-01 11:00") & (out["start"] < "2026-10-03")]
    assert (hole["on_time"] + hole["off_time"] == 0).all()
    assert hole["duty_cycle_pct"].isna().all()
    assert (hole["unknown_seconds"] == 3600).all()
    assert (out["duration_seconds"] >= out["on_time"] + out["off_time"]).all()


def test_ts_shape_exports_still_import() -> None:
    assert ts_shape.OEEEvents is OEEEvents
