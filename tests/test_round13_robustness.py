"""Robustness fixes: tz-aware data_end, missing states, value columns, ratios.

* ``state_intervals`` reads a naive / text ``data_end`` on tz-aware data as
  local time of the data (every MicroStopEvents method, DutyCycleEvents).
* A missing run/on state is unknown time, not idle / off.
* A signal whose values are in another ``value_*`` column warns once.
* A ``value_column`` the data does not have raises (PerformanceLossEvents,
  ``state_mask``).
* Variance / sigma ratios are NaN, never ``inf``.
* DutyCycleEvents windows extend to the constructor's ``data_end``.
* ``object`` numbers are features / cross-signal signals.
* ProductionScheduleAdherenceEvents raises ColumnNotFoundError.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import ts_shape
from ts_shape.errors import ColumnNotFoundError, DataQualityWarning
from ts_shape.events._compat import state_mask
from ts_shape.events.engineering.disturbance_recovery import DisturbanceRecoveryEvents
from ts_shape.events.engineering.process_window import ProcessWindowEvents
from ts_shape.events.production.duty_cycle import DutyCycleEvents
from ts_shape.events.production.machine_state import MachineStateEvents
from ts_shape.events.production.micro_stop_detection import MicroStopEvents
from ts_shape.events.production.performance_loss import PerformanceLossEvents
from ts_shape.events.production.schedule_adherence import (
    ProductionScheduleAdherenceEvents,
)
from ts_shape.events.quality.change_point import ChangePointEvents
from ts_shape.features.cross_signal import CrossSignalAnalytics
from ts_shape.features.export import FeatureMatrixExporter
from ts_shape.utils.intervals import state_intervals
from ts_shape.utils.signal_checks import warn_if_values_elsewhere

BERLIN = "Europe/Berlin"
VALUE_COLUMNS = ("value_double", "value_integer", "value_bool", "value_string")
MSG = "has no values in"


def _long(uuid: str, times: pd.DatetimeIndex, **values: object) -> pd.DataFrame:
    df = pd.DataFrame({"systime": times, "uuid": uuid, **values})
    for column in VALUE_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan
    return df


def _ours(record: list[warnings.WarningMessage]) -> list[str]:
    return [str(w.message) for w in record if MSG in str(w.message)]


# ---------------------------------------------------------------------------
# 1. data_end on tz-aware data
# ---------------------------------------------------------------------------


def _berlin_run() -> pd.DataFrame:
    # DST day: 02:00 -> 03:00 on 2024-03-31.
    t = pd.date_range("2024-03-31 05:00", periods=4, freq="30min", tz=BERLIN)
    return _long("run", t, value_bool=[True, False, True, False])


def test_state_intervals_text_data_end_on_tz_aware_frame() -> None:
    df = _berlin_run()
    df["state"] = df["value_bool"]
    out = state_intervals(
        df, time_column="systime", state_column="state", data_end="2024-03-31 08:00"
    )
    assert out["end"].iloc[-1] == pd.Timestamp("2024-03-31 08:00", tz=BERLIN)
    assert out["duration_seconds"].iloc[-1] == 5400.0  # from the 06:30 sample


def test_state_intervals_aware_data_end_is_converted() -> None:
    df = _berlin_run()
    df["state"] = df["value_bool"]
    out = state_intervals(
        df,
        time_column="systime",
        state_column="state",
        data_end=pd.Timestamp("2024-03-31 06:00", tz="UTC"),
    )
    assert out["end"].iloc[-1] == pd.Timestamp("2024-03-31 08:00", tz=BERLIN)


def test_micro_stop_methods_accept_text_data_end_on_tz_aware_frame() -> None:
    ev = MicroStopEvents(_berlin_run(), "run", data_end="2024-03-31 08:00")
    stops = ev.detect_micro_stops(max_duration="2h")
    # The last stop (07:30 idle) is closed at data_end 08:00 Berlin.
    assert stops["end"].max() == pd.Timestamp("2024-03-31 08:00", tz=BERLIN)
    ev.micro_stop_frequency(window="1h", max_duration="2h")
    ev.micro_stop_impact(window="1h", max_duration="2h")
    ev.micro_stop_patterns(hour_grouping=True, max_duration="2h")


def test_duty_cycle_intervals_accept_text_data_end_on_tz_aware_frame() -> None:
    dc = DutyCycleEvents(_berlin_run(), "run", data_end="2024-03-31 08:00")
    out = dc.on_off_intervals()
    assert out["end"].iloc[-1] == pd.Timestamp("2024-03-31 08:00", tz=BERLIN)
    assert out["duration_seconds"].iloc[-1] == 5400.0


# ---------------------------------------------------------------------------
# 2. Missing states are unknown time
# ---------------------------------------------------------------------------


def _with_na() -> pd.DataFrame:
    # on 06:00-06:30, unknown 06:30-07:00, off 07:00-07:30, on 07:30-08:00.
    t = pd.date_range("2024-01-01 06:00", periods=5, freq="30min")
    states = pd.array([True, pd.NA, False, True, True], dtype="boolean")
    return _long("m", t, value_bool=states)


def test_duty_cycle_off_time_excludes_missing_state() -> None:
    dc = DutyCycleEvents(_with_na(), "m")
    out = dc.duty_cycle_by_window(window="1h")
    first = out.iloc[0]
    assert first["on_time_seconds"] == 1800.0
    assert first["off_time_seconds"] == 0.0  # the 30 min of pd.NA are not off
    assert first["duty_cycle_pct"] == 100.0
    second = out.iloc[1]
    assert (second["on_time_seconds"], second["off_time_seconds"]) == (1800.0, 1800.0)


def test_duty_cycle_intervals_and_counts_skip_missing_state() -> None:
    dc = DutyCycleEvents(_with_na(), "m")
    intervals = dc.on_off_intervals()
    assert list(intervals["state"]) == ["on", "off", "on"]
    assert intervals["duration_seconds"].sum() == 5400.0  # 30 min unknown left out
    counts = dc.cycle_count(window="1h")
    assert counts["total_transitions"].sum() == 3  # first on, off, on


def test_duty_cycle_window_without_known_state_is_nan() -> None:
    t = pd.date_range("2024-01-01 06:00", periods=4, freq="30min")
    df = _long("m", t, value_bool=pd.array([pd.NA, pd.NA, True, False], "boolean"))
    out = DutyCycleEvents(df, "m").duty_cycle_by_window(window="1h")
    assert np.isnan(out.iloc[0]["duty_cycle_pct"])
    assert (
        out.iloc[0]["on_time_seconds"] == 0.0 and out.iloc[0]["off_time_seconds"] == 0.0
    )


def test_duty_cycle_cycle_count_all_missing_is_empty() -> None:
    t = pd.date_range("2024-01-01 06:00", periods=3, freq="30min")
    df = _long("m", t, value_double=[1.0, 2.0, 3.0])  # value_bool all NaN
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DataQualityWarning)
        out = DutyCycleEvents(df, "m").cycle_count(window="1h")
    assert out.empty
    assert "total_transitions" in out.columns


def test_machine_state_missing_state_is_neither_run_nor_idle() -> None:
    ms = MachineStateEvents(_with_na(), "m")
    out = ms.detect_run_idle()
    assert list(out["state"]) == ["run", "idle", "run"]
    idle = out.loc[out["state"] == "idle", "duration_seconds"].sum()
    assert idle == 1800.0
    transitions = ms.transition_events()
    assert list(transitions["transition"]) == ["run_to_idle", "idle_to_run"]
    metrics = ms.state_quality_metrics().iloc[0]
    assert metrics["avg_idle_duration_seconds"] == 1800.0
    assert metrics["run_idle_factor"] == pytest.approx(1800.0 / 1800.0 * 2)


def test_machine_state_nan_state_float_column() -> None:
    t = pd.date_range("2024-01-01 06:00", periods=3, freq="30min")
    df = _long("m", t, value_double=[1.0, np.nan, 0.0])
    out = MachineStateEvents(df, "m", value_column="value_double").detect_run_idle()
    assert list(out["state"]) == ["run", "idle"]
    assert out["duration_seconds"].tolist() == [1800.0, 0.0]


def test_machine_state_value_range_missing_is_unknown() -> None:
    t = pd.date_range("2024-01-01 06:00", periods=3, freq="30min")
    df = _long("m", t, value_double=[5.0, np.nan, 0.0])
    ms = MachineStateEvents(df, "m", value_column="value_double", value_range=(1, 10))
    assert list(ms.detect_run_idle()["state"]) == ["run", "idle"]


def test_state_quality_metrics_on_empty_input_averages_are_nan() -> None:
    empty = _long("m", pd.DatetimeIndex([]))
    metrics = MachineStateEvents(empty, "m").state_quality_metrics().iloc[0]
    assert np.isnan(metrics["avg_run_duration_seconds"])
    assert np.isnan(metrics["avg_idle_duration_seconds"])
    assert np.isnan(metrics["run_idle_factor"])


# ---------------------------------------------------------------------------
# 3. Values in another value_* column warn (results unchanged)
# ---------------------------------------------------------------------------


def _state_in_integer() -> pd.DataFrame:
    t = pd.date_range("2024-01-01", periods=120, freq="1min")
    return _long("st", t, value_integer=(np.arange(120) // 10) % 2)


def _counter_in_double() -> pd.DataFrame:
    t = pd.date_range("2024-01-01", periods=120, freq="1min")
    return _long("c", t, value_double=np.arange(120.0))


def test_process_window_warns_for_state_in_value_integer() -> None:
    with pytest.warns(DataQualityWarning) as record:
        out = ProcessWindowEvents(_state_in_integer(), "st").windowed_statistics(
            window="1h"
        )
    assert out.empty  # unchanged: nothing in value_double
    assert _ours(record) == [
        "ProcessWindowEvents: signal 'st' has no values in value_double; its "
        "values are in value_integer — pass value_column='value_integer'"
    ]


def test_line_throughput_warns_for_counter_in_value_double() -> None:
    with pytest.warns(DataQualityWarning) as record:
        out = ts_shape.LineThroughputEvents(_counter_in_double()).count_parts("c")
    assert out.empty
    (message,) = _ours(record)
    assert "'c' has no values in value_integer" in message
    assert "pass value_column='value_double'" in message


def test_machine_state_warns_for_state_in_value_integer() -> None:
    with pytest.warns(DataQualityWarning) as record:
        MachineStateEvents(_state_in_integer(), "st").detect_run_idle()
    (message,) = _ours(record)
    assert "in value_bool; its values are in value_integer" in message


def test_correct_value_column_does_not_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", DataQualityWarning)
        ProcessWindowEvents(_counter_in_double(), "c").windowed_statistics(window="1h")
        ts_shape.LineThroughputEvents(_state_in_integer()).count_parts("st")
        MachineStateEvents(
            _state_in_integer(), "st", value_column="value_integer"
        ).detect_run_idle()


def test_warning_once_per_signal_and_column() -> None:
    lt = ts_shape.LineThroughputEvents(_counter_in_double())
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        lt.count_parts("c")
        lt.count_parts("c", window="1h")
    assert len(_ours(record)) == 1


def test_helper_on_signal_rows() -> None:
    rows = _counter_in_double()
    with pytest.warns(DataQualityWarning, match="pass value_column='value_double'"):
        assert warn_if_values_elsewhere(rows, "c", "value_integer", where="X")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert not warn_if_values_elsewhere(rows, "c", "value_double")
        assert not warn_if_values_elsewhere(rows.iloc[:0], "c", "value_integer")
        # No value anywhere: nothing to point at.
        assert not warn_if_values_elsewhere(
            rows.assign(value_double=np.nan), "c", "value_integer"
        )


# ---------------------------------------------------------------------------
# 4. Unknown value columns raise
# ---------------------------------------------------------------------------


def test_performance_loss_unknown_value_column_raises() -> None:
    df = _counter_in_double().drop(columns=["value_integer"])
    perf = PerformanceLossEvents(df)
    with pytest.raises(ColumnNotFoundError, match="value_integr"):
        perf.performance_by_shift("c", 30.0, value_column="value_integr")
    with pytest.raises(ColumnNotFoundError):
        perf.performance_by_shift("c", 30.0)  # default value_integer is absent
    out = perf.performance_by_shift("c", 30.0, value_column="value_double")
    assert out["actual_parts"].sum() > 0


def test_state_mask_unknown_value_column_raises() -> None:
    data = pd.DataFrame({"value_bool": [True, False]})
    with pytest.raises(ColumnNotFoundError, match="value_strng"):
        state_mask(data, "value_strng", "Running", bool_value=True)
    with pytest.raises(ColumnNotFoundError):
        state_mask(data, "value_strng", True, bool_value=True)
    # The default value_string falls back to value_bool as before.
    assert state_mask(data, "value_string", "Running", bool_value=True).tolist() == [
        True,
        False,
    ]


def test_downtime_availability_trend_defaults_with_bool_state() -> None:
    t = pd.date_range("2024-01-01", periods=48, freq="1h")
    df = _long("run", t, value_bool=np.arange(48) % 4 != 0)
    out = ts_shape.DowntimeEvents(df).availability_trend("run", window="1D")
    assert out["availability_pct"].iloc[0] == pytest.approx(75.0)
    with pytest.raises(ColumnNotFoundError):
        ts_shape.DowntimeEvents(df).availability_trend(
            "run", window="1D", value_column="value_strng"
        )


# ---------------------------------------------------------------------------
# 5. Ratios are NaN, never inf
# ---------------------------------------------------------------------------


def test_process_window_single_sample_std_is_nan() -> None:
    t = pd.DatetimeIndex(["2024-01-01 00:10", "2024-01-01 01:10", "2024-01-01 01:20"])
    df = _long("s", t, value_double=[1.0, 2.0, 4.0])
    stats = ProcessWindowEvents(df, "s").windowed_statistics(window="1h")
    assert np.isnan(stats["std"].iloc[0])
    assert stats["std"].iloc[1] == pytest.approx(np.std([2.0, 4.0], ddof=1))


def test_process_window_variance_ratio_never_inf() -> None:
    t = pd.date_range("2024-01-01", periods=180, freq="1min")
    values = np.r_[np.full(60, 5.0), np.random.default_rng(0).normal(5, 1, 60)]
    values = np.r_[values, np.full(60, 5.0)]
    pw = ProcessWindowEvents(_long("s", t, value_double=values), "s")
    out = pw.detect_variance_change(window="1h")
    assert not np.isinf(out["variance_factor"]).any()
    # constant -> varying: flagged with an undefined ratio.
    first = out[out["start"] == pd.Timestamp("2024-01-01 00:00")]
    assert len(first) == 1 and np.isnan(first["variance_factor"].iloc[0])
    shifts = pw.detect_mean_shift(window="1h")
    assert not np.isinf(shifts["shift_sigma"].astype(float)).any()


def test_process_window_single_sample_windows_are_skipped() -> None:
    t = pd.DatetimeIndex(["2024-01-01 00:10", "2024-01-01 01:10", "2024-01-01 02:10"])
    pw = ProcessWindowEvents(_long("s", t, value_double=[1.0, 50.0, 1.0]), "s")
    assert pw.detect_variance_change(window="1h").empty
    assert pw.detect_mean_shift(window="1h").empty


def test_disturbance_recovery_variance_ratio_never_inf() -> None:
    t = pd.date_range("2024-01-01", periods=400, freq="1min")
    rng = np.random.default_rng(1)
    values = np.r_[
        rng.normal(10, 0.1, 150),
        np.full(50, 10.0),
        np.full(10, 30.0),
        rng.normal(10, 1, 190),
    ]
    dr = DisturbanceRecoveryEvents(_long("s", t, value_double=values), "s")
    out = dr.before_after_comparison(
        baseline_window="30min", n_sigma=3.0, comparison_window="20min"
    )
    assert not out.empty
    assert not np.isinf(out["variance_factor"]).any()
    assert out["variance_factor"].isna().any()


def test_change_point_shift_sigma_nan_for_constant_segments() -> None:
    t = pd.date_range("2024-01-01", periods=40, freq="1min")
    df = _long("s", t, value_double=np.r_[np.full(20, 1.0), np.full(20, 5.0)])
    out = ChangePointEvents(df, "s").detect_change_points(penalty=1.0)
    assert len(out) == 1
    assert np.isnan(out["shift_sigma"].iloc[0])


# ---------------------------------------------------------------------------
# 6. DutyCycleEvents windows extend to data_end
# ---------------------------------------------------------------------------


def test_duty_cycle_windows_extend_to_data_end() -> None:
    t = pd.DatetimeIndex(["2024-01-01 06:00", "2024-01-01 06:30"])
    df = _long("m", t, value_bool=[False, True])
    dc = DutyCycleEvents(df, "m", data_end="2024-01-01 09:00")
    duty = dc.duty_cycle_by_window(window="1h")
    assert duty["start"].tolist() == list(
        pd.date_range("2024-01-01 06:00", periods=3, freq="1h")
    )
    assert duty["on_time_seconds"].tolist() == [1800.0, 3600.0, 3600.0]
    assert duty["duty_cycle_pct"].tolist() == [50.0, 100.0, 100.0]
    counts = dc.cycle_count(window="1h")
    assert counts["start"].tolist() == duty["start"].tolist()
    assert counts["total_transitions"].tolist() == [2, 0, 0]
    # Without data_end: only the window with samples (as before).
    plain = DutyCycleEvents(df, "m")
    assert len(plain.duty_cycle_by_window(window="1h")) == 1
    assert len(plain.cycle_count(window="1h")) == 1


def test_duty_cycle_data_end_on_tz_aware_frame() -> None:
    t = pd.date_range("2024-03-31 01:00", periods=2, freq="30min", tz=BERLIN)
    df = _long("m", t, value_bool=[True, False])
    dc = DutyCycleEvents(df, "m", data_end="2024-03-31 05:00")
    duty = dc.duty_cycle_by_window(window="1h")
    # 01:00, 03:00 (02:00 does not exist), 04:00.
    assert len(duty) == 3
    assert duty["on_time_seconds"].sum() + duty["off_time_seconds"].sum() == 3 * 3600.0
    assert dc.cycle_count(window="1h")["end"].max() == pd.Timestamp(
        "2024-03-31 05:00", tz=BERLIN
    )


# ---------------------------------------------------------------------------
# 7. object numbers
# ---------------------------------------------------------------------------


def test_cross_signal_coerces_object_numbers() -> None:
    t = pd.date_range("2024-01-01", periods=60, freq="1min")
    x = np.sin(np.arange(60) / 4)
    wide = pd.DataFrame(
        {
            "systime": t,
            "a": pd.Series(x, dtype=object),
            "b": pd.Series(np.roll(x, 3).astype(str), dtype=object),
            "label": "line-1",
        }
    )
    cross = CrossSignalAnalytics(wide)
    assert cross.signals == ["a", "b"]
    assert cross.lead_lag("a", "b", max_lag=5)["optimal_lag"] == 3
    assert cross.lead_lag_matrix(max_lag=5).shape == (2, 2)


def test_cross_signal_text_without_numbers_still_raises() -> None:
    wide = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "z"]})
    with pytest.raises(ValueError, match="at least 2 numeric"):
        CrossSignalAnalytics(wide)


def test_feature_matrix_coerces_object_value_columns() -> None:
    df = pd.DataFrame(
        {
            "uuid": ["s"] * 4,
            "value_double": pd.Series([1.0, 2.0, 3.0, 4.0], dtype=object),
            "value_string": ["7", "8", "9", "10"],
        }
    )
    matrix = FeatureMatrixExporter.to_feature_matrix(df)
    assert set(matrix.columns) == {
        "s__value_double__mean",
        "s__value_double__std",
        "s__value_double__min",
        "s__value_double__max",
    }
    assert matrix.loc[0, "s__value_double__mean"] == 2.5


# ---------------------------------------------------------------------------
# 8. Schedule columns
# ---------------------------------------------------------------------------


def _schedule() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "start": pd.to_datetime(["2024-01-01 06:00"]),
            "end": pd.to_datetime(["2024-01-01 14:00"]),
            "planned_qty": [100],
            "order_id": ["WO-1"],
        }
    )


def test_schedule_missing_column_raises_column_not_found() -> None:
    df = _counter_in_double()
    with pytest.raises(ColumnNotFoundError, match="planned_qty"):
        ProductionScheduleAdherenceEvents(df, _schedule().drop(columns="planned_qty"))
    with pytest.raises(ValueError):  # still a ValueError
        ProductionScheduleAdherenceEvents(df, _schedule().drop(columns="end"))


def test_schedule_bad_order_id_column_raises() -> None:
    df = _counter_in_double()
    with pytest.raises(ColumnNotFoundError, match="wo_number"):
        ProductionScheduleAdherenceEvents(df, _schedule(), order_id_column="wo_number")
    # The default order_id column may be absent.
    ev = ProductionScheduleAdherenceEvents(df, _schedule().drop(columns="order_id"))
    assert ev.adherence_by_period("c")["order_id"].isna().all()
