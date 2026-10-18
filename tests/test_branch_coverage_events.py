"""Detector branches the scenario tests do not reach.

Monthly quality roll-ups, changeover reference methods, DOE response
statistics and level binning, and multi-signal startup logic.
"""

import numpy as np
import pandas as pd
import pytest

from ts_shape.errors import DataQualityWarning
from ts_shape.events.development.design_of_experiments import (
    DesignOfExperimentsEvents,
)
from ts_shape.events.engineering.startup_events import StartupDetectionEvents
from ts_shape.events.production.changeover import ChangeoverEvents
from ts_shape.events.production.period_summary import PeriodSummaryEvents

# --------------------------------------------------------------------------
# PeriodSummaryEvents.monthly_summary with OK / NOK counters
# --------------------------------------------------------------------------


def _daily_counters():
    days = pd.date_range("2024-01-30", "2024-02-02", freq="1D")
    rows = []
    for uuid, per_day in (("parts", 100), ("ok", 95), ("nok", 5)):
        # one reading at the start and end of each day
        for i, d in enumerate(days):
            rows.append((d + pd.Timedelta("6h"), uuid, i * per_day))
            rows.append((d + pd.Timedelta("22h"), uuid, (i + 1) * per_day))
    return pd.DataFrame(rows, columns=["systime", "uuid", "value_integer"])


def test_monthly_summary_adds_quality_per_month():
    out = PeriodSummaryEvents(_daily_counters()).monthly_summary(
        "parts", ok_counter_uuid="ok", nok_counter_uuid="nok"
    )
    assert out["month"].tolist() == [1, 2]
    assert (out["quality_pct"] == 95.0).all()
    assert (
        out["ok_parts"].sum() + out["nok_parts"].sum() == out["total_production"].sum()
    )
    assert out["start"].tolist() == list(pd.to_datetime(["2024-01-01", "2024-02-01"]))


def test_monthly_summary_quality_counter_without_data():
    df = _daily_counters()
    with pytest.warns(DataQualityWarning, match="'nok' not found"):
        out = PeriodSummaryEvents(df[df["uuid"] != "nok"]).monthly_summary(
            "parts", ok_counter_uuid="ok", nok_counter_uuid="nok"
        )
    assert "nok_parts" not in out.columns
    assert "quality_pct" not in out.columns
    assert out["ok_parts"].sum() > 0


def test_monthly_summary_quality_of_a_month_without_parts_is_zero():
    df = _daily_counters()
    df.loc[df["uuid"].isin(["ok", "nok"]), "value_integer"] = 0
    out = PeriodSummaryEvents(df).monthly_summary(
        "parts", ok_counter_uuid="ok", nok_counter_uuid="nok"
    )
    assert (out["quality_pct"] == 0).all()


def test_monthly_summary_empty_with_quality_columns():
    empty = _daily_counters().iloc[:0]
    out = PeriodSummaryEvents(empty).monthly_summary(
        "parts", ok_counter_uuid="ok", nok_counter_uuid="nok"
    )
    assert out.empty
    assert {"ok_parts", "nok_parts", "quality_pct", "resets"} <= set(out.columns)


# --------------------------------------------------------------------------
# ChangeoverEvents.changeover_window(until="stable_band") reference methods
# --------------------------------------------------------------------------


def _changeover_frame():
    t = pd.date_range("2024-01-01 06:00", periods=120, freq="10s")
    product = pd.DataFrame(
        {"systime": t[[0, 30]], "uuid": "line:product", "value_string": ["A", "B"]}
    )
    # the metric jumps at the change, settles at 50 after 2 minutes
    level = np.where(np.arange(120) < 42, 80.0, 50.0)
    metric = pd.DataFrame({"systime": t, "uuid": "line:temp", "value_double": level})
    return pd.concat([product, metric], ignore_index=True)


METRIC = {
    "uuid": "line:temp",
    "value_column": "value_double",
    "band": 1.0,
    "hold": "1min",
}


@pytest.mark.parametrize(
    "extra",
    [
        {"reference_method": "rolling_mean", "rolling_window": 4},
        {"reference_method": "ewma", "ewma_span": 3},
        {"reference_method": "target_value", "target_values": {"line:temp": 50.0}},
        {"reference_method": "not-a-method"},  # falls back to the expanding median
    ],
    ids=["rolling_mean", "ewma", "target_value", "unknown"],
)
def test_changeover_stable_band_reference_methods(extra):
    out = ChangeoverEvents(_changeover_frame()).changeover_window(
        "line:product", until="stable_band", config={"metrics": [METRIC], **extra}
    )
    change = out[out["start"] == pd.Timestamp("2024-01-01 06:05")].iloc[0]
    assert change["method"] == "stable_band"
    assert change["completed"]
    assert (
        pd.Timestamp("2024-01-01 06:05")
        < change["end"]
        <= pd.Timestamp("2024-01-01 06:20")
    )


def test_changeover_target_value_far_from_data_falls_back():
    config = {
        "metrics": [METRIC],
        "reference_method": "target_value",
        "target_values": {"line:temp": 500.0},
    }
    out = ChangeoverEvents(_changeover_frame()).changeover_window(
        "line:product",
        until="stable_band",
        config=config,
        fallback={"default_duration": "3min", "completed": False},
    )
    assert (out["method"] == "stable_band").all()
    assert not out["completed"].any()
    assert (out["duration_seconds"] == 180.0).all()


def _chunked(frame, at):
    """Two chunks concatenated without ignore_index: duplicate index labels."""
    return pd.concat(
        [frame.iloc[:at].reset_index(drop=True), frame.iloc[at:].reset_index(drop=True)]
    )


def test_changeover_stable_band_with_duplicate_index_labels():
    df = _changeover_frame()
    config = {
        "metrics": [METRIC],
        "reference_method": "target_value",
        "target_values": {"line:temp": 50.0},
    }
    expected = ChangeoverEvents(df).changeover_window(
        "line:product", until="stable_band", config=config
    )
    got = ChangeoverEvents(_chunked(df, 60)).changeover_window(
        "line:product", until="stable_band", config=config
    )
    pd.testing.assert_frame_equal(got, expected)


# --------------------------------------------------------------------------
# DesignOfExperimentsEvents: response statistics and level binning
# --------------------------------------------------------------------------


def _doe_frame():
    t = pd.date_range("2024-01-01", periods=480, freq="1min")
    temp = np.repeat([200.0, 220.0, 200.0, 220.0], 120)
    speed = np.tile(np.repeat([80.0, 120.0], 60), 4)
    visc = 500 - 2 * (temp - 200) - 0.5 * (speed - 80)
    visc = visc + np.random.default_rng(0).normal(0, 2, 480)
    return pd.concat(
        [
            pd.DataFrame({"systime": t, "uuid": u, "value_double": v})
            for u, v in (("temp", temp), ("speed", speed), ("visc", visc))
        ],
        ignore_index=True,
    )


@pytest.mark.parametrize("statistic", ["mean", "median", "max", "min", "settled"])
def test_doe_effects_statistics_rank_the_temperature_effect(statistic):
    doe = DesignOfExperimentsEvents(_doe_frame(), ["temp", "speed"])
    out = doe.effects("visc", statistic=statistic, min_duration="10min")
    temp = out[out["factor"] == "temp"].set_index("level")["response_mean"]
    # +20 degrees lower the viscosity by about 40
    assert temp.loc[200.0] - temp.loc[220.0] == pytest.approx(40, abs=8)


def test_doe_effects_unknown_statistic():
    doe = DesignOfExperimentsEvents(_doe_frame(), ["temp", "speed"])
    with pytest.raises(ValueError, match="Unknown statistic"):
        doe.effects("visc", statistic="mode", min_duration="10min")


def test_doe_levels_binned_into_n_levels():
    df = _doe_frame()
    jitter = np.random.default_rng(1).normal(0, 0.5, (df["uuid"] == "temp").sum())
    df.loc[df["uuid"] == "temp", "value_double"] += jitter
    doe = DesignOfExperimentsEvents(df, ["temp", "speed"])
    runs = doe.detect_runs(
        min_duration="10min", stability_tol=0.05, n_levels={"temp": 2}
    )
    assert set(runs["factor__temp_level"]) == {0.0, 1.0}


def test_doe_response_without_data_or_runs():
    df = _doe_frame()
    doe = DesignOfExperimentsEvents(df[df["uuid"] != "visc"], ["temp", "speed"])
    with pytest.raises(ValueError), pytest.warns(DataQualityWarning):
        doe.effects("visc")
    only_factors = DesignOfExperimentsEvents(df, ["temp", "speed"])
    assert only_factors.effects("visc", min_duration="1D").empty


def test_doe_factor_missing_from_data_is_named():
    with pytest.raises(ValueError, match="'pressure' not found"):
        DesignOfExperimentsEvents(_doe_frame(), ["pressure"])


# --------------------------------------------------------------------------
# StartupDetectionEvents.detect_startup_multi_signal
# --------------------------------------------------------------------------


def _startup_frame(shift_b="0s"):
    t = pd.date_range("2024-01-01 06:00", periods=120, freq="10s")
    ramp = np.clip((np.arange(120) - 20) * 2.0, 0, 100)
    b_times = t + pd.Timedelta(shift_b)
    return pd.concat(
        [
            pd.DataFrame({"systime": t, "uuid": "motor:current", "value_double": ramp}),
            pd.DataFrame(
                {"systime": b_times, "uuid": "oven:temp", "value_double": ramp}
            ),
        ],
        ignore_index=True,
    )


SIGNALS = {
    "motor:current": {"method": "threshold", "threshold": 50.0},
    "oven:temp": {"method": "slope", "min_slope": 0.1, "min_duration": "1min"},
}


def test_multi_signal_any_reports_each_signal():
    sd = StartupDetectionEvents(_startup_frame(), "motor:current")
    out = sd.detect_startup_multi_signal(SIGNALS, logic="any")
    assert sorted(s[0] for s in out["signals_triggered"]) == [
        "motor:current",
        "oven:temp",
    ]
    assert (out["method"] == "multi_signal_any").all()
    assert sd.target_uuid == "motor:current"  # restored after the scan


def test_multi_signal_all_within_tolerance():
    sd = StartupDetectionEvents(_startup_frame(), "motor:current")
    out = sd.detect_startup_multi_signal(SIGNALS, logic="all", time_tolerance="5min")
    assert len(out) == 1
    assert sorted(out["signals_triggered"].iloc[0]) == ["motor:current", "oven:temp"]


def test_multi_signal_all_outside_tolerance():
    sd = StartupDetectionEvents(_startup_frame(shift_b="2h"), "motor:current")
    out = sd.detect_startup_multi_signal(SIGNALS, logic="all", time_tolerance="30s")
    assert out.empty


def test_multi_signal_all_with_a_silent_signal():
    signals = {**SIGNALS, "oven:temp": {"method": "threshold", "threshold": 1000.0}}
    sd = StartupDetectionEvents(_startup_frame(), "motor:current")
    out = sd.detect_startup_multi_signal(signals, logic="all")
    assert out.empty
    assert "signals_triggered" in out.columns


@pytest.mark.parametrize(
    "signals,logic,match",
    [
        (SIGNALS, "most", "logic must be"),
        ({"motor:current": {"method": "fourier"}}, "any", "Unknown method"),
    ],
)
def test_multi_signal_rejects_bad_settings(signals, logic, match):
    sd = StartupDetectionEvents(_startup_frame(), "motor:current")
    with pytest.raises(ValueError, match=match):
        sd.detect_startup_multi_signal(signals, logic=logic)


def test_slope_startup_with_duplicate_index_labels():
    df = _startup_frame()
    signal = df[df["uuid"] == "oven:temp"].reset_index(drop=True)
    expected = StartupDetectionEvents(signal, "oven:temp").detect_startup_by_slope(
        min_slope=0.1, min_duration="1min"
    )
    got = StartupDetectionEvents(
        _chunked(signal, 60), "oven:temp"
    ).detect_startup_by_slope(min_slope=0.1, min_duration="1min")
    pd.testing.assert_frame_equal(got, expected)


def test_slope_startup_repeated_labels_hand_built():
    # Two 5-sample chunks concatenated without ignore_index: labels 0..4 twice.
    # Used to raise IndexError (label-based run search, mask of wrong length).
    t0 = pd.Timestamp("2024-01-01")
    df = pd.DataFrame(
        {
            "systime": t0 + pd.to_timedelta(np.arange(10), unit="s"),
            "uuid": "sig",
            "value_double": [0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 4.0, 5.0, 6.0],
        },
        index=[0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
    )
    out = StartupDetectionEvents(df, "sig").detect_startup_by_slope(
        min_slope=0.5, min_duration="1s"
    )
    # A run starts at the sample before its first rising step.
    assert out["start"].tolist() == [t0 + pd.Timedelta("1s"), t0 + pd.Timedelta("6s")]
    assert out["end"].tolist() == [t0 + pd.Timedelta("4s"), t0 + pd.Timedelta("9s")]
    assert out["duration_seconds"].tolist() == [3.0, 3.0]
    assert out["avg_slope"].tolist() == [1.0, 1.0]
    assert out["method"].tolist() == ["slope", "slope"]


def _flat_two_signals():
    t0 = pd.Timestamp("2024-01-01")
    return pd.DataFrame(
        {
            "systime": t0 + pd.to_timedelta([0, 1, 2, 0, 1, 2], unit="s"),
            "uuid": ["a", "a", "a", "b", "b", "b"],
            "value_double": [0.0, 0.0, 0.0, 0.0, 9.0, 9.0],
        }
    )


MULTI_COLS = [
    "start",
    "end",
    "uuid",
    "is_delta",
    "method",
    "signals_triggered",
    "signal_details",
]


def test_multi_signal_any_without_events_has_the_all_columns():
    # Used to return a frame with no columns for "any".
    sd = StartupDetectionEvents(_flat_two_signals(), "a")
    sig = {"a": {"method": "threshold", "threshold": 5.0}}
    got_any = sd.detect_startup_multi_signal(sig, logic="any")
    got_all = sd.detect_startup_multi_signal(sig, logic="all")
    assert list(got_any.columns) == MULTI_COLS
    assert len(got_any) == 0
    pd.testing.assert_frame_equal(got_any, got_all)


def test_multi_signal_all_without_a_match_keeps_the_columns():
    # "b" starts at 1s, "a" at 0s: no match within 0s, but the columns stay.
    sd = StartupDetectionEvents(_flat_two_signals(), "a")
    sig = {
        "b": {"method": "threshold", "threshold": 5.0},
        "a": {"method": "threshold", "threshold": -1.0},
    }
    out = sd.detect_startup_multi_signal(sig, logic="all", time_tolerance="0s")
    assert list(out.columns) == MULTI_COLS
    assert len(out) == 0
