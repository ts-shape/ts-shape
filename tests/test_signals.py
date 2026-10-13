"""Tests for the signal semantics core: specs, inference and primitives."""

import numpy as np
import pandas as pd
import pytest

from ts_shape.signals import SignalSpec, SignalSpecs, infer_specs
from ts_shape.signals.primitives import (
    align,
    increments,
    intervals,
    per_window,
    shift_windows,
    windows,
)

BERLIN = "Europe/Berlin"


def _ts(*clock, day="2024-01-01"):
    return pd.to_datetime([f"{day} {c}" for c in clock], format="ISO8601")


# --- SignalSpec / SignalSpecs ----------------------------------------------------


def test_spec_validates_and_coerces():
    spec = SignalSpec("a", kind="counter", nominal_period="10s", max_gap="1min")
    assert spec.nominal_period == pd.Timedelta("10s")
    assert spec.interval_end == "next_change"
    assert SignalSpec("e", kind="event").interval_end == "last_sample"
    with pytest.raises(ValueError):
        SignalSpec("a", kind="level")
    with pytest.raises(ValueError):
        SignalSpec("a", value_column="value_text")


def test_specs_round_trip_and_override():
    specs = SignalSpecs.from_dict(
        {"parts": {"kind": "counter", "value_column": "value_integer", "unit": "pcs"}}
    )
    specs = specs.override({"temp": {"max_gap": "5min"}})
    assert specs["temp"].max_gap == pd.Timedelta("5min")
    again = SignalSpecs.from_dict(specs.to_dict())
    assert again["parts"] == specs["parts"]
    assert list(specs.to_frame()["uuid"]) == ["parts", "temp"]


def test_specs_from_yaml(tmp_path):
    pytest.importorskip("yaml")
    path = tmp_path / "specs.yaml"
    path.write_text(
        "line1:state:\n  kind: state\n  value_column: value_bool\n  max_gap: 15min\n"
    )
    specs = SignalSpecs.from_yaml(path)
    assert specs["line1:state"].max_gap == pd.Timedelta("15min")


# --- inference -------------------------------------------------------------------


@pytest.fixture()
def plant_frame():
    rng = np.random.default_rng(0)
    t = pd.date_range("2024-01-01", periods=200, freq="10s", tz=BERLIN)
    parts = np.cumsum(rng.integers(0, 3, 200))
    parts[100:] -= parts[99]  # PLC restart: counter reset
    ce_t = pd.to_datetime(
        [
            "2024-01-01 00:00",
            "2024-01-01 00:07",
            "2024-01-01 01:31",
            "2024-01-01 01:33",
            "2024-01-01 05:00",
        ]
    ).tz_localize(BERLIN)
    return pd.concat(
        [
            pd.DataFrame({"systime": t, "uuid": "parts", "value_integer": parts}),
            pd.DataFrame(
                {
                    "systime": t,
                    "uuid": "temp",
                    "value_double": 20 + rng.normal(size=200),
                }
            ),
            pd.DataFrame(
                {
                    "systime": t,
                    "uuid": "run",
                    "value_bool": np.repeat([True, False] * 2, 50),
                }
            ),
            pd.DataFrame(
                {"systime": ce_t, "uuid": "recipe", "value_string": list("ABABA")}
            ),
            pd.DataFrame(
                {
                    "systime": t,
                    "uuid": "step",
                    "value_integer": np.repeat([1, 2, 3, 2], 50),
                }
            ),
        ],
        ignore_index=True,
    )


@pytest.mark.parametrize(
    ("uuid", "kind", "encoding", "column"),
    [
        ("parts", "counter", "sampled", "value_integer"),
        ("temp", "continuous", "sampled", "value_double"),
        ("run", "state", "sampled", "value_bool"),
        ("recipe", "state", "change_encoded", "value_string"),
        ("step", "state", "sampled", "value_integer"),
    ],
)
def test_infer_specs(plant_frame, uuid, kind, encoding, column):
    spec = infer_specs(plant_frame)[uuid]
    assert (spec.kind, spec.encoding, spec.value_column) == (kind, encoding, column)
    assert spec.tz == BERLIN
    assert spec.notes["kind"]


def test_infer_timing_and_overrides(plant_frame):
    specs = infer_specs(plant_frame, overrides={"recipe": {"max_gap": "4h"}})
    assert specs["temp"].nominal_period == pd.Timedelta("10s")
    assert specs["temp"].max_gap == pd.Timedelta("50s")
    assert specs["recipe"].max_gap == pd.Timedelta("4h")
    assert "not inferred" in infer_specs(plant_frame)["recipe"].notes["max_gap"]


# --- primitives ------------------------------------------------------------------


@pytest.fixture()
def run_state():
    t = _ts("08:00", "08:10", "09:00", "12:00", "12:05")
    df = pd.DataFrame(
        {"systime": t, "uuid": "run", "value_bool": [True, False, True, True, False]}
    )
    spec = SignalSpec(
        "run",
        kind="state",
        encoding="change_encoded",
        value_column="value_bool",
        max_gap="1h",
    )
    return df, spec


def test_intervals_close_at_next_change_and_respect_gaps(run_state):
    df, spec = run_state
    out = intervals(df, spec, data_end=pd.Timestamp("2024-01-01 13:00"))
    assert list(out["duration_seconds"]) == [600.0, 3000.0, 0.0, 300.0, 3300.0]
    # The 3 h gap after 09:00 exceeds max_gap: 09:00 and 12:00 (both True)
    # are separate intervals, and nothing is credited across the gap.
    assert list(out["state"]) == [True, False, True, True, False]


def test_intervals_without_gap_cap_merge_equal_values(run_state):
    df, spec = run_state
    out = intervals(df, spec.with_(max_gap=None))
    assert list(out["state"]) == [True, False, True, False]
    assert out["duration_seconds"].iloc[2] == 3 * 3600 + 300


def test_per_window_bool_state_is_time_fraction(run_state):
    df, spec = run_state
    out = per_window(df, spec, "1h")
    assert out.iloc[0] == pytest.approx(10 / 60)
    assert np.isnan(out.loc[pd.Timestamp("2024-01-01 10:00")])


@pytest.fixture()
def counter():
    t = _ts("08:00", "08:50", "09:10", "09:20", "09:40")
    df = pd.DataFrame({"systime": t, "uuid": "c", "value_integer": [0, 50, 70, 5, 25]})
    return df, SignalSpec("c", kind="counter", value_column="value_integer")


def test_counter_increments_and_windows(counter):
    df, spec = counter
    assert increments(df, spec).tolist() == [0.0, 50.0, 20.0, 5.0, 20.0]
    assert per_window(df, spec, "1h").tolist() == [50.0, 45.0]
    raw = increments(df, spec.with_(reset_policy="none"))
    assert raw.tolist() == [0.0, 50.0, 20.0, -65.0, 20.0]
    with pytest.raises(ValueError):
        increments(df, spec.with_(kind="continuous"))


def test_per_window_string_state_raises():
    t = _ts("08:00", "08:30")
    df = pd.DataFrame({"systime": t, "uuid": "r", "value_string": ["A", "B"]})
    with pytest.raises(ValueError, match="intervals"):
        per_window(df, SignalSpec("r", kind="state", value_column="value_string"), "1h")


def test_align_holds_backward_with_tolerance():
    pv = pd.DataFrame(
        {
            "systime": pd.date_range("2024-01-01 10:00", periods=5, freq="1min"),
            "uuid": "pv",
            "value_double": 1.0,
        }
    )
    sp = pd.DataFrame(
        {
            "systime": _ts("10:00:00", "10:02:30"),
            "uuid": "sp",
            "value_double": [50.0, 80.0],
        }
    )
    out = align(pv, SignalSpec("pv"), sp, SignalSpec("sp", max_gap="150s"))
    # 10:02 still uses 50 (no look-ahead); 10:04 is 90 s after 10:02:30 -> held.
    # With a 30 s max_gap the same value is stale.
    assert out["sp"].tolist()[:4] == [50.0, 50.0, 50.0, 80.0]
    assert out["sp"].iloc[4] == 80.0
    stale = align(pv, SignalSpec("pv"), sp, SignalSpec("sp", max_gap="30s"))
    assert np.isnan(stale["sp"].iloc[4])


def test_windows_follow_dst_calendar():
    out = windows("2024-03-30", "2024-04-02", "1D", tz=BERLIN)
    assert list(out["duration_seconds"]) == [86400.0, 82800.0, 86400.0]


def test_shift_windows_overnight_across_dst():
    out = shift_windows("2024-03-30 12:00", "2024-03-31 12:00", tz=BERLIN)
    night = out[out["shift"] == "shift_3"].iloc[0]
    assert str(night["production_day"]) == "2024-03-30"
    assert night["duration_seconds"] == 7 * 3600
    assert list(out["shift"]) == ["shift_1", "shift_2", "shift_3", "shift_1"]
