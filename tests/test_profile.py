"""Tests for ts_shape.profile()."""

import numpy as np
import pandas as pd
import pytest

import ts_shape
from ts_shape.signals import SignalSpecs

BERLIN = "Europe/Berlin"


@pytest.fixture()
def messy():
    rng = np.random.default_rng(0)
    t = pd.date_range("2024-01-01", periods=300, freq="10s", tz=BERLIN)
    parts = np.cumsum(rng.integers(0, 3, 300))
    parts[150:] -= parts[149]  # reset
    temp = 20 + rng.normal(size=300)
    temp[200:260] = temp[199]  # stuck for 10 min
    keep = np.ones(300, bool)
    keep[50:120] = False  # 11.5-minute logging outage
    frames = [
        pd.DataFrame(
            {"systime": t[keep], "uuid": "parts", "value_integer": parts[keep]}
        ),
        pd.DataFrame({"systime": t, "uuid": "temp", "value_double": temp}),
        pd.DataFrame(
            {
                "systime": list(t[:5]) + [t[4]],  # one duplicate
                "uuid": "dup",
                "value_double": [1.0, 2.0, np.nan, 4.0, 5.0, 5.0],
            }
        ),
    ]
    return pd.concat(frames, ignore_index=True)


def test_profile_finds_the_problems(messy):
    report = ts_shape.profile(messy)
    issues = {(r.uuid, r.issue) for r in report.issues().itertuples()}
    assert ("parts", "logging gaps") in issues
    assert ("parts", "counter resets") in issues
    assert ("temp", "possibly stuck") in issues
    assert ("dup", "duplicate timestamps") in issues
    assert ("dup", "missing values") in issues
    table = report.to_frame().set_index("uuid")
    assert table.loc["parts", "kind"] == "counter"
    assert table.loc["parts", "resets"] == 1
    assert table.loc["parts", "gap_count"] == 1
    assert table.loc["temp", "tz"] == BERLIN


def test_issues_sorted_by_severity_and_filterable(messy):
    report = ts_shape.profile(messy)
    order = {"error": 0, "warning": 1, "info": 2}
    ranks = [order[s] for s in report.issues()["severity"]]
    assert ranks == sorted(ranks)
    assert set(report.issues("warning")["severity"]) == {"warning"}
    assert "ts-shape profile: 3 signal(s)" in report.summary()


def test_profile_nat_and_mixed_columns():
    df = pd.DataFrame(
        {
            "systime": ["2024-01-01 00:00", "garbage", "2024-01-01 00:02"],
            "uuid": "x",
            "value_double": [1.0, 2.0, np.nan],
            "value_integer": [np.nan, np.nan, 3],
        }
    )
    issues = set(ts_shape.profile(df).issues()["issue"])
    assert {"unparseable timestamps", "mixed value columns"} <= issues


def test_profile_accepts_overrides_and_specs(messy):
    report = ts_shape.profile(messy, {"parts": {"unit": "pcs", "max_gap": "1h"}})
    assert report.specs["parts"].unit == "pcs"
    assert report.to_frame().set_index("uuid").loc["parts", "gap_count"] == 0
    fixed = SignalSpecs.from_dict({"temp": {"kind": "continuous", "max_gap": "1s"}})
    again = ts_shape.profile(messy, fixed)
    assert again.specs["temp"].max_gap == pd.Timedelta("1s")


def test_profile_repeated_access_stays_callable():
    df = ts_shape.make_timeseries(["a"], n_points=20)
    ts_shape.profile(df)
    assert callable(ts_shape.profile)
    assert ts_shape.profile(df).issues().empty


def test_profile_reports_non_numeric_values():
    df = pd.DataFrame(
        {
            "systime": pd.date_range("2024-01-01", periods=4, freq="1min"),
            "uuid": "x",
            "value_double": pd.Series([1.0, "n/a", 3.0, "bad"], dtype=object),
        }
    )
    issues = ts_shape.profile(df).issues()
    row = issues[issues["issue"] == "non-numeric values"]
    assert len(row) == 1
    assert "2 value(s)" in row["detail"].iloc[0]
