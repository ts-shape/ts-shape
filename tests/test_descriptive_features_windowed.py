"""DescriptiveFeatures.compute_windowed: one row per time window, all signals."""

import time

import numpy as np
import pandas as pd
import pytest

from ts_shape.features.stats.feature_table import (
    _KIND_FEATURES,
    _OVERALL_FEATURES,
    DescriptiveFeatures,
)

T0 = pd.Timestamp("2024-01-01 00:00:00")


def _frame() -> pd.DataFrame:
    """Numeric, integer, boolean and string signals over 6 s, gap at 3-4 s."""
    rng = np.random.default_rng(3)
    offsets = [0.1, 0.3, 0.35, 0.6, 0.8, 1.2, 1.5, 1.9, 2.05, 2.5, 2.7, 5.2, 5.9]
    times = [T0 + pd.Timedelta(seconds=s) for s in offsets]
    n = len(times)
    doubles = rng.normal(10, 2, n)
    doubles[2] = np.nan
    doubles[4] = doubles[3]  # a mode with a count of 2
    strings = [
        "ab",
        "B1",
        "ab",
        None,
        "cc",
        "dd",
        "dd",
        "x",
        "XY",
        "xy",
        "XY",
        "e",
        "e",
    ]
    bools = [
        True,
        False,
        True,
        np.nan,
        False,
        True,
        True,
        False,
        False,
        True,
        False,
        True,
        True,
    ]
    is_delta = [True, False] * 6 + [True]
    parts = [
        pd.DataFrame(
            {
                "systime": times,
                "uuid": "num",
                "value_double": doubles,
                "is_delta": is_delta,
            }
        ),
        pd.DataFrame(
            {
                "systime": times,
                "uuid": "int",
                "value_integer": pd.array(np.arange(n) % 4, dtype="int64"),
                "is_delta": is_delta,
            }
        ),
        pd.DataFrame(
            {
                "systime": times,
                "uuid": "flag",
                "value_bool": pd.Series(bools, dtype=object),
            }
        ),
        pd.DataFrame({"systime": times, "uuid": "text", "value_string": strings}),
    ]
    return pd.concat(parts, ignore_index=True)


def _same(a: object, b: object) -> bool:
    if a is None or (isinstance(a, float) and np.isnan(a)) or a is pd.NaT:
        return b is None or pd.isna(b)
    if isinstance(a, pd.Timedelta):
        return a == pd.Timedelta(b)
    if isinstance(a, (bool, np.bool_)):
        return float(a) == float(b)  # type: ignore[arg-type]
    if isinstance(a, (int, float, np.integer, np.floating)):
        return bool(np.isclose(float(a), float(b), rtol=1e-9, atol=1e-9))  # type: ignore[arg-type]
    return a == b


def test_each_window_matches_compute_on_its_slice():
    df = _frame()
    out = DescriptiveFeatures(df).compute_windowed("1s")
    kinds = {"num": "numeric", "int": "numeric", "flag": "boolean", "text": "string"}
    column = {
        "num": "value_double",
        "int": "value_integer",
        "flag": "value_bool",
        "text": "value_string",
    }
    checked = 0
    for _, row in out.iterrows():
        part = df[(df["systime"] >= row["start"]) & (df["systime"] < row["end"])]
        for uuid, group in part.groupby("uuid"):
            col = column[uuid]
            stats = DescriptiveFeatures(
                group[
                    [
                        "systime",
                        "uuid",
                        col,
                        *(["is_delta"] if group["is_delta"].notna().any() else []),
                    ]
                ]
            ).compute()[uuid]
            stats_key = {
                "numeric": "numeric_stats",
                "boolean": "boolean_stats",
                "string": "string_stats",
            }
            expected = dict(stats["overall"])
            expected.update(stats[col][stats_key[kinds[uuid]]])
            for name in _OVERALL_FEATURES + _KIND_FEATURES[kinds[uuid]]:
                got = row[f"{uuid}__{name}"]
                if name == "total_time_seconds":
                    want = expected["total_time"].total_seconds()
                else:
                    want = expected[name]
                assert _same(want, got), (row["start"], uuid, name, want, got)
                checked += 1
    assert checked > 100


def test_grid_gaps_and_window_columns():
    df = _frame()
    out = DescriptiveFeatures(df).compute_windowed("1s")
    assert list(out.columns[:3]) == ["start", "end", "duration_seconds"]
    assert list(out["start"]) == [T0 + pd.Timedelta(seconds=s) for s in range(6)]
    assert list(out["end"]) == [T0 + pd.Timedelta(seconds=s) for s in range(1, 7)]
    assert out["duration_seconds"].dtype == "float64"
    assert (out["duration_seconds"] == 1.0).all()
    # 3 s and 4 s hold no data: NaN, counts 0.
    gap = out.iloc[3:5]
    assert gap["num__mean"].isna().all()
    assert gap["text__most_frequent"].isna().all()
    assert gap["flag__mode"].isna().all()
    assert gap["num__total_time"].isna().all()
    assert (gap["num__total_rows"] == 0).all()
    assert (gap["text__count_null"] == 0).all()
    assert out["num__total_rows"].dtype == "int64"
    assert out["num__mean"].dtype == "float64"
    assert out["flag__is_balanced"].dtype == "float64"
    assert list(out["num__total_rows"]) == [5, 3, 3, 0, 0, 2]
    # Signals sorted by uuid, features in compute order.
    uuids = [c.split("__")[0] for c in out.columns[3:]]
    assert uuids == sorted(uuids)
    num_cols = [c.split("__", 1)[1] for c in out.columns if c.startswith("num__")]
    assert num_cols == list(_OVERALL_FEATURES + _KIND_FEATURES["numeric"])


def test_leading_and_trailing_grid_follow_grouper():
    df = pd.DataFrame(
        {
            "systime": pd.to_datetime(
                ["2024-01-01 00:00:03.5", "2024-01-01 00:00:20.0"]
            ),
            "uuid": "a",
            "value_double": [1.0, 2.0],
        }
    )
    out = DescriptiveFeatures(df).compute_windowed("7s", metrics=["mean"])
    grouper = df.groupby(pd.Grouper(key="systime", freq="7s"))["value_double"].mean()
    assert list(out["start"]) == list(grouper.index)
    np.testing.assert_allclose(out["a__mean"], grouper.to_numpy())
    assert list(out.columns) == ["start", "end", "duration_seconds", "a__mean"]


def test_tz_aware_dst_day_and_hour():
    times = pd.date_range(
        "2024-03-30 22:00", "2024-04-01 01:00", freq="30min", tz="Europe/Berlin"
    )
    df = pd.DataFrame(
        {
            "systime": times,
            "uuid": "a",
            "value_double": np.arange(len(times), dtype=float),
        }
    )
    day = DescriptiveFeatures(df).compute_windowed("1D", metrics=["total_rows"])
    assert str(day["start"].dt.tz) == "Europe/Berlin"
    assert list(day["duration_seconds"]) == [86400.0, 82800.0, 86400.0]
    assert list(day["a__total_rows"]) == [4, 46, 3]
    assert day["end"].iloc[1] == pd.Timestamp("2024-04-01", tz="Europe/Berlin")

    hour = DescriptiveFeatures(df).compute_windowed("1h", metrics=["total_rows"])
    assert (hour["duration_seconds"] == 3600.0).all()
    assert hour["a__total_rows"].sum() == len(df)
    # The skipped hour 02:00-03:00 does not exist: one window fewer than wall clock.
    assert len(hour) == 27


def test_naive_daily():
    times = pd.date_range("2024-01-01 12:00", periods=5, freq="12h")
    df = pd.DataFrame(
        {"systime": times, "uuid": "a", "value_double": [1.0, 2, 3, 4, 5]}
    )
    out = DescriptiveFeatures(df).compute_windowed("1D", metrics=["mean", "sum"])
    assert out["start"].dt.tz is None
    assert list(out["a__mean"]) == [1.0, 2.5, 4.5]
    assert list(out["a__sum"]) == [1.0, 5.0, 9.0]


def test_metrics_subset_and_skipped_kinds():
    out = DescriptiveFeatures(_frame()).compute_windowed(
        "2s", metrics=["mean", "true_count", "most_frequent"]
    )
    assert list(out.columns[3:]) == [
        "flag__true_count",
        "int__mean",
        "num__mean",
        "text__most_frequent",
    ]


def test_unknown_metric_raises():
    with pytest.raises(ValueError, match="unknown metrics.*Available"):
        DescriptiveFeatures(_frame()).compute_windowed(metrics=["mean", "nope"])


def test_compute_routes_to_windowed():
    df = _frame()
    via_compute = DescriptiveFeatures(df).compute(output_format="windowed", window="2s")
    direct = DescriptiveFeatures(df).compute_windowed("2s")
    pd.testing.assert_frame_equal(via_compute, direct)
    default = DescriptiveFeatures(df).compute(output_format="windowed")
    assert (default["duration_seconds"] == 1.0).all()
    with pytest.raises(ValueError, match="window applies only"):
        DescriptiveFeatures(df).compute(output_format="dict", window="1s")
    with pytest.raises(ValueError, match="'windowed'"):
        DescriptiveFeatures(df).compute(output_format="nope")


def test_empty_frame_keeps_window_columns():
    df = _frame().iloc[0:0]
    out = DescriptiveFeatures(df).compute_windowed("1s")
    assert list(out.columns) == ["start", "end", "duration_seconds"]
    assert out.empty


def test_unsorted_input_same_result():
    df = _frame()
    shuffled = df.sample(frac=1.0, random_state=0)
    a = DescriptiveFeatures(df).compute_windowed(
        "1s", metrics=["mean", "total_rows", "unique_values"]
    )
    b = DescriptiveFeatures(shuffled).compute_windowed(
        "1s", metrics=["mean", "total_rows", "unique_values"]
    )
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.scale
def test_scale_one_million_rows():
    n_per, uuids = 200_000, ["a", "b", "c", "d", "e"]
    rng = np.random.default_rng(0)
    times = pd.date_range("2024-01-01", periods=n_per, freq="900ms")
    parts = []
    for i, u in enumerate(uuids):
        part = pd.DataFrame({"systime": times, "uuid": u})
        if i < 3:
            part["value_double"] = rng.normal(size=n_per)
        elif i == 3:
            part["value_bool"] = rng.random(n_per) > 0.5
        else:
            part["value_string"] = rng.choice(["idle", "RUN", "stop2"], n_per)
        parts.append(part)
    df = pd.concat(parts, ignore_index=True)
    begin = time.perf_counter()
    out = DescriptiveFeatures(df).compute_windowed("1s")
    elapsed = time.perf_counter() - begin
    print(f"compute_windowed: {len(df)} rows -> {out.shape} in {elapsed:.2f} s")
    assert len(out) == 180_000
    assert out["a__total_rows"].sum() == n_per
    assert elapsed < 30
