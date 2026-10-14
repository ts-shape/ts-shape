import itertools

import numpy as np
import pandas as pd
import pytest

from ts_shape.events.quality.change_point import (
    ChangePointEvents,
    _robust_sigma,
    binseg,
    pelt,
)


def _frame(values, tz=None, uuid="sig"):
    t = pd.date_range("2024-03-30", periods=len(values), freq="1min", tz=tz)
    return pd.DataFrame({"systime": t, "uuid": uuid, "value_double": values})


def _total_cost(x, points, cost, penalty):
    sigma = _robust_sigma(x)
    floor = max(np.var(x), 1e-12) * 1e-6
    bounds = [0, *points, len(x)]
    total = penalty * len(points)
    for a, b in zip(bounds[:-1], bounds[1:]):
        v = x[a:b]
        if cost == "normal_mean":
            total += ((v - v.mean()) ** 2).sum() / sigma**2
        else:
            total += len(v) * np.log(max(v.var(), floor))
    return total


@pytest.mark.parametrize(
    "cost,min_size",
    [
        ("normal_mean", 1),
        ("normal_mean", 3),
        ("normal_meanvar", 2),
        ("normal_meanvar", 3),
    ],
)
def test_pelt_is_exact_against_brute_force(cost, min_size):
    rng = np.random.default_rng(1)
    for _ in range(25):
        n = 12
        x = rng.normal(0, 1, n) + np.repeat(rng.normal(0, 2, 3), [4, 4, 4])
        pen = float(rng.uniform(0.5, 6))
        got = _total_cost(
            x, pelt(x, cost=cost, penalty=pen, min_size=min_size), cost, pen
        )
        ms = max(min_size, 2 if cost == "normal_meanvar" else 1)
        best = min(
            _total_cost(x, list(c), cost, pen)
            for k in range(n)
            for c in itertools.combinations(range(1, n), k)
            if all(b - a >= ms for a, b in zip((0, *c), (*c, n)))
        )
        assert got == pytest.approx(best, abs=1e-8)


def test_finds_steps_at_true_indices():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(m, 0.5, 200) for m in (10, 12, 9, 9.8)])
    for method in (pelt, binseg):
        found = method(x)
        assert len(found) == 3
        assert np.allclose(found, [200, 400, 600], atol=3)


def test_no_false_changes_on_stationary_noise():
    rng = np.random.default_rng(5)
    for _ in range(20):
        x = rng.normal(50, 2, 1000)
        assert pelt(x) == []
        assert pelt(x, cost="normal_meanvar") == []


def test_meanvar_finds_variance_change():
    rng = np.random.default_rng(3)
    x = np.concatenate([rng.normal(0, 1, 400), rng.normal(0, 4, 400)])
    found = pelt(x, cost="normal_meanvar")
    assert len(found) == 1 and abs(found[0] - 400) < 15


def test_detect_change_points_output_and_tz():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(10, 0.5, 100), rng.normal(13, 0.5, 100)])
    df = _frame(x, tz="Europe/Berlin")  # spans the spring-forward night
    out = ChangePointEvents(df, "sig").detect_change_points()
    assert len(out) == 1
    row = out.iloc[0]
    assert row["systime"] == df["systime"].iloc[100]
    assert str(out["systime"].dt.tz) == "Europe/Berlin"
    assert row["shift"] == pytest.approx(3, abs=0.3)
    assert row["shift_sigma"] > 4
    assert row["source_uuid"] == "sig" and row["uuid"] == "quality:change_point"


def test_segment_statistics_tile_the_signal():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(m, 0.3, 60) for m in (1, 3, 2)])
    df = _frame(x)
    segs = ChangePointEvents(df, "sig").segment_statistics()
    assert segs["sample_count"].sum() == len(x)
    assert segs["start"].iloc[0] == df["systime"].iloc[0]
    assert (segs["end"].iloc[:-1].to_numpy() == segs["start"].iloc[1:].to_numpy()).all()
    assert np.allclose(segs["mean"], [1, 3, 2], atol=0.15)


def test_other_uuids_nan_and_short_input():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(0, 1, 50), rng.normal(5, 1, 50)])
    df = _frame(x)
    df.loc[10, "value_double"] = np.nan
    noise = _frame(rng.normal(0, 100, 100), uuid="other")
    a = ChangePointEvents(pd.concat([df, noise]), "sig").detect_change_points()
    b = ChangePointEvents(df, "sig").detect_change_points()
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))
    assert ChangePointEvents(_frame([1.0, 2.0]), "sig").detect_change_points().empty
    assert len(ChangePointEvents(_frame([1.0, 2.0]), "sig").segment_statistics()) == 1


def test_validation_and_large_input_warning(monkeypatch):
    import ts_shape.events.quality.change_point as cp

    df = _frame(np.r_[np.zeros(30), np.ones(30)])
    with pytest.raises(ValueError, match="cost"):
        ChangePointEvents(df, "sig").detect_change_points(cost="poisson")
    with pytest.raises(ValueError, match="method"):
        ChangePointEvents(df, "sig").detect_change_points(method="window")
    monkeypatch.setattr(cp, "PELT_WARN_SAMPLES", 10)
    with pytest.warns(UserWarning, match="binseg"):
        ChangePointEvents(df, "sig").detect_change_points()
