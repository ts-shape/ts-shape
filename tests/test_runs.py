"""Tests for ts_shape.utils.runs."""

import numpy as np
import pandas as pd
import pytest

from ts_shape.utils.runs import (
    RUN_FUNCS,
    run_agg,
    run_argmax,
    run_bounds,
    run_labels,
    run_positions,
    total_seconds,
    true_runs,
)


def _pandas_runs(values: pd.Series) -> list[tuple[int, int]]:
    """Reference: the ``(s != s.shift()).cumsum()`` idiom, as (start, end)."""
    s = pd.Series(values).reset_index(drop=True)
    gid = (s != s.shift()).cumsum()
    out = []
    for _, seg in s.groupby(gid):
        out.append((int(seg.index[0]), int(seg.index[-1]) + 1))
    return out


def _as_pairs(runs) -> list[tuple[int, int]]:
    return list(zip(runs.starts.tolist(), runs.ends.tolist(), strict=True))


class TestRunBounds:
    def test_empty(self):
        r = run_bounds(np.array([], dtype=float))
        assert r.n_runs == 0
        assert r.starts.dtype == np.int64
        assert r.ends.size == 0 and r.lengths.size == 0

    def test_empty_series_object(self):
        assert run_bounds(pd.Series([], dtype=object)).n_runs == 0

    def test_single(self):
        r = run_bounds([5])
        assert _as_pairs(r) == [(0, 1)]
        assert r.lengths.tolist() == [1]
        assert r.last.tolist() == [0]

    def test_all_same(self):
        r = run_bounds(np.full(7, 3))
        assert _as_pairs(r) == [(0, 7)]
        assert r.lengths.tolist() == [7]

    def test_ends_exclusive(self):
        r = run_bounds([1, 1, 2, 2, 2, 1])
        assert r.starts.tolist() == [0, 2, 5]
        assert r.ends.tolist() == [2, 5, 6]
        assert r.last.tolist() == [1, 4, 5]
        assert r.lengths.tolist() == [2, 3, 1]

    def test_bool_mask(self):
        r = run_bounds(np.array([True, True, False, True]))
        assert _as_pairs(r) == [(0, 2), (2, 3), (3, 4)]

    def test_nan_states_unequal_by_default(self):
        v = np.array([1.0, np.nan, np.nan, 1.0, 1.0])
        r = run_bounds(v)
        assert _as_pairs(r) == _pandas_runs(pd.Series(v))
        assert _as_pairs(r) == [(0, 1), (1, 2), (2, 3), (3, 5)]

    def test_nan_equal(self):
        v = np.array([1.0, np.nan, np.nan, 1.0, 1.0])
        assert _as_pairs(run_bounds(v, nan_equal=True)) == [(0, 1), (1, 3), (3, 5)]

    def test_all_nan(self):
        v = np.full(3, np.nan)
        assert run_bounds(v).n_runs == 3
        assert run_bounds(v, nan_equal=True).n_runs == 1

    @pytest.mark.parametrize(
        "values",
        [
            ["a", "a", None, None, "b", "a", "a"],
            ["a", np.nan, "a", "a", pd.NA, pd.NA],
            [1, 1, 1.0, True, 2, None],
            pd.Categorical(["x", "x", "y", None, None, "x"]),
            pd.to_datetime(
                ["2024-01-01", "2024-01-01", None, "2024-01-02"]
            ).tz_localize("UTC"),
        ],
    )
    def test_matches_pandas_idiom(self, values):
        s = pd.Series(values)
        assert _as_pairs(run_bounds(s)) == _pandas_runs(s)

    def test_nullable_missing_is_own_run(self):
        # For nullable dtypes ``s != s.shift()`` is <NA> next to a missing
        # value and the pandas idiom drops those rows; run_bounds keeps them.
        b = pd.array([True, None, None, False, False], dtype="boolean")
        assert _as_pairs(run_bounds(b)) == [(0, 1), (1, 2), (2, 3), (3, 5)]
        assert _as_pairs(run_bounds(b, nan_equal=True)) == [(0, 1), (1, 3), (3, 5)]
        i = pd.Series(pd.array([1, 1, None, 2], dtype="Int64"))
        assert _as_pairs(run_bounds(i)) == [(0, 2), (2, 3), (3, 4)]

    def test_object_nan_equal(self):
        s = pd.Series(["a", None, np.nan, "a"])
        assert _as_pairs(run_bounds(s, nan_equal=True)) == [(0, 1), (1, 3), (3, 4)]

    @pytest.mark.parametrize("seed", range(5))
    def test_random_matches_pandas_idiom(self, seed):
        rng = np.random.default_rng(seed)
        v = rng.integers(0, 3, size=500).astype(float)
        v[rng.random(500) < 0.1] = np.nan
        s = pd.Series(v)
        assert _as_pairs(run_bounds(s)) == _pandas_runs(s)
        strs = pd.Series(np.where(np.isnan(v), None, v.astype(str)), dtype=object)
        assert _as_pairs(run_bounds(strs)) == _pandas_runs(strs)

    def test_labels(self):
        r = run_bounds([1, 1, 2, 3, 3])
        assert run_labels(r).tolist() == [0, 0, 1, 2, 2]


class TestTrueRuns:
    def test_basic(self):
        r = true_runs(np.array([False, True, True, False, True]))
        assert _as_pairs(r) == [(1, 3), (4, 5)]
        assert r.lengths.tolist() == [2, 1]

    def test_empty_and_all_false(self):
        assert true_runs(np.array([], dtype=bool)).n_runs == 0
        assert true_runs(np.zeros(4, dtype=bool)).n_runs == 0

    def test_all_true(self):
        assert _as_pairs(true_runs(np.ones(4, dtype=bool))) == [(0, 4)]

    def test_nullable_na_is_false(self):
        m = pd.array([True, None, True, True], dtype="boolean")
        assert _as_pairs(true_runs(m)) == [(0, 1), (2, 4)]

    def test_series(self):
        assert _as_pairs(true_runs(pd.Series([True, False]))) == [(0, 1)]


class TestRunAgg:
    def test_first_last_size(self):
        r = run_bounds([1, 1, 2, 2, 2])
        out = run_agg(np.array([10, 11, 12, 13, 14]), r, ["first", "last", "size"])
        assert out["first"].tolist() == [10, 12]
        assert out["last"].tolist() == [11, 14]
        assert out["size"].tolist() == [2, 3]

    def test_single_name_returns_array(self):
        r = run_bounds([1, 2])
        assert run_agg([3.0, 4.0], r, "max").tolist() == [3.0, 4.0]

    def test_empty(self):
        r = run_bounds(np.array([], dtype=float))
        for f in RUN_FUNCS:
            assert run_agg(np.array([], dtype=float), r, f).size == 0

    def test_nan_handling(self):
        v = np.array([1.0, np.nan, 3.0, np.nan, np.nan])
        r = run_bounds([0, 0, 0, 1, 1])
        out = run_agg(v, r, RUN_FUNCS)
        assert np.isnan(out["sum"][0]) and np.isnan(out["mean"][0])
        assert out["nansum"].tolist() == [4.0, 0.0]
        assert out["nanmean"][0] == 2.0 and np.isnan(out["nanmean"][1])
        assert out["nanmin"][0] == 1.0 and np.isnan(out["nanmin"][1])
        assert out["nanmax"][0] == 3.0 and np.isnan(out["nanmax"][1])
        assert np.isnan(out["min"][0]) and np.isnan(out["max"][0])
        assert out["count"].tolist() == [2, 0]

    def test_any_all(self):
        r = run_bounds([0, 0, 1, 1])
        out = run_agg(np.array([True, False, True, True]), r, ["any", "all"])
        assert out["any"].tolist() == [True, True]
        assert out["all"].tolist() == [False, True]

    def test_true_runs_exclude_gaps(self):
        mask = np.array([True, False, False, True, True, False])
        v = np.array([1.0, 100.0, -100.0, 2.0, 3.0, 50.0])
        out = run_agg(v, true_runs(mask), ["min", "max", "sum", "count", "last"])
        assert out["min"].tolist() == [1.0, 2.0]
        assert out["max"].tolist() == [1.0, 3.0]
        assert out["sum"].tolist() == [1.0, 5.0]
        assert out["count"].tolist() == [1, 2]
        assert out["last"].tolist() == [1.0, 3.0]

    def test_unknown(self):
        with pytest.raises(ValueError, match="unknown run aggregation"):
            run_agg([1.0], run_bounds([1]), "median")

    @pytest.mark.parametrize("seed", range(6))
    def test_bit_identical_to_pandas(self, seed):
        rng = np.random.default_rng(seed)
        n = 4000
        lengths = rng.integers(1, [3, 12, 140, 600, 2, 40][seed], size=n)
        state = np.repeat(np.arange(n) % 3, lengths)[:n]
        v = rng.normal(size=n) * 10.0 ** rng.integers(-3, 8)
        v[rng.random(n) < 0.05] = np.nan
        v[rng.random(n) < 0.05] = -0.0
        r = run_bounds(state)
        out = run_agg(v, r, ["sum", "nansum", "nanmean", "mean", "nanmin", "nanmax"])
        segs = [pd.Series(v[s:e]) for s, e in zip(r.starts, r.ends, strict=True)]
        exp_sum = np.array([s.to_numpy().sum() for s in segs])
        np.testing.assert_array_equal(out["sum"], exp_sum)
        assert (np.signbit(out["sum"]) == np.signbit(exp_sum)).all()
        np.testing.assert_array_equal(out["nansum"], [s.sum() for s in segs])
        np.testing.assert_array_equal(out["nanmean"], [s.mean() for s in segs])
        np.testing.assert_array_equal(out["nanmin"], [s.min() for s in segs])
        np.testing.assert_array_equal(out["nanmax"], [s.max() for s in segs])

    def test_long_run_sum_exact(self):
        v = np.random.default_rng(9).normal(size=100_003) * 1e6
        r = run_bounds(np.zeros(v.size))
        assert run_agg(v, r, "sum")[0] == v.sum()


class TestHelpers:
    def test_positions_and_labels_for_true_runs(self):
        r = true_runs(np.array([False, True, True, False, True]))
        assert run_positions(r).tolist() == [1, 2, 4]
        assert run_labels(r).tolist() == [0, 0, 1]

    def test_argmax_first_on_ties_and_skips_nan(self):
        r = run_bounds([0, 0, 0, 1, 1, 2])
        v = np.array([1.0, 3.0, 3.0, np.nan, 2.0, np.nan])
        assert run_argmax(v, r).tolist() == [1, 4, 5]
        assert run_argmax(v, run_bounds(np.array([]))).size == 0

    @pytest.mark.parametrize("unit", ["ns", "us", "ms", "s"])
    def test_total_seconds_matches_scalar(self, unit):
        rng = np.random.default_rng(3)
        ns = rng.integers(-(10**15), 10**15, size=2000)
        td = pd.to_timedelta(ns, unit="ns").as_unit(unit)
        expected = [x.total_seconds() for x in td]
        np.testing.assert_array_equal(total_seconds(td), expected)

    def test_total_seconds_nat(self):
        out = total_seconds(pd.to_timedelta([1, None], unit="s"))
        assert out[0] == 1.0 and np.isnan(out[1])
