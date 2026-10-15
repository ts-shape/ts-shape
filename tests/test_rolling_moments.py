"""window_moments (O(n log w) pairwise blocks) == the former two-pass gather.

``_two_pass_reference`` is a frozen copy of the implementation it replaced
(O(sum of window lengths)): the new one must agree within 1e-9 relative, and
exactly where the two-pass result is rounding noise (constant windows).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ts_shape.events.maintenance._rolling import (
    count_window_bounds,
    time_window_bounds,
    window_moments,
)


def _two_pass_reference(
    x: np.ndarray, y: np.ndarray, lo: np.ndarray, hi: np.ndarray
) -> dict[str, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = len(lo)
    valid = ~np.isnan(y)
    w = valid.astype(float)
    y0 = np.where(valid, y, 0.0)
    n_total = (hi - lo).astype(np.int64)
    n, sy, sxx, sxy, syy = (np.zeros(m) for _ in range(5))
    for i in np.flatnonzero(n_total > 0):
        idx = np.arange(lo[i], hi[i])
        wx, xx, yy = w[idx], x[idx], y0[idx]
        cnt = np.add.reduceat(wx, [0])
        with np.errstate(invalid="ignore", divide="ignore"):
            mx = np.add.reduceat(xx * wx, [0]) / cnt
            my = np.add.reduceat(yy, [0]) / cnt
        dx = np.where(wx > 0, (xx - mx) * wx, 0.0)
        dy = np.where(wx > 0, (yy - my) * wx, 0.0)
        n[i], sy[i] = cnt[0], my[0]
        sxx[i] = np.add.reduceat(dx * dx, [0])[0]
        sxy[i] = np.add.reduceat(dx * dy, [0])[0]
        syy[i] = np.add.reduceat(dy * dy, [0])[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, sy, np.nan)
        var = np.where(n >= 2, syy / (n - 1), np.nan)
        slope = np.where((n >= 2) & (sxx > 0), sxy / sxx, np.nan)
    return {
        "n_total": n_total,
        "n": n,
        "mean": mean,
        "var": var,
        "sxx": sxx,
        "sxy": sxy,
        "syy": syy,
        "slope": slope,
    }


def _assert_close(ref: dict[str, np.ndarray], got: dict[str, np.ndarray]) -> None:
    assert set(ref) == set(got)
    for key in ref:
        a, b = ref[key], got[key]
        assert a.dtype == b.dtype, key
        if a.dtype.kind != "f":
            np.testing.assert_array_equal(b, a, err_msg=key)
            continue
        np.testing.assert_array_equal(np.isnan(b), np.isnan(a), err_msg=key)
        ok = ~np.isnan(a)
        # The mean of zero-mean noise is compared on the scale of its spread.
        scale = np.abs(a[ok])
        if key == "mean":
            scale = scale + np.sqrt(np.nan_to_num(ref["var"]))[ok]
        diff = np.abs(a[ok] - b[ok])
        assert np.all(diff <= 1e-9 * scale), (key, float(np.max(diff / scale)))


def _series(seed: int, n: int, kind: str) -> tuple[pd.Series, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = pd.Series(pd.date_range("2024-01-01", periods=n, freq="2s"))
    if kind == "duplicates":
        t = pd.Series(t.iloc[np.sort(rng.integers(0, n, n))].to_numpy())
    y = np.cumsum(rng.normal(0, 1, n)) + 1e4
    if kind == "quantized":
        y = np.round(y / 4) * 4  # long constant runs
    elif kind == "nan":
        y[rng.random(n) < 0.05] = np.nan
    elif kind == "noise":
        y = rng.normal(0, 1, n)
    elif kind == "constant":
        y = np.full(n, 0.1)
    return t, y


@pytest.mark.parametrize(
    "kind", ["walk", "duplicates", "quantized", "nan", "noise", "constant"]
)
@pytest.mark.parametrize("offset", [0.0, 1e7])
def test_window_moments_match_two_pass_reference(kind: str, offset: float) -> None:
    t, y = _series(7, 3000, kind)
    # Seconds since start, optionally far from 0 (a long series' tail).
    x = (t - t.iloc[0]).dt.total_seconds().to_numpy() + offset
    for window in ("9s", "10min", "3h"):
        lo, hi = time_window_bounds(t, pd.Timedelta(window))
        _assert_close(_two_pass_reference(x, y, lo, hi), window_moments(x, y, lo, hi))
        zeros = np.zeros(len(y))
        _assert_close(
            _two_pass_reference(zeros, y, lo, hi), window_moments(zeros, y, lo, hi)
        )


def test_constant_windows_are_exactly_the_two_pass_result() -> None:
    t, y = _series(1, 500, "constant")
    x = (t - t.iloc[0]).dt.total_seconds().to_numpy()
    lo, hi = time_window_bounds(t, pd.Timedelta("1min"))
    ref = _two_pass_reference(x, y, lo, hi)
    got = window_moments(x, y, lo, hi)
    for key in ("syy", "sxy", "var"):
        np.testing.assert_array_equal(got[key], ref[key])


def test_count_windows_and_empty_windows() -> None:
    t, y = _series(3, 400, "nan")
    y[:30] = np.nan  # windows without any value
    x = (t - t.iloc[0]).dt.total_seconds().to_numpy()
    lo, hi = count_window_bounds(len(y), 10)
    _assert_close(_two_pass_reference(x, y, lo, hi), window_moments(x, y, lo, hi))
    lo = np.array([0, 5, 5], dtype=np.int64)
    hi = np.array([0, 5, 6], dtype=np.int64)
    _assert_close(_two_pass_reference(x, y, lo, hi), window_moments(x, y, lo, hi))


def test_row_chunks_match_the_reference(monkeypatch) -> None:
    import ts_shape.events.maintenance._rolling as rolling

    monkeypatch.setattr(rolling, "_ROW_CHUNK", 97)  # windows straddle chunks
    t, y = _series(5, 1000, "nan")
    y[400:520] = np.nan  # a chunk whose windows are empty or partly empty
    x = (t - t.iloc[0]).dt.total_seconds().to_numpy() + 1e7
    lo, hi = time_window_bounds(t, pd.Timedelta("5min"))
    _assert_close(_two_pass_reference(x, y, lo, hi), window_moments(x, y, lo, hi))
