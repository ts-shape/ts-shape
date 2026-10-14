"""Vectorized per-row window statistics (internal).

The maintenance detectors evaluate statistics -- mean, variance, a least-squares
slope -- over one window per sample. Doing that with a boolean mask over the
whole series per row is O(n^2); this module computes the window bounds with
``searchsorted`` and the statistics for all windows at once.

The statistics are evaluated in two passes per window (window mean first, then
centred sums) on gathered window members rather than from prefix sums: prefix
sums of ``x**2`` / ``x*y`` cancel catastrophically on long series (seconds since
start grow into the millions), which would change results well beyond rounding.
The cost is O(sum of window lengths), processed in bounded chunks.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd

_MAX_CHUNK_ELEMENTS = 2_000_000


def time_window_bounds(
    times: pd.Series, window: pd.Timedelta
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(lo, hi)`` so row ``i``'s window is ``[lo[i], hi[i])``.

    The window holds every row with ``t_i - window < t <= t_i`` -- the same rows
    as the boolean mask ``(t > t_i - window) & (t <= t_i)`` over a sorted
    series, including later rows that share ``t_i`` (duplicate timestamps).
    """
    t = times_ns(times)
    w = np.int64(pd.Timedelta(window).value)
    hi = np.searchsorted(t, t, side="right")
    lo = np.searchsorted(t, t - w, side="right")
    return lo.astype(np.int64), hi.astype(np.int64)


def count_window_bounds(n: int, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Bounds of the trailing windows ``[max(0, i - size + 1), i + 1)``."""
    hi = np.arange(1, n + 1, dtype=np.int64)
    lo = np.maximum(hi - size, 0)
    return lo, hi


def times_ns(times: pd.Series) -> np.ndarray:
    """Sorted times as int64 nanoseconds since the epoch (UTC for tz-aware)."""
    ser = pd.Series(times)
    if isinstance(ser.dtype, pd.DatetimeTZDtype):
        ser = ser.dt.tz_convert("UTC").dt.tz_localize(None)
    return ser.to_numpy(dtype="datetime64[ns]").view(np.int64)


def interval_bounds(
    times: pd.Series, starts: pd.DatetimeIndex, width: pd.Timedelta
) -> tuple[np.ndarray, np.ndarray]:
    """Index ranges of the half-open intervals ``[start, start + width)``."""
    t = times_ns(times)
    s = times_ns(pd.Series(starts))
    w = np.int64(pd.Timedelta(width).value)
    lo = np.searchsorted(t, s, side="left")
    hi = np.searchsorted(t, s + w, side="left")
    return lo.astype(np.int64), hi.astype(np.int64)


def gather(lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flat member indices of windows ``[lo, hi)`` (all non-empty) and their
    offsets in the flat array, for ``np.ufunc.reduceat``."""
    lens = (hi - lo).astype(np.int64)
    starts = np.concatenate(([0], np.cumsum(lens)[:-1])).astype(np.int64)
    idx = np.arange(int(lens.sum()), dtype=np.int64) + np.repeat(lo - starts, lens)
    return idx, starts


def _chunks(lengths: np.ndarray) -> Iterator[tuple[int, int]]:
    """Split rows into consecutive chunks of at most ~_MAX_CHUNK_ELEMENTS members."""
    cum = np.cumsum(lengths)
    start, m = 0, len(lengths)
    while start < m:
        base = cum[start - 1] if start else 0
        end = int(np.searchsorted(cum, base + _MAX_CHUNK_ELEMENTS, side="right"))
        end = max(end, start + 1)
        yield start, end
        start = end


def window_moments(
    x: np.ndarray, y: np.ndarray, lo: np.ndarray, hi: np.ndarray
) -> dict[str, np.ndarray]:
    """Least-squares statistics of ``y`` against ``x`` over each window.

    NaN ``y`` samples are skipped (as pandas ``mean``/``var`` do). Returns per row:

    - ``n_total``: rows in the window, NaN or not;
    - ``n``: non-NaN rows;
    - ``mean``: mean of ``y`` (NaN when ``n == 0``);
    - ``var``: sample variance of ``y``, ``ddof=1`` (NaN when ``n < 2``);
    - ``sxx``, ``sxy``, ``syy``: centred sums of squares/products;
    - ``slope``: ``sxy / sxx`` -- the ``np.polyfit(x, y, 1)`` slope -- NaN when
      ``n < 2`` or all ``x`` coincide (``sxx == 0``, duplicate timestamps).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    lo = np.asarray(lo, dtype=np.int64)
    hi = np.asarray(hi, dtype=np.int64)
    m = len(lo)
    valid = ~np.isnan(y)
    w = valid.astype(float)
    y0 = np.where(valid, y, 0.0)

    n_total = (hi - lo).astype(np.int64)
    n = np.zeros(m)
    sy = np.zeros(m)
    sxx = np.zeros(m)
    sxy = np.zeros(m)
    syy = np.zeros(m)

    rows = np.flatnonzero(n_total > 0)
    lengths = n_total[rows]
    for a, b in _chunks(lengths):
        r = rows[a:b]
        lens = lengths[a:b]
        starts = np.concatenate(([0], np.cumsum(lens)[:-1]))
        idx = np.arange(int(lens.sum()), dtype=np.int64) + np.repeat(
            lo[r] - starts, lens
        )
        wx, xx, yy = w[idx], x[idx], y0[idx]
        cnt = np.add.reduceat(wx, starts)
        with np.errstate(invalid="ignore", divide="ignore"):
            mx = np.add.reduceat(xx * wx, starts) / cnt
            my = np.add.reduceat(yy, starts) / cnt
        dx = (xx - np.repeat(mx, lens)) * wx
        dy = (yy - np.repeat(my, lens)) * wx
        dx = np.where(wx > 0, dx, 0.0)
        dy = np.where(wx > 0, dy, 0.0)
        n[r] = cnt
        sy[r] = my
        sxx[r] = np.add.reduceat(dx * dx, starts)
        sxy[r] = np.add.reduceat(dx * dy, starts)
        syy[r] = np.add.reduceat(dy * dy, starts)

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


def py_min(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray:
    """Elementwise Python ``min(a, b)`` (keeps ``a`` unless ``b < a``; NaN-faithful)."""
    return np.where(np.asarray(b) < a, b, a)


def py_max(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray:
    """Elementwise Python ``max(a, b)`` (keeps ``a`` unless ``b > a``; NaN-faithful)."""
    return np.where(np.asarray(b) > a, b, a)
