"""Vectorized per-row window statistics (internal).

The maintenance detectors evaluate statistics -- mean, variance, a least-squares
slope -- over one window per sample. Doing that with a boolean mask over the
whole series per row is O(n^2); this module computes the window bounds with
``searchsorted`` and the statistics for all windows at once.

:func:`window_moments` does not visit every member of every window (that costs
O(sum of window lengths), i.e. O(n * window)). It builds the moments of
power-of-two blocks -- ``[i, i + 2**k)`` for every ``i`` -- level by level and
assembles each window ``[lo, hi)`` from the blocks given by the binary digits
of its length: O(n log w) for windows of up to ``w`` samples, independent of
how many samples a window holds.

Blocks are combined with the pairwise update of Chan, Golub & LeVeque (1983):
every block carries its count, its means and its *centred* sums of squares and
products, and two blocks merge as::

    n = na + nb,  d = mean_b - mean_a,  mean = mean_a + d * nb / n,
    S = Sa + Sb + d_x * d_y * na * nb / n

Each statistic is thus re-centred on its block's own mean at every step, so
there is no ``sum(x**2) - sum(x)**2 / n`` cancellation (prefix sums of ``x**2``
over seconds-since-start grow into the 1e18s and would lose every digit of a
window's spread).

Accuracy: the result agrees with the two-pass evaluation (window mean first,
then centred sums over the gathered members) to rounding -- within ``1e-9``
relative for every statistic that is not itself at rounding level. Windows
where the two-pass result *is* rounding noise (a constant window, all samples
at one timestamp, an exactly flat trend: the two-pass mean is not exactly the
constant, so it reports a spread of ~1e-30 where the pairwise update gives an
exact 0) are detected and recomputed with the two-pass evaluation, so those
match it exactly as well. A mean close to 0 relative to the window's spread
(zero-mean noise) agrees to ``1e-9`` relative of the spread rather than of the
mean itself.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd

from ts_shape.utils.frames import to_ns

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
    return to_ns(pd.Series(times))


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


_ROUNDOFF = 2.0**-53
#: Re-evaluate a window with the two-pass formula when one of its centred sums
#: is within this factor of the two-pass evaluation's own rounding error.
_DEGENERATE_FACTOR = 1e10


def _two_pass(
    x: np.ndarray,
    y: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    rows: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Two-pass moments of the (non-empty) windows ``rows``: per window the
    mean first, then centred sums over the gathered members. NaN ``y`` are
    skipped. Returns ``(n, mean_y, sxx, sxy, syy)`` for ``rows``.
    O(sum of the window lengths), processed in bounded chunks.
    """
    valid = ~np.isnan(y)
    w = valid.astype(float)
    y0 = np.where(valid, y, 0.0)
    k = len(rows)
    n = np.zeros(k)
    my_out = np.zeros(k)
    sxx = np.zeros(k)
    sxy = np.zeros(k)
    syy = np.zeros(k)
    lengths = (hi[rows] - lo[rows]).astype(np.int64)
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
        n[a:b] = cnt
        my_out[a:b] = my
        sxx[a:b] = np.add.reduceat(dx * dx, starts)
        sxy[a:b] = np.add.reduceat(dx * dy, starts)
        syy[a:b] = np.add.reduceat(dy * dy, starts)
    return n, my_out, sxx, sxy, syy


def _merge(
    a: list[np.ndarray],
    b: list[np.ndarray],
    shift_y: np.ndarray,
    shift_x: np.ndarray | None,
) -> list[np.ndarray]:
    """Chan et al. pairwise merge of block moments.

    A block is ``[n, my, syy]`` or, with ``shift_x``, ``[n, my, syy, mx, sxx,
    sxy]``; ``a`` precedes ``b``. Means are stored relative to a reference
    value of each block (the series' value at the block's first row), and
    ``shift_*`` is ``ref(b) - ref(a)``: the result is relative to ``ref(a)``.
    Keeping means small this way keeps their rounding at the scale of the
    block's spread instead of the series' magnitude. Empty blocks
    (``n == 0``) carry zero means and sums and merge as the identity.
    """
    na, nb = a[0], b[0]
    n = na + nb
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.where(n > 0, nb / n, 0.0)
    g = na * f
    dy = shift_y + (b[1] - a[1])
    out = [n, a[1] + dy * f, a[2] + b[2] + dy * dy * g]
    if shift_x is not None:
        dx = shift_x + (b[3] - a[3])
        out += [a[3] + dx * f, a[4] + b[4] + dx * dx * g, a[5] + b[5] + dx * dy * g]
    return out


def _reference(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """A finite value near each row: the row's own value, else the last valid
    one before it (else the first valid one; 0 for an all-missing series)."""
    if valid.all():
        return values
    if not valid.any():
        return np.zeros(len(values))
    idx = np.where(valid, np.arange(len(values)), -1)
    np.maximum.accumulate(idx, out=idx)
    idx[idx < 0] = int(np.argmax(valid))
    return values[idx]


#: Rows per chunk of :func:`window_moments`: the block levels are built over
#: the rows the chunk's windows span only, bounding memory (and staying in
#: cache) on long series.
_ROW_CHUNK = 1 << 20


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

    O(n log w) for windows of up to ``w`` rows (see the module docstring for
    the method and its accuracy).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    lo = np.asarray(lo, dtype=np.int64)
    hi = np.asarray(hi, dtype=np.int64)
    m = len(lo)
    if m <= _ROW_CHUNK:
        return _moments_of_windows(x, y, lo, hi)
    parts = []
    for r0 in range(0, m, _ROW_CHUNK):
        r1 = min(r0 + _ROW_CHUNK, m)
        clo, chi = lo[r0:r1], hi[r0:r1]
        nonempty = chi > clo
        a = int(clo[nonempty].min()) if nonempty.any() else 0
        b = int(chi[nonempty].max()) if nonempty.any() else 0
        # Empty windows keep lo == hi inside the span.
        parts.append(
            _moments_of_windows(
                x[a:b],
                y[a:b],
                np.where(nonempty, clo - a, 0),
                np.where(nonempty, chi - a, 0),
            )
        )
    return {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}


def _moments_of_windows(
    x: np.ndarray, y: np.ndarray, lo: np.ndarray, hi: np.ndarray
) -> dict[str, np.ndarray]:
    """:func:`window_moments` for one chunk of rows (arrays cover their span)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    lo = np.asarray(lo, dtype=np.int64)
    hi = np.asarray(hi, dtype=np.int64)
    m = len(lo)
    size = len(y)
    n_total = (hi - lo).astype(np.int64)
    valid = ~np.isnan(y)
    # x == 0 everywhere (variance / RMS callers): its sums are exactly 0.
    with_x = bool(np.any(x != 0))
    ref_y = _reference(y, valid)
    ref_x = _reference(x, ~np.isnan(x)) if with_x else None

    # Level 0: one block per sample (NaN samples are empty blocks); means are
    # relative to the block's first row, so 0 here.
    level = [valid.astype(float), np.zeros(size), np.zeros(size)]
    if with_x:
        level += [np.zeros(size), np.zeros(size), np.zeros(size)]
    # Each window's accumulator is relative to the window's first row.
    acc = [np.zeros(m) for _ in level]
    safe_lo = np.minimum(lo, max(size - 1, 0))
    pos = lo.copy()
    remaining = n_total.copy()
    k = 0
    while True:
        sel = np.flatnonzero(remaining & (1 << k))
        if len(sel):
            p = pos[sel]
            start = safe_lo[sel]
            merged = _merge(
                [arr[sel] for arr in acc],
                [arr[p] for arr in level],
                ref_y[p] - ref_y[start],
                None if ref_x is None else ref_x[p] - ref_x[start],
            )
            for arr, val in zip(acc, merged, strict=True):
                arr[sel] = val
            pos[sel] += 1 << k
            remaining[sel] -= 1 << k
        if not remaining.any():
            break
        half = 1 << k
        length = len(level[0]) - half
        level = _merge(
            [arr[:length] for arr in level],
            [arr[half:] for arr in level],
            ref_y[half : half + length] - ref_y[:length],
            None if ref_x is None else ref_x[half : half + length] - ref_x[:length],
        )
        k += 1

    if m and size:
        acc[1] = acc[1] + ref_y[safe_lo]
        if ref_x is not None:
            acc[3] = acc[3] + ref_x[safe_lo]
    n, my, syy = acc[0], acc[1], acc[2]
    if with_x:
        mx, sxx, sxy = acc[3], acc[4], acc[5]
    else:
        mx, sxx, sxy = np.zeros(m), np.zeros(m), np.zeros(m)

    # Windows whose two-pass result is rounding noise: recompute them with it.
    if m:
        dx = n * _ROUNDOFF * np.abs(mx)
        dy = n * _ROUNDOFF * np.abs(my)
        with np.errstate(invalid="ignore", over="ignore"):
            degenerate = (n >= 2) & (
                (syy <= _DEGENERATE_FACTOR * n * dy * dy)
                | (with_x & (sxx <= _DEGENERATE_FACTOR * n * dx * dx))
                | (with_x & (np.abs(sxy) <= _DEGENERATE_FACTOR * n * dx * dy))
            )
        rows = np.flatnonzero(degenerate)
        if len(rows):
            n_r, my_r, sxx_r, sxy_r, syy_r = _two_pass(x, y, lo, hi, rows)
            n[rows], my[rows], syy[rows] = n_r, my_r, syy_r
            if with_x:
                sxx[rows], sxy[rows] = sxx_r, sxy_r

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, my, np.nan)
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
