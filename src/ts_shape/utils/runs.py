"""Vectorized run-length helpers.

Many detectors collapse consecutive equal values -- a boolean condition mask
or a state / identifier column -- into *runs* and then compute something per
run. The pandas idiom ``df.groupby((s != s.shift()).cumsum())`` followed by a
Python loop over the groups costs tens of microseconds per run; these helpers
find the run boundaries with one comparison and aggregate per run with
``ufunc.reduceat``.

:func:`run_bounds` returns every run of equal consecutive values,
:func:`true_runs` only the runs where a boolean mask is ``True``, and
:func:`run_agg` aggregates a value array per run.

Run boundaries follow the pandas idiom ``(s != s.shift()).cumsum()`` exactly:
a missing value (``NaN``, ``None``, ``NaT``, ``pd.NA``) never equals anything,
so by default each missing value is a run of its own. Pass
``nan_equal=True`` to merge consecutive missing values into one run instead.
(For nullable extension dtypes the idiom compares to ``<NA>`` next to a
missing value and ``groupby`` then drops those rows; :func:`run_bounds` keeps
them as runs of their own.)
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, NamedTuple, TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd

__all__ = [
    "Runs",
    "run_bounds",
    "true_runs",
    "run_labels",
    "run_agg",
    "RUN_FUNCS",
    "total_seconds",
    "run_positions",
    "run_argmax",
]

_Ints: TypeAlias = npt.NDArray[np.int64]


class Runs(NamedTuple):
    """Positional bounds of runs.

    Attributes:
        starts: Position of the first element of each run.
        ends: Position *one past* the last element of each run (exclusive,
            like a slice stop), so ``values[starts[k]:ends[k]]`` is run ``k``
            and the last element is ``values[ends[k] - 1]``.
        lengths: Number of elements per run (``ends - starts``, always >= 1).
    """

    starts: _Ints
    ends: _Ints
    lengths: _Ints

    @property
    def last(self) -> _Ints:
        """Position of the last element of each run (inclusive end)."""
        return self.ends - 1

    @property
    def n_runs(self) -> int:
        """Number of runs."""
        return int(self.starts.size)


def _empty_runs() -> Runs:
    e = np.zeros(0, dtype=np.int64)
    return Runs(e, e.copy(), e.copy())


def _from_change(change: npt.NDArray[np.bool_], n: int) -> Runs:
    """Runs from ``change[i]`` = "element ``i + 1`` starts a new run"."""
    starts = np.concatenate(([0], np.flatnonzero(change) + 1)).astype(np.int64)
    ends = np.concatenate((starts[1:], [n])).astype(np.int64)
    return Runs(starts, ends, ends - starts)


def _as_array(values: Any) -> Any:
    if isinstance(values, pd.Series | pd.Index):
        return values.array
    if isinstance(values, np.ndarray | pd.api.extensions.ExtensionArray):
        return values
    return np.asarray(values)


def run_bounds(values: Any, *, nan_equal: bool = False) -> Runs:
    """Find the runs of equal consecutive values.

    Args:
        values: A 1-d array, Series or Index of any dtype (numeric, boolean,
            datetime, string, object, categorical, nullable extension types).
            The order is taken as given; sort first if needed.
        nan_equal: ``False`` (default) treats every missing value as unequal
            to everything, including another missing value -- exactly what
            ``(s != s.shift()).cumsum()`` does -- so each missing value is a
            run of length 1. ``True`` merges consecutive missing values into
            one run.

    Returns:
        :class:`Runs` with ``starts`` / ``ends`` (exclusive) / ``lengths`` as
        ``int64`` arrays, one entry per run in order. Empty input gives empty
        arrays.
    """
    arr = _as_array(values)
    n = len(arr)
    if n == 0:
        return _empty_runs()
    if n == 1:
        return _from_change(np.zeros(0, dtype=bool), 1)

    if isinstance(arr, np.ndarray) and arr.dtype.kind in "biu":
        return _from_change(arr[1:] != arr[:-1], n)
    if isinstance(arr, np.ndarray) and arr.dtype.kind in "fc":
        na = np.isnan(arr)
        change = arr[1:] != arr[:-1]  # NaN != NaN is True
        if nan_equal:
            change &= ~(na[1:] & na[:-1])
        return _from_change(change, n)

    # Everything else: compare hash codes; missing values get code -1.
    codes, _ = pd.factorize(arr, use_na_sentinel=True)
    codes = np.asarray(codes)
    change = codes[1:] != codes[:-1]
    if not nan_equal:
        na = codes < 0
        change |= na[1:] | na[:-1]
    return _from_change(change, n)


def true_runs(mask: Any) -> Runs:
    """Runs where a boolean mask is ``True``.

    Args:
        mask: Boolean array or Series. Missing values (nullable boolean) count
            as ``False``.

    Returns:
        :class:`Runs` of the ``True`` stretches only, in order.
    """
    arr = _as_array(mask)
    if not (isinstance(arr, np.ndarray) and arr.dtype == bool):
        arr = np.asarray(pd.array(arr, dtype="boolean").fillna(False), dtype=bool)
    n = arr.size
    if n == 0:
        return _empty_runs()
    padded = np.concatenate(([False], arr, [False])).astype(np.int8)
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1).astype(np.int64)
    ends = np.flatnonzero(edges == -1).astype(np.int64)
    return Runs(starts, ends, ends - starts)


def run_labels(runs: Runs) -> _Ints:
    """Run number (0, 1, ...) per element covered by ``runs``, in order.

    For runs that tile the array (from :func:`run_bounds`) this is one label
    per element; for :func:`true_runs` only the ``True`` elements are covered
    (see :func:`run_positions` for their positions).
    """
    return np.repeat(np.arange(runs.n_runs, dtype=np.int64), runs.lengths)


def _float(values: Any) -> npt.NDArray[np.float64]:
    return np.asarray(values, dtype=np.float64)


_PW_BLOCK = 128  # numpy's pairwise-summation block size


def _pairwise(
    values: npt.NDArray[np.float64], starts: _Ints, lengths: _Ints
) -> npt.NDArray[np.float64]:
    """numpy's pairwise summation of ``values[s:s+n]``, vectorized over runs.

    Mirrors ``pairwise_sum`` in numpy's ``loops_utils.h``: a left-to-right
    loop from ``0.0`` below 8 elements, eight interleaved accumulators up to
    128 elements, and a recursive split (at a multiple of 8) above that.
    """
    out = np.zeros(starts.size, dtype=np.float64)

    small = np.flatnonzero(lengths < 8)
    if small.size:
        s, n = starts[small], lengths[small]
        acc = np.zeros(small.size, dtype=np.float64)
        for j in range(7):
            m = n > j
            if not m.any():
                break
            acc[m] += values[s[m] + j]
        out[small] = acc

    mid = np.flatnonzero((lengths >= 8) & (lengths <= _PW_BLOCK))
    if mid.size:
        s, n = starts[mid], lengths[mid]
        lane = np.arange(8)
        r = values[s[:, None] + lane]
        full = n - n % 8
        for i in range(8, _PW_BLOCK, 8):
            m = full > i
            if not m.any():
                break
            r[m] += values[s[m, None] + i + lane]
        res = ((r[:, 0] + r[:, 1]) + (r[:, 2] + r[:, 3])) + (
            (r[:, 4] + r[:, 5]) + (r[:, 6] + r[:, 7])
        )
        rest = n % 8
        for j in range(7):
            m = rest > j
            if not m.any():
                break
            res[m] += values[s[m] + full[m] + j]
        out[mid] = res

    big = np.flatnonzero(lengths > _PW_BLOCK)
    if big.size:
        s, n = starts[big], lengths[big]
        n2 = n // 2
        n2 -= n2 % 8
        # Both halves of every run in one call, so the number of calls grows
        # with the tree depth (log n), not with the number of tree nodes.
        halves = _pairwise(values, np.r_[s, s + n2], np.r_[n2, n - n2])
        out[big] = halves[: big.size] + halves[big.size :]
    return out


_BUFFER = 8192  # numpy's default ufunc buffer size


def _sums_in_buffers() -> bool:
    """True when this numpy sums a long array buffer by buffer.

    Older numpy (2.0, for one) reduces a long contiguous array in blocks of
    ``_BUFFER`` elements, pairwise within a block and left to right across
    blocks; newer numpy sums the whole array pairwise. The two differ in the
    last bit, so the emulation follows whichever this numpy does.
    """
    probe = np.random.default_rng(0).normal(size=3 * _BUFFER + 5) * 1e6
    whole = _pairwise(probe, np.array([0]), np.array([probe.size]))[0]
    return bool(whole != probe.sum())


_BUFFERED = _sums_in_buffers()


def _blocked(
    values: npt.NDArray[np.float64], starts: _Ints, lengths: _Ints
) -> npt.NDArray[np.float64]:
    """Buffer-by-buffer sum of each run (see :func:`_sums_in_buffers`)."""
    out = np.zeros(starts.size, dtype=np.float64)
    blocks = -(-lengths // _BUFFER)
    for k in range(int(np.max(blocks)) if blocks.size else 0):
        m = blocks > k
        offset = k * _BUFFER
        size = np.minimum(lengths[m] - offset, _BUFFER)
        out[m] += _pairwise(values, starts[m] + offset, size)
    return out


def _sum_exact(values: npt.NDArray[np.float64], runs: Runs) -> npt.NDArray[np.float64]:
    """Per-run sum, bit-identical to ``values[s:e].sum()`` for every run."""
    values = np.ascontiguousarray(values)
    if _BUFFERED and runs.lengths.size and int(np.max(runs.lengths)) > _BUFFER:
        return _blocked(values, runs.starts, runs.lengths)
    return _pairwise(values, runs.starts, runs.lengths)


def _nan_filled(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    na = np.isnan(values)
    return np.where(na, 0.0, values) if na.any() else values


def _reduceat(ufunc: np.ufunc, values: npt.NDArray[Any], runs: Runs) -> Any:
    """``ufunc.reduceat`` over exactly ``[start, end)`` of each run.

    Cutting at both bounds keeps gaps between runs (e.g. the ``False``
    stretches left out by :func:`true_runs`) out of the reduction; one padding
    element keeps a cut at ``len(values)`` valid.
    """
    cuts = np.empty(2 * runs.starts.size, dtype=np.int64)
    cuts[0::2] = runs.starts
    cuts[1::2] = runs.ends
    padded = np.concatenate((values, values[-1:]))
    return ufunc.reduceat(padded, cuts)[0::2]


def _count(values: npt.NDArray[np.float64], runs: Runs) -> _Ints:
    ok = (~np.isnan(values)).astype(np.int64)
    return np.asarray(_reduceat(np.add, ok, runs), dtype=np.int64)


#: Aggregations understood by :func:`run_agg`.
RUN_FUNCS = (
    "first",
    "last",
    "size",
    "count",
    "sum",
    "nansum",
    "mean",
    "nanmean",
    "min",
    "max",
    "nanmin",
    "nanmax",
    "any",
    "all",
)


def _one(values: Any, runs: Runs, func: str) -> npt.NDArray[Any]:
    starts = runs.starts
    if func == "size":
        return runs.lengths.copy()
    arr = np.asarray(values)
    if func == "first":
        return arr[starts]
    if func == "last":
        return arr[runs.ends - 1]
    if func in ("any", "all"):
        b = arr.astype(bool)
        op = np.logical_or if func == "any" else np.logical_and
        return np.asarray(_reduceat(op, b, runs), dtype=bool) if starts.size else b[:0]
    f = _float(arr)
    if starts.size == 0:
        return f[:0].astype(np.int64) if func == "count" else f[:0]
    if func == "count":
        return _count(f, runs)
    if func == "sum":
        return _sum_exact(f, runs)
    if func == "nansum":
        return _sum_exact(_nan_filled(f), runs)
    if func == "mean":
        with np.errstate(invalid="ignore", divide="ignore"):
            return _sum_exact(f, runs) / runs.lengths
    if func == "nanmean":
        cnt = _count(f, runs)
        total = _sum_exact(_nan_filled(f), runs)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(cnt > 0, total / np.maximum(cnt, 1), np.nan)
    if func == "min":
        return np.asarray(_reduceat(np.minimum, f, runs), dtype=np.float64)
    if func == "max":
        return np.asarray(_reduceat(np.maximum, f, runs), dtype=np.float64)
    if func == "nanmin":
        return np.asarray(_reduceat(np.fmin, f, runs), dtype=np.float64)
    if func == "nanmax":
        return np.asarray(_reduceat(np.fmax, f, runs), dtype=np.float64)
    raise ValueError(f"unknown run aggregation {func!r}; expected one of {RUN_FUNCS}")


def run_agg(values: Any, runs: Runs, funcs: str | Iterable[str]) -> Any:
    """Aggregate ``values`` per run.

    Args:
        values: 1-d array or Series aligned with the array the runs were
            found on (positionally).
        runs: Runs from :func:`run_bounds` or :func:`true_runs`. Runs must
            not overlap and must be in increasing order.
        funcs: One aggregation name or several, from :data:`RUN_FUNCS`:

            - ``first`` / ``last``: the first / last element (any dtype).
            - ``size``: run length; ``count``: non-NaN elements.
            - ``sum`` / ``mean`` / ``min`` / ``max``: NaN propagates, like
              numpy.
            - ``nansum`` / ``nanmean`` / ``nanmin`` / ``nanmax``: NaN skipped,
              like pandas ``Series.sum()`` / ``mean()`` / ``min()`` /
              ``max()``; an all-NaN run gives ``0.0`` (``nansum``) or NaN.
            - ``any`` / ``all``: logical reductions.

            Numeric aggregations work in ``float64``. ``sum`` / ``mean`` (and
            their ``nan`` variants) are bit-identical to numpy's pairwise
            ``ndarray.sum()`` -- and so to pandas -- on each run's slice.

    Returns:
        One array (one entry per run) for a single name, else a dict of
        name -> array.
    """
    if isinstance(funcs, str):
        return _one(values, runs, funcs)
    return {f: _one(values, runs, f) for f in funcs}


def run_positions(runs: Runs) -> _Ints:
    """Positions of all elements covered by ``runs``, in order.

    Aligned with :func:`run_labels`: element ``i`` of both belongs together.
    """
    offsets = np.cumsum(runs.lengths) - runs.lengths
    total = int(runs.lengths.sum())
    shift = np.repeat(runs.starts - offsets, runs.lengths)
    return np.asarray(np.arange(total, dtype=np.int64) + shift, dtype=np.int64)


def run_argmax(values: Any, runs: Runs) -> _Ints:
    """Position of the first maximum of ``values`` in each run.

    Like ``Series.idxmax()`` per run (NaN skipped), but returns positions.
    A run whose values are all NaN gives its first position.
    """
    if runs.n_runs == 0:
        return np.zeros(0, dtype=np.int64)
    f = _float(values)
    f = np.where(np.isnan(f), -np.inf, f)
    peak = np.asarray(_reduceat(np.maximum, f, runs), dtype=np.float64)
    labels = run_labels(runs)
    covered = run_positions(runs)
    hit = f[covered] == peak[labels]
    first = np.full(runs.n_runs, -1, dtype=np.int64)
    # Hits come in increasing position: assigning in reverse keeps the first.
    first[labels[hit][::-1]] = covered[hit][::-1]
    return np.where(first < 0, runs.starts, first)


def total_seconds(deltas: Any) -> npt.NDArray[np.float64]:
    """Seconds per timedelta, bit-identical to scalar ``Timedelta.total_seconds()``.

    The scalar method computes ``days * 86400 + seconds + microseconds / 1e6``
    (dropping sub-microsecond precision); the vectorized
    ``TimedeltaIndex.total_seconds()`` divides nanoseconds by ``1e9``, which
    can differ in the last bit. Use this when replacing a per-row
    ``(end - start).total_seconds()`` loop. ``NaT`` gives NaN.
    """
    td = pd.TimedeltaIndex(deltas)
    if td.unit == "ns":
        us = td.asi8 // 1000
    else:
        us = td.asi8 * {"us": 1, "ms": 1000, "s": 1_000_000}[td.unit]
    out = (us // 1_000_000).astype(np.float64) + (us % 1_000_000) / 1e6
    out[np.asarray(td.isna())] = np.nan
    return np.asarray(out, dtype=np.float64)
