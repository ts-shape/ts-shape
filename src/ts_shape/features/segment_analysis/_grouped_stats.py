"""Vectorized per-group version of ``NumericStatistics.summary_as_dict``.

``SegmentProcessor.compute_metric_profiles`` and
``TimeWindowedFeatureTable.compute_long`` compute the same 19 statistics for
every (uuid, segment[, window]) group. Calling ``summary_as_dict`` once per
group costs ~2 ms per group (thousands of groups on a day of data); this
module computes all groups at once with the same definitions:

- ``std``, ``var``, ``sem``: sample statistics (``ddof=1``).
- ``skewness``, ``kurtosis``: pandas' bias-corrected estimators, 0 for a
  constant group, NaN below 3 / 4 values.
- ``q1``, ``q3``, ``median``, ``percentile_*``: linear interpolation.
- ``iqr``: ``q3 - q1``; ``range``: ``max - min``.
- ``mad``: mean absolute deviation from the mean.
- ``coeff_var``: ``std / mean`` (NaN when the mean is 0).
- ``mode``: the most frequent value, the smallest one on a tie.

Missing values are ignored by every statistic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

STAT_ORDER = [
    "min",
    "max",
    "mean",
    "median",
    "std",
    "var",
    "sum",
    "kurtosis",
    "skewness",
    "q1",
    "q3",
    "iqr",
    "range",
    "mad",
    "coeff_var",
    "sem",
    "mode",
    "percentile_90",
    "percentile_10",
]


def _zero_out_fperr(values: np.ndarray, tolerance: np.ndarray) -> np.ndarray:
    out: np.ndarray = np.where(np.abs(values) < tolerance, 0.0, values)
    return out


def grouped_summary(
    codes: np.ndarray,
    values: np.ndarray,
    n_groups: int,
) -> pd.DataFrame:
    """Statistics of ``values`` per group code.

    Args:
        codes: Group code (0..n_groups-1) per value; ``-1`` rows are ignored.
        values: float64 values aligned with ``codes``; NaN is ignored.
        n_groups: Number of groups.

    Returns:
        DataFrame indexed 0..n_groups-1 with ``sample_count`` (non-NaN values)
        and one column per statistic in :data:`STAT_ORDER`.
    """
    keep = (codes >= 0) & ~np.isnan(values)
    c = codes[keep]
    v = values[keep]
    order = np.lexsort((v, c))
    c = c[order]
    v = v[order]

    count = np.bincount(c, minlength=n_groups).astype("float64")
    with np.errstate(invalid="ignore", divide="ignore"):
        total = np.bincount(c, weights=v, minlength=n_groups)
        mean = total / count
        dev = v - mean[c]
        m2 = np.bincount(c, weights=dev**2, minlength=n_groups)
        m3 = np.bincount(c, weights=dev**3, minlength=n_groups)
        m4 = np.bincount(c, weights=dev**4, minlength=n_groups)
        mad = np.bincount(c, weights=np.abs(dev), minlength=n_groups) / count

        var = m2 / (count - 1)
        var[count < 2] = np.nan
        std = np.sqrt(var)
        sem = std / np.sqrt(count)

        absmax = np.zeros(n_groups)
        if len(v):
            np.maximum.at(absmax, c, np.abs(v))
        eps = np.finfo(np.float64).eps
        m2z = _zero_out_fperr(m2, ((eps * absmax) ** 2) * count)
        m3z = _zero_out_fperr(m3, ((eps * absmax) ** 3) * count)
        m4z = _zero_out_fperr(m4, ((eps * absmax) ** 4) * count)

        skew = (count * (count - 1) ** 0.5 / (count - 2)) * (m3z / m2z**1.5)
        skew = np.where(m2z == 0, 0.0, skew)
        skew[count < 3] = np.nan

        adj = 3 * (count - 1) ** 2 / ((count - 2) * (count - 3))
        numer = count * (count + 1) * (count - 1) * m4z
        denom = (count - 2) * (count - 3) * m2z**2
        kurt = numer / denom - adj
        kurt = np.where(denom == 0, 0.0, kurt)
        kurt[count < 4] = np.nan

    # Sorted by (code, value): group g occupies [first[g], first[g] + n_g).
    n_int = count.astype(np.int64)
    first = np.concatenate(([0], np.cumsum(n_int)[:-1]))
    present = n_int > 0

    def _quantile(q: float) -> np.ndarray:
        out = np.full(n_groups, np.nan)
        pos = q * (n_int[present] - 1)
        lo = np.floor(pos).astype(np.int64)
        hi = np.minimum(lo + 1, n_int[present] - 1)
        frac = pos - lo
        base = first[present]
        v_lo = v[base + lo]
        v_hi = v[base + hi]
        out[present] = v_lo + (v_hi - v_lo) * frac
        return out

    vmin = np.full(n_groups, np.nan)
    vmax = np.full(n_groups, np.nan)
    vmin[present] = v[first[present]]
    vmax[present] = v[first[present] + n_int[present] - 1]

    q1 = _quantile(0.25)
    q3 = _quantile(0.75)

    # Mode: longest run of equal values in each (sorted) group; the first
    # (smallest) value wins a tie.
    mode = np.full(n_groups, np.nan)
    if len(v):
        new_run = np.ones(len(v), dtype=bool)
        new_run[1:] = (c[1:] != c[:-1]) | (v[1:] != v[:-1])
        run_start = np.flatnonzero(new_run)
        run_len = np.diff(np.append(run_start, len(v)))
        run_code = c[run_start]
        run_val = v[run_start]
        pick = np.lexsort((run_val, -run_len, run_code))
        run_code_sorted = run_code[pick]
        first_of_group = np.ones(len(pick), dtype=bool)
        first_of_group[1:] = run_code_sorted[1:] != run_code_sorted[:-1]
        mode[run_code_sorted[first_of_group]] = run_val[pick][first_of_group]

    with np.errstate(invalid="ignore", divide="ignore"):
        coeff_var = np.where(mean == 0, np.nan, std / mean)
    total = np.where(present, total, 0.0)

    stats = {
        "min": vmin,
        "max": vmax,
        "mean": mean,
        "median": _quantile(0.5),
        "std": std,
        "var": var,
        "sum": total,
        "kurtosis": kurt,
        "skewness": skew,
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "range": vmax - vmin,
        "mad": mad,
        "coeff_var": coeff_var,
        "sem": sem,
        "mode": mode,
        "percentile_90": _quantile(0.9),
        "percentile_10": _quantile(0.1),
    }
    out = pd.DataFrame(stats, columns=STAT_ORDER)
    out.insert(0, "sample_count", n_int)
    return out


def summarize_groups(
    dataframe: pd.DataFrame,
    keys: list[object],
    value_column: str,
    min_count: int = 2,
) -> pd.DataFrame:
    """Statistics of ``value_column`` per group of ``keys``.

    Args:
        dataframe: Input frame.
        keys: Anything ``DataFrame.groupby`` accepts as a list (column names,
            a :class:`pandas.Grouper`).
        value_column: Column holding the values; text is read as NaN.
        min_count: Groups with fewer non-NaN values are dropped.

    Returns:
        One row per group, ordered by key: the group keys as columns (named
            after the groupby levels), ``sample_count`` and the statistics of
            :data:`STAT_ORDER`.
    """
    gb = dataframe.groupby(keys, sort=True, observed=True, dropna=True)
    codes = gb.ngroup().to_numpy(dtype=np.int64, na_value=-1)
    index = gb.size().index
    values = pd.to_numeric(dataframe[value_column], errors="coerce")
    vals = np.asarray(values.to_numpy(dtype="float64", na_value=np.nan))
    stats = grouped_summary(codes, vals, len(index))
    keys_frame = index.to_frame(index=False)
    result = pd.concat([keys_frame, stats], axis=1)
    return result[result["sample_count"] >= min_count].reset_index(drop=True)
