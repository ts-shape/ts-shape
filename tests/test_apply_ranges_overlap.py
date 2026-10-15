"""apply_ranges with overlapping ranges == the former per-range scan."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from ts_shape.features.segment_analysis.segment_processor import SegmentProcessor


def _reference_rows(times: pd.Series, ranges: pd.DataFrame, closed: np.ndarray):
    """Frozen copy of the former overlap path: one full scan per range, in
    (start, end) order. Returns the (row, range) pairs in output order."""
    t_i8 = pd.DatetimeIndex(times).as_unit("ns").asi8
    has_time = ~np.asarray(pd.isna(times))
    s = pd.DatetimeIndex(ranges["segment_start"]).as_unit("ns").asi8
    e = pd.DatetimeIndex(ranges["segment_end"]).as_unit("ns").asi8
    order = np.lexsort((e, s))
    rows, owners = [], []
    for k in order:
        hit = has_time & (t_i8 >= s[k]) & (t_i8 < e[k])
        if closed[k]:
            hit |= has_time & (t_i8 == e[k])
        r = np.flatnonzero(hit)
        rows.append(r)
        owners.append(np.full(len(r), k, dtype=np.int64))
    row_pos, range_pos = np.concatenate(rows), np.concatenate(owners)
    out = np.lexsort((row_pos, range_pos))
    return row_pos[out], range_pos[out]


@pytest.mark.parametrize("seed", range(6))
def test_overlapping_ranges_match_per_range_scan(seed):
    rng = np.random.default_rng(seed)
    n = 400
    t = pd.Series(
        pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 600, n), "s")
    )
    t[rng.random(n) < 0.03] = pd.NaT
    df = pd.DataFrame({"systime": t, "uuid": "a", "value_double": np.arange(n)})
    k = 25
    starts = pd.Timestamp("2024-01-01") + pd.to_timedelta(
        rng.integers(-30, 600, k), "s"
    )
    # Mostly forward ranges, some zero-length and some ending before they start.
    ends = starts + pd.to_timedelta(rng.integers(-20, 200, k), "s")
    ends = ends.where(rng.random(k) > 0.15, starts)
    # Ends exactly on sample times, so closed ends matter.
    ends = ends.where(rng.random(k) > 0.3, t.dropna().sample(k, random_state=seed))
    closed = rng.random(k) < 0.5
    ranges = pd.DataFrame(
        {
            "segment_start": starts,
            "segment_end": ends.to_numpy(),
            "segment_value": [f"v{i}" for i in range(k)],
            "segment_index": np.arange(k),
            "end_inclusive": closed,
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = SegmentProcessor.apply_ranges(df, ranges)
    rows, owners = _reference_rows(t, ranges, closed)
    assert out["value_double"].tolist() == rows.tolist()
    assert out["segment_index"].tolist() == owners.tolist()
    assert out["segment_value"].tolist() == [f"v{i}" for i in owners]
