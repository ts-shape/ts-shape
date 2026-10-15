"""STOMP matrix profile and vectorized / pruned DTW == the former loops.

The references below are frozen copies of the implementations they replaced:
one MASS (FFT) distance profile per row for the matrix profile, and the
scalar per-window dynamic program for DTW.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ts_shape.features.pattern_recognition import PatternRecognition


def _matrix_profile_reference(
    series: np.ndarray, window_size: int, exclusion_zone: int
) -> tuple[np.ndarray, np.ndarray]:
    n = len(series)
    profile_len = max(n - window_size + 1, 0)
    mp = np.full(profile_len, np.inf)
    mpi = np.full(profile_len, -1, dtype=int)
    valid = PatternRecognition._valid_windows(series, window_size)
    for i in range(profile_len):
        if not valid[i]:
            continue
        query = series[i : i + window_size]
        dist_profile = PatternRecognition._mass(series, query)
        dist_profile = np.where(np.isnan(dist_profile), np.inf, dist_profile)
        ez_start = max(0, i - exclusion_zone)
        ez_end = min(profile_len, i + exclusion_zone + 1)
        dist_profile[ez_start:ez_end] = np.inf
        min_idx = int(np.argmin(dist_profile))
        min_val = dist_profile[min_idx]
        if np.isfinite(min_val):
            mp[i] = min_val
            mpi[i] = min_idx
    return mp, mpi


def _dtw_reference(a: np.ndarray, b: np.ndarray) -> float:
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = (a[i - 1] - b[j - 1]) ** 2
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(np.sqrt(D[n, m]))


def _dtw_profile_reference(
    series: np.ndarray, query: np.ndarray, normalize: bool
) -> np.ndarray:
    m = len(query)
    q = PatternRecognition._znorm(query) if normalize else query
    valid = PatternRecognition._valid_windows(series, m)
    out = np.full(len(series) - m + 1, np.nan)
    for i in np.flatnonzero(valid):
        sub = series[i : i + m].copy()
        if normalize:
            sub = PatternRecognition._znorm(sub)
        out[i] = _dtw_reference(q, sub)
    return out


def _signal(seed: int, n: int, kind: str) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if kind == "periodic":
        v = np.sin(np.arange(n) / 6.0) + rng.normal(0, 0.2, n)
    else:
        v = np.cumsum(rng.normal(0, 1, n))
    if kind == "gaps":
        v[rng.random(n) < 0.02] = np.nan
    if kind == "steps":
        v = np.round(v)  # many exactly equal windows / ties
    return v


@pytest.mark.parametrize("kind", ["walk", "periodic", "gaps"])
@pytest.mark.parametrize("window", [5, 24])
def test_stomp_matches_per_row_mass(kind: str, window: int) -> None:
    # 300 rows span several exact-FFT refreshes of the running dot products.
    series = _signal(11, 300, kind)
    ref_mp, ref_mpi = _matrix_profile_reference(series, window, window // 2)
    mp, mpi = PatternRecognition._compute_matrix_profile(series, window, window // 2)
    np.testing.assert_array_equal(mpi, ref_mpi)
    np.testing.assert_array_equal(np.isfinite(mp), np.isfinite(ref_mp))
    ok = np.isfinite(ref_mp)
    np.testing.assert_allclose(mp[ok], ref_mp[ok], rtol=1e-9, atol=0)


def test_stomp_short_and_missing_series() -> None:
    for series in (np.arange(3.0), np.full(40, np.nan)):
        mp, mpi = PatternRecognition._compute_matrix_profile(series, 5, 2)
        ref_mp, ref_mpi = _matrix_profile_reference(series, 5, 2)
        np.testing.assert_array_equal(mp, ref_mp)
        np.testing.assert_array_equal(mpi, ref_mpi)


@pytest.mark.parametrize("kind", ["walk", "periodic", "gaps", "steps"])
@pytest.mark.parametrize("normalize", [True, False])
def test_dtw_profile_and_search_are_identical(kind: str, normalize: bool) -> None:
    series = _signal(5, 400, kind)
    rng = np.random.default_rng(1)
    query = np.nan_to_num(series[30:42]) + rng.normal(0, 0.1, 12)
    ref = _dtw_profile_reference(series, query, normalize)
    df = pd.DataFrame({"value_double": series})
    got = PatternRecognition.compute_distance_profile(
        df, query, metric="dtw", normalize=normalize
    )
    np.testing.assert_array_equal(got, ref)

    candidates = np.flatnonzero(~np.isnan(ref))
    ranked = candidates[np.argsort(ref[candidates], kind="stable")]
    for top_k in (1, 3, 10, len(candidates) + 5, 0):
        out = PatternRecognition.similarity_search(
            df, query, top_k=top_k, normalize=normalize
        )
        assert out["start_index"].tolist() == ranked[:top_k].tolist()
        assert out["dtw_distance"].tolist() == ref[ranked[:top_k]].tolist()
