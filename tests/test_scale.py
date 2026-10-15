"""Scale checks: doubling the rows must not much more than double the cost.

Marked ``scale`` and left out of the default run (``-m "not scale"`` in the
pytest addopts); run them with ``pytest -m scale``. Each check times the
same call at 200k and 400k rows (best of five after a warm-up, to ride out
noise) and fails when the larger run takes 3x as long or more -- a sign of
a row loop or a sort that grows faster than the data. ``Base`` construction also has a
memory budget, measured with ``tracemalloc`` (NumPy allocations are traced).

The synthetic frame comes from ``benchmarks/run.py``.
"""

from __future__ import annotations

import gc
import importlib.util
import time
import tracemalloc
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import ts_shape
from ts_shape.utils.base import Base

pytestmark = pytest.mark.scale

_RUN = Path(__file__).resolve().parents[1] / "benchmarks" / "run.py"
_spec = importlib.util.spec_from_file_location("ts_shape_benchmarks_run", _RUN)
assert _spec is not None and _spec.loader is not None
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)

SMALL, LARGE = 200_000, 400_000
MAX_RATIO = 3.0
#: Peak traced bytes per input row while constructing ``Base`` (about 55
#: sorted and 90 shuffled; it was ~190 when ``Base`` copied the frame
#: several times).
BASE_BYTES_PER_ROW = 120

_frames: dict[tuple[int, bool], pd.DataFrame] = {}


def _frame(n_rows: int, shuffled: bool = False) -> pd.DataFrame:
    key = (n_rows, shuffled)
    if key not in _frames:
        df = bench.make_frame(n_rows)
        if shuffled:
            df = df.sample(frac=1.0, random_state=0)
        _frames[key] = df
    return _frames[key]


CASES: dict[str, Callable[[pd.DataFrame], Any]] = {
    "base": lambda df: Base(df),
    "machine_state.detect_run_idle": lambda df: ts_shape.MachineStateEvents(
        df, bench.STATE
    ).detect_run_idle(),
    "signal_quality.detect_missing_data": lambda df: ts_shape.SignalQualityEvents(
        df, bench.TEMP
    ).detect_missing_data(),
    "stability_index.stability_score": lambda df: (
        ts_shape.ProcessStabilityIndexEvents(df, bench.PRESSURE).stability_score()
    ),
}


def _best_seconds(fn: Callable[[pd.DataFrame], Any], df: pd.DataFrame) -> float:
    # One untimed warm-up call, then the best of five with the garbage
    # collector off: inside a long test session, a collection or first-touch
    # page faults can otherwise land in one timing and fake a slow-down.
    best = float("inf")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fn(df)
        gc.collect()
        gc.disable()
        try:
            for _ in range(5):
                start = time.perf_counter()
                fn(df)
                best = min(best, time.perf_counter() - start)
        finally:
            gc.enable()
    return best


@pytest.mark.parametrize("name", sorted(CASES))
def test_doubling_rows_at_most_triples_time(name: str) -> None:
    fn = CASES[name]
    small = _best_seconds(fn, _frame(SMALL))
    large = _best_seconds(fn, _frame(LARGE))
    # Below a few milliseconds the ratio is noise.
    assert large < max(MAX_RATIO * small, 0.05), (name, small, large)


@pytest.mark.parametrize("shuffled", [False, True], ids=["sorted", "shuffled"])
def test_base_construction_memory_per_row(shuffled: bool) -> None:
    df = _frame(SMALL, shuffled)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tracemalloc.start()
        try:
            Base(df)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
    assert peak / len(df) < BASE_BYTES_PER_ROW, peak / len(df)
