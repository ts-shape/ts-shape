"""Lightweight perf guard for the adapter hot path.

The adapter previously used row-wise ``.apply`` for timestamp conversion,
activity templating, and event-id generation, which made ``adapt()``
slow on legacy DataFrames with thousands of rows. This test asserts a
generous-but-meaningful upper bound so a future refactor that
re-introduces a row-wise ``.apply`` will fail CI rather than silently
ship a regression.

The threshold (1.5 s for a 10 000-row outlier-shaped frame on most
machines) is ~10x slower than the current vectorised implementation,
leaving headroom for slow CI runners while still catching the worst
pessimisations.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ts_shape.eventlog import to_event_log


def test_adapt_under_perf_budget_for_10k_rows():
    rng = np.random.default_rng(0)
    n = 10_000

    # Synthesize a legacy outlier-shaped DataFrame directly (skip the
    # detector — we're benchmarking the adapter, not the detector).
    legacy = pd.DataFrame({
        "systime": pd.date_range("2026-05-07", periods=n, freq="1s", tz="UTC"),
        "value_double": rng.normal(loc=10.0, scale=1.0, size=n),
        "uuid": ["sensor"] * n,
        "is_delta": [False] * n,
        "source_uuid": ["asset-A"] * n,
        "severity_score": rng.uniform(0.5, 5.5, size=n),
    })

    t0 = time.perf_counter()
    log = to_event_log(legacy, detector="OutlierDetectionEvents.detect_outliers_zscore")
    elapsed = time.perf_counter() - t0

    assert len(log.events) == n
    # Generous threshold: catches re-introduction of row-wise apply.
    assert elapsed < 1.5, (
        f"adapt() took {elapsed:.2f}s on a {n}-row legacy frame; "
        "threshold is 1.5s. Likely a row-wise apply regression."
    )


def test_adapt_under_perf_budget_for_templated_interval():
    """Templated activity rendering on an interval-shape detector — exercises
    the vectorised template path explicitly."""
    n = 10_000
    states = ["run", "idle"] * (n // 2)
    legacy = pd.DataFrame({
        "start": pd.date_range("2026-05-07", periods=n, freq="30s", tz="UTC"),
        "end":   pd.date_range("2026-05-07 00:00:30", periods=n, freq="30s", tz="UTC"),
        "uuid": ["evt"] * n,
        "source_uuid": ["asset-A"] * n,
        "is_delta": [True] * n,
        "state": states,
        "duration_seconds": [30.0] * n,
    })

    t0 = time.perf_counter()
    log = to_event_log(legacy, detector="MachineStateEvents.detect_run_idle")
    elapsed = time.perf_counter() - t0

    assert len(log.events) == n
    activities = set(log.events["ocel:activity"])
    assert {"production.machine_state.run", "production.machine_state.idle"} <= activities
    assert elapsed < 2.0, (
        f"templated-interval adapt() took {elapsed:.2f}s on a {n}-row frame; "
        "threshold is 2.0s. Likely a row-wise apply regression in template rendering."
    )
