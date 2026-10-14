"""Scale smoke test: formerly quadratic / slow methods of the maintenance,
engineering, supply-chain, quality, correlation and development packs run on a
~50k-row frame well within a generous time budget.

Before the vectorization several of these took minutes (per-row full-series
masks plus a ``np.polyfit`` per row); the bound only catches a return to that
behaviour, not small regressions.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
import pytest

import ts_shape
from tests.invariants import harness as H

ROWS_PER_SIGNAL = 10_000  # five dense signals + a setpoint -> ~50k rows
TOTAL_BUDGET_S = 90.0

METHODS = [
    ("DegradationDetectionEvents", "detect_trend_degradation"),
    ("DegradationDetectionEvents", "detect_variance_increase"),
    ("DegradationDetectionEvents", "detect_level_shift"),
    ("DegradationDetectionEvents", "health_score"),
    ("FailurePredictionEvents", "remaining_useful_life"),
    ("FailurePredictionEvents", "time_to_threshold"),
    ("FailurePredictionEvents", "detect_exceedance_pattern"),
    ("VibrationAnalysisEvents", "bearing_health_indicators"),
    ("VibrationAnalysisEvents", "detect_amplitude_growth"),
    ("WarmUpCoolDownEvents", "time_to_target"),
    ("InventoryMonitoringEvents", "stockout_prediction"),
    ("ChangePointEvents", "detect_change_points"),
    ("ChangePointEvents", "segment_statistics"),
    ("SetpointChangeEvents", "control_quality_metrics"),
    ("SetpointChangeEvents", "detect_setpoint_steps"),
    ("StartupDetectionEvents", "detect_startup_by_threshold"),
    ("AnomalyCorrelationEvents", "coincident_anomalies"),
    ("AnomalyCorrelationEvents", "cascade_detection"),
    ("AnomalyClassificationEvents", "detect_flatline"),
    ("AnomalyClassificationEvents", "detect_oscillation"),
    ("AnomalyClassificationEvents", "classify_anomalies"),
    ("DesignOfExperimentsEvents", "detect_runs"),
    ("DesignOfExperimentsEvents", "effects"),
    ("MultiSensorValidationEvents", "detect_disagreement"),
    ("ThresholdMonitoringEvents", "multi_level_threshold"),
    ("ToleranceDeviationEvents", "detect_deviations"),
]


def _large_plant(n: int = ROWS_PER_SIGNAL) -> pd.DataFrame:
    """The harness plant frame's six signals, ``n`` one-minute samples each."""
    rng = np.random.default_rng(7)
    t = pd.date_range("2024-03-01", periods=n, freq="1min")
    counter = np.cumsum(rng.integers(0, 3, n))
    state = (np.arange(n) // 30) % 3 != 0
    sp_t = t[::60]
    frames = [
        pd.DataFrame(
            {"systime": t, "uuid": H.CONT, "value_double": 50 + rng.normal(0, 2, n)}
        ),
        pd.DataFrame(
            {"systime": t, "uuid": H.CONT2, "value_double": 20 + rng.normal(0, 1, n)}
        ),
        pd.DataFrame({"systime": t, "uuid": H.STATE, "value_bool": state}),
        pd.DataFrame({"systime": t, "uuid": H.COUNTER, "value_integer": counter}),
        pd.DataFrame(
            {
                "systime": sp_t,
                "uuid": H.SETPOINT,
                "value_double": np.tile([50.0, 55.0], len(sp_t))[: len(sp_t)],
            }
        ),
        pd.DataFrame(
            {
                "systime": t,
                "uuid": H.TEXT,
                "value_string": np.where(state, "Running", "Stopped"),
            }
        ),
    ]
    df = pd.concat(frames, ignore_index=True)
    for col in ("value_double", "value_integer", "value_bool", "value_string"):
        if col not in df.columns:
            df[col] = np.nan
    df["is_delta"] = True
    return df.sort_values(["systime", "uuid"]).reset_index(drop=True)


@pytest.fixture(scope="module")
def large_plant() -> pd.DataFrame:
    return _large_plant()


def test_formerly_slow_methods_scale(large_plant: pd.DataFrame) -> None:
    assert len(large_plant) > 45_000
    timings: dict[str, float] = {}
    total = 0.0
    for cname, method in METHODS:
        call = H.build_call(getattr(ts_shape, cname), method, large_plant)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t0 = time.perf_counter()
            out = call()
            dt = time.perf_counter() - t0
        assert out is not None
        timings[f"{cname}.{method}"] = dt
        total += dt
    slowest = sorted(timings.items(), key=lambda kv: -kv[1])[:5]
    assert total < TOTAL_BUDGET_S, f"total {total:.1f}s; slowest: {slowest}"
