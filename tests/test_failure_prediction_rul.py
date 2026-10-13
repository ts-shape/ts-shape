"""Regression tests: RUL direction handling and NaN samples."""

import numpy as np
import pandas as pd

from ts_shape.events.maintenance.failure_prediction import FailurePredictionEvents


def _sig(values):
    t = pd.date_range("2024-01-01", periods=len(values), freq="min")
    return pd.DataFrame({"systime": t, "uuid": "s", "value_double": values})


def test_improving_signal_projects_no_failure():
    ev = FailurePredictionEvents(_sig([50.0, 49.0, 48.0, 47.0, 46.0]), "s")
    out = ev.remaining_useful_life(degradation_rate=1 / 60, failure_threshold=100.0)
    assert pd.isna(out["rul_seconds"].iloc[-1])


def test_past_threshold_is_zero():
    ev = FailurePredictionEvents(_sig([98.0, 99.0, 100.0, 101.0]), "s")
    out = ev.remaining_useful_life(degradation_rate=1 / 60, failure_threshold=100.0)
    assert out["rul_seconds"].iloc[-1] == 0.0


def test_falling_failure_direction():
    ev = FailurePredictionEvents(_sig([20.0, 19.0, 18.0, 17.0]), "s")
    out = ev.remaining_useful_life(degradation_rate=-1 / 60, failure_threshold=10.0)
    assert np.isclose(out["rul_seconds"].iloc[-1], 7 * 60)


def test_nan_sample_does_not_poison_following_rows():
    vals = [0.0, 1.0, np.nan, 3.0, 4.0, 5.0]
    ev = FailurePredictionEvents(_sig(vals), "s")
    out = ev.remaining_useful_life(degradation_rate=1 / 60, failure_threshold=100.0)
    rul = out["rul_seconds"].iloc[-1]
    assert np.isclose(rul, 95 * 60)
