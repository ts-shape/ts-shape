"""CUSUM counts a repeated sample once and matches the reference recurrence."""

import warnings

import numpy as np
import pandas as pd
import pytest

from ts_shape.errors import DataQualityWarning
from ts_shape.events.quality.statistical_process_control import (
    StatisticalProcessControlEvents,
    _cusum,
)


def _frame(values, tol):
    t = pd.date_range("2024-01-01", periods=len(values), freq="1min")
    return pd.concat(
        [
            pd.DataFrame({"systime": t, "uuid": "act", "value_double": values}),
            pd.DataFrame(
                {"systime": t[: len(tol)], "uuid": "tol", "value_double": tol}
            ),
        ],
        ignore_index=True,
    )


def _shifts(df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DataQualityWarning)
        return StatisticalProcessControlEvents(
            df,
            value_column="value_double",
            tolerance_uuid="tol",
            actual_uuid="act",
            event_uuid="spc",
        ).detect_cusum_shifts()


def test_repeated_samples_do_not_change_the_result():
    rng = np.random.default_rng(3)
    values = np.r_[rng.normal(10, 1, 60), rng.normal(11.5, 1, 40)]
    tol = rng.normal(10, 1, 50)
    clean = _frame(values, tol)
    acts = clean[clean["uuid"] == "act"]
    dup = pd.concat([clean, acts.iloc[::7]], ignore_index=True)
    expected = _shifts(clean).reset_index(drop=True)
    with pytest.warns(DataQualityWarning, match="repeated sample"):
        StatisticalProcessControlEvents(
            dup,
            value_column="value_double",
            tolerance_uuid="tol",
            actual_uuid="act",
            event_uuid="spc",
        ).detect_cusum_shifts()
    pd.testing.assert_frame_equal(_shifts(dup).reset_index(drop=True), expected)


def test_cusum_matches_reference_loop():
    rng = np.random.default_rng(7)
    x = rng.normal(0, 1, 500)
    x[100] = np.nan
    hi, lo = _cusum(x, 0.2, 0.5)
    rh = rl = 0.0
    for i, v in enumerate(x):
        rh = max(0, rh + v - 0.2 - 0.5) if i else max(0, v - 0.2 - 0.5)
        rl = max(0, rl + 0.2 - v - 0.5) if i else max(0, 0.2 - v - 0.5)
        assert hi[i] == rh and lo[i] == rl
