"""Regression tests for the round-7 maturity fixes in the maintenance,
engineering, supply-chain, quality, correlation, development and features
packs: vectorized rolling statistics (checked against the naive per-row
reference), degenerate inputs and duplicate timestamps.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from ts_shape.errors import DataQualityWarning
from ts_shape.events.engineering.setpoint_events import SetpointChangeEvents
from ts_shape.events.engineering.startup_events import StartupDetectionEvents
from ts_shape.events.maintenance._rolling import (
    count_window_bounds,
    time_window_bounds,
    window_moments,
)
from ts_shape.events.maintenance.degradation_detection import (
    DegradationDetectionEvents,
)
from ts_shape.events.maintenance.failure_prediction import FailurePredictionEvents
from ts_shape.events.quality.anomaly_classification import (
    AnomalyClassificationEvents,
)
from ts_shape.events.quality.change_point import _cost_function, _setup, pelt
from ts_shape.features.stats.feature_table import DescriptiveFeatures
from ts_shape.features.stats.string_stats import StringStatistics


def _irregular(n: int = 400, seed: int = 0, dup_every: int = 17) -> pd.DataFrame:
    """One signal with jittered, partly duplicated timestamps and some NaNs."""
    rng = np.random.default_rng(seed)
    t = pd.Timestamp("2024-01-01") + pd.to_timedelta(
        np.cumsum(rng.integers(5, 120, n)), unit="s"
    )
    t = pd.Series(t)
    t.iloc[::dup_every] = t.shift(1).iloc[::dup_every].fillna(t.iloc[0])
    v = 50 + np.cumsum(rng.normal(0, 0.3, n))
    df = pd.DataFrame({"systime": t, "uuid": "s", "value_double": v})
    return df.sort_values("systime", kind="stable").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Vectorized window statistics == naive per-row reference
# ---------------------------------------------------------------------------


def test_window_moments_match_polyfit_and_var() -> None:
    df = _irregular()
    t = df["systime"]
    x = (t - t.iloc[0]).dt.total_seconds().to_numpy()
    y = df["value_double"].to_numpy().copy()
    y[[5, 90]] = np.nan
    w = pd.Timedelta("20min")
    lo, hi = time_window_bounds(t, w)
    mom = window_moments(x, y, lo, hi)
    for i in range(len(df)):
        mask = ((t > t.iloc[i] - w) & (t <= t.iloc[i])).to_numpy()
        assert hi[i] - lo[i] == mask.sum()
        xs, ys = x[mask], y[mask]
        ok = ~np.isnan(ys)
        if ok.sum() >= 2:
            assert mom["var"][i] == pytest.approx(pd.Series(ys).var(), rel=1e-9)
        if ok.sum() >= 2 and np.ptp(xs[ok]) > 0:
            ref = np.polyfit(xs[ok], ys[ok], 1)[0]
            assert mom["slope"][i] == pytest.approx(ref, rel=1e-7, abs=1e-12)


def test_count_window_bounds() -> None:
    lo, hi = count_window_bounds(5, 3)
    assert lo.tolist() == [0, 0, 0, 1, 2]
    assert hi.tolist() == [1, 2, 3, 4, 5]


def test_health_score_matches_naive_reference() -> None:
    df = _irregular(seed=3)
    out = DegradationDetectionEvents(df, "s").health_score(
        window="15min", baseline_window="1h"
    )
    t = df["systime"]
    x = (t - t.iloc[0]).dt.total_seconds().to_numpy()
    y = df["value_double"].to_numpy()
    base = y[(t <= t.iloc[0] + pd.Timedelta("1h")).to_numpy()]
    base_var = base.var(ddof=1)
    for i in range(0, len(df), 7):
        m = ((t > t.iloc[i] - pd.Timedelta("15min")) & (t <= t.iloc[i])).to_numpy()
        if m.sum() < 2:
            continue
        assert out["variance_ratio"].iloc[i] == pytest.approx(
            y[m].var(ddof=1) / base_var, rel=1e-9
        )
        if np.ptp(x[m]) > 0:
            assert out["trend_slope"].iloc[i] == pytest.approx(
                np.polyfit(x[m], y[m], 1)[0], rel=1e-7, abs=1e-12
            )


def test_trend_slope_of_same_timestamp_window_is_undefined() -> None:
    # Every sample of the first window shares one timestamp: np.polyfit gave
    # an arbitrary slope (or raised); now there is no slope.
    t = pd.to_datetime(["2024-01-01 00:00"] * 3 + ["2024-01-01 02:00"])
    df = pd.DataFrame({"systime": t, "uuid": "s", "value_double": [1.0, 2, 3, 4]})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DataQualityWarning)
        hs = DegradationDetectionEvents(df, "s").health_score(window="10min")
    assert hs["trend_slope"].iloc[:3].tolist() == [0.0, 0.0, 0.0]


def test_rul_and_time_to_threshold_match_polyfit() -> None:
    df = _irregular(seed=5)
    fp = FailurePredictionEvents(df, "s")
    out = fp.time_to_threshold(threshold=1000.0)
    rate = out["rate_of_change"].to_numpy()
    # The class's own sample order (ties between duplicates included).
    t = out["systime"]
    x = (t - t.iloc[0]).dt.total_seconds().to_numpy()
    y = out["current_value"].to_numpy()
    for i in range(1, len(df)):
        xs, ys = x[max(0, i - 9) : i + 1], y[max(0, i - 9) : i + 1]
        if np.ptp(xs) > 0:
            assert rate[i] == pytest.approx(
                np.polyfit(xs, ys, 1)[0], rel=1e-7, abs=1e-12
            )


def test_time_to_threshold_nan_value_has_no_estimate() -> None:
    t = pd.date_range("2024-01-01", periods=6, freq="1min")
    v = [1.0, 2.0, 3.0, np.nan, 5.0, 6.0]
    df = pd.DataFrame({"systime": t, "uuid": "s", "value_double": v})
    out = FailurePredictionEvents(df, "s").time_to_threshold(threshold=10.0)
    # A NaN reading was reported as 0 s ("at the threshold now").
    assert out["estimated_time_seconds"].iloc[3] is None or pd.isna(
        out["estimated_time_seconds"].iloc[3]
    )
    assert out["estimated_time_seconds"].iloc[5] == pytest.approx(240.0)


# ---------------------------------------------------------------------------
# PELT: the buffered loop returns exactly what the plain loop returns
# ---------------------------------------------------------------------------


def _pelt_reference(x, cost, penalty, min_size):
    x, min_size, penalty = _setup(x, cost, penalty, min_size)
    n = len(x)
    if n < 2 * min_size:
        return []
    seg_cost = _cost_function(x, cost)
    f = np.full(n + 1, np.inf)
    f[0] = -penalty
    last = np.zeros(n + 1, dtype=int)
    cand = np.array([0])
    pruned_at = np.array([n + 1])
    for t in range(min_size, n + 1):
        alive = pruned_at > t - min_size
        cand, pruned_at = cand[alive], pruned_at[alive]
        evaluable = t - cand >= min_size
        if not evaluable.any():
            continue
        eligible = cand[evaluable]
        costs = f[eligible] + seg_cost(eligible, t)
        best = int(np.argmin(costs))
        f[t] = costs[best] + penalty
        last[t] = eligible[best]
        newly = np.zeros(len(cand), dtype=bool)
        newly[evaluable] = costs > f[t]
        pruned_at[newly & (pruned_at > n)] = t
        cand = np.append(cand, t)
        pruned_at = np.append(pruned_at, n + 1)
    points, t = [], n
    while t > 0:
        s = int(last[t])
        if s > 0:
            points.append(s)
        t = s
    return sorted(points)


@pytest.mark.parametrize("cost", ["normal_mean", "normal_meanvar"])
@pytest.mark.parametrize("min_size", [2, 5])
def test_pelt_matches_reference(cost: str, min_size: int) -> None:
    rng = np.random.default_rng(11)
    x = np.repeat(rng.normal(0, 4, 12), 60) + rng.normal(0, 1, 720)
    x[300:420] *= 3
    assert pelt(x, cost=cost, min_size=min_size) == _pelt_reference(
        x, cost, None, min_size
    )


# ---------------------------------------------------------------------------
# Degenerate inputs
# ---------------------------------------------------------------------------


def _setpoint_frame(sp_values, sp_times=None):
    t = pd.date_range("2024-01-01", periods=20, freq="1min")
    sp_times = t[: len(sp_values)] if sp_times is None else sp_times
    act = pd.DataFrame(
        {"systime": t, "uuid": "act", "value_double": np.linspace(0, 10, 20)}
    )
    sp = pd.DataFrame({"systime": sp_times, "uuid": "sp", "value_double": sp_values})
    return pd.concat([sp, act], ignore_index=True)


@pytest.mark.parametrize("sp_values", [[5.0], [5.0] * 20])
def test_control_quality_metrics_without_changes(sp_values) -> None:
    # One setpoint row / a constant setpoint: no change -> an empty result
    # with the full column set (raised KeyError: 'start').
    full = SetpointChangeEvents(
        _setpoint_frame(
            [1.0, 5.0, 1.0, 5.0], pd.date_range("2024-01-01", periods=4, freq="5min")
        ),
        "sp",
    ).control_quality_metrics("act")
    out = SetpointChangeEvents(
        _setpoint_frame(sp_values), "sp"
    ).control_quality_metrics("act")
    assert out.empty
    assert list(out.columns) == list(full.columns)


def test_setpoint_ramps_duplicate_timestamps_have_no_infinite_rate() -> None:
    t = pd.to_datetime(
        ["2024-01-01 00:00", "2024-01-01 00:01", "2024-01-01 00:01", "2024-01-01 00:10"]
    )
    df = pd.DataFrame(
        {"systime": t, "uuid": "sp", "value_double": [0.0, 0.0, 5.0, 5.0]}
    )
    ev = SetpointChangeEvents(df, "sp")
    with pytest.warns(DataQualityWarning, match="share a timestamp"):
        out = ev.detect_setpoint_ramps(min_rate=0.001)
    # The jump at the duplicated instant was an infinite-rate "ramp".
    assert out.empty


def test_startup_by_slope_duplicate_timestamps_have_no_infinite_slope() -> None:
    t = pd.to_datetime(
        ["2024-01-01 00:00", "2024-01-01 00:01", "2024-01-01 00:01", "2024-01-01 00:10"]
    )
    df = pd.DataFrame({"systime": t, "uuid": "m", "value_double": [0.0, 0.0, 5.0, 5.0]})
    ev = StartupDetectionEvents(df, "m")
    with pytest.warns(DataQualityWarning, match="share a timestamp"):
        out = ev.detect_startup_by_slope(min_slope=0.001)
    assert out.empty or np.isfinite(out["avg_slope"].astype(float)).all()


def test_string_stats_all_null_column() -> None:
    t = pd.date_range("2024-01-01", periods=3, freq="1min")
    df = pd.DataFrame(
        {
            "systime": t,
            "uuid": "a",
            "value_string": pd.Series([None, None, None], dtype=object),
            "is_delta": True,
        }
    )
    assert StringStatistics.most_frequent(df) is None
    assert StringStatistics.count_most_frequent(df) == 0
    assert StringStatistics.longest_string(df) is None
    assert StringStatistics.shortest_string(df) is None
    stats = DescriptiveFeatures(df).compute()["a"]["value_string"]["string_stats"]
    assert stats["most_frequent"] is None
    assert stats["count_most_frequent"] == 0
    assert stats["least_common"] is None
    wide = DescriptiveFeatures(df).compute(output_format="dataframe")
    assert wide["a::value_string::string_stats::count_most_frequent"].iloc[0] == 0


def test_detect_oscillation_rejects_non_positive_window() -> None:
    df = _irregular(n=20)
    with pytest.raises(ValueError, match="window must be positive"):
        AnomalyClassificationEvents(df, "s").detect_oscillation(window="0s")


def test_count_most_frequent_with_boolean_values():
    import pandas as pd

    from ts_shape.features.stats.string_stats import StringStatistics

    df = pd.DataFrame({"value_string": [True, True, False, None]}, dtype=object)
    assert StringStatistics.most_frequent(df) is True
    assert StringStatistics.count_most_frequent(df) == 2


def test_feature_table_routes_object_columns_by_their_values():
    import numpy as np
    import pandas as pd

    from ts_shape.features.stats.feature_table import DescriptiveFeatures

    t = pd.date_range("2024-01-01", periods=4, freq="1min")
    df = pd.DataFrame(
        {
            "systime": list(t) * 2,
            "uuid": ["flag"] * 4 + ["temp"] * 4,
            "value_bool": [True, False, True, True] + [np.nan] * 4,
            "value_double": [np.nan] * 4 + [1.0, 2.0, 3.0, 4.0],
            "value_string": [None] * 8,
            "is_delta": False,
        }
    )
    out = DescriptiveFeatures(df).compute(output_format="dict")
    assert "boolean_stats" in out["flag"]["value_bool"]
    assert "string_stats" not in out["flag"]["value_bool"]
