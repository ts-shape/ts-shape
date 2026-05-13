import pandas as pd  # type: ignore
import numpy as np  # type: ignore

from ts_shape.events.engineering.process_stability_index import ProcessStabilityIndex


def _times(start: str, count: int, freq: str) -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=count, freq=freq)


def _make_df(uuid: str, times, values) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "uuid": [uuid] * len(times),
            "systime": times,
            "value_double": values,
            "is_delta": [True] * len(times),
        }
    )


def test_stability_score_stable_signal():
    """A perfectly stable signal should score high."""
    t = _times("2024-01-01", 480, "1min")  # 8 hours
    vals = [50.0] * len(t)
    df = _make_df("sensor", t, vals)

    psi = ProcessStabilityIndex(
        df, "sensor", target=50.0, lower_spec=40.0, upper_spec=60.0
    )
    result = psi.stability_score(window="8h")
    assert not result.empty
    assert result["stability_score"].iloc[0] >= 90
    assert result["grade"].iloc[0] == "A"


def test_stability_score_noisy_signal():
    """A noisy signal should score lower."""
    t = _times("2024-01-01", 480, "1min")
    rng = np.random.RandomState(42)
    vals = 50 + rng.normal(0, 15, len(t))  # high noise
    df = _make_df("sensor", t, vals)

    psi = ProcessStabilityIndex(
        df, "sensor", target=50.0, lower_spec=40.0, upper_spec=60.0
    )
    result = psi.stability_score(window="8h")
    assert not result.empty
    assert result["stability_score"].iloc[0] < 90


def test_stability_score_subscores():
    """All four sub-scores should be present and sum to total."""
    t = _times("2024-01-01", 480, "1min")
    vals = [50.0] * len(t)
    df = _make_df("sensor", t, vals)

    psi = ProcessStabilityIndex(df, "sensor")
    result = psi.stability_score(window="8h")
    assert not result.empty
    sub = (
        result["variance_score"]
        + result["bias_score"]
        + result["excursion_score"]
        + result["smoothness_score"]
    )
    assert abs(sub.iloc[0] - result["stability_score"].iloc[0]) < 0.2


def test_score_trend():
    """Score trend should have direction column."""
    t = _times("2024-01-01", 1440, "1min")  # 24 hours
    vals = [50.0] * len(t)
    df = _make_df("sensor", t, vals)

    psi = ProcessStabilityIndex(df, "sensor")
    result = psi.score_trend(window="4h")
    assert not result.empty
    assert "trend_direction" in result.columns
    assert all(
        d in ("improving", "degrading", "stable") for d in result["trend_direction"]
    )


def test_worst_periods():
    """Should return worst-scoring windows with primary_issue."""
    t = _times("2024-01-01", 480, "1min")
    rng = np.random.RandomState(42)
    vals = 50 + rng.normal(0, 5, len(t))
    df = _make_df("sensor", t, vals)

    psi = ProcessStabilityIndex(
        df, "sensor", target=50.0, lower_spec=40.0, upper_spec=60.0
    )
    result = psi.worst_periods(window="1h", n=3)
    assert not result.empty
    assert len(result) <= 3
    assert "primary_issue" in result.columns


def test_stability_comparison():
    """Comparison should include gap_to_best and pct_of_best."""
    t = _times("2024-01-01", 1440, "1min")  # 24 hours
    vals = [50.0] * len(t)
    df = _make_df("sensor", t, vals)

    psi = ProcessStabilityIndex(df, "sensor")
    result = psi.stability_comparison(window="4h")
    assert not result.empty
    assert "gap_to_best" in result.columns
    assert "pct_of_best" in result.columns
    assert all(result["gap_to_best"] >= 0)


def test_auto_derived_specs():
    """When no target/specs given, should auto-derive from data."""
    t = _times("2024-01-01", 480, "1min")
    vals = [50.0] * len(t)
    df = _make_df("sensor", t, vals)

    psi = ProcessStabilityIndex(df, "sensor")
    assert psi.target is not None
    assert psi.lower_spec is not None
    assert psi.upper_spec is not None
    result = psi.stability_score(window="8h")
    assert not result.empty


def test_empty_dataframe():
    df = pd.DataFrame(columns=["uuid", "systime", "value_double", "is_delta"])
    psi = ProcessStabilityIndex(df, "sensor")
    assert psi.stability_score().empty
    assert psi.score_trend().empty
    assert psi.worst_periods().empty
    assert psi.stability_comparison().empty
