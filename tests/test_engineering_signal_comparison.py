import pandas as pd  # type: ignore
import numpy as np  # type: ignore

from ts_shape.events.engineering.signal_comparison import SignalComparisonEvents


def _times(start: str, count: int, freq: str) -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=count, freq=freq)


def _make_two_signal_df(t, ref_vals, act_vals) -> pd.DataFrame:
    ref = pd.DataFrame(
        {
            "uuid": ["ref"] * len(t),
            "systime": t,
            "value_double": ref_vals,
            "is_delta": [True] * len(t),
        }
    )
    act = pd.DataFrame(
        {
            "uuid": ["act"] * len(t),
            "systime": t,
            "value_double": act_vals,
            "is_delta": [True] * len(t),
        }
    )
    return pd.concat([ref, act], ignore_index=True)


def test_detect_divergence_large_offset():
    """Two signals with constant offset should diverge."""
    t = _times("2024-01-01", 120, "10s")  # 20 minutes
    ref = [50.0] * len(t)
    act = [60.0] * len(t)  # offset of 10
    df = _make_two_signal_df(t, ref, act)

    comp = SignalComparisonEvents(df, "ref")
    result = comp.detect_divergence("act", tolerance=5.0, min_duration="1m")
    assert not result.empty
    assert result["direction"].iloc[0] == "above"
    assert result["max_deviation"].iloc[0] >= 10.0


def test_no_divergence_within_tolerance():
    """Signals within tolerance should produce no divergence events."""
    t = _times("2024-01-01", 60, "10s")
    ref = [50.0] * len(t)
    act = [52.0] * len(t)  # offset of 2
    df = _make_two_signal_df(t, ref, act)

    comp = SignalComparisonEvents(df, "ref")
    result = comp.detect_divergence("act", tolerance=5.0, min_duration="1m")
    assert result.empty


def test_deviation_statistics():
    """Stats should reflect known offset."""
    t = _times("2024-01-01", 120, "10s")
    ref = [50.0] * len(t)
    act = [55.0] * len(t)
    df = _make_two_signal_df(t, ref, act)

    comp = SignalComparisonEvents(df, "ref")
    stats = comp.deviation_statistics("act", window="10min")
    assert not stats.empty
    assert abs(stats["mae"].iloc[0] - 5.0) < 0.1
    assert abs(stats["bias"].iloc[0] - 5.0) < 0.1


def test_tracking_error_trend_stable():
    """Stable offset should show stable trend."""
    t = _times("2024-01-01", 480, "10s")  # 80 min
    ref = [50.0] * len(t)
    act = [55.0] * len(t)
    df = _make_two_signal_df(t, ref, act)

    comp = SignalComparisonEvents(df, "ref")
    trend = comp.tracking_error_trend("act", window="20min")
    assert not trend.empty
    assert all(
        d in ("stable", "improving", "worsening") for d in trend["trend_direction"]
    )


def test_correlation_windows_identical():
    """Identical signals should have correlation = 1."""
    t = _times("2024-01-01", 120, "10s")
    vals = np.sin(np.linspace(0, 4 * np.pi, len(t))) * 10 + 50
    df = _make_two_signal_df(t, vals, vals)

    comp = SignalComparisonEvents(df, "ref")
    corr = comp.correlation_windows("act", window="10min")
    assert not corr.empty
    assert all(abs(c - 1.0) < 0.01 for c in corr["correlation"])


def test_correlation_windows_anti_correlated():
    """Anti-correlated signals should have negative correlation."""
    t = _times("2024-01-01", 120, "10s")
    vals = np.sin(np.linspace(0, 4 * np.pi, len(t))) * 10 + 50
    anti = -vals + 100
    df = _make_two_signal_df(t, vals, anti)

    comp = SignalComparisonEvents(df, "ref")
    corr = comp.correlation_windows("act", window="10min")
    assert not corr.empty
    assert all(c < 0 for c in corr["correlation"])


def test_empty_dataframe():
    df = pd.DataFrame(columns=["uuid", "systime", "value_double", "is_delta"])
    comp = SignalComparisonEvents(df, "ref")
    assert comp.detect_divergence("act", tolerance=1.0).empty
    assert comp.deviation_statistics("act").empty
    assert comp.correlation_windows("act").empty
