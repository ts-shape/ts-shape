import pandas as pd  # type: ignore
import numpy as np  # type: ignore

from ts_shape.events.engineering.operating_range import OperatingRangeEvents


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


def test_operating_envelope():
    """Envelope should capture min/max/percentiles."""
    t = _times("2024-01-01", 120, "10s")
    rng = np.random.RandomState(42)
    vals = 50 + rng.normal(0, 5, len(t))
    df = _make_df("sensor", t, vals)

    det = OperatingRangeEvents(df, "sensor")
    result = det.operating_envelope(window="10min")
    assert not result.empty
    assert "p5" in result.columns
    assert "p95" in result.columns
    assert all(result["min_value"] <= result["p5"])
    assert all(result["p95"] <= result["max_value"])


def test_detect_regime_change():
    """A sudden jump in mean should trigger regime change."""
    t = _times("2024-01-01", 240, "10s")  # 40 minutes
    vals = [50.0] * 120 + [100.0] * 120  # jump at midpoint
    df = _make_df("sensor", t, vals)

    det = OperatingRangeEvents(df, "sensor")
    result = det.detect_regime_change(window="10min", shift_threshold=2.0)
    assert not result.empty
    assert result["shift_magnitude"].iloc[0] >= 2.0


def test_no_regime_change_stable():
    """A stable signal should produce no regime changes."""
    t = _times("2024-01-01", 240, "10s")
    rng = np.random.RandomState(42)
    vals = 50 + rng.normal(0, 1, len(t))
    df = _make_df("sensor", t, vals)

    det = OperatingRangeEvents(df, "sensor")
    result = det.detect_regime_change(window="10min", shift_threshold=5.0)
    assert result.empty


def test_time_in_range():
    """All values within range should give 100%."""
    t = _times("2024-01-01", 120, "10s")
    vals = [50.0] * len(t)
    df = _make_df("sensor", t, vals)

    det = OperatingRangeEvents(df, "sensor")
    result = det.time_in_range(lower=40.0, upper=60.0, window="10min")
    assert not result.empty
    assert all(abs(r - 100.0) < 0.01 for r in result["time_in_range_pct"])


def test_time_in_range_partial():
    """Half in, half out should be ~50%."""
    t = _times("2024-01-01", 120, "10s")
    vals = [50.0] * 60 + [200.0] * 60
    df = _make_df("sensor", t, vals)

    det = OperatingRangeEvents(df, "sensor")
    result = det.time_in_range(lower=40.0, upper=60.0, window="20min")
    assert not result.empty
    # At least one window should have < 100% in range
    assert any(r < 100.0 for r in result["time_in_range_pct"])


def test_value_distribution():
    """Distribution should sum to 100%."""
    t = _times("2024-01-01", 100, "10s")
    vals = list(range(100))
    df = _make_df("sensor", t, vals)

    det = OperatingRangeEvents(df, "sensor")
    result = det.value_distribution(n_bins=5)
    assert not result.empty
    assert abs(result["pct"].sum() - 100.0) < 0.1
    assert abs(result["cumulative_pct"].iloc[-1] - 100.0) < 0.1


def test_empty_dataframe():
    df = pd.DataFrame(columns=["uuid", "systime", "value_double", "is_delta"])
    det = OperatingRangeEvents(df, "sensor")
    assert det.operating_envelope().empty
    assert det.detect_regime_change().empty
    assert det.time_in_range(0, 100).empty
    assert det.value_distribution().empty
