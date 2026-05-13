import pandas as pd  # type: ignore
import numpy as np  # type: ignore

from ts_shape.events.engineering.steady_state_detection import (
    SteadyStateDetectionEvents,
)


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


def test_detect_steady_state_flat_signal():
    """A flat signal should be detected as steady state."""
    t = _times("2024-01-01", 120, "10s")  # 20 minutes
    vals = [50.0] * len(t)
    df = _make_df("temp", t, vals)

    det = SteadyStateDetectionEvents(df, "temp")
    result = det.detect_steady_state(window="1m", std_threshold=1.0, min_duration="5m")
    assert not result.empty
    assert "mean_value" in result.columns
    assert result["mean_value"].iloc[0] == 50.0


def test_detect_steady_state_with_noise():
    """Small noise should still be detected as steady."""
    rng = np.random.RandomState(42)
    t = _times("2024-01-01", 120, "10s")
    vals = 50.0 + rng.normal(0, 0.1, len(t))  # very small noise
    df = _make_df("temp", t, vals)

    det = SteadyStateDetectionEvents(df, "temp")
    result = det.detect_steady_state(window="1m", std_threshold=1.0, min_duration="5m")
    assert not result.empty


def test_detect_transient_during_ramp():
    """A ramp should be detected as transient."""
    t = _times("2024-01-01", 60, "10s")  # 10 minutes
    vals = np.linspace(0, 100, len(t))
    df = _make_df("temp", t, vals)

    det = SteadyStateDetectionEvents(df, "temp")
    result = det.detect_transient_periods(window="1m", std_threshold=1.0)
    assert not result.empty
    assert result["max_std"].iloc[0] > 1.0


def test_steady_state_statistics():
    """Stats should report steady time for a flat signal."""
    t = _times("2024-01-01", 120, "10s")
    vals = [50.0] * len(t)
    df = _make_df("temp", t, vals)

    det = SteadyStateDetectionEvents(df, "temp")
    stats = det.steady_state_statistics(
        window="1m", std_threshold=1.0, min_duration="5m"
    )
    assert stats["num_steady_periods"] > 0
    assert stats["steady_pct"] > 0


def test_steady_state_value_bands():
    """Value bands should be computed for steady intervals."""
    t = _times("2024-01-01", 120, "10s")
    vals = [50.0] * len(t)
    df = _make_df("temp", t, vals)

    det = SteadyStateDetectionEvents(df, "temp")
    bands = det.steady_state_value_bands(
        window="1m", std_threshold=1.0, min_duration="5m"
    )
    assert not bands.empty
    assert "lower_band" in bands.columns
    assert "upper_band" in bands.columns


def test_no_steady_state_during_oscillation():
    """A highly oscillating signal should not produce steady intervals."""
    t = _times("2024-01-01", 60, "10s")
    vals = [0, 100] * 30  # oscillating wildly
    df = _make_df("temp", t, vals)

    det = SteadyStateDetectionEvents(df, "temp")
    result = det.detect_steady_state(window="1m", std_threshold=5.0, min_duration="5m")
    assert result.empty


def test_empty_dataframe():
    """Empty input should return empty results."""
    df = pd.DataFrame(columns=["uuid", "systime", "value_double", "is_delta"])
    det = SteadyStateDetectionEvents(df, "temp")
    assert det.detect_steady_state().empty
    assert det.detect_transient_periods().empty
    stats = det.steady_state_statistics()
    assert stats["num_steady_periods"] == 0
