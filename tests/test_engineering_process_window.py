import numpy as np
import pandas as pd

from ts_shape.events.engineering.process_window import ProcessWindowEvents


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


def test_windowed_statistics():
    """Stats should be computed per window."""
    t = _times("2024-01-01", 120, "10s")  # 20 minutes
    rng = np.random.RandomState(42)
    vals = 50 + rng.normal(0, 2, len(t))
    df = _make_df("sensor", t, vals)

    pw = ProcessWindowEvents(df, "sensor")
    result = pw.windowed_statistics(window="10min")
    assert not result.empty
    assert "mean" in result.columns
    assert "std" in result.columns
    assert "p25" in result.columns
    assert "p75" in result.columns
    assert all(result["count"] > 0)


def test_detect_mean_shift():
    """A jump in mean should be detected."""
    t = _times("2024-01-01", 240, "10s")  # 40 minutes
    vals = [50.0] * 120 + [80.0] * 120  # jump at midpoint
    df = _make_df("sensor", t, vals)

    pw = ProcessWindowEvents(df, "sensor")
    result = pw.detect_mean_shift(window="10min", sensitivity=2.0)
    assert not result.empty
    assert result["shift_sigma"].iloc[0] >= 2.0


def test_no_mean_shift_stable():
    """A stable signal should produce no mean shift events."""
    t = _times("2024-01-01", 240, "10s")
    vals = [50.0] * len(t)
    df = _make_df("sensor", t, vals)

    pw = ProcessWindowEvents(df, "sensor")
    result = pw.detect_mean_shift(window="10min", sensitivity=2.0)
    assert result.empty


def test_detect_variance_change():
    """A change in noise level should be detected."""
    rng = np.random.RandomState(42)
    t = _times("2024-01-01", 240, "10s")  # 40 minutes
    low_var = 50 + rng.normal(0, 0.5, 120)  # low noise
    high_var = 50 + rng.normal(0, 10, 120)  # high noise
    vals = np.concatenate([low_var, high_var])
    df = _make_df("sensor", t, vals)

    pw = ProcessWindowEvents(df, "sensor")
    result = pw.detect_variance_change(window="10min", ratio_threshold=2.0)
    assert not result.empty


def test_window_comparison_anomalous():
    """An anomalous window should be flagged."""
    t = _times("2024-01-01", 360, "10s")  # 60 minutes
    vals = [50.0] * 300 + [200.0] * 60  # last window is anomalous
    df = _make_df("sensor", t, vals)

    pw = ProcessWindowEvents(df, "sensor")
    result = pw.window_comparison(window="10min")
    assert not result.empty
    assert any(result["is_anomalous"])


def test_window_comparison_stable():
    """A stable signal should have no anomalous windows."""
    t = _times("2024-01-01", 360, "10s")
    vals = [50.0] * len(t)
    df = _make_df("sensor", t, vals)

    pw = ProcessWindowEvents(df, "sensor")
    result = pw.window_comparison(window="10min")
    assert not result.empty
    assert not any(result["is_anomalous"])


def test_empty_dataframe():
    df = pd.DataFrame(columns=["uuid", "systime", "value_double", "is_delta"])
    pw = ProcessWindowEvents(df, "sensor")
    assert pw.windowed_statistics().empty
    assert pw.detect_mean_shift().empty
    assert pw.detect_variance_change().empty
    assert pw.window_comparison().empty


def test_mean_shift_and_variance_change_thresholds_are_strict():
    """Documented as '>': a value exactly at the threshold is not flagged."""
    t = _times("2024-01-01", 4, "30min")
    df = _make_df("sensor", t, [1.0, 3.0, 5.0, 9.0])
    pw = ProcessWindowEvents(df, "sensor")
    stats = pw.windowed_statistics(window="1h")
    prev, curr = stats.iloc[0], stats.iloc[1]
    shift = abs(curr["mean"] - prev["mean"]) / prev["std"]
    ratio = curr["std"] / prev["std"]

    assert pw.detect_mean_shift(window="1h", sensitivity=shift).empty
    assert len(pw.detect_mean_shift(window="1h", sensitivity=shift * 0.999)) == 1
    assert pw.detect_variance_change(window="1h", ratio_threshold=ratio).empty
    assert (
        len(pw.detect_variance_change(window="1h", ratio_threshold=ratio * 0.999)) == 1
    )
    # Shrinking variance: exactly 1/ratio_threshold is not flagged either.
    df2 = _make_df("sensor", t, [5.0, 9.0, 1.0, 3.0])
    pw2 = ProcessWindowEvents(df2, "sensor")
    assert pw2.detect_variance_change(window="1h", ratio_threshold=ratio).empty
