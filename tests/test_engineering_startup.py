import pandas as pd  # type: ignore

from ts_shape.events.engineering import StartupDetectionEvents


def _times(start: str, count: int, freq: str) -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=count, freq=freq)


def test_startup_by_threshold():
    # Metric crosses threshold=500 at t=30s and remains above for at least 60s
    t = _times("2024-01-01 00:00:00", 7, "30s")  # 0..180s
    vals = [400, 480, 520, 550, 560, 570, 580]
    df = pd.DataFrame(
        {
            "uuid": ["motor"] * len(t),
            "systime": t,
            "value_integer": vals,
            "is_delta": [True] * len(t),
        }
    )

    sde = StartupDetectionEvents(df, target_uuid="motor", value_column="value_integer")
    events = sde.detect_startup_by_threshold(
        threshold=500, hysteresis=(500, 480), min_above="60s"
    )
    assert not events.empty
    assert events["method"].iloc[0] == "threshold"
    assert events["end"].iloc[0] >= events["start"].iloc[0]


def test_startup_by_slope():
    # Value increases by 10 every 10s => slope ~1 unit/s for a duration of ~50s
    t = _times("2024-01-01 00:00:00", 7, "10s")  # 0..60s
    vals = [0, 10, 20, 30, 40, 50, 60]
    df = pd.DataFrame(
        {
            "uuid": ["metric"] * len(t),
            "systime": t,
            "value_integer": vals,
            "is_delta": [True] * len(t),
        }
    )

    sde = StartupDetectionEvents(df, target_uuid="metric", value_column="value_integer")
    events = sde.detect_startup_by_slope(min_slope=0.5, min_duration="30s")
    assert not events.empty
    assert events["method"].iloc[0] == "slope"
    assert (events["end"] >= events["start"]).all()
