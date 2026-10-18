import pandas as pd

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


def test_startup_by_slope_window_smooths_noise():
    # Zig-zag ramp: +3, -1 alternating every 10s -> net slope 0.1 unit/s,
    # instantaneous slope alternates +0.3 / -0.1.
    t = _times("2024-01-01 00:00:00", 13, "10s")
    vals = [0]
    for i in range(12):
        vals.append(vals[-1] + (3 if i % 2 == 0 else -1))
    df = pd.DataFrame(
        {
            "uuid": ["metric"] * len(t),
            "systime": t,
            "value_integer": vals,
            "is_delta": [True] * len(t),
        }
    )
    sde = StartupDetectionEvents(df, target_uuid="metric", value_column="value_integer")
    # Instantaneous: no run of rising samples lasts 30s.
    assert sde.detect_startup_by_slope(min_slope=0.05, min_duration="30s").empty
    # 20s trailing window: slope is 0.1 everywhere it is defined.
    ev = sde.detect_startup_by_slope(
        min_slope=0.05, min_duration="30s", slope_window="20s"
    )
    assert len(ev) == 1
    # The first defined slope (at t[2]) describes t[0] -> t[2]: the run starts
    # at its reference sample t[0].
    assert ev["start"].iloc[0] == t[0] and ev["end"].iloc[0] == t[-1]
    assert abs(ev["avg_slope"].iloc[0] - 0.1) < 1e-9
