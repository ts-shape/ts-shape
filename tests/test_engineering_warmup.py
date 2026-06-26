import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from ts_shape.events.engineering.warmup_analysis import WarmUpCoolDownEvents


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


def test_detect_warmup_ramp():
    """A rising ramp should be detected as warmup."""
    t = _times("2024-01-01", 60, "10s")  # 10 minutes
    vals = np.linspace(20, 120, len(t))  # rise of 100
    df = _make_df("oven", t, vals)

    det = WarmUpCoolDownEvents(df, "oven")
    result = det.detect_warmup(min_rise=50.0, min_duration="2m")
    assert not result.empty
    assert result["rise"].iloc[0] >= 50.0
    assert result["avg_rate"].iloc[0] > 0


def test_detect_cooldown_ramp():
    """A falling ramp should be detected as cooldown."""
    t = _times("2024-01-01", 60, "10s")
    vals = np.linspace(120, 20, len(t))  # fall of 100
    df = _make_df("oven", t, vals)

    det = WarmUpCoolDownEvents(df, "oven")
    result = det.detect_cooldown(min_fall=50.0, min_duration="2m")
    assert not result.empty
    assert result["fall"].iloc[0] >= 50.0


def test_no_warmup_flat_signal():
    """A flat signal should produce no warmup events."""
    t = _times("2024-01-01", 60, "10s")
    vals = [50.0] * len(t)
    df = _make_df("oven", t, vals)

    det = WarmUpCoolDownEvents(df, "oven")
    result = det.detect_warmup(min_rise=10.0, min_duration="1m")
    assert result.empty


def test_warmup_consistency_multiple_ramps():
    """Multiple ramps should be compared for consistency."""
    # Two warmup ramps separated by a flat period
    t1 = _times("2024-01-01 00:00", 30, "10s")
    v1 = np.linspace(20, 80, 30)

    t2 = _times("2024-01-01 01:00", 30, "10s")
    v2 = np.linspace(20, 80, 30)

    t_flat = _times("2024-01-01 00:10", 30, "10s")
    v_flat = [80.0] * 30

    t = t1.append(t_flat).append(t2)
    vals = list(v1) + list(v_flat) + list(v2)
    df = _make_df("oven", t, vals)

    det = WarmUpCoolDownEvents(df, "oven")
    consistency = det.warmup_consistency(min_rise=30.0, min_duration="1m")
    assert not consistency.empty
    assert "deviation_from_median_duration" in consistency.columns


def test_time_to_target():
    """Should detect when target value is reached."""
    t = _times("2024-01-01", 60, "10s")
    vals = np.linspace(20, 120, len(t))
    df = _make_df("oven", t, vals)

    det = WarmUpCoolDownEvents(df, "oven")
    result = det.time_to_target(target_value=80.0, direction="rising")
    assert not result.empty
    assert result["time_to_target_seconds"].iloc[0] > 0


def test_time_to_target_not_reached():
    """If target is never reached, result should be empty."""
    t = _times("2024-01-01", 60, "10s")
    vals = np.linspace(20, 50, len(t))  # never reaches 100
    df = _make_df("oven", t, vals)

    det = WarmUpCoolDownEvents(df, "oven")
    result = det.time_to_target(target_value=100.0, direction="rising")
    assert result.empty


def test_empty_dataframe():
    df = pd.DataFrame(columns=["uuid", "systime", "value_double", "is_delta"])
    det = WarmUpCoolDownEvents(df, "oven")
    assert det.detect_warmup(min_rise=10.0).empty
    assert det.detect_cooldown(min_fall=10.0).empty
    assert det.warmup_consistency(min_rise=10.0).empty
    assert det.time_to_target(target_value=100.0).empty
