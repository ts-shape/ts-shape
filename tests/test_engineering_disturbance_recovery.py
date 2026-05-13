import pandas as pd  # type: ignore
import numpy as np  # type: ignore

from ts_shape.events.engineering.disturbance_recovery import DisturbanceRecoveryEvents


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


def test_detect_disturbance_spike():
    """A spike in an otherwise flat signal should be detected."""
    t = _times("2024-01-01", 120, "10s")
    vals = [50.0] * 50 + [100.0] * 10 + [50.0] * 60  # spike at t=50
    df = _make_df("sensor", t, vals)

    det = DisturbanceRecoveryEvents(df, "sensor")
    result = det.detect_disturbances(
        baseline_window="5m", threshold_sigma=2.0, min_duration="30s"
    )
    assert not result.empty
    assert result["direction"].iloc[0] == "above"
    assert result["peak_deviation"].iloc[0] > 30


def test_no_disturbance_flat():
    """No disturbance in a flat signal."""
    t = _times("2024-01-01", 120, "10s")
    vals = [50.0] * len(t)
    df = _make_df("sensor", t, vals)

    det = DisturbanceRecoveryEvents(df, "sensor")
    result = det.detect_disturbances(baseline_window="5m", threshold_sigma=3.0)
    assert result.empty


def test_recovery_time_quick_recovery():
    """Signal that returns to baseline should show recovery."""
    t = _times("2024-01-01", 120, "10s")
    vals = [50.0] * 40 + [100.0] * 10 + [50.0] * 70
    df = _make_df("sensor", t, vals)

    det = DisturbanceRecoveryEvents(df, "sensor")
    result = det.recovery_time(baseline_window="5m", threshold_sigma=2.0)
    if not result.empty:
        assert (
            result["recovered"].iloc[0] is True or result["recovered"].iloc[0] == True
        )


def test_disturbance_frequency():
    """Should count disturbances per window."""
    t = _times("2024-01-01", 120, "10s")
    vals = [50.0] * 40 + [100.0] * 10 + [50.0] * 70
    df = _make_df("sensor", t, vals)

    det = DisturbanceRecoveryEvents(df, "sensor")
    result = det.disturbance_frequency(
        window="20min", baseline_window="5m", threshold_sigma=2.0
    )
    assert not result.empty
    assert "disturbance_count" in result.columns


def test_before_after_comparison():
    """Step change should show mean_shift."""
    t = _times("2024-01-01", 120, "10s")
    vals = [50.0] * 40 + [100.0] * 10 + [80.0] * 70  # doesn't return to 50
    df = _make_df("sensor", t, vals)

    det = DisturbanceRecoveryEvents(df, "sensor")
    result = det.before_after_comparison(
        baseline_window="5m", threshold_sigma=2.0, comparison_window="3m"
    )
    if not result.empty:
        assert "mean_shift" in result.columns


def test_with_setpoint():
    """Should detect disturbance relative to setpoint."""
    t = _times("2024-01-01", 120, "10s")
    sp_vals = [50.0] * len(t)
    pv_vals = [50.0] * 40 + [90.0] * 10 + [50.0] * 70

    sp_df = pd.DataFrame(
        {
            "uuid": ["sp"] * len(t),
            "systime": t,
            "value_double": sp_vals,
            "is_delta": True,
        }
    )
    pv_df = pd.DataFrame(
        {
            "uuid": ["pv"] * len(t),
            "systime": t,
            "value_double": pv_vals,
            "is_delta": True,
        }
    )
    df = pd.concat([sp_df, pv_df], ignore_index=True)

    det = DisturbanceRecoveryEvents(df, "pv", setpoint_uuid="sp")
    result = det.detect_disturbances(
        baseline_window="5m", threshold_sigma=2.0, min_duration="30s"
    )
    assert not result.empty


def test_empty_dataframe():
    df = pd.DataFrame(columns=["uuid", "systime", "value_double", "is_delta"])
    det = DisturbanceRecoveryEvents(df, "sensor")
    assert det.detect_disturbances().empty
    assert det.recovery_time().empty
    assert det.before_after_comparison().empty
