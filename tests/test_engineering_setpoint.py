import pandas as pd  # type: ignore

from ts_shape.events.engineering import SetpointChangeEvents


def _times(start: str, count: int, freq: str) -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=count, freq=freq)


def test_setpoint_step_detection_and_unified_changes():
    # Setpoint with clear steps at t=120s and t=240s
    t = _times("2024-01-01 00:00:00", 5, "60s")
    df_sp = pd.DataFrame(
        {
            "uuid": ["sp"] * len(t),
            "systime": t,
            "value_double": [10.0, 10.0, 11.0, 11.0, 12.0],
            "is_delta": [True] * len(t),
        }
    )

    spe = SetpointChangeEvents(
        pd.concat([df_sp], ignore_index=True), setpoint_uuid="sp"
    )

    steps = spe.detect_setpoint_steps(min_delta=0.5, min_hold="30s")
    assert not steps.empty
    # Expect two step change points
    assert steps["change_type"].unique().tolist() == ["step"]
    assert len(steps) == 2
    assert steps["magnitude"].abs().tolist() == [1.0, 1.0]

    unified = spe.detect_setpoint_changes(min_delta=0.5, min_rate=None, min_hold="30s")
    # Should include the same step events and standard columns
    assert {"start", "end", "uuid", "is_delta", "change_type"}.issubset(unified.columns)
    assert (unified["change_type"] == "step").all()


def test_setpoint_ramp_detection():
    # Build a ramp: values increase by 1.0 every 10s over 1 minute
    t = _times("2024-01-01 00:00:00", 7, "10s")  # 0..60s
    vals = [0, 1, 2, 3, 4, 5, 6]
    df_sp = pd.DataFrame(
        {
            "uuid": ["sp"] * len(t),
            "systime": t,
            "value_double": [float(v) for v in vals],
            "is_delta": [True] * len(t),
        }
    )

    spe = SetpointChangeEvents(df_sp, setpoint_uuid="sp")
    # dv/dt = 1 per 10s = 0.1 per second
    ramps = spe.detect_setpoint_ramps(min_rate=0.05, min_duration="20s")
    assert not ramps.empty
    assert (ramps["change_type"] == "ramp").all()
    # One continuous ramp interval over most of the series
    assert (ramps["end"] >= ramps["start"]).all()


def test_setpoint_time_to_settle_and_overshoot():
    # Setpoint changes at 120s and 240s
    t_sp = _times("2024-01-01 00:00:00", 5, "60s")
    df_sp = pd.DataFrame(
        {
            "uuid": ["sp"] * len(t_sp),
            "systime": t_sp,
            "value_double": [10.0, 10.0, 11.0, 11.0, 12.0],
            "is_delta": [True] * len(t_sp),
        }
    )

    # Actual follows setpoint with settling after 45s and 30s respectively; add overshoot on first change
    t_pv = _times("2024-01-01 00:00:00", 25, "15s")  # 0..6m in 15s steps
    pv_vals = []
    for ts in t_pv:
        sec = (ts - t_pv[0]).total_seconds()
        if sec < 120:
            pv_vals.append(10.0)
        elif sec < 165:  # settling begins at 165s for first step
            pv_vals.append(10.8)  # approaching, not within tol yet
        elif sec < 240:
            # within tol around 11; add an overshoot peak at 180s (11.5)
            pv_vals.append(11.5 if sec == 180 else 11.05)
        elif sec < 270:  # second step not yet settled
            pv_vals.append(11.6)
        else:
            pv_vals.append(12.0)  # within tol after 30s

    df_pv = pd.DataFrame(
        {
            "uuid": ["pv"] * len(t_pv),
            "systime": t_pv,
            "value_double": pv_vals,
            "is_delta": [True] * len(t_pv),
        }
    )

    df = pd.concat([df_sp, df_pv], ignore_index=True)
    spe = SetpointChangeEvents(df, setpoint_uuid="sp")

    settle = spe.time_to_settle(actual_uuid="pv", tol=0.1, hold="30s", lookahead="10m")
    assert not settle.empty
    assert {"start", "t_settle_seconds", "settled"}.issubset(settle.columns)
    # First change settles after ~45s, second after ~30s
    assert sorted([s for s in settle["t_settle_seconds"] if s is not None]) == [
        30.0,
        45.0,
    ]

    overs = spe.overshoot_metrics(actual_uuid="pv", window="10m")
    assert not overs.empty
    assert {"overshoot_abs", "overshoot_pct", "t_peak_seconds"}.issubset(overs.columns)
    # Expect a positive overshoot on at least one change
    assert overs["overshoot_abs"].fillna(0).max() >= 0.4
