import pandas as pd  # type: ignore
import numpy as np  # type: ignore

from ts_shape.events.engineering.control_loop_health import ControlLoopHealthEvents


def _times(start: str, count: int, freq: str) -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=count, freq=freq)


def _make_loop_df(t, sp_vals, pv_vals, out_vals=None) -> pd.DataFrame:
    frames = [
        pd.DataFrame(
            {
                "uuid": ["sp"] * len(t),
                "systime": t,
                "value_double": sp_vals,
                "is_delta": True,
            }
        ),
        pd.DataFrame(
            {
                "uuid": ["pv"] * len(t),
                "systime": t,
                "value_double": pv_vals,
                "is_delta": True,
            }
        ),
    ]
    if out_vals is not None:
        frames.append(
            pd.DataFrame(
                {
                    "uuid": ["out"] * len(t),
                    "systime": t,
                    "value_double": out_vals,
                    "is_delta": True,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_error_integrals_zero_error():
    """Perfect tracking should have IAE ≈ 0."""
    t = _times("2024-01-01", 120, "10s")
    vals = [50.0] * len(t)
    df = _make_loop_df(t, vals, vals)

    clh = ControlLoopHealthEvents(df, "sp", "pv")
    result = clh.error_integrals(window="10min")
    assert not result.empty
    assert all(result["iae"] < 0.01)
    assert all(abs(result["bias"]) < 0.01)


def test_error_integrals_constant_offset():
    """Constant offset should produce nonzero IAE and bias."""
    t = _times("2024-01-01", 120, "10s")
    sp = [50.0] * len(t)
    pv = [55.0] * len(t)
    df = _make_loop_df(t, sp, pv)

    clh = ControlLoopHealthEvents(df, "sp", "pv")
    result = clh.error_integrals(window="10min")
    assert not result.empty
    assert all(result["iae"] > 0)
    assert all(result["bias"] > 4.5)


def test_detect_oscillation_sine():
    """A sine wave error should be detected as oscillation."""
    t = _times("2024-01-01", 120, "10s")  # 20 min
    sp = [50.0] * len(t)
    # PV oscillates around SP with period ~60s
    pv = [50 + 10 * np.sin(2 * np.pi * i / 6) for i in range(len(t))]
    df = _make_loop_df(t, sp, pv)

    clh = ControlLoopHealthEvents(df, "sp", "pv")
    result = clh.detect_oscillation(window="5min", min_crossings=4)
    assert not result.empty
    assert result["crossing_count"].iloc[0] >= 4
    assert result["amplitude"].iloc[0] > 5


def test_no_oscillation_flat():
    """No oscillation in a flat signal."""
    t = _times("2024-01-01", 120, "10s")
    vals = [50.0] * len(t)
    df = _make_loop_df(t, vals, vals)

    clh = ControlLoopHealthEvents(df, "sp", "pv")
    result = clh.detect_oscillation(window="5min", min_crossings=4)
    assert result.empty


def test_output_saturation():
    """Valve at 100% should show saturation."""
    t = _times("2024-01-01", 120, "10s")
    sp = [50.0] * len(t)
    pv = [50.0] * len(t)
    out = [100.0] * len(t)  # fully open
    df = _make_loop_df(t, sp, pv, out)

    clh = ControlLoopHealthEvents(df, "sp", "pv", output_uuid="out")
    result = clh.output_saturation(window="10min")
    assert not result.empty
    assert all(result["pct_time_at_high"] > 90)


def test_output_saturation_no_output():
    """No output_uuid should return empty."""
    t = _times("2024-01-01", 60, "10s")
    vals = [50.0] * len(t)
    df = _make_loop_df(t, vals, vals)

    clh = ControlLoopHealthEvents(df, "sp", "pv")
    result = clh.output_saturation()
    assert result.empty


def test_loop_health_summary():
    """Summary should return grades."""
    t = _times("2024-01-01", 360, "10s")  # 1 hour
    sp = [50.0] * len(t)
    pv = [50.0] * len(t)
    df = _make_loop_df(t, sp, pv)

    clh = ControlLoopHealthEvents(df, "sp", "pv")
    result = clh.loop_health_summary(window="30min")
    assert not result.empty
    assert "health_grade" in result.columns


def test_empty_dataframe():
    df = pd.DataFrame(columns=["uuid", "systime", "value_double", "is_delta"])
    clh = ControlLoopHealthEvents(df, "sp", "pv")
    assert clh.error_integrals().empty
    assert clh.detect_oscillation().empty
    assert clh.output_saturation().empty
    assert clh.loop_health_summary().empty
