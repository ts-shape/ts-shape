import pandas as pd  # type: ignore
import numpy as np  # type: ignore

from ts_shape.events.engineering.material_balance import MaterialBalanceEvents


def _times(start: str, count: int, freq: str) -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=count, freq=freq)


def _make_signal(uuid: str, times, values) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "uuid": [uuid] * len(times),
            "systime": times,
            "value_double": values,
            "is_delta": [True] * len(times),
        }
    )


def test_balance_check_balanced():
    """Equal input and output should be balanced."""
    t = _times("2024-01-01", 120, "1min")
    inp = _make_signal("in1", t, [100.0] * len(t))
    out = _make_signal("out1", t, [100.0] * len(t))
    df = pd.concat([inp, out], ignore_index=True)

    mb = MaterialBalanceEvents(df, ["in1"], ["out1"])
    result = mb.balance_check(window="1h", tolerance_pct=5.0)
    assert not result.empty
    assert all(result["balanced"])
    assert all(abs(r) < 0.01 for r in result["imbalance"])


def test_balance_check_imbalanced():
    """Large difference should be flagged as imbalanced."""
    t = _times("2024-01-01", 120, "1min")
    inp = _make_signal("in1", t, [100.0] * len(t))
    out = _make_signal("out1", t, [50.0] * len(t))
    df = pd.concat([inp, out], ignore_index=True)

    mb = MaterialBalanceEvents(df, ["in1"], ["out1"])
    result = mb.balance_check(window="1h", tolerance_pct=5.0)
    assert not result.empty
    assert not all(result["balanced"])
    assert all(result["imbalance_pct"] > 5.0)


def test_imbalance_trend():
    """Should return trend with direction column."""
    t = _times("2024-01-01", 360, "1min")  # 6 hours
    inp = _make_signal("in1", t, [100.0] * len(t))
    out = _make_signal("out1", t, [80.0] * len(t))
    df = pd.concat([inp, out], ignore_index=True)

    mb = MaterialBalanceEvents(df, ["in1"], ["out1"])
    result = mb.imbalance_trend(window="1h")
    assert not result.empty
    assert "trend_direction" in result.columns
    assert all(
        d in ("growing", "shrinking", "stable") for d in result["trend_direction"]
    )


def test_detect_balance_exceedance():
    """Sustained imbalance should be detected."""
    t = _times("2024-01-01", 360, "1min")  # 6 hours
    inp = _make_signal("in1", t, [100.0] * len(t))
    out = _make_signal("out1", t, [50.0] * len(t))
    df = pd.concat([inp, out], ignore_index=True)

    mb = MaterialBalanceEvents(df, ["in1"], ["out1"])
    result = mb.detect_balance_exceedance(
        window="1h", tolerance_pct=5.0, min_duration="2h"
    )
    assert not result.empty
    assert result["likely_cause"].iloc[0] == "accumulation"


def test_contribution_breakdown():
    """Should show each signal's contribution."""
    t = _times("2024-01-01", 120, "1min")
    in1 = _make_signal("in1", t, [60.0] * len(t))
    in2 = _make_signal("in2", t, [40.0] * len(t))
    out1 = _make_signal("out1", t, [100.0] * len(t))
    df = pd.concat([in1, in2, out1], ignore_index=True)

    mb = MaterialBalanceEvents(df, ["in1", "in2"], ["out1"])
    result = mb.contribution_breakdown(window="1h")
    assert not result.empty
    assert "pct_of_total" in result.columns
    # Input contributions should sum to ~100%
    input_rows = result[result["role"] == "input"]
    for ws in input_rows["window_start"].unique():
        ws_rows = input_rows[input_rows["window_start"] == ws]
        assert abs(ws_rows["pct_of_total"].sum() - 100.0) < 0.1


def test_multiple_inputs_outputs():
    """Should handle multiple input and output streams."""
    t = _times("2024-01-01", 120, "1min")
    in1 = _make_signal("in1", t, [30.0] * len(t))
    in2 = _make_signal("in2", t, [70.0] * len(t))
    out1 = _make_signal("out1", t, [60.0] * len(t))
    out2 = _make_signal("out2", t, [40.0] * len(t))
    df = pd.concat([in1, in2, out1, out2], ignore_index=True)

    mb = MaterialBalanceEvents(df, ["in1", "in2"], ["out1", "out2"])
    result = mb.balance_check(window="1h", tolerance_pct=5.0)
    assert not result.empty
    assert all(result["balanced"])


def test_empty_dataframe():
    df = pd.DataFrame(columns=["uuid", "systime", "value_double", "is_delta"])
    mb = MaterialBalanceEvents(df, ["in1"], ["out1"])
    assert mb.balance_check().empty
    assert mb.imbalance_trend().empty
    assert mb.detect_balance_exceedance().empty
    assert mb.contribution_breakdown().empty
