"""Hand-computed value tests for methods that had no test at all.

``StatisticalProcessControlEvents.rule_1`` is covered in ``tests/test_spc.py``
and ``tests/reference/test_spc_rules_reference.py``.
"""

import pandas as pd
import pytest

from ts_shape.events.production.line_throughput import LineThroughputEvents
from ts_shape.events.production.machine_state import MachineStateEvents

T0 = pd.Timestamp("2024-01-01 00:00")


def _counter():
    """Readings every 30 min: 0, 5 | 10, 20 | 30, 2 | 6, 26 | 27, 28.

    30 -> 2 at 02:30 is a reset. Reset-aware increments per hour:
    0+5, 5+10, 10+2, 4+20, 1+1 = 5, 15, 12, 24, 2.
    """
    vals = [0, 5, 10, 20, 30, 2, 6, 26, 27, 28]
    return pd.DataFrame(
        {
            "systime": [T0 + pd.Timedelta(minutes=30 * i) for i in range(len(vals))],
            "uuid": "parts",
            "value_integer": vals,
        }
    )


def test_throughput_trends_hand_computed():
    out = LineThroughputEvents(_counter()).throughput_trends(
        "parts", window="1h", trend_window=2
    )
    assert list(out["start"]) == [T0 + pd.Timedelta(hours=h) for h in range(5)]
    assert list(out["end"]) == [T0 + pd.Timedelta(hours=h + 1) for h in range(5)]
    assert list(out["count"]) == [5, 15, 12, 24, 2]
    # Rolling mean over the current and previous window (min_periods=1).
    assert out["moving_avg"].tolist() == pytest.approx([5, 10, 13.5, 18, 13])
    # Slope +5, +3.5, +4.5, -5 against the +-0.5 stable band.
    assert pd.isna(out["trend_direction"].iloc[0])
    assert list(out["trend_direction"].iloc[1:].astype(str)) == [
        "increasing",
        "increasing",
        "increasing",
        "decreasing",
    ]
    # Degradation: count below 85 % of the moving average (2 < 11.05).
    assert list(out["degradation_detected"]) == [False, False, False, False, True]
    assert set(out["uuid"]) == {"prod:throughput"}
    assert set(out["source_uuid"]) == {"parts"}


def test_throughput_trends_needs_trend_window_windows():
    out = LineThroughputEvents(_counter()).throughput_trends(
        "parts", window="1h", trend_window=6
    )
    assert out.empty
    assert "degradation_detected" in out.columns


def _state():
    # Transitions at 10 s, 12 s, 14 s, 16 s (a burst), 100 s and 200 s.
    secs = [0, 10, 12, 14, 16, 100, 200]
    vals = [False, True, False, True, False, True, False]
    return pd.DataFrame(
        {
            "systime": [T0 + pd.Timedelta(seconds=s) for s in secs],
            "uuid": "run",
            "value_bool": vals,
        }
    )


def _sec(s):
    return T0 + pd.Timedelta(seconds=s)


def test_detect_rapid_transitions_hand_computed():
    out = MachineStateEvents(_state(), "run").detect_rapid_transitions(
        threshold="5s", min_count=3
    )
    # Every run of >= 3 consecutive transitions spanning <= 5 s:
    # 10..14 (3 transitions, 4 s) and 12..16 (3, 4 s); 10..16 spans 6 s.
    assert list(out["start"]) == [_sec(10), _sec(12)]
    assert list(out["end"]) == [_sec(14), _sec(16)]
    assert list(out["transition_count"]) == [3, 3]
    assert list(out["duration_seconds"]) == [4.0, 4.0]
    assert list(out["systime"]) == list(out["start"])
    assert set(out["uuid"]) == {"prod:run_idle"}
    assert set(out["source_uuid"]) == {"run"}


def test_detect_rapid_transitions_wider_threshold_reports_each_span():
    out = MachineStateEvents(_state(), "run").detect_rapid_transitions(
        threshold="6s", min_count=3
    )
    # 10..14, 10..16 (4 transitions in 6 s) and 12..16; 100 and 200 are far.
    assert list(zip(out["start"], out["end"], out["transition_count"])) == [
        (_sec(10), _sec(14), 3),
        (_sec(10), _sec(16), 4),
        (_sec(12), _sec(16), 3),
    ]
    assert list(out["duration_seconds"]) == [4.0, 6.0, 4.0]


def test_detect_rapid_transitions_none_when_spread_out():
    out = MachineStateEvents(_state(), "run").detect_rapid_transitions(
        threshold="1s", min_count=3
    )
    assert out.empty
    assert list(out.columns) == [
        "systime",
        "uuid",
        "source_uuid",
        "start",
        "end",
        "transition_count",
        "duration_seconds",
    ]
