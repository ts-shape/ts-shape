"""Reset-aware behaviour of LineThroughputEvents.count_parts."""

import pandas as pd  # type: ignore
import pytest

from ts_shape.events.production import LineThroughputEvents


@pytest.fixture
def reset_counter_data():
    """A counter that resets mid-way through the second 5-minute window.

    Values: 0,100,200,300,400,500,600,[reset]0,100,200 at 1-minute spacing.
    Two 5-minute windows:
      - window 06:00-06:04: 0,100,200,300,400  (true production 400)
      - window 06:05-06:09: 500,600,0,100,200  (true production 400, 1 reset)
    True total production = 800.
    """
    times = pd.date_range("2024-01-15 06:00:00", periods=10, freq="1min")
    values = [0, 100, 200, 300, 400, 500, 600, 0, 100, 200]
    rows = [
        {
            "systime": t,
            "uuid": "line_counter",
            "value_integer": v,
            "is_delta": True,
        }
        for t, v in zip(times, values)
    ]
    return pd.DataFrame(rows)


def test_count_parts_default_under_reports(reset_counter_data):
    lt = LineThroughputEvents(reset_counter_data)
    result = lt.count_parts("line_counter", window="5min")
    assert list(result.columns) == [
        "start",
        "uuid",
        "source_uuid",
        "is_delta",
        "count",
    ]
    assert "resets" not in result.columns
    # window-over-window diff loses the first window and the reset drop
    assert result["count"].sum() == 200


def test_count_parts_handles_resets(reset_counter_data):
    lt = LineThroughputEvents(reset_counter_data)
    result = lt.count_parts("line_counter", window="5min", handle_resets=True)
    assert "resets" in result.columns
    # per-reading increments recover the true production in both windows
    assert result["count"].sum() == 800
    assert result["count"].tolist() == [400, 400]
    assert result["resets"].sum() == 1
