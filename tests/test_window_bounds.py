"""Window bounds of summaries: calendar-correct, also for right-labeled bins."""

import pandas as pd
import pytest

from ts_shape.events._windows import window_bounds
from ts_shape.events.production.line_throughput import LineThroughputEvents


@pytest.mark.parametrize(
    "window,label,start,end",
    [
        ("1D", "2024-03-31", "2024-03-31", "2024-04-01"),
        ("MS", "2024-02-01", "2024-02-01", "2024-03-01"),
        # pandas labels 'W' (W-SUN) and 'ME' bins with their last day
        ("W", "2024-01-07", "2024-01-01", "2024-01-08"),
        ("ME", "2024-02-29", "2024-02-01", "2024-03-01"),
    ],
)
def test_window_bounds(window, label, start, end):
    s, e = window_bounds(pd.Series(pd.to_datetime([label])), window)
    assert s.iloc[0] == pd.Timestamp(start)
    assert e.iloc[0] == pd.Timestamp(end)


def test_dst_day_is_23_hours():
    lab = pd.Series([pd.Timestamp("2024-03-31", tz="Europe/Berlin")])
    s, e = window_bounds(lab, "1D")
    assert (e - s).iloc[0] == pd.Timedelta(hours=23)


@pytest.mark.parametrize("window", ["W", "ME", "1D"])
def test_resampled_summary_windows_contain_their_data(window):
    t = pd.date_range("2024-01-01", "2024-03-31", freq="6h")
    df = pd.DataFrame({"systime": t, "uuid": "parts", "value_integer": range(len(t))})
    out = LineThroughputEvents(df).count_parts("parts", window=window)
    for row in out.itertuples():
        inside = df[(df.systime >= row.start) & (df.systime < row.end)]
        assert len(inside), f"{window}: window {row.start}..{row.end} holds no samples"
    assert out["start"].min() <= t[0] < out["end"].iloc[0]
