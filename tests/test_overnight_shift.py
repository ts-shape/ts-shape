"""Overnight-shift attribution (production_day) regression tests."""

import pandas as pd  # type: ignore

from ts_shape.events.production.shift_reporting import ShiftReporting
from ts_shape.events.production.target_tracking import TargetTracking


def _counter_frame():
    # A monotonic counter sampled hourly across a full day. The night shift
    # (22:00 -> 06:00) straddles midnight. Counter climbs by ~100/hour.
    times = pd.date_range("2024-01-01 00:00:00", "2024-01-02 05:00:00", freq="1h")
    values = list(range(0, 100 * len(times), 100))
    return pd.DataFrame(
        {
            "uuid": "counter",
            "systime": times,
            "value_integer": values,
        }
    )


def test_target_tracking_overnight_split_without_production_day():
    tt = TargetTracking(_counter_frame())
    naive = tt.compare_to_target("counter", {"shift_3": 800})
    # Jan 1 shift_3 group spans 00:00-06:00 AND 22:00-23:59 -> last-first sweeps
    # in the whole day, hugely overstating achievement.
    jan1_night = naive[
        (naive["date"] == pd.Timestamp("2024-01-01").date())
        & (naive["shift"] == "shift_3")
    ]
    assert (jan1_night["achievement_pct"] > 200).any()


def test_target_tracking_production_day_contiguous_shift():
    tt = TargetTracking(_counter_frame())
    fixed = tt.compare_to_target("counter", {"shift_3": 800}, production_day=True)
    # With production_day, the night shift starting 2024-01-01 22:00 runs to
    # 2024-01-02 06:00 as one group: 22:00->05:00 counter delta = ~700.
    jan1_night = fixed[
        (fixed["date"] == pd.Timestamp("2024-01-01").date())
        & (fixed["shift"] == "shift_3")
    ]
    assert not jan1_night.empty
    # Achievement is now in a sane range, not >300%.
    assert (jan1_night["achievement_pct"] < 150).all()


def test_shift_reporting_production_day_merges_night_shift():
    sr = ShiftReporting(_counter_frame())
    naive = sr.shift_production("counter")
    fixed = sr.shift_production("counter", production_day=True)
    # Naive splits the night shift across two calendar dates; production_day
    # yields a single contiguous shift_3 row for 2024-01-01.
    naive_night = naive[naive["shift"] == "shift_3"]
    fixed_night = fixed[fixed["shift"] == "shift_3"]
    assert len(naive_night) > len(fixed_night[fixed_night["date"] == pd.Timestamp("2024-01-02").date()])
    jan1 = fixed_night[fixed_night["date"] == pd.Timestamp("2024-01-01").date()]
    assert len(jan1) == 1
