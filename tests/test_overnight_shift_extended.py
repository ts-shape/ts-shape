"""Overnight production_day extended to scrap/rework/quality/operator."""

import pandas as pd  # type: ignore

from ts_shape.events.production.quality_tracking import QualityTracking
from ts_shape.events.production.scrap_tracking import ScrapTracking


def _night_counter(uuid, value_column, start_val=0, step=10):
    # Hourly readings 22:00 Jan 1 -> 05:00 Jan 2 (the night shift, straddling
    # midnight) plus a couple of day-shift readings on Jan 2.
    times = pd.date_range("2024-01-01 22:00:00", "2024-01-02 09:00:00", freq="1h")
    return pd.DataFrame(
        {
            "uuid": uuid,
            "systime": times,
            value_column: [start_val + step * i for i in range(len(times))],
        }
    )


def test_scrap_by_shift_production_day_merges_night():
    df = _night_counter("scrap", "value_double")
    sc = ScrapTracking(df)
    naive = sc.scrap_by_shift("scrap", production_day=False)
    fixed = sc.scrap_by_shift("scrap", production_day=True)

    naive_night = naive[naive["shift"] == "shift_3"]
    fixed_night = fixed[
        (fixed["shift"] == "shift_3")
        & (fixed["date"] == pd.Timestamp("2024-01-01").date())
    ]
    # Naive splits the night shift across Jan 1 / Jan 2; production_day makes it
    # one contiguous shift_3 group anchored on Jan 1.
    assert len(naive_night) >= 2
    assert len(fixed_night) == 1


def test_nok_by_shift_production_day():
    ok = _night_counter("ok", "value_integer", start_val=0, step=100)
    nok = _night_counter("nok", "value_integer", start_val=0, step=5)
    df = pd.concat([ok, nok], ignore_index=True)
    qt = QualityTracking(df)
    fixed = qt.nok_by_shift("ok", "nok", production_day=True)
    night = fixed[
        (fixed["shift"] == "shift_3")
        & (fixed["date"] == pd.Timestamp("2024-01-01").date())
    ]
    assert len(night) == 1


def test_night_shift_is_one_row_by_default_everywhere():
    """Default production_day: one night shift (22:00-06:00) is one row whose
    start/end are that shift's real window."""
    from ts_shape.events.production.downtime_tracking import DowntimeTracking

    t = pd.date_range("2024-01-01 22:00", "2024-01-02 05:30", freq="30min")
    df = pd.DataFrame({"systime": t, "uuid": "state", "value_string": "Running"})
    out = DowntimeTracking(df).downtime_by_shift("state")
    night = out[out["shift"] == "shift_3"]
    assert len(night) == 1
    assert night["start"].iloc[0] == pd.Timestamp("2024-01-01 22:00")
    assert night["end"].iloc[0] == pd.Timestamp("2024-01-02 06:00")
    split = DowntimeTracking(df).downtime_by_shift("state", production_day=False)
    assert len(split[split["shift"] == "shift_3"]) == 2
