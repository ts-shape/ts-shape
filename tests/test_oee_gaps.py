"""OEE-related gap fixes: handle_resets threading, max_gap, availability proxy."""

import pandas as pd  # type: ignore

from ts_shape.events.production.line_throughput import LineThroughputEvents
from ts_shape.events.production.oee_calculator import OEECalculator
from ts_shape.events.production.runtime_accounting import RuntimeAccountingEvents


def _oee_frame(counter_values):
    """One day: run-state always True (hourly), a part counter, no rejects."""
    times = pd.date_range("2024-01-01 00:00:00", periods=len(counter_values), freq="1h")
    state = pd.DataFrame({"uuid": "state", "systime": times, "value_bool": True})
    counter = pd.DataFrame(
        {"uuid": "counter", "systime": times, "value_integer": counter_values}
    )
    return pd.concat([state, counter], ignore_index=True)


def test_calculate_oee_threads_handle_resets():
    # Counter resets mid-day: 0,100,200,0,100 -> true output 300, naive 100.
    df = _oee_frame([0, 100, 200, 0, 100])
    oee = OEECalculator(df)
    naive = oee.calculate_oee("state", "counter", ideal_cycle_time=30.0)
    fixed = oee.calculate_oee(
        "state", "counter", ideal_cycle_time=30.0, handle_resets=True
    )
    # Reset handling recovers the lost production -> higher performance -> higher OEE.
    assert fixed["performance_pct"].iloc[0] > naive["performance_pct"].iloc[0]
    assert fixed["oee_pct"].iloc[0] > naive["oee_pct"].iloc[0]


def test_calculate_availability_max_gap_excludes_outage():
    # Running samples, then a 6h gap, then running again -- all same day.
    times = [
        pd.Timestamp("2024-01-01 08:00:00"),
        pd.Timestamp("2024-01-01 08:05:00"),
        pd.Timestamp("2024-01-01 14:00:00"),  # 6h gap while "running"
        pd.Timestamp("2024-01-01 14:05:00"),
    ]
    df = pd.DataFrame(
        {"uuid": "state", "systime": times, "value_bool": [True, True, True, True]}
    )
    oee = OEECalculator(df)
    naive = oee.calculate_availability("state")
    capped = oee.calculate_availability("state", max_gap="15min")
    # The 6h gap is booked as run time by default; max_gap excludes it.
    assert naive["run_seconds"].iloc[0] > capped["run_seconds"].iloc[0]
    assert capped["run_seconds"].iloc[0] < 15 * 60


def test_throughput_oee_availability_threshold_now_honored():
    # Windows with widely varying counts; target_rate high so some windows are
    # below the availability threshold.
    times = pd.date_range("2024-01-01 00:00:00", periods=200, freq="1min")
    # Counter climbs fast in the first hour (~1/min), slowly afterward (~1/5min).
    values = list(range(0, 100)) + [100 + i // 5 for i in range(100)]
    df = pd.DataFrame({"uuid": "c", "systime": times, "value_integer": values})
    lt = LineThroughputEvents(df)
    lenient = lt.throughput_oee("c", window="1h", availability_threshold=0.0)
    strict = lt.throughput_oee("c", window="1h", availability_threshold=0.95)
    # Under a strict threshold, low-output windows are no longer "available",
    # so total availability is not greater than the lenient case.
    assert strict["availability"].sum() <= lenient["availability"].sum()


def test_runtime_summary_max_gap():
    times = [
        pd.Timestamp("2024-01-01 08:00:00"),
        pd.Timestamp("2024-01-01 08:05:00"),
        pd.Timestamp("2024-01-01 20:00:00"),  # 12h gap while running
        pd.Timestamp("2024-01-01 20:05:00"),
    ]
    df = pd.DataFrame(
        {"uuid": "run", "systime": times, "value_bool": [True, True, True, True]}
    )
    rt = RuntimeAccountingEvents(df, run_uuid="run")
    naive = rt.runtime_summary()
    capped = rt.runtime_summary(max_gap="15min")
    assert naive["run_seconds"].iloc[0] > capped["run_seconds"].iloc[0]
    assert capped["run_seconds"].iloc[0] < 15 * 60
