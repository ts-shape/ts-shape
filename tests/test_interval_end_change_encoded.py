"""End-to-end tests: interval_end='next_change' fixes change-encoded signals.

Change-encoded (is_delta=True) signals log one row per transition. With
'last_sample' interval closing, every interval collapses to zero duration and
downtime/micro-stop/duty-cycle analyses report nothing. The default
'next_change' closes each interval at the following transition.
"""

import pandas as pd

from ts_shape.events.production.duty_cycle import DutyCycleEvents
from ts_shape.events.production.long_downtime_events import LongDowntimeEvents
from ts_shape.events.production.machine_state import MachineStateEvents
from ts_shape.events.production.micro_stop_detection import MicroStopEvents


def _state_frame(uuid, pairs):
    return pd.DataFrame(
        [
            {
                "uuid": uuid,
                "systime": pd.Timestamp(t),
                "value_bool": v,
                "is_delta": True,
            }
            for t, v in pairs
        ]
    )


# ---------------------------------------------------------------------------
# machine_state
# ---------------------------------------------------------------------------
def test_machine_state_change_encoded():
    df = _state_frame(
        "sig",
        [
            ("2024-01-01 08:00:00", True),
            ("2024-01-01 08:10:00", False),
            ("2024-01-01 08:12:00", True),
            ("2024-01-01 08:30:00", False),
        ],
    )
    ms = MachineStateEvents(df, "sig")

    # last_sample: single-row intervals collapse to zero duration.
    naive = ms.detect_run_idle(interval_end="last_sample")
    assert (naive["duration_seconds"] == 0).all()

    # Default (next_change) measures each interval to the next transition.
    fixed = ms.detect_run_idle()
    durations = dict(zip(fixed["state"], fixed["duration_seconds"], strict=False))
    # First run 08:00->08:10 = 600s; idle 08:10->08:12 = 120s.
    assert fixed.iloc[0]["duration_seconds"] == 600.0
    assert fixed.iloc[1]["duration_seconds"] == 120.0
    assert durations  # non-empty

    # min_duration now actually keeps real intervals under next_change.
    kept = ms.detect_run_idle(min_duration="60s")
    assert not kept.empty


# ---------------------------------------------------------------------------
# micro_stop_detection
# ---------------------------------------------------------------------------
def test_micro_stop_change_encoded():
    df = _state_frame(
        "sig",
        [
            ("2024-01-01 09:00:00", True),
            ("2024-01-01 09:01:00", False),  # 20s micro stop
            ("2024-01-01 09:01:20", True),
            ("2024-01-01 09:03:00", False),  # 10s micro stop
            ("2024-01-01 09:03:10", True),
        ],
    )
    naive = MicroStopEvents(df, "sig", interval_end="last_sample")
    assert naive.micro_stop_impact()["total_micro_stop_time"].sum() == 0

    fixed = MicroStopEvents(df, "sig")
    impact = fixed.micro_stop_impact()
    assert impact["total_micro_stop_time"].sum() == 30.0  # 20s + 10s


# ---------------------------------------------------------------------------
# long_downtime_events
# ---------------------------------------------------------------------------
def test_long_downtime_change_encoded():
    df = _state_frame(
        "sig",
        [
            ("2024-01-01 09:00:00", True),
            ("2024-01-01 09:30:00", False),  # 4.5h stop
            ("2024-01-01 14:00:00", True),
        ],
    )
    ev = LongDowntimeEvents(df, "sig")
    # last_sample misses the stop entirely.
    assert ev.detect_long_downtime(min_gap="3h", interval_end="last_sample").empty

    result = ev.detect_long_downtime(min_gap="3h")
    assert len(result) == 1
    assert result.iloc[0]["duration_seconds"] == 4.5 * 3600


# ---------------------------------------------------------------------------
# duty_cycle
# ---------------------------------------------------------------------------
def test_duty_cycle_change_encoded():
    df = _state_frame(
        "sig",
        [
            ("2024-01-01 09:00:00", True),
            ("2024-01-01 09:10:00", False),
            ("2024-01-01 09:15:00", True),
        ],
    )
    naive = DutyCycleEvents(df, "sig", interval_end="last_sample")
    assert (naive.on_off_intervals()["duration"] == pd.Timedelta(0)).all()

    fixed = DutyCycleEvents(df, "sig")
    intervals = fixed.on_off_intervals()
    # on 09:00->09:10 = 600s, off 09:10->09:15 = 300s.
    assert intervals.iloc[0]["duration"] == pd.Timedelta(seconds=600)
    assert intervals.iloc[1]["duration"] == pd.Timedelta(seconds=300)
