"""Proof of concept: detectors take their defaults from SignalSpecs."""

import pandas as pd

from ts_shape.events.production.downtime_tracking import DowntimeEvents
from ts_shape.events.production.machine_state import MachineStateEvents
from ts_shape.signals import infer_specs


def _change_encoded_run():
    t = pd.to_datetime(
        ["2024-01-01 08:00", "2024-01-01 08:10", "2024-01-01 09:00", "2024-01-01 12:00"]
    )
    return pd.DataFrame(
        {"systime": t, "uuid": "run", "value_bool": [True, False, True, False]}
    )


def test_machine_state_without_specs_closes_at_next_change():
    out = MachineStateEvents(_change_encoded_run(), "run").detect_run_idle()
    # Without specs the default interval_end="next_change" applies: every
    # interval but the final open one has a real duration.
    assert (out["duration_seconds"].iloc[:-1] > 0).all()
    naive = MachineStateEvents(_change_encoded_run(), "run").detect_run_idle(
        interval_end="last_sample"
    )
    assert (naive["duration_seconds"] == 0).all()


def test_machine_state_with_specs_closes_at_next_change_and_caps_gaps():
    df = _change_encoded_run()
    specs = infer_specs(df, overrides={"run": {"max_gap": "1h"}})
    assert specs["run"].encoding == "change_encoded"
    out = MachineStateEvents(df, "run", specs=specs).detect_run_idle()
    # 08:00-08:10 run, 08:10-09:00 idle; the 3 h hold after 09:00 exceeds
    # max_gap and is "no data" (0 s), the final row has no successor.
    assert list(out["state"]) == ["run", "idle", "run", "idle"]
    assert list(out["duration_seconds"]) == [600.0, 3000.0, 0.0, 0.0]
    # An explicit argument still wins over the spec.
    legacy = MachineStateEvents(df, "run", specs=specs).detect_run_idle(
        interval_end="last_sample"
    )
    assert (legacy["duration_seconds"] == 0).all()


def test_state_quality_metrics_follows_spec_like_detect_run_idle():
    df = _change_encoded_run()
    specs = infer_specs(df, overrides={"run": {"max_gap": "1h"}})
    ev = MachineStateEvents(df, "run", specs=specs)
    metrics = ev.state_quality_metrics().iloc[0]
    # Same spec-derived intervals as detect_run_idle(): run 600 s, idle 3000 s.
    assert metrics["avg_idle_duration_seconds"] == 1500.0
    assert metrics["run_idle_factor"] == 600.0 / 3000.0


def test_state_quality_metrics_run_idle_ratio_nan_without_idle_time():
    import math

    t = pd.date_range("2024-01-01", periods=4, freq="1min")
    df = pd.DataFrame({"systime": t, "uuid": "run", "value_bool": [True] * 4})
    metrics = MachineStateEvents(df, "run").state_quality_metrics().iloc[0]
    assert math.isnan(metrics["run_idle_factor"])
    empty = MachineStateEvents(df.iloc[:0], "run").state_quality_metrics().iloc[0]
    assert math.isnan(empty["run_idle_factor"])


def test_downtime_by_shift_takes_max_gap_from_spec():
    t = pd.to_datetime(["2024-01-01 07:00", "2024-01-01 07:30", "2024-01-01 11:00"])
    df = pd.DataFrame(
        {"systime": t, "uuid": "st", "value_string": ["Running", "Running", "Stopped"]}
    )
    plain = DowntimeEvents(df).downtime_by_shift("st")
    specs = infer_specs(df, overrides={"st": {"max_gap": "1h"}})
    capped = DowntimeEvents(df, specs=specs).downtime_by_shift("st")
    # Without a cap the 3.5 h outage after 07:30 counts as uptime.
    assert plain["uptime_minutes"].iloc[0] > capped["uptime_minutes"].iloc[0]
    assert capped["uptime_minutes"].iloc[0] == 30.0
