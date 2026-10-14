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


def test_machine_state_without_specs_is_unchanged():
    out = MachineStateEvents(_change_encoded_run(), "run").detect_run_idle()
    assert (out["duration_seconds"] == 0).all()  # legacy: zero-length intervals


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
