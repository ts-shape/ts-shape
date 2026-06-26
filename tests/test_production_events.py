import pandas as pd  # type: ignore

from ts_shape.events.production import (
    ChangeoverEvents,
    FlowConstraintEvents,
    LineThroughputEvents,
    MachineStateEvents,
)


def test_machine_state_intervals_and_transitions():
    t = pd.date_range("2024-01-01 00:00:00", periods=6, freq="30s")
    run = [False, False, True, True, False, False]
    df = pd.DataFrame(
        {
            "uuid": ["run"] * len(t),
            "systime": t,
            "value_bool": run,
            "is_delta": [True] * len(t),
        }
    )

    mse = MachineStateEvents(df, run_state_uuid="run")
    intervals = mse.detect_run_idle(min_duration="30s")
    assert not intervals.empty
    assert set(intervals["state"].unique()) == {"run", "idle"}

    transitions = mse.transition_events()
    assert not transitions.empty
    assert set(transitions["transition"].unique()) == {"idle_to_run", "run_to_idle"}


def test_machine_state_integer_range():
    t = pd.date_range("2024-01-01 00:00:00", periods=6, freq="30s")
    rpm = [0, 50, 150, 200, 80, 0]
    df = pd.DataFrame(
        {
            "uuid": ["rpm"] * len(t),
            "systime": t,
            "value_integer": rpm,
            "is_delta": [True] * len(t),
        }
    )

    # running when rpm >= 100
    mse = MachineStateEvents(
        df, run_state_uuid="rpm", value_column="value_integer", value_range=(100, None)
    )
    intervals = mse.detect_run_idle()
    assert not intervals.empty
    run_intervals = intervals[intervals["state"] == "run"]
    assert not run_intervals.empty
    # only the two points at rpm=150 and rpm=200 are "run"
    assert all(run_intervals["duration_seconds"] >= 0)

    transitions = mse.transition_events()
    assert not transitions.empty
    assert "idle_to_run" in transitions["transition"].values
    assert "run_to_idle" in transitions["transition"].values


def test_machine_state_double_range():
    t = pd.date_range("2024-01-01 00:00:00", periods=6, freq="30s")
    current = [0.1, 0.3, 1.5, 2.0, 0.4, 0.2]
    df = pd.DataFrame(
        {
            "uuid": ["cur"] * len(t),
            "systime": t,
            "value_double": current,
            "is_delta": [True] * len(t),
        }
    )

    # running when 0.5 <= current <= 3.0
    mse = MachineStateEvents(
        df, run_state_uuid="cur", value_column="value_double", value_range=(0.5, 3.0)
    )
    intervals = mse.detect_run_idle()
    assert not intervals.empty
    states = set(intervals["state"].unique())
    assert "run" in states
    assert "idle" in states


def test_machine_state_open_upper_range():
    t = pd.date_range("2024-01-01 00:00:00", periods=4, freq="30s")
    df = pd.DataFrame(
        {
            "uuid": ["s"] * len(t),
            "systime": t,
            "value_double": [0.0, 5.0, 10.0, 0.0],
            "is_delta": [True] * len(t),
        }
    )

    # running when value >= 5.0 (no upper bound)
    mse = MachineStateEvents(
        df, run_state_uuid="s", value_column="value_double", value_range=(5.0, None)
    )
    intervals = mse.detect_run_idle()
    run_intervals = intervals[intervals["state"] == "run"]
    idle_intervals = intervals[intervals["state"] == "idle"]
    assert not run_intervals.empty
    assert not idle_intervals.empty


def test_machine_state_range_backward_compat():
    # value_range=None must behave identically to the original boolean path
    t = pd.date_range("2024-01-01 00:00:00", periods=4, freq="30s")
    df = pd.DataFrame(
        {
            "uuid": ["b"] * len(t),
            "systime": t,
            "value_bool": [False, True, True, False],
            "is_delta": [True] * len(t),
        }
    )
    mse = MachineStateEvents(df, run_state_uuid="b", value_range=None)
    intervals = mse.detect_run_idle()
    assert set(intervals["state"].unique()) == {"run", "idle"}


def test_line_throughput_count_and_takt():
    # Counter increments every minute by 5 parts
    t = pd.date_range("2024-01-01 00:00:00", periods=6, freq="1min")
    counter = [0, 5, 10, 15, 20, 25]
    df = pd.DataFrame(
        {
            "uuid": ["cnt"] * len(t),
            "systime": t,
            "value_integer": counter,
            "is_delta": [True] * len(t),
        }
    )
    lte = LineThroughputEvents(df)
    counts = lte.count_parts(counter_uuid="cnt", window="2min")
    assert not counts.empty
    assert "count" in counts.columns

    # Use boolean trigger to define cycle boundaries, takt=90s
    tb = pd.DataFrame(
        {
            "uuid": ["cyc"] * len(t),
            "systime": t,
            "value_bool": [True, False, True, False, True, False],
            "is_delta": [True] * len(t),
        }
    )
    lte2 = LineThroughputEvents(tb)
    takt = lte2.takt_adherence(
        cycle_uuid="cyc", value_column="value_bool", takt_time="90s", min_violation="0s"
    )
    assert not takt.empty
    assert "cycle_time_seconds" in takt.columns


def test_changeover_detect_and_window_stable_band():
    # Product changes A->B at t=60s, metric stabilizes after 2 minutes
    t = pd.date_range("2024-01-01 00:00:00", periods=7, freq="1min")
    prod = ["A", "A", "B", "B", "B", "B", "B"]
    df_prod = pd.DataFrame(
        {
            "uuid": ["prod"] * len(t),
            "systime": t,
            "value_string": prod,
            "is_delta": [True] * len(t),
        }
    )
    metric_vals = [
        10.0,
        10.2,
        12.5,
        12.2,
        12.1,
        12.05,
        12.0,
    ]  # settles near 12 ±0.2 for ≥2m
    df_m = pd.DataFrame(
        {
            "uuid": ["m1"] * len(t),
            "systime": t,
            "value_double": metric_vals,
            "is_delta": [True] * len(t),
        }
    )
    df = pd.concat([df_prod, df_m], ignore_index=True)

    co = ChangeoverEvents(df)
    changes = co.detect_changeover(
        product_uuid="prod", value_column="value_string", min_hold="0s"
    )
    assert not changes.empty
    win = co.changeover_window(
        product_uuid="prod",
        value_column="value_string",
        until="stable_band",
        config={
            "metrics": [
                {
                    "uuid": "m1",
                    "value_column": "value_double",
                    "band": 0.25,
                    "hold": "2m",
                }
            ]
        },
    )
    assert not win.empty
    assert (win["end"] >= win["start"]).all()


def test_flow_blocked_and_starved():
    # Create data where upstream runs while downstream is idle for multiple timestamps
    t = pd.date_range("2024-01-01 00:00:00", periods=8, freq="30s")
    up = [False, True, True, True, True, False, False, False]
    dn = [False, False, False, True, True, True, True, False]
    df_up = pd.DataFrame(
        {
            "uuid": ["up"] * len(t),
            "systime": t,
            "value_bool": up,
            "is_delta": [True] * len(t),
        }
    )
    df_dn = pd.DataFrame(
        {
            "uuid": ["dn"] * len(t),
            "systime": t,
            "value_bool": dn,
            "is_delta": [True] * len(t),
        }
    )
    df = pd.concat([df_up, df_dn], ignore_index=True)

    fce = FlowConstraintEvents(df)
    # Blocked: upstream True while downstream False at t1,t2 (duration=30s)
    blocked = fce.blocked_events(
        roles={"upstream_run": "up", "downstream_run": "dn"},
        tolerance="1s",
        min_duration="30s",
    )
    assert not blocked.empty
    assert (blocked["type"] == "blocked").all()

    # Starved: downstream True while upstream False at t5,t6 (duration=30s)
    starved = fce.starved_events(
        roles={"upstream_run": "up", "downstream_run": "dn"},
        tolerance="1s",
        min_duration="30s",
    )
    assert isinstance(starved, pd.DataFrame)
