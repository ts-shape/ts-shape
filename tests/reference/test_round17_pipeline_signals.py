"""Round 17 (PR I, part b): pipeline routing and isolation, signal inference.

Pipeline: ``init__`` / ``method__`` kwarg prefixes, detector identity and
build-time checks for class/static-method steps, plain callables that take
their DataFrame as a keyword, private input copies per step, and
``run_stream`` feeding each detect step the data as transformed by the steps
before it. Signals: robust sampled/change-encoded rule, slow counters,
calendar windows that start at the period start, intervals that do not hold
the last value across an outage, and the ``uuid_column`` of the primitives.

Each test states the definition it checks in its docstring; the expected
values follow from that definition, not from the implementation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ts_shape import Pipeline
from ts_shape.events.production.long_downtime_events import LongDowntimeEvents
from ts_shape.events.production.machine_state import MachineStateEvents
from ts_shape.events.production.period_summary import PeriodSummaryEvents
from ts_shape.profiling import profile
from ts_shape.signals import infer_spec
from ts_shape.signals.primitives import intervals, signal_series, windows
from ts_shape.signals.spec import SignalSpec

T0 = pd.Timestamp("2024-01-01")


def _machine() -> pd.DataFrame:
    """48 hourly rows: run state ``m:run`` and counter ``m:count``.

    The machine is stopped at hours 5-9 and 30-34 (two 5 h downtimes); the
    counter reads 0, 1, ..., 47.
    """
    t = pd.date_range(T0, periods=48, freq="h")
    run = np.ones(48, dtype=bool)
    run[5:10] = False
    run[30:35] = False
    return pd.concat(
        [
            pd.DataFrame({"systime": t, "uuid": "m:run", "value_bool": run}),
            pd.DataFrame(
                {
                    "systime": t,
                    "uuid": "m:count",
                    "value_integer": pd.array(range(48), dtype="Int64"),
                }
            ),
        ],
        ignore_index=True,
    )


# --- 1. init__ / method__ prefixes ----------------------------------------


def _between_gaps(**kwargs) -> Pipeline:
    return Pipeline("between").detect(
        LongDowntimeEvents,
        "count_events_between_gaps",
        name="w",
        run_state_uuid="m:run",
        production_df="$input",
        production_uuid="m:count",
        aggregation="sum",
        **kwargs,
    )


def test_method_prefix_sends_value_column_to_the_method_only():
    """The downtimes close at 10:00 and open at 30:00 (day 2, 06:00).

    The one window between them holds the counter readings strictly inside
    (10:00, 30:00): hours 11..29, whose values sum to (11 + 29) * 19 / 2 =
    380. ``method__value_column`` reaches the method only, so the constructor
    keeps reading the run state from ``value_bool``.
    """
    df = _machine()
    out = _between_gaps(method__value_column="value_integer").run(df).events["w"]
    direct = LongDowntimeEvents(df, "m:run").count_events_between_gaps(
        df, "m:count", value_column="value_integer", aggregation="sum"
    )
    pd.testing.assert_frame_equal(out, direct)
    assert len(out) == 1
    assert out["start"].iloc[0] == T0 + pd.Timedelta(hours=10)
    assert out["end"].iloc[0] == T0 + pd.Timedelta(hours=30)
    assert out["event_count"].iloc[0] == 380


def test_unprefixed_shared_name_still_goes_to_both():
    """Kept default: a flat name both sides accept goes to both.

    The constructor then reads the run state from ``value_integer`` (no
    values), so no downtime is found and there is no window.
    """
    out = _between_gaps(value_column="value_integer").run(_machine()).events["w"]
    assert len(out) == 0


def test_init_prefix_and_validation():
    """``init__`` reaches the constructor only; prefixed names are checked."""
    df = _machine()
    out = (
        _between_gaps(init__value_column="value_bool", value_column="value_integer")
        .run(df)
        .events["w"]
    )
    assert out["event_count"].tolist() == [380]
    with pytest.raises(ValueError, match="has no parameter 'bogus'"):
        _between_gaps(method__bogus=1)
    with pytest.raises(ValueError, match="has no parameter 'aggregation'"):
        _between_gaps(init__aggregation="sum")


# --- 2. class/static-method detect steps ----------------------------------


def _daily() -> pd.DataFrame:
    """14 days from Monday 2024-01-01, ``parts`` = 1 each day."""
    return pd.DataFrame(
        {"date": pd.date_range(T0, periods=14, freq="D"), "parts": [1.0] * 14}
    )


def test_static_step_has_detector_identity():
    """A classmethod step is named ``Class.method`` and reaches the event log."""
    pipe = Pipeline("s").detect(
        PeriodSummaryEvents, "from_daily_data", name="w", window="W"
    )
    assert pipe._steps[0].detector_id == "PeriodSummaryEvents.from_daily_data"
    result = pipe.run(_daily())
    assert len(result.events["w"]) == 2  # two Monday-to-Sunday weeks
    logs = result.to_event_log(concat=False)
    assert list(logs) == ["w"]


def test_static_step_rejects_unknown_kwarg_at_build_time():
    """The check runs against the bound method's own signature."""
    with pytest.raises(ValueError, match="unknown argument"):
        Pipeline("s").detect(PeriodSummaryEvents, "from_daily_data", name="w", bogus=1)
    with pytest.raises(ValueError, match="no constructor"):
        Pipeline("s").detect(
            PeriodSummaryEvents, "from_daily_data", name="w", init__window="W"
        )


# --- 3. plain callables taking their frame as a keyword -------------------


def test_callable_takes_dataframe_from_keyword():
    """``dataframe="$prev"`` replaces the positional frame: 96 rows -> 48."""

    def half(dataframe, other):
        return dataframe.iloc[: len(other) // 2]

    out = Pipeline("c").transform(half, dataframe="$prev", other="$input")
    assert len(out.run(_machine()).data) == 48


# --- 4. steps cannot change each other's input ----------------------------


def _true_count(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"n": [int(frame["value_bool"].fillna(False).sum())]})


def test_detect_step_mutation_does_not_reach_later_steps():
    """The run state is True for 48 - 10 = 38 hours, whatever step 1 does."""

    def clobber(dataframe):
        dataframe["value_bool"] = False
        return dataframe.iloc[:0]

    df = _machine()
    before = df.copy()
    out = Pipeline("m").detect(clobber, name="a").detect(_true_count, name="b")
    assert out.run(df).events["b"]["n"].tolist() == [38]
    pd.testing.assert_frame_equal(df, before)


def test_input_sentinel_is_a_fresh_copy_per_step():
    """A transform that changes ``$input`` in place leaves later ``$input`` intact."""

    def clobber_input(dataframe, other):
        other["value_bool"] = False
        return dataframe

    def count_input(dataframe, original):
        return _true_count(original)

    out = (
        Pipeline("m")
        .transform(clobber_input, other="$input")
        .detect(count_input, name="b", original="$input")
    )
    assert out.run(_machine()).events["b"]["n"].tolist() == [38]


# --- 5. run_stream: each detect step sees only earlier transforms ---------


def test_run_stream_detect_sees_only_earlier_transforms():
    """``before`` sees the real run state: run 0-5, idle 5-10, run 10-30,
    idle 30-35, run 35-47 = 5 intervals. After the transform sets every
    value to True, ``after`` sees one run interval."""
    st = _machine().query("uuid == 'm:run'").reset_index(drop=True)

    def all_true(dataframe):
        out = dataframe.copy()
        out["value_bool"] = True
        return out

    pipe = (
        Pipeline("s")
        .detect(
            MachineStateEvents, "detect_run_idle", name="before", run_state_uuid="m:run"
        )
        .transform(all_true)
        .detect(
            MachineStateEvents, "detect_run_idle", name="after", run_state_uuid="m:run"
        )
    )
    full = pipe.run(st)
    streamed = pipe.run_stream([st.iloc[:20], st.iloc[20:]], chunk="12h")
    assert len(streamed.events["before"]) == 5
    assert len(streamed.events["after"]) == 1
    for name in ("before", "after"):
        pd.testing.assert_frame_equal(
            streamed.events[name], full.events[name].reset_index(drop=True)
        )


# --- 6. robust sampled / change-encoded rule ------------------------------


def _one_gap() -> pd.DataFrame:
    """60 one-minute samples, a 31-minute interval, 30 more samples."""
    first = pd.date_range(T0, periods=60, freq="min")
    second = pd.date_range(first[-1] + pd.Timedelta("31min"), periods=30, freq="min")
    t = first.append(second)
    return pd.DataFrame(
        {"systime": t, "uuid": "x", "value_double": np.sin(np.arange(90))}
    )


def test_one_long_gap_keeps_a_signal_sampled():
    """88 of 89 intervals are 1 min (99 % regular, >= 80 %): sampled.

    max_gap = 5 x 1 min; the 31-minute interval (1860 s) is one gap.
    """
    df = _one_gap()
    spec = infer_spec(df, "x")
    assert spec.encoding == "sampled"
    assert spec.max_gap == pd.Timedelta("5min")
    row = profile(df).to_frame().iloc[0]
    assert row["gap_count"] == 1
    assert row["gap_seconds"] == 1860.0


def test_irregular_signal_is_change_encoded():
    """Intervals cycle 1, 2, 5, 13, 30 min: at most 1 in 5 is within 10 %
    of the median, under 80 %: change-encoded."""
    steps = np.tile([1, 2, 5, 13, 30], 10)
    t = T0 + pd.to_timedelta(np.r_[0, np.cumsum(steps)], unit="min")
    df = pd.DataFrame(
        {"systime": t, "uuid": "x", "value_double": np.linspace(0, 1, len(t))}
    )
    assert infer_spec(df, "x").encoding == "change_encoded"


# --- 7. slow counters -----------------------------------------------------


def _integers(values) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "systime": pd.date_range(T0, periods=len(values), freq="min"),
            "uuid": "c",
            "value_integer": values,
        }
    )


def test_slow_counter_is_a_counter():
    """``arange(1000) // 10`` never decreases and rises 99 times (on 10 % of
    readings): a counter."""
    assert infer_spec(_integers(np.arange(1000) // 10), "c").kind == "counter"


def test_constant_and_two_level_are_not_counters():
    """A constant never rises; an on/off signal has two values -> state."""
    assert infer_spec(_integers(np.full(1000, 5)), "c").kind == "state"
    toggles = np.zeros(1000, dtype=int)
    for k in range(3):
        toggles[200 * k + 100 : 200 * k + 150] = 1  # rises 3 times, falls 3 times
    assert infer_spec(_integers(toggles), "c").kind == "state"


# --- 8. calendar windows start at the period start ------------------------


def test_weekly_windows_cover_start():
    """2024-01-01 is a Monday: weeks [Jan 1, Jan 8) and [Jan 8, Jan 15)."""
    w = windows("2024-01-01", "2024-01-15", "1W")
    assert w["start"].tolist() == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-08"),
    ]
    assert w["end"].tolist() == [pd.Timestamp("2024-01-08"), pd.Timestamp("2024-01-15")]
    assert w["duration_seconds"].tolist() == [7 * 86400.0] * 2


def test_month_end_windows_cover_start():
    """``"ME"``: January (31 d) and February 2024 (29 d), from the 1st."""
    w = windows("2024-01-01", "2024-03-01", "ME")
    assert w["start"].tolist() == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-02-01"),
    ]
    assert w["end"].tolist() == [pd.Timestamp("2024-02-01"), pd.Timestamp("2024-03-01")]
    assert w["duration_seconds"].tolist() == [31 * 86400.0, 29 * 86400.0]


def test_mid_week_start_is_inside_the_first_window():
    """Wednesday 2024-01-03 lies in the week [Mon Jan 1, Mon Jan 8)."""
    w = windows("2024-01-03", "2024-01-10", "W")
    assert w["start"].iloc[0] == pd.Timestamp("2024-01-01")
    assert w["start"].iloc[0] <= pd.Timestamp("2024-01-03") < w["end"].iloc[0]


# --- 9. intervals do not hold across an outage ----------------------------


SPEC = SignalSpec(
    uuid="x",
    kind="state",
    encoding="sampled",
    value_column="value_double",
    nominal_period=pd.Timedelta("1min"),
    max_gap=pd.Timedelta("5min"),
)


def _three() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "systime": pd.date_range(T0, periods=3, freq="min"),
            "uuid": "x",
            "value_double": [1.0, 1.0, 2.0],
        }
    )


def test_last_interval_ends_at_last_sample_after_long_silence():
    """Samples at 00:00, 00:01, 00:02; data_end 01:00 is 58 min > 5 min after
    the last sample, so value 2.0 is not held: [00:02, 00:02], 0 s."""
    out = intervals(_three(), SPEC, data_end=T0 + pd.Timedelta("1h"))
    assert out["end"].tolist() == [T0 + pd.Timedelta("2min")] * 2
    assert out["duration_seconds"].tolist() == [120.0, 0.0]


def test_last_interval_holds_to_data_end_within_max_gap():
    """data_end 00:05 is 3 min <= 5 min after 00:02: held, 180 s."""
    out = intervals(_three(), SPEC, data_end=T0 + pd.Timedelta("5min"))
    assert out["end"].iloc[-1] == T0 + pd.Timedelta("5min")
    assert out["duration_seconds"].tolist() == [120.0, 180.0]


# --- 10. uuid_column ------------------------------------------------------


def _tagged() -> pd.DataFrame:
    t = pd.date_range(T0, periods=3, freq="min")
    return pd.DataFrame(
        {
            "systime": t.append(t),
            "tag": ["x"] * 3 + ["y"] * 3,
            "value_double": [1.0, 1.0, 2.0, 10.0, 20.0, 30.0],
        }
    )


def test_uuid_column_filters_the_signal():
    """``uuid_column="tag"`` keeps the 3 rows of x: intervals 1.0 then 2.0."""
    out = intervals(_tagged(), SPEC, uuid_column="tag")
    assert out["state"].tolist() == [1.0, 2.0]
    assert out["sample_count"].tolist() == [2, 1]


def test_missing_uuid_column_with_several_signals_raises():
    """No ``uuid`` column and 3 of 6 timestamps repeat: several signals."""
    with pytest.raises(ValueError, match="looks like several signals"):
        signal_series(_tagged(), SPEC)


def test_single_signal_without_uuid_column_still_works():
    """One signal, no id column: all rows are the signal."""
    s = signal_series(_three().drop(columns="uuid"), SPEC)
    assert s.tolist() == [1.0, 1.0, 2.0]
