"""Chunked processing (:mod:`ts_shape.stream`): edge cases, sources, pipeline."""

from __future__ import annotations

import gc
import tracemalloc
import warnings

import numpy as np
import pandas as pd
import pytest

import ts_shape
from ts_shape import (
    DowntimeEvents,
    DutyCycleEvents,
    LineThroughputEvents,
    LongDowntimeEvents,
    MachineStateEvents,
    MicroStopEvents,
    NotChunkableError,
    OutlierDetectionEvents,
    PartProductionEvents,
    Pipeline,
    RuntimeAccountingEvents,
    ThresholdMonitoringEvents,
    iter_chunked,
    run_chunked,
    time_range_source,
)
from ts_shape.errors import ColumnNotFoundError, DataQualityWarning
from ts_shape.stream import SUPPORT

BERLIN = "Europe/Berlin"
SHIFTS = {
    "early": ("06:00", "14:00"),
    "late": ("14:00", "22:00"),
    "night": ("22:00", "06:00"),
}


def _frame(
    start: str,
    end: str,
    signals: dict[str, tuple[str, np.ndarray]],
    *,
    freq: str = "1min",
    tz: str | None = BERLIN,
) -> pd.DataFrame:
    t = pd.date_range(start, end, freq=freq, inclusive="left", tz=tz)
    parts = []
    for uuid, (column, values) in signals.items():
        parts.append(pd.DataFrame({"systime": t, "uuid": uuid, column: values(t)}))
    df = pd.concat(parts, ignore_index=True)
    return df.sort_values(["systime", "uuid"], kind="stable").reset_index(drop=True)


def _days(df: pd.DataFrame) -> list[pd.DataFrame]:
    return [g for _, g in df.groupby(df["systime"].dt.normalize(), sort=True)]


def _same(memory: pd.DataFrame, chunked: pd.DataFrame, keys: list[str]) -> None:
    assert list(memory.columns) == list(chunked.columns)
    a = memory.sort_values(keys, kind="stable").reset_index(drop=True)
    b = chunked.sort_values(keys, kind="stable").reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_dtype=False, rtol=1e-9, atol=1e-9)


def _wall(t: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(t.tz_localize(None) if t.tz is not None else t)


def _run_pattern(t: pd.DatetimeIndex) -> np.ndarray:
    """Run 10:00 day 1 to 10:00 day 3, then short alternating runs."""
    wall = _wall(t)
    state = ((np.arange(len(t)) // 50) % 2).astype(bool)
    long_run = (wall >= pd.Timestamp("2024-03-30 10:00")) & (
        wall < pd.Timestamp("2024-04-01 10:00")
    )
    state[long_run.to_numpy()] = True
    return state


def _noise(t: pd.DatetimeIndex) -> np.ndarray:
    return np.random.default_rng(3).normal(size=len(t))


@pytest.fixture(scope="module")
def plant() -> pd.DataFrame:
    """Four days across the DST change with a run spanning three slices."""
    return _frame(
        "2024-03-29",
        "2024-04-02",
        {"run": ("value_bool", _run_pattern), "temp": ("value_double", _noise)},
    )


# ---------------------------------------------------------------------------
# Edge cases at slice borders
# ---------------------------------------------------------------------------


def test_interval_spanning_three_slices(plant):
    memory = MachineStateEvents(plant, "run").detect_run_idle(min_duration="30h")
    chunked = run_chunked(
        _days(plant),
        MachineStateEvents,
        "detect_run_idle",
        run_state_uuid="run",
        min_duration="30h",
    )
    assert len(memory) == 1
    start, end = memory["start"].iloc[0], memory["end"].iloc[0]
    assert start.day == 30 and end.day == 1  # three slices, one a 23-hour day
    _same(memory, chunked, ["start"])


def test_all_intervals_match(plant):
    memory = MachineStateEvents(plant, "run").detect_run_idle()
    chunked = run_chunked(
        iter(_days(plant)), MachineStateEvents, "detect_run_idle", run_state_uuid="run"
    )
    _same(memory, chunked, ["start"])


def _stop_pattern(t: pd.DatetimeIndex) -> np.ndarray:
    wall = _wall(t)
    state = np.ones(len(t), dtype=bool)
    for lo, hi in (
        ("2024-03-30 23:58", "2024-03-31 00:02"),  # micro stop across midnight
        ("2024-03-31 22:00", "2024-04-01 02:00"),  # long stop across midnight
        ("2024-04-01 08:00", "2024-04-01 08:03"),
    ):
        state[((wall >= pd.Timestamp(lo)) & (wall < pd.Timestamp(hi))).to_numpy()] = (
            False
        )
    return state


@pytest.fixture(scope="module")
def stops() -> pd.DataFrame:
    return _frame(
        "2024-03-30", "2024-04-02", {"run": ("value_bool", _stop_pattern)}, freq="30s"
    )


def test_stop_across_midnight(stops):
    memory = MicroStopEvents(stops, "run").detect_micro_stops(max_duration="5min")
    chunked = run_chunked(
        _days(stops),
        MicroStopEvents,
        "detect_micro_stops",
        run_state_uuid="run",
        max_duration="5min",
    )
    assert len(memory) == 2
    assert memory["start"].dt.date.iloc[0] != memory["end"].dt.date.iloc[0]
    _same(memory, chunked, ["start"])


def test_long_downtime_across_midnight_is_numbered_over_the_run(stops):
    memory = LongDowntimeEvents(stops, "run").detect_long_downtime(min_gap="1h")
    chunked = run_chunked(
        _days(stops),
        LongDowntimeEvents,
        "detect_long_downtime",
        run_state_uuid="run",
        min_gap="1h",
    )
    assert len(memory) == 1 and memory["duration_seconds"].iloc[0] == 4 * 3600
    _same(memory, chunked, ["start"])


def test_downtime_ordinal_continues_across_slices():
    def pattern(t):
        wall = _wall(t)
        state = np.ones(len(t), dtype=bool)
        for day in ("2024-01-01", "2024-01-02", "2024-01-03"):
            lo = pd.Timestamp(f"{day} 03:00")
            state[((wall >= lo) & (wall < lo + pd.Timedelta("2h"))).to_numpy()] = False
        return state

    df = _frame("2024-01-01", "2024-01-04", {"run": ("value_bool", pattern)}, tz=None)
    memory = LongDowntimeEvents(df, "run").detect_long_downtime(min_gap="1h")
    chunked = run_chunked(
        _days(df),
        LongDowntimeEvents,
        "detect_long_downtime",
        run_state_uuid="run",
        min_gap="1h",
    )
    assert chunked["downtime_index"].tolist() == [0, 1, 2]
    _same(memory, chunked, ["start"])


def test_overnight_shift_downtime(stops):
    df = stops.assign(
        value_string=np.where(stops["value_bool"], "Running", "Stopped")
    ).drop(columns="value_bool")
    memory = DowntimeEvents(df, shift_definitions=SHIFTS).downtime_by_shift("run")
    chunked = run_chunked(
        _days(df),
        DowntimeEvents,
        "downtime_by_shift",
        shift_definitions=SHIFTS,
        run_state_uuid="run",
    )
    night = memory[(memory["shift"] == "night") & (memory["downtime_minutes"] > 60)]
    assert len(night) == 1
    assert night["start"].iloc[0].hour == 22
    assert night["end"].iloc[0].hour == 6
    _same(memory, chunked, ["start"])


def test_counter_reset_at_slice_border():
    def counter(t):
        values = np.arange(len(t)) * 2
        border = int(np.flatnonzero(_wall(t) >= pd.Timestamp("2024-03-31"))[0])
        values[border:] = np.arange(len(t) - border) * 2 + 1  # reset at midnight
        return values

    def part(t):
        return np.full(len(t), "A")

    df = _frame(
        "2024-03-30",
        "2024-04-01",
        {"parts": ("value_integer", counter), "part": ("value_string", part)},
    )
    memory = LineThroughputEvents(df).count_parts("parts", window="1h")
    chunked = run_chunked(
        _days(df),
        LineThroughputEvents,
        "count_parts",
        counter_uuid="parts",
        window="1h",
    )
    _same(memory, chunked, ["start"])
    resets = PartProductionEvents(df).detect_resets("part", "parts")
    chunked_resets = run_chunked(
        _days(df),
        PartProductionEvents,
        "detect_resets",
        part_id_uuid="part",
        counter_uuid="parts",
    )
    assert len(resets) == 1
    _same(resets, chunked_resets, ["systime"])


def test_dst_day_slice_lasts_23_hours(plant):
    memory = RuntimeAccountingEvents(plant, "run").runtime_by_window(window="1D")
    chunked = run_chunked(
        _days(plant),
        RuntimeAccountingEvents,
        "runtime_by_window",
        run_state_uuid="run",
        window="1D",
    )
    dst = chunked[chunked["start"] == pd.Timestamp("2024-03-31", tz=BERLIN)]
    assert dst["duration_seconds"].iloc[0] == 23 * 3600
    _same(memory, chunked, ["start"])


def test_naive_timestamps():
    df = _frame(
        "2024-01-01", "2024-01-04", {"run": ("value_bool", _run_pattern)}, tz=None
    )
    memory = MachineStateEvents(df, "run").detect_run_idle()
    chunked = run_chunked(
        _days(df), MachineStateEvents, "detect_run_idle", run_state_uuid="run"
    )
    _same(memory, chunked, ["start"])


def test_twelve_hour_chunks_and_threshold(plant):
    memory = ThresholdMonitoringEvents(plant, "temp").threshold_with_hysteresis(
        1.0, 0.0
    )
    chunked = run_chunked(
        _days(plant),
        ThresholdMonitoringEvents,
        "threshold_with_hysteresis",
        chunk="12h",
        signal_uuid="temp",
        upper_limit=1.0,
        lower_limit=0.0,
    )
    _same(memory, chunked, ["start"])


def test_custom_time_column(plant):
    df = plant.rename(columns={"systime": "ts"})
    memory = MachineStateEvents(df, "run", time_column="ts").detect_run_idle()
    chunked = run_chunked(
        [df],
        MachineStateEvents,
        "detect_run_idle",
        time_column="ts",
        run_state_uuid="run",
    )
    _same(memory, chunked, ["start"])


def test_deprecated_method_name_resolves(plant):
    with pytest.warns(DeprecationWarning):
        memory = DutyCycleEvents(plant, "run").duty_cycle_per_window(window="1h")
    with pytest.warns(DeprecationWarning):
        chunked = run_chunked(
            _days(plant),
            DutyCycleEvents,
            "duty_cycle_per_window",
            signal_uuid="run",
            window="1h",
        )
    _same(memory, chunked, ["start"])


def test_explicit_overlap(plant):
    out = run_chunked(
        _days(plant),
        MachineStateEvents,
        "detect_run_idle",
        overlap="2h",
        run_state_uuid="run",
    )
    _same(MachineStateEvents(plant, "run").detect_run_idle(), out, ["start"])


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


def test_named_frames_and_single_frame(plant):
    expected = MachineStateEvents(plant, "run").detect_run_idle()
    named = ((f"day-{i}.parquet", d) for i, d in enumerate(_days(plant)))
    _same(
        expected,
        run_chunked(named, MachineStateEvents, "detect_run_idle", run_state_uuid="run"),
        ["start"],
    )
    _same(
        expected,
        run_chunked(plant, MachineStateEvents, "detect_run_idle", run_state_uuid="run"),
        ["start"],
    )


def test_frames_split_within_a_slice_and_unsorted(plant):
    pieces = []
    for day in _days(plant):
        half = len(day) // 2
        pieces.extend([day.iloc[half:], day.iloc[:half]])  # unordered inside a slice
    out = run_chunked(
        pieces, MachineStateEvents, "detect_run_idle", run_state_uuid="run"
    )
    _same(MachineStateEvents(plant, "run").detect_run_idle(), out, ["start"])


def test_late_frame_raises(plant):
    days = _days(plant)
    with pytest.raises(ValueError, match="already processed.*time_range_source"):
        run_chunked(
            [days[1], days[2], days[0]],
            MachineStateEvents,
            "detect_run_idle",
            run_state_uuid="run",
        )


def test_time_range_source(plant):
    calls = []

    def load(start, end):
        calls.append((start, end))
        # An inclusive end: the row at ``end`` is dropped by the source.
        return plant[(plant["systime"] >= start) & (plant["systime"] <= end)]

    source = time_range_source(
        load,
        pd.Timestamp("2024-03-29 12:00", tz=BERLIN),
        pd.Timestamp("2024-04-02", tz=BERLIN),
        chunk="1D",
    )
    out = run_chunked(
        source, MachineStateEvents, "detect_run_idle", run_state_uuid="run"
    )
    assert calls[0][0] == pd.Timestamp("2024-03-29 12:00", tz=BERLIN)
    assert calls[1] == (
        pd.Timestamp("2024-03-30", tz=BERLIN),
        pd.Timestamp("2024-03-31", tz=BERLIN),
    )
    assert calls[2][1] - calls[2][0] == pd.Timedelta(hours=23)
    assert len(calls) == 4
    part = plant[plant["systime"] >= pd.Timestamp("2024-03-29 12:00", tz=BERLIN)]
    _same(MachineStateEvents(part, "run").detect_run_idle(), out, ["start"])


def test_iter_chunked_yields_every_row_once(plant):
    parts = list(
        iter_chunked(
            _days(plant), MachineStateEvents, "detect_run_idle", run_state_uuid="run"
        )
    )
    assert len(parts) >= 2
    assert all(len(p) for p in parts)
    combined = pd.concat(parts, ignore_index=True)
    assert not combined.duplicated(["start", "state"]).any()
    _same(MachineStateEvents(plant, "run").detect_run_idle(), combined, ["start"])


def test_empty_source_gives_the_in_memory_empty_columns(empty_df):
    for target, method, kwargs in (
        (MachineStateEvents, "detect_run_idle", {"run_state_uuid": "run"}),
        (LineThroughputEvents, "count_parts", {"counter_uuid": "c", "window": "1h"}),
    ):
        out = run_chunked([], target, method, **kwargs)
        assert out.empty
        init = {k: v for k, v in kwargs.items() if k == "run_state_uuid"}
        call = {k: v for k, v in kwargs.items() if k not in init}
        memory = getattr(target(empty_df, **init), method)(**call)
        assert list(out.columns) == list(memory.columns)
        typed = run_chunked([empty_df], target, method, **kwargs)
        assert list(typed.columns) == list(memory.columns)
    assert (
        list(
            iter_chunked([], MachineStateEvents, "detect_run_idle", run_state_uuid="r")
        )
        == []
    )


def test_bad_source_items_and_columns(plant):
    with pytest.raises(TypeError, match="DataFrames or"):
        run_chunked([1, 2], MachineStateEvents, "detect_run_idle", run_state_uuid="run")
    with pytest.raises(ColumnNotFoundError, match="time column"):
        run_chunked(
            [plant.drop(columns="systime")],
            MachineStateEvents,
            "detect_run_idle",
            run_state_uuid="run",
        )


def test_missing_times_are_dropped(plant):
    df = plant.copy()
    df["systime"] = df["systime"].astype(object)
    df.loc[df.index[:5], "systime"] = pd.NaT
    df.loc[df.index[5:], "systime"] = plant["systime"].iloc[5:]
    out = run_chunked(
        [df.iloc[:5], df.iloc[5:]],
        MachineStateEvents,
        "detect_run_idle",
        run_state_uuid="run",
    )
    expected = MachineStateEvents(plant.iloc[5:], "run").detect_run_idle()
    _same(expected, out, ["start"])


def test_signal_without_later_reading_warns_after_max_carry():
    def blocks(t):
        return (np.arange(len(t)) // 120) % 2 == 0

    df = _frame("2024-01-01", "2024-01-05", {"other": ("value_bool", blocks)}, tz=None)
    run = df[df["systime"] < pd.Timestamp("2024-01-01 12:00")].assign(uuid="run")
    full = pd.concat([df, run]).sort_values("systime", kind="stable")
    pipe = (
        Pipeline()
        .detect(MachineStateEvents, "detect_run_idle", name="run", run_state_uuid="run")
        .detect(
            MachineStateEvents, "detect_run_idle", name="other", run_state_uuid="other"
        )
    )
    with pytest.warns(DataQualityWarning, match="no reading after"):
        out = pipe.run_stream(_days(full), max_carry="1D")
    expected = pipe.run(full)
    for name in ("run", "other"):
        _same(expected.events[name], out.events[name], ["start"])


# ---------------------------------------------------------------------------
# Validation and errors
# ---------------------------------------------------------------------------


def test_unsupported_method_raises_with_reason_and_alternative():
    with pytest.raises(NotChunkableError) as info:
        run_chunked([], OutlierDetectionEvents, "detect_outliers_zscore")
    message = str(info.value)
    assert "whole signal" in message
    assert "in memory" in message
    assert "OutlierDetectionEvents(df, ...).detect_outliers_zscore(...)" in message
    assert "Chunkable methods of OutlierDetectionEvents: none" in message
    with pytest.raises(
        NotChunkableError, match="detect_rapid_transitions, detect_run_idle"
    ):
        run_chunked([], MachineStateEvents, "transition_events")
    with pytest.raises(NotChunkableError, match="one summary of the whole data"):
        run_chunked([], DowntimeEvents, "downtime_by_reason")
    assert issubclass(NotChunkableError, ValueError)
    assert ts_shape.NotChunkableError is NotChunkableError


def test_plain_callables_and_unknown_targets():
    with pytest.raises(NotChunkableError, match="only"):
        run_chunked([], lambda df: df)
    with pytest.raises(AttributeError):
        run_chunked([], MachineStateEvents, "no_such_method")

    class NotRegistered:
        def method(self):  # pragma: no cover - never called
            return None

    with pytest.raises(NotChunkableError, match="no registered output shape"):
        run_chunked([], NotRegistered, "method")


def test_chunk_and_window_validation(plant):
    with pytest.raises(ValueError, match="whole multiple"):
        run_chunked(
            [plant],
            RuntimeAccountingEvents,
            "runtime_by_window",
            run_state_uuid="run",
            window="7h",
        )
    with pytest.raises(ValueError, match="fixed length"):
        run_chunked(
            [plant],
            MachineStateEvents,
            "detect_run_idle",
            chunk="MS",
            run_state_uuid="run",
        )
    with pytest.raises(ValueError, match="positive"):
        run_chunked(
            [plant],
            MachineStateEvents,
            "detect_run_idle",
            chunk="-1D",
            run_state_uuid="run",
        )
    with pytest.raises(ValueError, match="whole number of days"):
        run_chunked(
            [plant],
            DowntimeEvents,
            "downtime_by_shift",
            chunk="12h",
            run_state_uuid="run",
        )
    with pytest.raises(ValueError, match="overlap"):
        run_chunked(
            [plant],
            MachineStateEvents,
            "detect_run_idle",
            overlap="x",
            run_state_uuid="run",
        )
    with pytest.raises(ValueError, match="max_carry"):
        run_chunked(
            [plant],
            MachineStateEvents,
            "detect_run_idle",
            max_carry="0s",
            run_state_uuid="run",
        )
    with pytest.raises(ValueError, match="unknown argument"):
        run_chunked(
            [plant],
            MachineStateEvents,
            "detect_run_idle",
            run_state_uuid="run",
            bogus=1,
        )


def test_event_open_longer_than_max_carry_raises(plant):
    with pytest.raises(ValueError, match="longer than max_carry"):
        run_chunked(
            _days(plant),
            MachineStateEvents,
            "detect_run_idle",
            run_state_uuid="run",
            max_carry="20h",
        )


def test_support_names_are_registered():
    from ts_shape.eventlog.taxonomy import REGISTRY

    assert len(SUPPORT) > 50
    for name in SUPPORT:
        assert tuple(name.split(".", 1)) in REGISTRY


# ---------------------------------------------------------------------------
# Pipeline.run_stream
# ---------------------------------------------------------------------------


def _pipeline() -> Pipeline:
    from ts_shape import IntegerCalc

    return (
        Pipeline(name="stream")
        .transform(
            IntegerCalc, "scale_column", value_column="value_double", factor=10.0
        )
        .detect(
            MachineStateEvents, "detect_run_idle", name="runs", run_state_uuid="run"
        )
        .detect(
            ThresholdMonitoringEvents,
            "time_above_threshold",
            name="above",
            signal_uuid="temp",
            threshold=5.0,
            window="1h",
        )
    )


def test_run_stream_equals_run(plant):
    pipe = _pipeline()
    expected = pipe.run(plant)
    streamed = pipe.run_stream(_days(plant))
    assert sorted(streamed.events) == ["above", "runs"]
    for name in expected.events:
        _same(expected.events[name], streamed.events[name], ["start"])
    assert streamed.data.empty
    assert list(streamed.data.columns) == list(expected.data.columns)
    kept = pipe.run_stream(_days(plant), keep_data=True)
    _same(expected.data, kept.data, ["systime", "uuid"])
    log = streamed.to_event_log()
    assert len(log.events) == len(expected.to_event_log().events)


def test_run_stream_empty_source_and_errors(plant):
    pipe = _pipeline()
    empty = pipe.run_stream([])
    assert empty.events["runs"].empty and empty.data.empty
    with pytest.raises(TypeError, match="source step"):
        Pipeline().source(lambda: plant).run_stream([plant])
    with pytest.raises(NotChunkableError):
        Pipeline().detect(lambda df: df, name="x").run_stream([plant])
    with pytest.raises(NotChunkableError):
        Pipeline().detect(
            OutlierDetectionEvents,
            "detect_outliers_zscore",
            value_column="value_double",
        ).run_stream([plant])
    bad = (
        Pipeline()
        .transform(lambda df: 1)
        .detect(MachineStateEvents, "detect_run_idle", run_state_uuid="run")
    )
    with pytest.raises(TypeError, match="expected a DataFrame"):
        bad.run_stream([plant])
    broken = (
        Pipeline()
        .transform(lambda df: df["nope"])
        .detect(MachineStateEvents, "detect_run_idle", run_state_uuid="run")
    )
    with pytest.raises(RuntimeError, match="failed"):
        broken.run_stream([plant])


def test_run_stream_with_unknown_signals_watches_all(plant):
    pipe = (
        Pipeline()
        .detect(
            ThresholdMonitoringEvents,
            "time_above_threshold",
            signal_uuid="temp",
            threshold=0.5,
            window="1h",
        )
        .detect(
            MachineStateEvents, "detect_run_idle", name="runs", run_state_uuid="run"
        )
    )
    streamed = pipe.run_stream(_days(plant), chunk="1D")
    expected = pipe.run(plant)
    for name in expected.events:
        _same(expected.events[name], streamed.events[name], ["start"])


# ---------------------------------------------------------------------------
# Memory at scale
# ---------------------------------------------------------------------------


def _scale_day(day: int, signals: int = 5) -> pd.DataFrame:
    """One day at 1 Hz: a run state and ``signals - 1`` sensors (432k rows)."""
    n = 86_400
    start = pd.Timestamp("2024-01-01") + pd.Timedelta(days=day)
    t = pd.date_range(start, periods=n, freq="1s")
    rng = np.random.default_rng(day)
    names = np.array(
        ["run", *(f"sensor-{i}" for i in range(signals - 1))], dtype=object
    )
    values = np.full(n * signals, np.nan)
    rng.standard_normal(out=values[n:])
    state = pd.array(np.zeros(n * signals, dtype=bool), dtype="boolean")
    state[:n] = (np.arange(n) // 2_700 + day) % 3 != 0
    state[n:] = pd.NA
    return pd.DataFrame(
        {
            "systime": np.tile(t.to_numpy(), signals),
            # Dictionary-encoded, as parquet loads it with categories.
            "uuid": pd.Categorical.from_codes(
                np.repeat(np.arange(signals, dtype=np.int8), n), names
            ),
            "value_double": values,
            "value_bool": state,
        },
        copy=False,
    )


@pytest.mark.scale
def test_chunked_memory_stays_near_one_day():
    days = 20
    one_day = _scale_day(0).memory_usage(deep=True).sum()

    def source():
        for day in range(days):
            yield _scale_day(day)

    gc.collect()
    tracemalloc.start()
    try:
        out = run_chunked(
            source(), MachineStateEvents, "detect_run_idle", run_state_uuid="run"
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert (
        peak < 3 * one_day
    ), f"peak {peak / 1e6:.0f} MB vs one day {one_day / 1e6:.0f} MB"
    # Equal to the in-memory run on the first five days.
    sample = pd.concat([_scale_day(d) for d in range(5)], ignore_index=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        expected = MachineStateEvents(sample, "run").detect_run_idle()
    part = run_chunked(
        (_scale_day(d) for d in range(5)),
        MachineStateEvents,
        "detect_run_idle",
        run_state_uuid="run",
    )
    _same(expected, part, ["start"])
    assert len(out) > len(part)
