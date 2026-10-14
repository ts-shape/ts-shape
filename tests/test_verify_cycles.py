"""Verification of CycleExtractor: every extraction method on realistic data.

Expected boundaries, counts and ids are derived by hand from the scenario
frames. See the module docstring of ``cycles_extractor`` for the contract.
"""

import time
import uuid as uuidlib
import warnings

import numpy as np
import pandas as pd
import pytest

from ts_shape.errors import DataQualityWarning
from ts_shape.features.cycles.cycles_extractor import CycleExtractor

BERLIN = "Europe/Berlin"
COLUMNS = [
    "cycle_start",
    "cycle_end",
    "cycle_uuid",
    "is_complete",
    "start",
    "end",
    "duration_seconds",
    "uuid",
    "source_uuid",
]


def T(minutes: float, tz: str | None = None) -> pd.Timestamp:
    return pd.Timestamp("2024-03-05 06:00", tz=tz) + pd.Timedelta(minutes=minutes)


def long_frame(rows: list[tuple], tz: str | None = None) -> pd.DataFrame:
    """rows: (minute, uuid, column, value) -> long-format frame."""
    records = []
    for minute, signal, column, value in rows:
        rec = {
            "systime": T(minute, tz),
            "uuid": signal,
            "value_bool": None,
            "value_integer": None,
            "value_double": None,
            "value_string": None,
        }
        rec[column] = value
        records.append(rec)
    df = pd.DataFrame(records)
    df["value_double"] = df["value_double"].astype("float64")
    return df


def bools(signal: str, pattern: list[tuple[float, object]]) -> list[tuple]:
    return [(m, signal, "value_bool", v) for m, v in pattern]


def cycles_as_tuples(out: pd.DataFrame, tz: str | None = None) -> list[tuple]:
    """(start minute, end minute or None, is_complete) per cycle."""
    base = T(0, tz)
    res = []
    for s, e, c in zip(out["cycle_start"], out["cycle_end"], out["is_complete"]):
        sm = (s - base) / pd.Timedelta(minutes=1)
        em = None if pd.isna(e) else (e - base) / pd.Timedelta(minutes=1)
        res.append((sm, em, bool(c)))
    return res


def quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


# ---------------------------------------------------------------------------
# persistent
# ---------------------------------------------------------------------------


def test_persistent_interleaved_signals_mid_cycle_start_and_end():
    rows = (
        # machine A: already running at the first sample, 3 real cycles, the
        # last one still running at the end of the data. NaN inside a run holds.
        bools(
            "A.run",
            [
                (0, True),
                (2, False),
                (5, True),
                (6, None),
                (7, True),
                (9, False),
                (12, True),
                (14, False),
                (20, True),
            ],
        )
        # machine B and a temperature interleaved (must be ignored)
        + bools("B.run", [(1, True), (3, False), (8, True), (15, False)])
        + [(m, "A.temp", "value_double", 20.0 + m) for m in range(0, 21, 3)]
    )
    df = long_frame(rows)
    ce = CycleExtractor(df, "A.run")
    out = ce.process_persistent_cycle()
    assert list(out.columns) == COLUMNS
    assert cycles_as_tuples(out) == [
        (0, 2, False),  # began before the data: partial
        (5, 9, True),  # NaN at 6 does not split the run
        (12, 14, True),
        (20, None, False),  # still running
    ]
    assert list(out["duration_seconds"].iloc[:3]) == [120.0, 240.0, 120.0]
    assert np.isnan(out["duration_seconds"].iloc[3])
    assert (out["start"] == out["cycle_start"]).all()
    assert out["uuid"].eq("cycles:persistent").all()
    assert out["source_uuid"].eq("A.run").all()
    stats = ce.get_extraction_stats()
    assert stats["total_cycles"] == 4
    assert stats["complete_cycles"] == 2
    assert stats["incomplete_cycles"] == 2
    assert stats["unmatched_starts"] == 1
    assert stats["partial_first_cycles"] == 1

    out_b = CycleExtractor(df, "B.run").process_persistent_cycle()
    # B is True at its first sample: the first cycle is partial
    assert cycles_as_tuples(out_b) == [(1, 3, False), (8, 15, True)]


def test_persistent_dst_fall_back_night_keeps_tz_and_absolute_durations():
    # 2024-10-27: 03:00 CEST -> 02:00 CET. Samples every 10 min in UTC.
    t = pd.date_range("2024-10-26 23:30", periods=12, freq="10min", tz="UTC")
    t = t.tz_convert(BERLIN)
    pattern = [
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        True,
        False,
    ]
    df = pd.DataFrame({"systime": t, "uuid": "press", "value_bool": pattern})
    out = CycleExtractor(df, "press").process_persistent_cycle()
    assert str(out["cycle_start"].dt.tz) == BERLIN
    assert str(out["cycle_end"].dt.tz) == BERLIN
    assert str(out["start"].dt.tz) == BERLIN
    # first cycle: 01:40 CEST -> 02:50 CET = 23:40 -> 01:00 UTC = 80 minutes
    assert out["cycle_start"].iloc[0] == t[1]
    assert out["cycle_end"].iloc[0] == t[9]
    assert out["duration_seconds"].tolist() == [80 * 60.0, 10 * 60.0]
    assert out["is_complete"].tolist() == [True, True]


def test_persistent_nullable_and_pyarrow_bool():
    t = [T(m) for m in range(6)]
    for dtype in ("boolean", "bool[pyarrow]"):
        df = pd.DataFrame(
            {
                "systime": t,
                "uuid": "s",
                "value_bool": pd.array(
                    [False, True, None, False, True, False], dtype=dtype
                ),
            }
        )
        out = CycleExtractor(df, "s").process_persistent_cycle()
        assert cycles_as_tuples(out) == [(1, 3, True), (4, 5, True)]


def test_persistent_without_uuid_column_uses_all_rows():
    df = pd.DataFrame(
        {"systime": [T(m) for m in range(4)], "value_bool": [False, True, False, True]}
    )
    out = CycleExtractor(df, "whatever").process_persistent_cycle()
    assert cycles_as_tuples(out) == [(1, 2, True), (3, None, False)]


def test_unknown_start_uuid_in_multi_signal_frame_warns_and_is_empty():
    df = long_frame(bools("a", [(0, True), (1, False)]) + bools("b", [(0, True)]))
    ce = CycleExtractor(df, "zzz")
    with pytest.warns(DataQualityWarning, match="not found"):
        out = ce.process_persistent_cycle()
    assert out.empty
    assert list(out.columns) == COLUMNS


def test_single_other_signal_is_used_with_warning():
    df = long_frame(bools("a", [(0, False), (1, True), (2, False)]))
    with pytest.warns(DataQualityWarning, match="one signal"):
        out = CycleExtractor(df, "label").process_persistent_cycle()
    assert cycles_as_tuples(out) == [(1, 2, True)]


@pytest.mark.parametrize("tz", [None, BERLIN])
def test_empty_results_keep_full_schema(tz):
    df = long_frame(bools("a", [(0, False), (1, False)]), tz=tz)
    ce = CycleExtractor(df, "a")
    for out in (
        ce.process_persistent_cycle(),
        ce.process_trigger_cycle(),
        ce.process_step_sequence(1, 2),
    ):
        assert out.empty
        assert list(out.columns) == COLUMNS
        assert out["cycle_start"].dtype == df["systime"].dtype
        assert out["end"].dtype == df["systime"].dtype
        assert out["is_complete"].dtype == bool


def test_empty_input_frame():
    df = pd.DataFrame(columns=["systime", "uuid", "value_bool"])
    out = CycleExtractor(df, "a").process_persistent_cycle()
    assert out.empty and list(out.columns) == COLUMNS


# ---------------------------------------------------------------------------
# trigger
# ---------------------------------------------------------------------------


def test_trigger_held_pulse_counts_once_and_last_is_open():
    rows = bools(
        "trig",
        [
            (0, False),
            (1, True),
            (1.5, True),
            (2, False),
            (4, True),
            (5, False),
            (9, True),
        ],
    )
    out = CycleExtractor(long_frame(rows), "trig").process_trigger_cycle()
    assert cycles_as_tuples(out) == [(1, 4, True), (4, 9, True), (9, None, False)]


def test_trigger_event_log_of_true_rows_counts_every_row():
    rows = bools("ev", [(0, True), (3, True), (7, True)])
    out = CycleExtractor(long_frame(rows), "ev").process_trigger_cycle()
    assert cycles_as_tuples(out) == [(0, 3, True), (3, 7, True), (7, None, False)]


def test_trigger_signal_high_at_first_sample_is_partial():
    rows = bools("t", [(0, True), (1, False), (3, True), (4, False), (6, True)])
    out = CycleExtractor(long_frame(rows), "t").process_trigger_cycle()
    assert cycles_as_tuples(out) == [(0, 3, False), (3, 6, True), (6, None, False)]


# ---------------------------------------------------------------------------
# separate start / end
# ---------------------------------------------------------------------------


def test_separate_back_to_back_cycles_share_a_timestamp():
    rows = bools("s", [(0, True), (5, True)]) + bools("e", [(5, True), (9, True)])
    ce = CycleExtractor(long_frame(rows), "s", "e")
    out = ce.process_separate_start_end_cycle()
    assert cycles_as_tuples(out) == [(0, 5, True), (5, 9, True)]
    assert ce.get_extraction_stats()["unmatched_ends"] == 0


def test_separate_nested_start_aborts_the_open_cycle():
    rows = bools("s", [(0, True), (2, True), (10, True)]) + bools(
        "e", [(5, True), (12, True)]
    )
    out = CycleExtractor(long_frame(rows), "s", "e").process_separate_start_end_cycle()
    assert cycles_as_tuples(out) == [(0, None, False), (2, 5, True), (10, 12, True)]


def test_separate_end_before_start_and_double_end_are_unmatched():
    rows = bools("s", [(3, True), (10, True)]) + bools(
        "e", [(1, True), (6, True), (7, True)]
    )
    ce = CycleExtractor(long_frame(rows), "s", "e")
    out = ce.process_separate_start_end_cycle()
    assert cycles_as_tuples(out) == [(3, 6, True), (10, None, False)]
    stats = ce.get_extraction_stats()
    assert stats["unmatched_ends"] == 2
    assert stats["unmatched_starts"] == 1


def test_separate_sampled_pulses_use_rising_edges():
    # start held True for 3 samples, end held True for 2 samples
    rows = bools(
        "s",
        [
            (0, False),
            (1, True),
            (2, True),
            (3, True),
            (4, False),
            (8, True),
            (9, False),
        ],
    ) + bools("e", [(0, False), (5, True), (6, True), (7, False), (10, True)])
    out = CycleExtractor(long_frame(rows), "s", "e").process_separate_start_end_cycle()
    assert cycles_as_tuples(out) == [(1, 5, True), (8, 10, True)]


def test_separate_missing_end_signal_warns_and_leaves_cycles_open():
    rows = bools("s", [(0, True), (4, True)]) + bools("other", [(1, True)])
    ce = CycleExtractor(long_frame(rows), "s", "e")
    with pytest.warns(DataQualityWarning, match="'e' not found"):
        out = ce.process_separate_start_end_cycle()
    assert cycles_as_tuples(out) == [(0, None, False), (4, None, False)]


def test_separate_same_uuid_behaves_like_trigger():
    rows = bools("s", [(0, True), (4, True), (6, True)])
    out = CycleExtractor(long_frame(rows), "s").process_separate_start_end_cycle()
    assert cycles_as_tuples(out) == [(0, 4, True), (4, 6, True), (6, None, False)]


# ---------------------------------------------------------------------------
# step sequence
# ---------------------------------------------------------------------------


def _steps(values: list, dtype: str | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "systime": [T(m) for m in range(len(values))],
            "uuid": "step",
            "value_integer": pd.array(values, dtype=dtype) if dtype else values,
        }
    )


def test_step_sequence_starts_mid_cycle_holds_steps_and_ends_open():
    #           0  1  2  3  4  5  6  7  8  9 10 11
    values = [3, 4, 5, 0, 1, 1, 2, 3, 5, 5, 0, 1]
    ce = CycleExtractor(_steps(values), "step")
    out = ce.process_step_sequence(start_step=1, end_step=5)
    assert cycles_as_tuples(out) == [(4, 8, True), (11, None, False)]
    # start at the first sample: partial
    out2 = ce.process_step_sequence(start_step=3, end_step=5)
    assert cycles_as_tuples(out2) == [(0, 2, False), (7, 8, True)]


@pytest.mark.parametrize("dtype", ["Int64", "int64[pyarrow]", "float64"])
def test_step_sequence_missing_samples_hold_the_step(dtype):
    values = [1, None, 1, 2, 0, 1, None, 2]
    out = CycleExtractor(_steps(values, dtype), "step").process_step_sequence(1, 2)
    assert cycles_as_tuples(out) == [(0, 3, False), (5, 7, True)]


def test_step_sequence_same_start_and_end_step():
    values = [0, 7, 7, 3, 7, 1, 7]
    out = CycleExtractor(_steps(values), "step").process_step_sequence(7, 7)
    assert cycles_as_tuples(out) == [(1, 4, True), (4, 6, True), (6, None, False)]


def test_step_sequence_restart_without_end_aborts():
    values = [0, 1, 2, 1, 2, 3, 0]
    out = CycleExtractor(_steps(values), "step").process_step_sequence(1, 3)
    assert cycles_as_tuples(out) == [(1, None, False), (3, 5, True)]


# ---------------------------------------------------------------------------
# state change / value change
# ---------------------------------------------------------------------------


def test_state_change_every_sample_is_a_boundary_duplicates_collapse():
    rows = [
        (0, "st", "value_string", "IDLE"),
        (2, "st", "value_string", "RUN"),
        (2, "st", "value_string", "RUN"),  # duplicate row
        (5, "st", "value_string", "STOP"),
        (1, "noise", "value_string", "X"),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DataQualityWarning)
        out = CycleExtractor(long_frame(rows), "st").process_state_change_cycle()
    assert cycles_as_tuples(out) == [(0, 2, False), (2, 5, True), (5, None, False)]


def test_value_change_ignores_other_signals_and_holds_missing_values():
    rows = [
        (0, "a", "value_double", 1.0),
        (1, "a", "value_double", 1.05),  # below threshold
        (2, "a", "value_double", None),  # missing: holds 1.05
        (3, "a", "value_double", 1.05),
        (4, "a", "value_double", 2.0),
        (6, "a", "value_double", 2.0),
        (7, "a", "value_double", 0.5),
    ] + [(m, "b", "value_double", float(m)) for m in range(8)]
    ce = CycleExtractor(long_frame(rows), "a", value_change_threshold=0.1)
    out = ce.process_value_change_cycle()
    assert cycles_as_tuples(out) == [(0, 4, False), (4, 7, True), (7, None, False)]


def test_value_change_strings_and_bools():
    rows = [
        (0, "p", "value_string", "PART_A"),
        (1, "p", "value_string", "PART_A"),
        (2, "p", "value_string", None),
        (3, "p", "value_string", "PART_B"),
        (5, "p", "value_string", "PART_A"),
    ]
    out = CycleExtractor(long_frame(rows), "p").process_value_change_cycle()
    assert cycles_as_tuples(out) == [(0, 3, False), (3, 5, True), (5, None, False)]

    rows_b = bools("b", [(0, False), (1, False), (2, True), (3, None), (4, False)])
    out_b = CycleExtractor(long_frame(rows_b), "b").process_value_change_cycle()
    assert cycles_as_tuples(out_b) == [(0, 2, False), (2, 4, True), (4, None, False)]


def test_value_change_does_not_mutate_input():
    df = long_frame([(0, "a", "value_double", None), (1, "a", "value_double", 1.0)])
    before = df.copy()
    ce = CycleExtractor(df, "a")
    stored = ce.df.copy()
    ce.process_value_change_cycle()
    pd.testing.assert_frame_equal(df, before)
    pd.testing.assert_frame_equal(ce.df, stored)


# ---------------------------------------------------------------------------
# cycle ids
# ---------------------------------------------------------------------------


def test_cycle_ids_are_stable_unique_and_scoped():
    rows = bools("A", [(0, False), (1, True), (2, False), (3, True), (4, False)])
    rows += bools("B", [(0, False), (1, True), (2, False)])
    df = long_frame(rows, tz=BERLIN)
    first = CycleExtractor(df, "A").process_persistent_cycle()
    again = CycleExtractor(
        df.sample(frac=1, random_state=1), "A"
    ).process_persistent_cycle()
    assert first["cycle_uuid"].tolist() == again["cycle_uuid"].tolist()
    assert first["cycle_uuid"].is_unique
    for value in first["cycle_uuid"]:
        assert str(uuidlib.UUID(value)) == value
    other_signal = CycleExtractor(df, "B").process_persistent_cycle()
    assert other_signal["cycle_uuid"].iloc[0] != first["cycle_uuid"].iloc[0]
    other_method = CycleExtractor(df, "A").process_trigger_cycle()
    assert set(other_method["cycle_uuid"]).isdisjoint(first["cycle_uuid"])
    # the same cycle keeps its id when the query window grows
    longer = pd.concat(
        [df, long_frame(bools("A", [(9, True)]), tz=BERLIN)], ignore_index=True
    )
    grown = CycleExtractor(longer, "A").process_persistent_cycle()
    assert grown["cycle_uuid"].iloc[:2].tolist() == first["cycle_uuid"].tolist()


# ---------------------------------------------------------------------------
# validate / overlap / suggest
# ---------------------------------------------------------------------------


def _cycles(spans: list[tuple], tz: str | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cycle_start": [T(s, tz) for s, _ in spans],
            "cycle_end": [pd.NaT if e is None else T(e, tz) for _, e in spans],
            "cycle_uuid": [f"c{i}" for i in range(len(spans))],
            "is_complete": [e is not None for _, e in spans],
        }
    )


def test_validate_cycles_bounds_are_inclusive_and_partial_is_invalid():
    rows = bools(
        "a",
        [
            (0, True),
            (1, False),
            (2, True),
            (3, False),
            (4, True),
            (14, False),
            (20, True),
            (40, False),
            (50, True),
        ],
    )
    ce = CycleExtractor(long_frame(rows), "a")
    out = ce.validate_cycles(ce.process_persistent_cycle(), "1m", "10m")
    assert out["validation_issue"].tolist() == [
        "incomplete_cycle",  # partial first cycle
        "",  # exactly 1 min
        "",  # exactly 10 min
        "too_long;",
        "incomplete_cycle",
    ]
    assert out["is_valid"].tolist() == [False, True, True, False, False]
    assert out["duration_seconds"].iloc[1] == 60.0


def test_validate_cycles_empty_keeps_columns():
    ce = CycleExtractor(long_frame(bools("a", [(0, False)])), "a")
    out = ce.validate_cycles(ce.process_persistent_cycle())
    assert out.empty
    for column in (
        "cycle_duration",
        "duration_seconds",
        "is_valid",
        "validation_issue",
    ):
        assert column in out.columns


def test_overlap_flag_and_pair_count():
    spans = [(0, 10), (5, 12), (10, 20), (11, 13), (20, 25), (30, None)]
    ce = CycleExtractor(long_frame(bools("a", [(0, True)])), "a")
    out = ce.detect_overlapping_cycles(_cycles(spans))
    # pairs: (0,5..10)/(5..12) yes; (5..12, 10..20) yes; (5..12, 11..13) yes;
    # (10..20, 11..13) yes; back-to-back 10..20 / 20..25 no.
    assert out["has_overlap"].tolist() == [True, True, True, True, False, False]
    assert ce.get_extraction_stats()["overlapping_cycles"] == 4


@pytest.mark.parametrize(
    "resolve,kept",
    [
        ("keep_first", ["c0", "c2", "c4"]),
        ("keep_last", ["c1", "c3", "c4"]),
        ("keep_longest", ["c0", "c3", "c4"]),
    ],
)
def test_overlap_resolution_is_greedy(resolve, kept):
    # durations 10, 2, 10, 35; c4 has no end and is always kept
    spans = [(0, 10), (2, 4), (20, 30), (25, 60), (70, None)]
    ce = CycleExtractor(long_frame(bools("a", [(0, True)])), "a")
    out = ce.detect_overlapping_cycles(_cycles(spans), resolve=resolve)
    assert out["cycle_uuid"].tolist() == kept


def test_overlap_keep_first_on_chain_keeps_non_clashing_third():
    spans = [(0, 10), (5, 15), (12, 20)]
    ce = CycleExtractor(long_frame(bools("a", [(0, True)])), "a")
    out = ce.detect_overlapping_cycles(_cycles(spans), resolve="keep_first")
    assert out["cycle_uuid"].tolist() == ["c0", "c2"]


def test_overlap_unknown_resolve_raises():
    ce = CycleExtractor(long_frame(bools("a", [(0, True)])), "a")
    with pytest.raises(ValueError, match="resolve"):
        ce.detect_overlapping_cycles(_cycles([(0, 1)]), resolve="nope")


def test_extracted_cycles_never_overlap():
    rows = bools("s", [(0, True), (2, True), (5, True), (6, True)]) + bools(
        "e", [(3, True), (5, True), (8, True), (9, True)]
    )
    ce = CycleExtractor(long_frame(rows), "s", "e")
    out = ce.detect_overlapping_cycles(ce.process_separate_start_end_cycle())
    assert not out["has_overlap"].any()


def test_suggest_method_reads_the_start_signal():
    rows = bools("a", [(0, False), (1, True), (2, False)]) + [
        (0, "b", "value_integer", 1),
        (1, "b", "value_integer", 2),
    ]
    sugg = CycleExtractor(long_frame(rows), "a").suggest_method()
    assert sugg["recommended_methods"][:2] == [
        "process_persistent_cycle",
        "process_trigger_cycle",
    ]
    assert "process_step_sequence" not in sugg["recommended_methods"]
    assert sugg["data_characteristics"]["row_count"] == 3


# ---------------------------------------------------------------------------
# speed
# ---------------------------------------------------------------------------


def test_100k_rows_every_method_is_fast_and_correct():
    n = 100_000
    t = pd.date_range("2024-01-01", periods=n, freq="1s", tz=BERLIN)
    phase = np.arange(n) % 20
    df = pd.DataFrame(
        {
            "systime": t,
            "uuid": "m",
            "value_bool": phase < 12,
            "value_integer": phase,
            "value_double": (phase // 5).astype(float),
            "value_string": None,
        }
    )
    ce = CycleExtractor(df, "m")
    timings = {}
    for name, call, expected in [
        ("persistent", ce.process_persistent_cycle, n // 20),
        ("trigger", ce.process_trigger_cycle, n // 20),
        ("separate", ce.process_separate_start_end_cycle, n // 20),
        ("step", lambda: ce.process_step_sequence(0, 10), n // 20),
        ("state", ce.process_state_change_cycle, n),
        (
            "value",
            CycleExtractor(
                df.drop(columns=["value_integer", "value_bool"]), "m"
            ).process_value_change_cycle,
            n // 5,
        ),
    ]:
        started = time.perf_counter()
        out = call()
        timings[name] = time.perf_counter() - started
        assert len(out) == expected, name
        assert out["cycle_uuid"].is_unique
    started = time.perf_counter()
    cycles = ce.process_persistent_cycle()
    ce.detect_overlapping_cycles(cycles, resolve="keep_longest")
    timings["overlap"] = time.perf_counter() - started
    assert max(timings.values()) < 3.0, timings
    # persistent cycles are 12 s long, back to back every 20 s
    assert cycles["duration_seconds"].iloc[1:].eq(12.0).all()


def test_validate_too_short_without_is_complete_column_and_duration_formats():
    ce = CycleExtractor(long_frame(bools("a", [(0, True)])), "a")
    cyc = _cycles([(0, 0.5), (1, 3), (4, None)]).drop(columns="is_complete")
    out = ce.validate_cycles(cyc, min_duration="45s", max_duration="1h30min")
    assert out["validation_issue"].tolist() == ["too_short;", "", "incomplete_cycle"]
    assert ce._parse_duration("1.5 m") == pd.Timedelta(seconds=90)
    with pytest.raises(ValueError, match="Invalid duration"):
        ce._parse_duration("soon")


def test_suggest_method_for_steps_and_stats_reset():
    df = _steps([0, 1, 2, 3, 0, 1])
    ce = CycleExtractor(df, "step", "other")
    sugg = ce.suggest_method()
    assert sugg["recommended_methods"] == [
        "process_separate_start_end_cycle",
        "process_step_sequence",
        "process_value_change_cycle",
        "process_state_change_cycle",
    ]
    ce.process_step_sequence(1, 3)
    assert ce.get_extraction_stats()["total_cycles"] == 2
    assert ce.get_extraction_stats()["success_rate"] == 0.5
    ce.reset_stats()
    stats = ce.get_extraction_stats()
    assert stats["total_cycles"] == 0 and stats["success_rate"] == 0.0
