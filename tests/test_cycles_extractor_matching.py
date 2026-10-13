"""Regression tests: cycle start/end matching in CycleExtractor.

Before the fix every ``process_*`` method returned ``cycle_end = NaT`` and
``is_complete = False`` because the end time was lost in the merge_asof key.
"""

import pandas as pd
import pytest

from ts_shape.features.cycles.cycles_extractor import CycleExtractor


def _frame(tz=None):
    t = pd.date_range("2024-03-31 00:00", periods=8, freq="min", tz=tz)
    return pd.DataFrame(
        {
            "systime": t,
            "uuid": "a",
            "value_bool": [False, True, True, False, True, False, False, True],
            "value_integer": [0, 1, 1, 2, 0, 1, 2, 0],
            "value_double": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0],
            "value_string": list("aabbccdd"),
        }
    )


def _ts(minutes, tz=None):
    return pd.Timestamp("2024-03-31 00:00", tz=tz) + pd.Timedelta(minutes=minutes)


def test_persistent_cycle_rising_to_falling_edge():
    out = CycleExtractor(_frame(), "a").process_persistent_cycle()
    assert list(out["cycle_start"]) == [_ts(1), _ts(4), _ts(7)]
    assert list(out["cycle_end"][:2]) == [_ts(3), _ts(5)]
    assert list(out["is_complete"]) == [True, True, False]


def test_trigger_cycle_runs_to_next_trigger():
    out = CycleExtractor(_frame(), "a").process_trigger_cycle()
    assert list(out["cycle_start"]) == [_ts(1), _ts(4), _ts(7)]
    assert list(out["cycle_end"][:2]) == [_ts(4), _ts(7)]
    assert out["is_complete"].sum() == 2


def test_same_rows_never_close_at_themselves():
    ce = CycleExtractor(_frame(), "a")
    for out in (
        ce.process_state_change_cycle(),
        ce.process_value_change_cycle(),
        ce.process_separate_start_end_cycle(),
    ):
        done = out[out["is_complete"]]
        assert len(done) > 0
        assert (done["cycle_end"] > done["cycle_start"]).all()


def test_step_sequence_counts_step_entries_once():
    out = CycleExtractor(_frame(), "a").process_step_sequence(1, 2)
    assert list(out["cycle_start"]) == [_ts(1), _ts(5)]
    assert list(out["cycle_end"]) == [_ts(3), _ts(6)]


def test_tz_aware_systime_is_preserved():
    tz = "Europe/Berlin"
    out = CycleExtractor(_frame(tz), "a").process_persistent_cycle()
    assert str(out["cycle_start"].dt.tz) == tz
    assert str(out["cycle_end"].dt.tz) == tz
    assert out["cycle_start"].iloc[0] == _ts(1, tz)


def test_other_uuids_are_ignored():
    df = _frame()
    noise = df.assign(uuid="b", value_bool=True)
    out = CycleExtractor(pd.concat([df, noise]), "a").process_persistent_cycle()
    assert len(out) == 3


@pytest.mark.parametrize("tz", [None, "Europe/Berlin"])
def test_empty_result_keeps_systime_dtype(tz):
    df = _frame(tz).assign(value_bool=False)
    out = CycleExtractor(df, "a").process_persistent_cycle()
    assert out.empty
    assert out["cycle_start"].dtype == df["systime"].dtype
