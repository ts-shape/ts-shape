"""Tests for RuntimeAccountingEvents (operating-hours accounting)."""

import pandas as pd  # type: ignore
import pytest

import ts_shape
from ts_shape.events.production.runtime_accounting import RuntimeAccountingEvents


def _run_signal(segments, start="2025-01-01 06:00:00"):
    """Build a change-event run signal.

    ``segments`` is a list of ``(state, duration_seconds)`` pairs; a trailing
    ``False`` row closes the last segment.
    """
    t = pd.Timestamp(start)
    rows = []
    for state, dur in segments:
        rows.append({"systime": t, "uuid": "m", "value_bool": state})
        t += pd.Timedelta(seconds=dur)
    rows.append({"systime": t, "uuid": "m", "value_bool": False})
    df = pd.DataFrame(rows)
    df["value_double"] = float("nan")
    return df


def test_runtime_summary_totals():
    # run 2 h, idle 1 h, run 1 h.
    df = _run_signal([(True, 7200), (False, 3600), (True, 3600)])
    s = RuntimeAccountingEvents(df, run_uuid="m").runtime_summary()
    assert s["run_hours"].iloc[0] == pytest.approx(3.0)
    assert s["idle_seconds"].iloc[0] == pytest.approx(3600.0)
    assert s["start_count"].iloc[0] == 2
    assert s["longest_run_seconds"].iloc[0] == pytest.approx(7200.0)
    assert s["utilization_pct"].iloc[0] == pytest.approx(75.0)


def test_runtime_summary_canonical_columns():
    df = _run_signal([(True, 3600)])
    s = RuntimeAccountingEvents(df, run_uuid="m").runtime_summary()
    for col in ("start", "end", "duration_seconds"):
        assert col in s.columns


def test_runtime_per_window_totals():
    df = _run_signal([(True, 7200), (False, 3600), (True, 3600)])
    pw = RuntimeAccountingEvents(df, run_uuid="m").runtime_per_window(window="1D")
    assert pw["run_hours"].sum() == pytest.approx(3.0)
    assert pw["start_count"].sum() == 2


def test_operating_hours_meter_is_monotonic():
    df = _run_signal([(True, 7200), (False, 3600), (True, 3600)])
    meter = RuntimeAccountingEvents(df, run_uuid="m").operating_hours_meter(window="1h")
    cumulative = meter["cumulative_run_hours"].tolist()
    assert cumulative == sorted(cumulative)  # non-decreasing
    assert cumulative[-1] == pytest.approx(3.0)


def test_empty_input_returns_empty(empty_df):
    rt = RuntimeAccountingEvents(empty_df, run_uuid="m")
    assert rt.runtime_summary().empty
    assert rt.runtime_per_window().empty
    assert rt.operating_hours_meter().empty


def test_wrong_uuid_raises_clear_error():
    df = _run_signal([(True, 3600)])
    with pytest.raises(ValueError, match="ghost"):
        RuntimeAccountingEvents(df, run_uuid="ghost")


def test_exported_at_top_level():
    assert ts_shape.RuntimeAccountingEvents is RuntimeAccountingEvents
