"""Tests for FlowMetricsEvents (WIP, throughput, lead time, Little's Law)."""

import pandas as pd  # type: ignore
import pytest

import ts_shape
from ts_shape.events.production.flow_metrics import FlowMetricsEvents


def _make_flow(
    entry_period=60,
    exit_period=60,
    lag=300,
    n=200,
    start="2025-01-01 06:00:00",
):
    """Build entry/exit pulse signals; ``lag`` seconds entry-to-exit offset."""
    start = pd.Timestamp(start)
    rows = []
    t = start
    for _ in range(n):
        rows.append({"systime": t, "uuid": "in", "value_bool": True})
        rows.append(
            {"systime": t + pd.Timedelta(seconds=1), "uuid": "in", "value_bool": False}
        )
        t += pd.Timedelta(seconds=entry_period)
    t = start + pd.Timedelta(seconds=lag)
    for _ in range(n):
        rows.append({"systime": t, "uuid": "out", "value_bool": True})
        rows.append(
            {"systime": t + pd.Timedelta(seconds=1), "uuid": "out", "value_bool": False}
        )
        t += pd.Timedelta(seconds=exit_period)
    df = pd.DataFrame(rows)
    df["value_double"] = float("nan")
    return df


def test_lead_time_is_fifo_matched():
    df = _make_flow(lag=300)
    flow = FlowMetricsEvents(df, entry_uuid="in", exit_uuid="out")
    lt = flow.lead_time()
    assert not lt.empty
    assert (lt["lead_time_seconds"].round() == 300).all()
    for col in ("systime", "uuid", "source_uuid"):
        assert col in lt.columns


def test_throughput_counts_units_per_window():
    df = _make_flow(exit_period=60)
    flow = FlowMetricsEvents(df, entry_uuid="in", exit_uuid="out")
    tp = flow.throughput("1h")
    full_hours = tp[tp["units_out"] == 60]
    assert not full_hours.empty
    assert (full_hours["throughput_per_hour"] == 60.0).all()


def test_wip_reaches_expected_steady_state():
    # entry & exit both every 60 s, exit lagged 300 s -> steady-state WIP = 5.
    df = _make_flow(entry_period=60, exit_period=60, lag=300, n=300)
    flow = FlowMetricsEvents(df, entry_uuid="in", exit_uuid="out")
    wip = flow.wip_over_time("1h")
    interior = wip.iloc[1]  # 07:00-08:00, fully steady
    assert abs(interior["wip_mean"] - 5.0) < 0.3
    assert interior["wip_max"] == 5.0


def test_littles_law_consistency_holds_in_steady_state():
    df = _make_flow(entry_period=60, exit_period=60, lag=300, n=300)
    flow = FlowMetricsEvents(df, entry_uuid="in", exit_uuid="out")
    fs = flow.flow_summary(value_add_seconds=120, window="1h")
    interior = fs["consistency_ratio"].dropna()
    assert (abs(interior - 1.0) < 0.2).any()
    # PCE = value-add / lead time = 120 / 300 = 40 %
    pce = fs["process_cycle_efficiency_pct"].dropna()
    assert (pce.round() == 40).any()


def test_flow_summary_emits_canonical_summary_columns():
    df = _make_flow()
    flow = FlowMetricsEvents(df, entry_uuid="in", exit_uuid="out")
    fs = flow.flow_summary(window="1h")
    for col in ("start", "end", "duration_seconds"):
        assert col in fs.columns
    # PCE column only appears when value_add_seconds is supplied.
    assert "process_cycle_efficiency_pct" not in fs.columns


def test_empty_input_returns_empty(empty_df):
    flow = FlowMetricsEvents(empty_df, entry_uuid="in", exit_uuid="out")
    assert flow.wip_over_time("1h").empty
    assert flow.throughput("1h").empty
    assert flow.lead_time().empty


def test_wrong_uuid_raises_clear_error():
    df = _make_flow()
    with pytest.raises(ValueError, match="ghost"):
        FlowMetricsEvents(df, entry_uuid="ghost", exit_uuid="out")


def test_exported_at_top_level():
    assert ts_shape.FlowMetricsEvents is FlowMetricsEvents
