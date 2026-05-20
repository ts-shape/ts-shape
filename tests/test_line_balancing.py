"""Tests for LineBalancingEvents (line balancing and takt analysis)."""

import pandas as pd  # type: ignore
import pytest

import ts_shape
from ts_shape.events.production.line_balancing import LineBalancingEvents


def _make_line(station_cts, duration="2h", start="2025-01-01 06:00:00"):
    """Build pulse cycle-trigger signals for a line of stations.

    ``station_cts`` maps station UUID -> cycle time in seconds.
    """
    start = pd.Timestamp(start)
    end = start + pd.Timedelta(duration)
    rows = []
    for uuid, ct in station_cts.items():
        t = start
        while t < end:
            rows.append({"systime": t, "uuid": uuid, "value_bool": True})
            rows.append(
                {
                    "systime": t + pd.Timedelta(seconds=1),
                    "uuid": uuid,
                    "value_bool": False,
                }
            )
            t += pd.Timedelta(seconds=ct)
    df = pd.DataFrame(rows)
    df["value_double"] = float("nan")
    return df


def test_balanced_line_is_fully_efficient():
    df = _make_line({"a": 50, "b": 50, "c": 50})
    lb = LineBalancingEvents(df, station_uuids={"a": "A", "b": "B", "c": "C"})
    bm = lb.balance_metrics(takt_time="55s", window="1h")
    assert not bm.empty
    assert (bm["balance_efficiency_pct"] == 100.0).all()
    assert (bm["smoothness_index"] == 0.0).all()


def test_unbalanced_line_identifies_bottleneck():
    df = _make_line({"a": 40, "b": 70, "c": 50})
    lb = LineBalancingEvents(df, station_uuids={"a": "A", "b": "B", "c": "C"})
    bm = lb.balance_metrics(takt_time="75s", window="1h")
    assert (bm["bottleneck_uuid"] == "b").all()
    assert (bm["bottleneck_cycle_time"].round() == 70).all()
    assert (bm["balance_efficiency_pct"] < 100.0).all()
    # theoretical min = ceil((40 + 70 + 50) / 75) = ceil(2.13) = 3
    assert (bm["theoretical_min_stations"] == 3).all()


def test_balance_metrics_emits_canonical_summary_columns():
    df = _make_line({"a": 50, "b": 60})
    lb = LineBalancingEvents(df, station_uuids={"a": "A", "b": "B"})
    for out in (lb.station_cycle_times("1h"), lb.balance_metrics(window="1h")):
        for col in ("start", "end", "duration_seconds"):
            assert col in out.columns


def test_takt_resolved_from_demand_and_available_time():
    df = _make_line({"a": 50, "b": 50})
    lb = LineBalancingEvents(df, station_uuids={"a": "A", "b": "B"})
    # 120 units in 2h (7200 s) -> takt = 60 s
    ym = lb.yamazumi(demand=120, available_time="2h")
    assert (ym["takt_seconds"] == 60.0).all()
    assert (ym["loading_pct"].round(1) == 83.3).all()


def test_yamazumi_flags_the_bottleneck_station():
    df = _make_line({"a": 40, "b": 70})
    lb = LineBalancingEvents(df, station_uuids={"a": "A", "b": "B"})
    ym = lb.yamazumi(takt_time="75s")
    assert ym.loc[ym["uuid"] == "b", "is_bottleneck"].all()
    assert not ym.loc[ym["uuid"] == "a", "is_bottleneck"].any()


def test_empty_input_returns_empty_summary(empty_df):
    lb = LineBalancingEvents(empty_df, station_uuids={"x": "X"})
    out = lb.balance_metrics(window="1h")
    assert out.empty
    assert "balance_efficiency_pct" in out.columns


def test_wrong_uuid_raises_clear_error():
    df = _make_line({"a": 50})
    with pytest.raises(ValueError, match="missing"):
        LineBalancingEvents(df, station_uuids={"missing": "M"})


def test_exported_at_top_level():
    assert ts_shape.LineBalancingEvents is LineBalancingEvents
