"""Theme B follow-up: infer_dwell_from_next for traceability modules."""

import pandas as pd  # type: ignore

from ts_shape.events.production.order_traceability import OrderTraceabilityEvents


def _single_arrival_frame():
    # Each station logs ORD-001 exactly once, on arrival (event-encoded).
    rows = [
        {
            "uuid": "stA",
            "systime": pd.Timestamp("2024-01-01 08:00:00"),
            "value_string": "ORD-001",
        },
        {
            "uuid": "stB",
            "systime": pd.Timestamp("2024-01-01 08:10:00"),
            "value_string": "ORD-001",
        },
        {
            "uuid": "stC",
            "systime": pd.Timestamp("2024-01-01 08:25:00"),
            "value_string": "ORD-001",
        },
    ]
    return pd.DataFrame(rows)


def test_build_timeline_dwell_collapses_by_default():
    df = _single_arrival_frame()
    trace = OrderTraceabilityEvents(
        df, station_uuids={"stA": "A", "stB": "B", "stC": "C"}
    )
    timeline = trace.build_timeline()
    # Single-arrival events -> every dwell is zero by default.
    assert (timeline["duration_seconds"] == 0).all()


def test_build_timeline_infer_dwell_from_next():
    df = _single_arrival_frame()
    trace = OrderTraceabilityEvents(
        df, station_uuids={"stA": "A", "stB": "B", "stC": "C"}
    )
    timeline = trace.build_timeline(infer_dwell_from_next=True).sort_values(
        "station_sequence"
    )
    durations = list(timeline["duration_seconds"])
    # A: 08:00->08:10 = 600s, B: 08:10->08:25 = 900s, C: last visit stays open (0).
    assert durations == [600.0, 900.0, 0.0]


def test_station_dwell_statistics_infer_dwell():
    df = _single_arrival_frame()
    trace = OrderTraceabilityEvents(
        df, station_uuids={"stA": "A", "stB": "B", "stC": "C"}
    )
    stats = trace.station_dwell_statistics(infer_dwell_from_next=True)
    a = stats[stats["station_name"] == "A"]
    assert a["avg_dwell_seconds"].iloc[0] == 600.0
