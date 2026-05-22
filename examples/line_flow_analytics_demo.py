#!/usr/bin/env python3
"""
Line & Flow Analytics Demo for ts-shape
========================================

Demonstrates the industrial-engineering line-and-flow analytics:

1. LineBalancingEvents - station cycle times, line balance efficiency,
   smoothness index, theoretical minimum stations, and a Yamazumi table.
2. FlowMetricsEvents   - WIP over time, throughput, FIFO lead time, and a
   Little's Law consistency check with Process Cycle Efficiency.

Scenario: a 4-station manual assembly line. Each station emits a boolean
cycle-completion pulse. The line feeds a downstream process whose entry and
exit are tracked to derive flow metrics.
"""

import pandas as pd

from ts_shape.events.production.line_balancing import LineBalancingEvents
from ts_shape.events.production.flow_metrics import FlowMetricsEvents

# ---------------------------------------------------------------------------
# Helper: build pulse signals (a boolean that toggles True->False per event)
# ---------------------------------------------------------------------------


def _pulses(rows, uuid, start, period_seconds, duration_seconds):
    """Append True/False pulse rows for one signal at a fixed period."""
    t = start
    end = start + pd.Timedelta(seconds=duration_seconds)
    while t < end:
        rows.append({"systime": t, "uuid": uuid, "value_bool": True})
        rows.append(
            {"systime": t + pd.Timedelta(seconds=1), "uuid": uuid, "value_bool": False}
        )
        t += pd.Timedelta(seconds=period_seconds)


def build_line_data() -> pd.DataFrame:
    """Four assembly stations with deliberately unbalanced cycle times."""
    start = pd.Timestamp("2025-06-10 06:00:00")
    rows: list = []
    # Station cycle times (seconds): St3 is the bottleneck.
    for uuid, ct in [("st1", 48), ("st2", 52), ("st3", 71), ("st4", 44)]:
        _pulses(rows, uuid, start, ct, duration_seconds=4 * 3600)
    df = pd.DataFrame(rows)
    df["value_double"] = float("nan")
    return df


def build_flow_data() -> pd.DataFrame:
    """Entry / exit pulses for a downstream process (~6 units of WIP)."""
    start = pd.Timestamp("2025-06-10 06:00:00")
    rows: list = []
    _pulses(rows, "process:in", start, 60, duration_seconds=4 * 3600)
    # Exit lagged 360 s behind entry -> steady-state WIP = 360 / 60 = 6.
    exit_start = start + pd.Timedelta(seconds=360)
    t = exit_start
    end = exit_start + pd.Timedelta(seconds=4 * 3600)
    while t < end:
        rows.append({"systime": t, "uuid": "process:out", "value_bool": True})
        rows.append(
            {
                "systime": t + pd.Timedelta(seconds=1),
                "uuid": "process:out",
                "value_bool": False,
            }
        )
        t += pd.Timedelta(seconds=60)
    df = pd.DataFrame(rows)
    df["value_double"] = float("nan")
    return df


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------


def demo_line_balancing():
    print("=" * 72)
    print("1. LINE BALANCING")
    print("=" * 72)

    df = build_line_data()
    stations = {
        "st1": "Station 1",
        "st2": "Station 2",
        "st3": "Station 3",
        "st4": "Station 4",
    }
    lb = LineBalancingEvents(df, station_uuids=stations)

    # Takt from customer demand: 220 units across the 4-hour shift.
    takt = "4h"
    print("\n--- Yamazumi (station loading vs takt) ---")
    yam = lb.yamazumi(demand=220, available_time=takt)
    print(yam.to_string(index=False))

    print("\n--- Balance metrics per hour ---")
    bm = lb.balance_metrics(demand=220, available_time=takt, window="1h")
    cols = [
        "start",
        "n_stations",
        "bottleneck_uuid",
        "bottleneck_cycle_time",
        "balance_efficiency_pct",
        "balance_delay_pct",
        "smoothness_index",
        "theoretical_min_stations",
    ]
    print(bm[cols].to_string(index=False))
    print()


def demo_flow_metrics():
    print("=" * 72)
    print("2. FLOW METRICS (Little's Law)")
    print("=" * 72)

    df = build_flow_data()
    flow = FlowMetricsEvents(df, entry_uuid="process:in", exit_uuid="process:out")

    print("\n--- WIP over time ---")
    wip = flow.wip_over_time(window="1h")
    print(wip[["start", "wip_mean", "wip_max", "wip_min"]].to_string(index=False))

    print("\n--- Throughput ---")
    tp = flow.throughput(window="1h")
    print(tp[["start", "units_out", "throughput_per_hour"]].to_string(index=False))

    print("\n--- FIFO lead time ---")
    lt = flow.lead_time()
    print(f"  units measured: {len(lt)}")
    if not lt.empty:
        print(f"  mean lead time: {lt['lead_time_seconds'].mean():.1f} s")

    print("\n--- Flow summary (Little's Law + PCE) ---")
    fs = flow.flow_summary(value_add_seconds=150, window="1h")
    cols = [
        "start",
        "wip_mean",
        "throughput_per_hour",
        "lead_time_mean_seconds",
        "littles_law_lead_time_seconds",
        "consistency_ratio",
        "process_cycle_efficiency_pct",
    ]
    print(fs[cols].to_string(index=False))
    print()


if __name__ == "__main__":
    demo_line_balancing()
    demo_flow_metrics()

    print("=" * 72)
    print("Line & flow analytics demo complete.")
    print("=" * 72)
