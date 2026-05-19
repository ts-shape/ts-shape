#!/usr/bin/env python3
"""
Comprehensive demonstration of all production event classes in ts-shape.

This script demonstrates every production event class and method:

  Event Detection:
    1. MachineStateEvents     -- run/idle intervals, transitions, rapid transitions, quality metrics
    2. LineThroughputEvents   -- part counting, takt adherence, OEE, trends, cycle quality
    3. ChangeoverEvents       -- product changeover detection, changeover windows
    4. FlowConstraintEvents   -- blocked/starved interval detection

  Daily Production Tracking:
    5. PartProductionTracking  -- production by part, daily summary, totals
    6. CycleTimeTracking       -- cycle times, statistics, slow cycle detection, trends
    7. ShiftReporting          -- shift production, comparison, targets, best/worst
    8. DowntimeTracking        -- downtime by shift/reason, top reasons, availability trend
    9. QualityTracking         -- NOK by shift/part/reason, daily quality summary

  OEE and Advanced Analytics:
   10. OEECalculator           -- availability, performance, quality, combined OEE
   11. AlarmManagementEvents   -- frequency, duration stats, chattering, standing alarms
   12. BatchTrackingEvents     -- batch detection, duration stats, yield, transition matrix

All classes work with a standard timeseries DataFrame whose columns are:
    systime (datetime), uuid (string), value_bool, value_integer,
    value_double, value_string, is_delta (bool).

Each class filters by uuid to isolate specific signals from the shared DataFrame.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# ---------------------------------------------------------------------------
# Allow import when running from the examples/ directory
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ts_shape.events.production import (
    MachineStateEvents,
    LineThroughputEvents,
    ChangeoverEvents,
    FlowConstraintEvents,
    PartProductionTracking,
    CycleTimeTracking,
    ShiftReporting,
    DowntimeTracking,
    QualityTracking,
    OEECalculator,
    AlarmManagementEvents,
    BatchTrackingEvents,
)

# ---------------------------------------------------------------------------
# Constants: signal UUIDs used throughout the demo
# ---------------------------------------------------------------------------
UUID_MACHINE_STATE = "machine_run_state"       # bool: True = running
UUID_PART_COUNTER = "part_counter"             # int: monotonic counter of produced parts
UUID_CYCLE_TRIGGER = "cycle_trigger"           # bool: rising edge marks cycle end
UUID_PRODUCT_ID = "product_id"                 # string: current product/recipe
UUID_PART_NUMBER = "part_number_signal"        # string: current part type
UUID_UPSTREAM_RUN = "upstream_run"             # bool: upstream station running
UUID_DOWNSTREAM_RUN = "downstream_run"         # bool: downstream station running
UUID_ALARM_TEMP = "alarm_temp_high"            # bool: temperature high alarm
UUID_BATCH_ID = "batch_id_signal"              # string: current batch identifier
UUID_MACHINE_STATE_STR = "machine_state_str"   # string: Running/Stopped/Idle
UUID_DOWNTIME_REASON = "downtime_reason"       # string: reason code
UUID_OK_COUNTER = "ok_parts_counter"           # int: good parts counter
UUID_NOK_COUNTER = "nok_parts_counter"         # int: reject parts counter
UUID_REJECT_COUNTER = "reject_counter"         # int: reject counter for OEE
UUID_TOTAL_COUNTER = "total_counter"           # int: total counter for OEE
UUID_DEFECT_REASON = "defect_reason"           # string: defect reason code


# ========================================================================
# Data generation helpers
# ========================================================================

def _row(t, uuid, *, value_bool=None, value_integer=None,
         value_double=None, value_string=None, is_delta=True):
    """Convenience helper to build a single data row."""
    return {
        "systime": t,
        "uuid": uuid,
        "value_bool": value_bool,
        "value_integer": value_integer,
        "value_double": value_double,
        "value_string": value_string,
        "is_delta": is_delta,
    }


def generate_manufacturing_data(
    start: datetime,
    days: int = 3,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Build a realistic multi-day manufacturing dataset.

    The factory runs 24 h in three 8-hour shifts.  Signals include machine
    state, part counters, product IDs, alarms, batch IDs, downtime reasons,
    OK/NOK counters, and upstream/downstream flow states.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    rows: list[dict] = []

    for day_offset in range(days):
        day_start = start + timedelta(days=day_offset)

        # -- Machine boolean run state (10-second resolution) ---------------
        t = day_start
        running = True
        while t < day_start + timedelta(days=1):
            if running:
                duration = int(rng.integers(180, 600))  # run 3-10 min
            else:
                duration = int(rng.integers(30, 180))   # idle 0.5-3 min
            for s in range(0, duration, 10):
                rows.append(_row(t + timedelta(seconds=s), UUID_MACHINE_STATE,
                                 value_bool=running))
            t += timedelta(seconds=duration)
            running = not running

        # -- Machine string state (Running / Stopped) ----------------------
        t = day_start
        state = "Running"
        reasons = ["Material_Shortage", "Tool_Change", "Quality_Issue",
                    "Planned_Maintenance", "Operator_Break"]
        while t < day_start + timedelta(days=1):
            if state == "Running":
                dur = int(rng.integers(300, 900))  # 5-15 min running
            else:
                dur = int(rng.integers(60, 300))   # 1-5 min stopped
            for s in range(0, dur, 30):
                rows.append(_row(t + timedelta(seconds=s),
                                 UUID_MACHINE_STATE_STR,
                                 value_string=state))
                # Attach a downtime reason when stopped
                if state == "Stopped":
                    rows.append(_row(t + timedelta(seconds=s),
                                     UUID_DOWNTIME_REASON,
                                     value_string=rng.choice(reasons)))
            t += timedelta(seconds=dur)
            state = "Stopped" if state == "Running" else "Running"

        # -- Monotonic part counter (every minute) --------------------------
        counter = 1000 + day_offset * 1500
        for minute in range(24 * 60):
            t = day_start + timedelta(minutes=minute)
            increment = int(rng.integers(1, 6))
            counter += increment
            rows.append(_row(t, UUID_PART_COUNTER,
                             value_integer=int(counter)))

        # -- Total / reject counters for OEE (every minute) -----------------
        total_ctr = 1000 + day_offset * 1400
        reject_ctr = 10 + day_offset * 40
        for minute in range(24 * 60):
            t = day_start + timedelta(minutes=minute)
            total_ctr += int(rng.integers(1, 6))
            if rng.random() < 0.04:
                reject_ctr += 1
            rows.append(_row(t, UUID_TOTAL_COUNTER,
                             value_integer=int(total_ctr)))
            rows.append(_row(t, UUID_REJECT_COUNTER,
                             value_integer=int(reject_ctr)))

        # -- OK / NOK counters (every minute) --------------------------------
        ok_ctr = 500 + day_offset * 1300
        nok_ctr = 5 + day_offset * 35
        for minute in range(24 * 60):
            t = day_start + timedelta(minutes=minute)
            ok_ctr += int(rng.integers(1, 6))
            if rng.random() < 0.05:
                nok_ctr += 1
            rows.append(_row(t, UUID_OK_COUNTER,
                             value_integer=int(ok_ctr)))
            rows.append(_row(t, UUID_NOK_COUNTER,
                             value_integer=int(nok_ctr)))

        # -- Defect reason (changes occasionally) ---------------------------
        defect_reasons = ["Dimension_Error", "Surface_Defect",
                          "Wrong_Color", "Missing_Feature"]
        current_reason = rng.choice(defect_reasons)
        for minute in range(0, 24 * 60, 15):
            t = day_start + timedelta(minutes=minute)
            if rng.random() < 0.3:
                current_reason = rng.choice(defect_reasons)
            rows.append(_row(t, UUID_DEFECT_REASON,
                             value_string=current_reason))

        # -- Boolean cycle trigger (rising edge every ~45 s) -----------------
        t = day_start
        while t < day_start + timedelta(days=1):
            cycle_time = max(20, 45 + rng.normal(0, 5))
            t += timedelta(seconds=cycle_time)
            rows.append(_row(t - timedelta(seconds=1), UUID_CYCLE_TRIGGER,
                             value_bool=False))
            rows.append(_row(t, UUID_CYCLE_TRIGGER, value_bool=True))

        # -- Part number / product ID (changes every ~2 h) ------------------
        parts = ["PART_A", "PART_B", "PART_C"]
        products = ["PRODUCT_X", "PRODUCT_Y", "PRODUCT_Z"]
        part_idx = 0
        prod_idx = 0
        for minute in range(24 * 60):
            t = day_start + timedelta(minutes=minute)
            if minute > 0 and minute % 120 == 0:
                part_idx = (part_idx + 1) % len(parts)
                prod_idx = (prod_idx + 1) % len(products)
            rows.append(_row(t, UUID_PART_NUMBER,
                             value_string=parts[part_idx]))
            rows.append(_row(t, UUID_PRODUCT_ID,
                             value_string=products[prod_idx]))

        # -- Upstream / downstream flow states (5-second resolution) --------
        for i in range(0, 24 * 720, 1):  # every 5 s for 24 h = 17 280 pts
            t = day_start + timedelta(seconds=i * 5)
            if t >= day_start + timedelta(days=1):
                break
            up_run = rng.random() > 0.08
            dn_run = rng.random() > 0.12
            # Inject deliberate blocked window (10 min around hour 4)
            hour = (t - day_start).total_seconds() / 3600
            if 4.0 <= hour <= 4.17:
                up_run, dn_run = True, False
            if 8.0 <= hour <= 8.1:
                up_run, dn_run = False, True
            rows.append(_row(t, UUID_UPSTREAM_RUN, value_bool=bool(up_run)))
            rows.append(_row(t, UUID_DOWNSTREAM_RUN, value_bool=bool(dn_run)))

        # -- Temperature high alarm (bool, with chattering window) ----------
        t = day_start
        alarm_on = False
        while t < day_start + timedelta(days=1):
            if not alarm_on:
                gap = int(rng.integers(600, 7200))
                t += timedelta(seconds=gap)
                alarm_on = True
            else:
                # Stay on for 30 s - 2 h (some become standing alarms)
                on_dur = int(rng.integers(30, 7200))
                for s in range(0, on_dur, 10):
                    rows.append(_row(t + timedelta(seconds=s),
                                     UUID_ALARM_TEMP, value_bool=True))
                t += timedelta(seconds=on_dur)
                alarm_on = False
                # Insert chattering burst around hour 16
                if 15.5 < (t - day_start).total_seconds() / 3600 < 16.5:
                    for chatter in range(12):
                        rows.append(_row(t, UUID_ALARM_TEMP,
                                         value_bool=bool(chatter % 2 == 0)))
                        t += timedelta(seconds=3)
            if not alarm_on:
                rows.append(_row(t, UUID_ALARM_TEMP, value_bool=False))

        # -- Batch ID (string, changes every ~4 h) -------------------------
        batch_names = [f"BATCH-{day_offset * 10 + b:04d}" for b in range(6)]
        batch_idx = 0
        for minute in range(24 * 60):
            t = day_start + timedelta(minutes=minute)
            if minute > 0 and minute % 240 == 0:
                batch_idx = min(batch_idx + 1, len(batch_names) - 1)
            rows.append(_row(t, UUID_BATCH_ID,
                             value_string=batch_names[batch_idx]))

    df = pd.DataFrame(rows)
    df["systime"] = pd.to_datetime(df["systime"])
    return df.sort_values("systime").reset_index(drop=True)


# ========================================================================
# Utility for printing section headers
# ========================================================================

def _header(title: str, char: str = "=", width: int = 72) -> None:
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def _subheader(title: str) -> None:
    print(f"\n  --- {title} ---")


# ========================================================================
# Demonstration functions -- one per class
# ========================================================================

def demo_machine_state(df: pd.DataFrame) -> None:
    """1. MachineStateEvents: detect_run_idle, transition_events,
    detect_rapid_transitions, state_quality_metrics."""
    _header("1. MachineStateEvents")

    machine = MachineStateEvents(
        dataframe=df,
        run_state_uuid=UUID_MACHINE_STATE,
        event_uuid="demo:machine_state",
    )

    # detect_run_idle -- full
    _subheader("detect_run_idle()")
    intervals = machine.detect_run_idle()
    print(f"  Total intervals detected: {len(intervals)}")
    if not intervals.empty:
        run = intervals[intervals["state"] == "run"]
        idle = intervals[intervals["state"] == "idle"]
        print(f"  Run intervals : {len(run)}, avg {run['duration_seconds'].mean():.0f}s")
        print(f"  Idle intervals: {len(idle)}, avg {idle['duration_seconds'].mean():.0f}s")
        print(intervals[["start", "end", "state", "duration_seconds"]].head())

    # detect_run_idle -- with minimum duration filter
    _subheader("detect_run_idle(min_duration='60s')")
    filtered = machine.detect_run_idle(min_duration="60s")
    print(f"  Intervals after 60 s filter: {len(filtered)}")

    # transition_events
    _subheader("transition_events()")
    transitions = machine.transition_events()
    print(f"  Total transitions: {len(transitions)}")
    if not transitions.empty:
        print(transitions[["systime", "transition",
                           "time_since_last_transition_seconds"]].head())

    # detect_rapid_transitions
    _subheader("detect_rapid_transitions(threshold='30s', min_count=3)")
    rapid = machine.detect_rapid_transitions(threshold="30s", min_count=3)
    print(f"  Rapid transition bursts found: {len(rapid)}")
    if not rapid.empty:
        print(rapid.head())

    # state_quality_metrics
    _subheader("state_quality_metrics()")
    metrics = machine.state_quality_metrics()
    for k, v in metrics.items():
        print(f"  {k}: {v}")


def demo_line_throughput(df: pd.DataFrame) -> None:
    """2. LineThroughputEvents: count_parts, takt_adherence, throughput_oee,
    throughput_trends, cycle_quality_check."""
    _header("2. LineThroughputEvents")

    lt = LineThroughputEvents(dataframe=df, event_uuid="demo:throughput")

    # count_parts
    _subheader("count_parts(window='5min')")
    parts = lt.count_parts(UUID_PART_COUNTER, value_column="value_integer",
                           window="5min")
    print(f"  Windows: {len(parts)}")
    if not parts.empty:
        print(f"  Avg parts/window: {parts['count'].mean():.1f}")
        print(parts[["start", "count"]].head())

    # takt_adherence
    _subheader("takt_adherence(takt_time='50s')")
    takt = lt.takt_adherence(UUID_CYCLE_TRIGGER, value_column="value_bool",
                             takt_time="50s")
    print(f"  Cycles measured: {len(takt)}")
    if not takt.empty:
        violations = takt[takt["violation"]]
        print(f"  Violations: {len(violations)} "
              f"({len(violations) / len(takt) * 100:.1f}%)")
        print(f"  Avg cycle time: {takt['cycle_time_seconds'].mean():.1f}s")
        print(takt[["systime", "cycle_time_seconds", "violation"]].head())

    # throughput_oee
    _subheader("throughput_oee(window='1h')")
    oee = lt.throughput_oee(UUID_PART_COUNTER, value_column="value_integer",
                            window="1h")
    if not oee.empty:
        print(f"  Avg OEE score: {oee['oee_score'].mean():.2f}")
        print(oee[["start", "actual_count", "performance",
                    "oee_score"]].head())

    # throughput_trends
    _subheader("throughput_trends(window='1h', trend_window=6)")
    trends = lt.throughput_trends(UUID_PART_COUNTER,
                                  value_column="value_integer",
                                  window="1h", trend_window=6)
    if not trends.empty:
        print(f"  Windows with trend data: {len(trends)}")
        print(trends[["start", "count", "moving_avg",
                       "trend_direction", "degradation_detected"]].head())

    # cycle_quality_check
    _subheader("cycle_quality_check(tolerance_pct=0.15)")
    cqc = lt.cycle_quality_check(UUID_CYCLE_TRIGGER,
                                  value_column="value_bool",
                                  tolerance_pct=0.15)
    if not cqc.empty:
        print(f"  Cycles checked: {len(cqc)}")
        good = cqc[cqc["quality_flag"] == "good"]
        print(f"  Good cycles: {len(good)} "
              f"({len(good) / len(cqc) * 100:.1f}%)")
        print(cqc[["systime", "cycle_time_seconds", "deviation_pct",
                    "quality_flag"]].head())


def demo_changeover(df: pd.DataFrame) -> None:
    """3. ChangeoverEvents: detect_changeover, changeover_window."""
    _header("3. ChangeoverEvents")

    co = ChangeoverEvents(dataframe=df, event_uuid="demo:changeover")

    # detect_changeover
    _subheader("detect_changeover()")
    changes = co.detect_changeover(UUID_PRODUCT_ID,
                                    value_column="value_string")
    print(f"  Changeovers detected: {len(changes)}")
    if not changes.empty:
        print(changes[["systime", "new_value"]].head(10))

    # changeover_window -- fixed
    _subheader("changeover_window(until='fixed_window', duration='15m')")
    windows = co.changeover_window(
        UUID_PRODUCT_ID,
        value_column="value_string",
        until="fixed_window",
        config={"duration": "15m"},
    )
    if not windows.empty:
        print(f"  Changeover windows: {len(windows)}")
        print(windows[["start", "end", "method", "completed"]].head())


def demo_flow_constraints(df: pd.DataFrame) -> None:
    """4. FlowConstraintEvents: blocked_events, starved_events."""
    _header("4. FlowConstraintEvents")

    flow = FlowConstraintEvents(dataframe=df, event_uuid="demo:flow")
    roles = {
        "upstream_run": UUID_UPSTREAM_RUN,
        "downstream_run": UUID_DOWNSTREAM_RUN,
    }

    # blocked_events
    _subheader("blocked_events(min_duration='10s')")
    blocked = flow.blocked_events(roles=roles, tolerance="10s",
                                   min_duration="10s")
    print(f"  Blocked events: {len(blocked)}")
    if not blocked.empty:
        print(blocked[["start", "end", "type", "severity",
                        "duration"]].head())

    # starved_events
    _subheader("starved_events(min_duration='10s')")
    starved = flow.starved_events(roles=roles, tolerance="10s",
                                   min_duration="10s")
    print(f"  Starved events: {len(starved)}")
    if not starved.empty:
        print(starved[["start", "end", "type", "severity",
                        "duration"]].head())

    # flow_constraint_analytics (combines both)
    _subheader("flow_constraint_analytics()")
    analytics = flow.flow_constraint_analytics(roles=roles, tolerance="10s",
                                                min_duration="10s")
    s = analytics["summary"]
    print(f"  Total constraint events : {s['total_constraint_events']}")
    print(f"  Blocked count / duration: {s['blocked_count']} / "
          f"{s['blocked_total_duration']}")
    print(f"  Starved count / duration: {s['starved_count']} / "
          f"{s['starved_total_duration']}")
    print(f"  Alignment quality       : {s['overall_alignment_quality']:.2f}")


def demo_part_production_tracking(df: pd.DataFrame) -> None:
    """5. PartProductionTracking: production_by_part, daily_production_summary,
    production_totals."""
    _header("5. PartProductionTracking")

    ppt = PartProductionTracking(dataframe=df)

    # production_by_part (hourly)
    _subheader("production_by_part(window='1h')")
    hourly = ppt.production_by_part(
        part_id_uuid=UUID_PART_NUMBER,
        counter_uuid=UUID_PART_COUNTER,
        window="1h",
    )
    print(f"  Rows: {len(hourly)}")
    if not hourly.empty:
        print(hourly[["start", "part_number", "quantity"]].head(8))

    # daily_production_summary
    _subheader("daily_production_summary()")
    daily = ppt.daily_production_summary(
        part_id_uuid=UUID_PART_NUMBER,
        counter_uuid=UUID_PART_COUNTER,
    )
    print(f"  Rows: {len(daily)}")
    if not daily.empty:
        print(daily.head(8))

    # production_totals
    _subheader("production_totals()")
    totals = ppt.production_totals(
        part_id_uuid=UUID_PART_NUMBER,
        counter_uuid=UUID_PART_COUNTER,
    )
    print(f"  Parts tracked: {len(totals)}")
    if not totals.empty:
        print(totals)


def demo_cycle_time_tracking(df: pd.DataFrame) -> None:
    """6. CycleTimeTracking: cycle_time_by_part, cycle_time_statistics,
    detect_slow_cycles, cycle_time_trend."""
    _header("6. CycleTimeTracking")

    ctt = CycleTimeTracking(dataframe=df)

    # cycle_time_by_part
    _subheader("cycle_time_by_part()")
    cycles = ctt.cycle_time_by_part(
        part_id_uuid=UUID_PART_NUMBER,
        cycle_trigger_uuid=UUID_CYCLE_TRIGGER,
    )
    print(f"  Cycle records: {len(cycles)}")
    if not cycles.empty:
        print(cycles[["systime", "part_number",
                       "cycle_time_seconds"]].head())

    # cycle_time_statistics
    _subheader("cycle_time_statistics()")
    stats = ctt.cycle_time_statistics(
        part_id_uuid=UUID_PART_NUMBER,
        cycle_trigger_uuid=UUID_CYCLE_TRIGGER,
    )
    if not stats.empty:
        print(stats)

    # detect_slow_cycles
    _subheader("detect_slow_cycles(threshold_factor=1.5)")
    slow = ctt.detect_slow_cycles(
        part_id_uuid=UUID_PART_NUMBER,
        cycle_trigger_uuid=UUID_CYCLE_TRIGGER,
        threshold_factor=1.5,
    )
    print(f"  Slow cycles detected: {len(slow)}")
    if not slow.empty:
        print(slow[["systime", "part_number", "cycle_time_seconds",
                     "deviation_factor"]].head())

    # cycle_time_trend for PART_A
    _subheader("cycle_time_trend(part_number='PART_A', window_size=10)")
    trend = ctt.cycle_time_trend(
        part_id_uuid=UUID_PART_NUMBER,
        cycle_trigger_uuid=UUID_CYCLE_TRIGGER,
        part_number="PART_A",
        window_size=10,
    )
    print(f"  Trend rows: {len(trend)}")
    if not trend.empty:
        print(trend[["systime", "cycle_time_seconds", "moving_avg",
                      "trend"]].head())


def demo_shift_reporting(df: pd.DataFrame) -> None:
    """7. ShiftReporting: shift_production, shift_comparison, shift_targets,
    best_and_worst_shifts."""
    _header("7. ShiftReporting")

    shifts = {
        "day": ("06:00", "14:00"),
        "afternoon": ("14:00", "22:00"),
        "night": ("22:00", "06:00"),
    }
    sr = ShiftReporting(dataframe=df, shift_definitions=shifts)

    # shift_production
    _subheader("shift_production()")
    prod = sr.shift_production(
        counter_uuid=UUID_PART_COUNTER,
        part_id_uuid=UUID_PART_NUMBER,
    )
    print(f"  Rows: {len(prod)}")
    if not prod.empty:
        print(prod.head(8))

    # shift_comparison
    _subheader("shift_comparison(days=3)")
    comp = sr.shift_comparison(counter_uuid=UUID_PART_COUNTER, days=3)
    if not comp.empty:
        print(comp)

    # shift_targets
    _subheader("shift_targets()")
    targets = {"day": 500, "afternoon": 480, "night": 450}
    target_df = sr.shift_targets(counter_uuid=UUID_PART_COUNTER,
                                  targets=targets)
    if not target_df.empty:
        print(target_df.head(8))

    # best_and_worst_shifts
    _subheader("best_and_worst_shifts(days=3)")
    bw = sr.best_and_worst_shifts(counter_uuid=UUID_PART_COUNTER, days=3)
    print("  Best shifts:")
    print(bw["best"])
    print("  Worst shifts:")
    print(bw["worst"])


def demo_downtime_tracking(df: pd.DataFrame) -> None:
    """8. DowntimeTracking: downtime_by_shift, downtime_by_reason,
    top_downtime_reasons, availability_trend."""
    _header("8. DowntimeTracking")

    shifts = {
        "day": ("06:00", "14:00"),
        "afternoon": ("14:00", "22:00"),
        "night": ("22:00", "06:00"),
    }
    dt = DowntimeTracking(dataframe=df, shift_definitions=shifts)

    # downtime_by_shift
    _subheader("downtime_by_shift()")
    by_shift = dt.downtime_by_shift(
        state_uuid=UUID_MACHINE_STATE_STR,
        running_value="Running",
    )
    print(f"  Rows: {len(by_shift)}")
    if not by_shift.empty:
        print(by_shift.head(8))

    # downtime_by_reason
    _subheader("downtime_by_reason()")
    by_reason = dt.downtime_by_reason(
        state_uuid=UUID_MACHINE_STATE_STR,
        reason_uuid=UUID_DOWNTIME_REASON,
        stopped_value="Stopped",
    )
    if not by_reason.empty:
        print(by_reason)

    # top_downtime_reasons
    _subheader("top_downtime_reasons(top_n=3)")
    top = dt.top_downtime_reasons(
        state_uuid=UUID_MACHINE_STATE_STR,
        reason_uuid=UUID_DOWNTIME_REASON,
        top_n=3,
        stopped_value="Stopped",
    )
    if not top.empty:
        print(top)

    # availability_trend
    _subheader("availability_trend(window='1D')")
    avail = dt.availability_trend(
        state_uuid=UUID_MACHINE_STATE_STR,
        running_value="Running",
        window="1D",
    )
    if not avail.empty:
        print(avail)


def demo_quality_tracking(df: pd.DataFrame) -> None:
    """9. QualityTracking: nok_by_shift, quality_by_part, nok_by_reason,
    daily_quality_summary."""
    _header("9. QualityTracking")

    shifts = {
        "day": ("06:00", "14:00"),
        "afternoon": ("14:00", "22:00"),
        "night": ("22:00", "06:00"),
    }
    qt = QualityTracking(dataframe=df, shift_definitions=shifts)

    # nok_by_shift
    _subheader("nok_by_shift()")
    nok_shift = qt.nok_by_shift(
        ok_counter_uuid=UUID_OK_COUNTER,
        nok_counter_uuid=UUID_NOK_COUNTER,
    )
    print(f"  Rows: {len(nok_shift)}")
    if not nok_shift.empty:
        print(nok_shift.head(8))

    # quality_by_part
    _subheader("quality_by_part()")
    by_part = qt.quality_by_part(
        ok_counter_uuid=UUID_OK_COUNTER,
        nok_counter_uuid=UUID_NOK_COUNTER,
        part_id_uuid=UUID_PART_NUMBER,
    )
    if not by_part.empty:
        print(by_part)

    # nok_by_reason
    _subheader("nok_by_reason()")
    by_reason = qt.nok_by_reason(
        nok_counter_uuid=UUID_NOK_COUNTER,
        defect_reason_uuid=UUID_DEFECT_REASON,
    )
    if not by_reason.empty:
        print(by_reason)

    # daily_quality_summary
    _subheader("daily_quality_summary()")
    daily = qt.daily_quality_summary(
        ok_counter_uuid=UUID_OK_COUNTER,
        nok_counter_uuid=UUID_NOK_COUNTER,
    )
    if not daily.empty:
        print(daily)


def demo_oee_calculator(df: pd.DataFrame) -> None:
    """10. OEECalculator: calculate_availability, calculate_performance,
    calculate_quality, calculate_oee."""
    _header("10. OEECalculator")

    oee = OEECalculator(dataframe=df)

    # calculate_availability
    _subheader("calculate_availability()")
    avail = oee.calculate_availability(UUID_MACHINE_STATE)
    if not avail.empty:
        print(avail)

    # calculate_performance (ideal cycle time = 30 s per part)
    _subheader("calculate_performance(ideal_cycle_time=30)")
    perf = oee.calculate_performance(
        UUID_PART_COUNTER,
        ideal_cycle_time=30.0,
        run_state_uuid=UUID_MACHINE_STATE,
    )
    if not perf.empty:
        print(perf)

    # calculate_quality
    _subheader("calculate_quality()")
    qual = oee.calculate_quality(UUID_TOTAL_COUNTER, UUID_REJECT_COUNTER)
    if not qual.empty:
        print(qual)

    # calculate_oee (combined daily OEE)
    _subheader("calculate_oee()")
    daily_oee = oee.calculate_oee(
        run_state_uuid=UUID_MACHINE_STATE,
        counter_uuid=UUID_PART_COUNTER,
        ideal_cycle_time=30.0,
        total_uuid=UUID_TOTAL_COUNTER,
        reject_uuid=UUID_REJECT_COUNTER,
    )
    if not daily_oee.empty:
        print(daily_oee)


def demo_alarm_management(df: pd.DataFrame) -> None:
    """11. AlarmManagementEvents: alarm_frequency, alarm_duration_stats,
    chattering_detection, standing_alarms."""
    _header("11. AlarmManagementEvents")

    alarms = AlarmManagementEvents(
        dataframe=df,
        alarm_uuid=UUID_ALARM_TEMP,
        event_uuid="demo:alarm",
    )

    # alarm_frequency
    _subheader("alarm_frequency(window='4h')")
    freq = alarms.alarm_frequency(window="4h")
    if not freq.empty:
        print(f"  Windows with alarm activity: "
              f"{(freq['alarm_count'] > 0).sum()} / {len(freq)}")
        print(freq[freq["alarm_count"] > 0].head(8))

    # alarm_duration_stats
    _subheader("alarm_duration_stats()")
    stats = alarms.alarm_duration_stats()
    if not stats.empty:
        print(stats)

    # chattering_detection
    _subheader("chattering_detection(min_transitions=5, window='5min')")
    chatter = alarms.chattering_detection(min_transitions=5, window="5min")
    print(f"  Chattering windows: {len(chatter)}")
    if not chatter.empty:
        print(chatter.head())

    # standing_alarms
    _subheader("standing_alarms(min_duration='30min')")
    standing = alarms.standing_alarms(min_duration="30min")
    print(f"  Standing alarms: {len(standing)}")
    if not standing.empty:
        print(standing[["start", "end", "duration_seconds"]].head())


def demo_batch_tracking(df: pd.DataFrame) -> None:
    """12. BatchTrackingEvents: detect_batches, batch_duration_stats,
    batch_yield, batch_transition_matrix."""
    _header("12. BatchTrackingEvents")

    bt = BatchTrackingEvents(
        dataframe=df,
        batch_uuid=UUID_BATCH_ID,
        event_uuid="demo:batch",
    )

    # detect_batches
    _subheader("detect_batches()")
    batches = bt.detect_batches()
    print(f"  Batches detected: {len(batches)}")
    if not batches.empty:
        print(batches[["batch_id", "start", "end",
                        "duration_seconds", "sample_count"]].head(8))

    # batch_duration_stats
    _subheader("batch_duration_stats()")
    stats = bt.batch_duration_stats()
    if not stats.empty:
        print(stats)

    # batch_yield (using part counter)
    _subheader("batch_yield(counter_uuid)")
    by = bt.batch_yield(UUID_PART_COUNTER)
    if not by.empty:
        print(by[["batch_id", "start", "end", "quantity"]].head(8))

    # batch_transition_matrix
    _subheader("batch_transition_matrix()")
    matrix = bt.batch_transition_matrix()
    if not matrix.empty:
        print(matrix)


# ========================================================================
# Main entry point
# ========================================================================

def main() -> int:
    _header("ts-shape Production Events -- Comprehensive Demo", "=", 72)
    print("\n  Generating 3 days of synthetic manufacturing data ...")

    df = generate_manufacturing_data(
        start=datetime(2024, 6, 1, 0, 0, 0),
        days=3,
    )
    print(f"  Dataset: {len(df):,} rows, "
          f"{df['uuid'].nunique()} distinct UUIDs, "
          f"{df['systime'].min()} to {df['systime'].max()}")

    # Show the distinct signal UUIDs present in the dataset
    print("\n  Signal UUIDs in dataset:")
    for uid in sorted(df["uuid"].unique()):
        count = (df["uuid"] == uid).sum()
        print(f"    {uid:30s}  ({count:>6,} samples)")

    # Run every demo
    try:
        demo_machine_state(df)
        demo_line_throughput(df)
        demo_changeover(df)
        demo_flow_constraints(df)
        demo_part_production_tracking(df)
        demo_cycle_time_tracking(df)
        demo_shift_reporting(df)
        demo_downtime_tracking(df)
        demo_quality_tracking(df)
        demo_oee_calculator(df)
        demo_alarm_management(df)
        demo_batch_tracking(df)
    except Exception as e:
        print(f"\n  ERROR during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1

    _header("All 12 production event demos completed successfully!", "=", 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
