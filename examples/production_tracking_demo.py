"""
Demonstration of production tracking in ts-shape.

This script shows how to use:
1. ShiftReportEvents (shift production, comparison, targets)
2. DowntimeEvents (downtime by shift and reason)
3. CycleTimeEvents (cycle time analysis by part number)
"""

import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from ts_shape.events.production.cycle_time_tracking import CycleTimeEvents
from ts_shape.events.production.downtime_tracking import DowntimeEvents
from ts_shape.events.production.shift_reporting import ShiftReportEvents


def create_shift_production_data():
    """Create synthetic production counter data spanning multiple shifts and days."""
    rows = []
    np.random.seed(42)

    for day_offset in range(5):
        base_date = datetime(2024, 1, 1 + day_offset)
        counter = 10000 + day_offset * 1500

        # Generate data every 10 minutes across all 3 shifts
        for hour in range(24):
            for minute in range(0, 60, 10):
                t = base_date + timedelta(hours=hour, minutes=minute)
                # Different production rates per shift
                if 6 <= hour < 14:
                    rate = np.random.randint(6, 10)  # Shift 1: higher rate
                elif 14 <= hour < 22:
                    rate = np.random.randint(5, 9)  # Shift 2: medium rate
                else:
                    rate = np.random.randint(4, 7)  # Shift 3: lower rate

                counter += rate
                rows.append(
                    {
                        "systime": t,
                        "uuid": "prod_counter",
                        "value_integer": counter,
                        "value_string": None,
                        "is_delta": True,
                    }
                )

        # Part number signal (changes once per day)
        parts = ["PART_X100", "PART_Y200", "PART_X100", "PART_Z300", "PART_Y200"]
        rows.append(
            {
                "systime": base_date + timedelta(hours=6),
                "uuid": "part_number",
                "value_integer": None,
                "value_string": parts[day_offset],
                "is_delta": True,
            }
        )

    return pd.DataFrame(rows)


def create_downtime_data():
    """Create synthetic machine state and downtime reason data."""
    rows = []
    np.random.seed(77)

    base_date = datetime(2024, 1, 1, 6, 0, 0)
    reasons = [
        "Material_Shortage",
        "Tool_Change",
        "Quality_Issue",
        "Operator_Break",
        "Mechanical_Failure",
    ]

    t = base_date
    for i in range(500):
        # Mostly running, with occasional stops
        if np.random.random() > 0.85:
            state = np.random.choice(["Stopped", "Idle"], p=[0.7, 0.3])
            reason = np.random.choice(reasons)
        else:
            state = "Running"
            reason = reasons[0]  # placeholder, not used when running

        rows.append(
            {
                "systime": t,
                "uuid": "machine_state",
                "value_string": state,
                "is_delta": True,
            }
        )

        rows.append(
            {
                "systime": t,
                "uuid": "downtime_reason",
                "value_string": reason,
                "is_delta": True,
            }
        )

        t += timedelta(minutes=np.random.randint(2, 8))

    return pd.DataFrame(rows)


def create_cycle_time_data():
    """Create synthetic cycle time data for multiple part numbers."""
    rows = []
    np.random.seed(33)

    base_time = datetime(2024, 1, 1, 8, 0, 0)
    parts = {
        "PART_A": {"cycle_time": 45.0, "std": 3.0, "count": 80},
        "PART_B": {"cycle_time": 62.0, "std": 5.0, "count": 60},
    }

    t = base_time
    for part_name, config in parts.items():
        # Set part number
        rows.append(
            {
                "systime": t,
                "uuid": "part_id_signal",
                "value_string": part_name,
                "value_bool": None,
                "is_delta": True,
            }
        )

        for i in range(config["count"]):
            cycle_time = max(20, np.random.normal(config["cycle_time"], config["std"]))
            # Inject a few slow cycles
            if i in [15, 40, 65]:
                cycle_time = config["cycle_time"] * 1.8

            t += timedelta(seconds=cycle_time)

            # Cycle trigger rising edge (False -> True)
            rows.append(
                {
                    "systime": t - timedelta(seconds=1),
                    "uuid": "cycle_complete",
                    "value_string": None,
                    "value_bool": False,
                    "is_delta": True,
                }
            )
            rows.append(
                {
                    "systime": t,
                    "uuid": "cycle_complete",
                    "value_string": None,
                    "value_bool": True,
                    "is_delta": True,
                }
            )

    return pd.DataFrame(rows)


def demo_shift_reporting():
    """Demo 1: Shift-based production reporting."""
    print("\n" + "=" * 70)
    print("DEMO 1: Shift Reporting")
    print("=" * 70)

    df = create_shift_production_data()
    print(f"\nDataset: {len(df)} records over 5 days")

    reporter = ShiftReportEvents(
        df,
        shift_definitions={
            "shift_1": ("06:00", "14:00"),
            "shift_2": ("14:00", "22:00"),
            "shift_3": ("22:00", "06:00"),
        },
    )

    # Production per shift
    print("\n--- Shift Production ---")
    shift_prod = reporter.shift_production(
        counter_uuid="prod_counter",
        value_column_counter="value_integer",
    )
    if not shift_prod.empty:
        print(shift_prod.to_string(index=False))

    # Shift comparison
    print("\n--- Shift Comparison (last 5 days) ---")
    comparison = reporter.shift_comparison(
        counter_uuid="prod_counter",
        value_column_counter="value_integer",
        days=5,
    )
    if not comparison.empty:
        print(comparison.to_string(index=False))

    # Shift targets
    print("\n--- Shift Target Achievement ---")
    targets = {"shift_1": 400, "shift_2": 380, "shift_3": 300}
    target_results = reporter.shift_targets(
        counter_uuid="prod_counter",
        targets=targets,
        value_column_counter="value_integer",
    )
    if not target_results.empty:
        print(target_results.to_string(index=False))

    # Best and worst shifts
    print("\n--- Best & Worst Shifts ---")
    results = reporter.best_and_worst_shifts(
        counter_uuid="prod_counter",
        value_column_counter="value_integer",
        days=5,
    )
    for rank_type in ("best", "worst"):
        ranked = results[results["rank_type"] == rank_type]
        print(f"  {rank_type.capitalize()} shifts:")
        if not ranked.empty:
            print(ranked.to_string(index=False))


def demo_downtime_tracking():
    """Demo 2: Downtime tracking by shift and reason."""
    print("\n" + "=" * 70)
    print("DEMO 2: Downtime Tracking")
    print("=" * 70)

    df = create_downtime_data()
    print(f"\nDataset: {len(df)} records")

    tracker = DowntimeEvents(
        df,
        shift_definitions={
            "shift_1": ("06:00", "14:00"),
            "shift_2": ("14:00", "22:00"),
            "shift_3": ("22:00", "06:00"),
        },
    )

    # Downtime by shift
    print("\n--- Downtime by Shift ---")
    shift_downtime = tracker.downtime_by_shift(
        state_uuid="machine_state",
        running_value="Running",
    )
    if not shift_downtime.empty:
        print(shift_downtime.to_string(index=False))

    # Downtime by reason
    print("\n--- Downtime by Reason ---")
    reason_analysis = tracker.downtime_by_reason(
        state_uuid="machine_state",
        reason_uuid="downtime_reason",
        stopped_value="Stopped",
    )
    if not reason_analysis.empty:
        print(reason_analysis.to_string(index=False))

    # Top downtime reasons (Pareto)
    print("\n--- Top 3 Downtime Reasons (Pareto) ---")
    top_reasons = tracker.top_downtime_reasons(
        state_uuid="machine_state",
        reason_uuid="downtime_reason",
        top_n=3,
        stopped_value="Stopped",
    )
    if not top_reasons.empty:
        print(top_reasons.to_string(index=False))

    # Availability trend
    print("\n--- Availability Trend (daily) ---")
    trend = tracker.availability_trend(
        state_uuid="machine_state",
        running_value="Running",
        window="1D",
    )
    if not trend.empty:
        print(trend.to_string(index=False))


def demo_cycle_time_tracking():
    """Demo 3: Cycle time analysis by part number."""
    print("\n" + "=" * 70)
    print("DEMO 3: Cycle Time Tracking")
    print("=" * 70)

    df = create_cycle_time_data()
    print(f"\nDataset: {len(df)} records (2 part types)")

    tracker = CycleTimeEvents(df)

    # Cycle times by part
    print("\n--- Cycle Times by Part ---")
    cycles = tracker.cycle_time_by_part(
        part_id_uuid="part_id_signal",
        cycle_trigger_uuid="cycle_complete",
    )
    if not cycles.empty:
        print(f"  Total cycles: {len(cycles)}")
        print(cycles.head(10).to_string(index=False))

    # Statistics
    print("\n--- Cycle Time Statistics ---")
    stats = tracker.cycle_time_statistics(
        part_id_uuid="part_id_signal",
        cycle_trigger_uuid="cycle_complete",
    )
    if not stats.empty:
        print(stats.to_string(index=False))

    # Detect slow cycles
    print("\n--- Slow Cycle Detection (threshold_factor=1.5) ---")
    slow = tracker.detect_slow_cycles(
        part_id_uuid="part_id_signal",
        cycle_trigger_uuid="cycle_complete",
        threshold_factor=1.5,
    )
    print(f"  Slow cycles found: {len(slow)}")
    if not slow.empty:
        print(
            slow[
                [
                    "systime",
                    "part_number",
                    "cycle_time_seconds",
                    "median_seconds",
                    "deviation_factor",
                ]
            ].to_string(index=False)
        )

    # Cycle time trend for PART_A
    print("\n--- Cycle Time Trend (PART_A, window=10) ---")
    trend = tracker.cycle_time_trend(
        part_id_uuid="part_id_signal",
        cycle_trigger_uuid="cycle_complete",
        part_number="PART_A",
        window_size=10,
    )
    if not trend.empty:
        print(f"  Trend data points: {len(trend)}")
        print(
            trend[["systime", "cycle_time_seconds", "moving_avg", "trend"]]
            .tail(10)
            .to_string(index=False)
        )

    # Hourly summary
    print("\n--- Hourly Cycle Time Summary ---")
    hourly = tracker.hourly_cycle_time_summary(
        part_id_uuid="part_id_signal",
        cycle_trigger_uuid="cycle_complete",
    )
    if not hourly.empty:
        print(hourly.to_string(index=False))


def main():
    """Run all production tracking demonstrations."""
    print("\n" + "=" * 70)
    print("Production Tracking Demonstration")
    print("=" * 70)

    try:
        demo_shift_reporting()
        demo_downtime_tracking()
        demo_cycle_time_tracking()

        print("\n" + "=" * 70)
        print("All demonstrations completed successfully!")
        print("=" * 70)

    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
