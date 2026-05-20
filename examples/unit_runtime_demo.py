#!/usr/bin/env python3
"""
Unit Conversion & Runtime Accounting Demo for ts-shape
=======================================================

Demonstrates two practical engineering utilities:

1. UnitConverter          - engineering unit conversion backed by ``pint``
   (install with ``pip install ts-shape[units]``).
2. RuntimeAccountingEvents - operating-hours accounting from a run signal:
   total run time, start count, longest run, and an hour-meter.

Scenario: a pump logs discharge pressure in bar and a boolean run state.
"""

import pandas as pd

from ts_shape.events.production.runtime_accounting import RuntimeAccountingEvents
from ts_shape.transform.calculator.unit_conversion import PINT_AVAILABLE, UnitConverter


def build_run_signal() -> pd.DataFrame:
    """Pump run state: change-events over an 8-hour shift."""
    # (state, duration_seconds): run 3 h, stop 1 h, run 2 h, stop 0.5 h, run 1.5 h
    segments = [
        (True, 3 * 3600),
        (False, 1 * 3600),
        (True, 2 * 3600),
        (False, 1800),
        (True, 5400),
    ]
    t = pd.Timestamp("2025-06-10 06:00:00")
    rows = []
    for state, dur in segments:
        rows.append({"systime": t, "uuid": "pump:run", "value_bool": state})
        t += pd.Timedelta(seconds=dur)
    rows.append({"systime": t, "uuid": "pump:run", "value_bool": False})
    df = pd.DataFrame(rows)
    df["value_double"] = float("nan")
    return df


def demo_unit_conversion():
    print("=" * 72)
    print("1. UNIT CONVERSION")
    print("=" * 72)

    if not PINT_AVAILABLE:
        print("\n  pint not installed -- run: pip install ts-shape[units]\n")
        return

    print("\n--- Scalar conversions ---")
    print(f"  100 C    -> {UnitConverter.convert_value(100, 'C', 'F'):.2f} F")
    print(f"  10 bar   -> {UnitConverter.convert_value(10, 'bar', 'psi'):.2f} psi")
    print(
        f"  5 m^3/h  -> {UnitConverter.convert_value(5, 'm^3/hour', 'L/min'):.2f} L/min"
    )

    print("\n--- Automatic conversion factor (scale, offset) ---")
    print(f"  bar -> psi: {UnitConverter.conversion_factor('bar', 'psi')}")
    print(f"  C   -> F:   {UnitConverter.conversion_factor('C', 'F')}")

    print("\n--- Convert a DataFrame column (bar -> psi) ---")
    df = pd.DataFrame(
        {
            "systime": pd.date_range("2025-06-10 06:00", periods=4, freq="h"),
            "uuid": "pump:pressure",
            "value_double": [4.2, 5.1, 4.8, 5.5],
        }
    )
    out = UnitConverter.convert_column(
        df, "bar", "psi", column_name="value_double", target_column="value_psi"
    )
    print(out[["systime", "value_double", "value_psi"]].round(2).to_string(index=False))
    print()


def demo_runtime_accounting():
    print("=" * 72)
    print("2. RUNTIME ACCOUNTING")
    print("=" * 72)

    df = build_run_signal()
    rt = RuntimeAccountingEvents(df, run_uuid="pump:run")

    print("\n--- Runtime summary ---")
    summary = rt.runtime_summary()
    cols = [
        "run_hours",
        "idle_seconds",
        "start_count",
        "longest_run_seconds",
        "mean_run_seconds",
        "utilization_pct",
    ]
    print(summary[cols].to_string(index=False))

    print("\n--- Runtime per window (1 h) ---")
    per_window = rt.runtime_per_window(window="1h")
    print(
        per_window[["start", "run_hours", "start_count", "utilization_pct"]]
        .head(8)
        .to_string(index=False)
    )

    print("\n--- Operating-hours meter (1 h windows) ---")
    meter = rt.operating_hours_meter(window="1h")
    print(meter[["start", "cumulative_run_hours"]].to_string(index=False))
    print()


if __name__ == "__main__":
    demo_unit_conversion()
    demo_runtime_accounting()

    print("=" * 72)
    print("Unit conversion & runtime accounting demo complete.")
    print("=" * 72)
