"""Energy Events Demo

Demonstrates the energy consumption and efficiency tracking packs
for manufacturing energy management.

Run: python examples/energy_events_demo.py
"""

import pandas as pd
import numpy as np

from ts_shape.events.energy.consumption_analysis import EnergyConsumptionEvents
from ts_shape.events.energy.efficiency_tracking import EnergyEfficiencyEvents


def create_energy_data(days: int = 7) -> pd.DataFrame:
    """Create realistic energy + production + machine state timeseries."""
    np.random.seed(42)
    base = pd.Timestamp("2024-01-01")
    rows = []

    for i in range(days * 24 * 60):  # per-minute readings
        t = base + pd.Timedelta(minutes=i)
        hour = t.hour

        # Energy meter: higher consumption during production shifts
        if 6 <= hour < 14:
            energy = 55 + np.random.normal(0, 5)  # Shift 1: highest
        elif 14 <= hour < 22:
            energy = 48 + np.random.normal(0, 5)  # Shift 2: moderate
        else:
            energy = 8 + np.random.normal(0, 2)  # Night: standby

        rows.append({
            "systime": t, "uuid": "meter:main",
            "value_double": max(0, energy), "value_integer": None,
            "value_bool": None, "is_delta": True,
        })

        # Production counter (monotonically increasing during shifts)
        if 6 <= hour < 22:
            counter = 1000 + i
        else:
            counter = 1000 + max(0, (i // (24 * 60)) * 16 * 60)

        rows.append({
            "systime": t, "uuid": "counter:line1",
            "value_double": None, "value_integer": counter,
            "value_bool": None, "is_delta": True,
        })

        # Machine state (boolean)
        running = 6 <= hour < 22
        rows.append({
            "systime": t, "uuid": "state:machine1",
            "value_double": None, "value_integer": None,
            "value_bool": running, "is_delta": True,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("=" * 70)
    print("ENERGY EVENTS DEMO")
    print("=" * 70)

    df = create_energy_data(days=7)
    print(f"\nCreated dataset: {len(df)} rows, {df['uuid'].nunique()} signals, "
          f"{(df['systime'].max() - df['systime'].min()).days} days\n")

    # -----------------------------------------------------------------------
    # 1. Consumption Analysis
    # -----------------------------------------------------------------------
    print("-" * 70)
    print("1. ENERGY CONSUMPTION ANALYSIS")
    print("-" * 70)

    ec = EnergyConsumptionEvents(df)

    # Hourly consumption
    hourly = ec.consumption_by_window("meter:main", window="1h")
    print(f"\nHourly consumption (first 5 rows):")
    print(hourly.head())

    # Peak demand detection
    peaks = ec.peak_demand_detection("meter:main", window="1h", percentile=0.90)
    peak_hours = peaks[peaks["is_peak"]]
    print(f"\nPeak demand windows detected: {len(peak_hours)} out of {len(peaks)}")
    print(f"Threshold: {peaks['threshold'].iloc[0]:.1f}")
    if not peak_hours.empty:
        print(f"Sample peak windows:")
        print(peak_hours[["start", "demand", "is_peak"]].head())

    # Baseline deviation
    deviation = ec.consumption_baseline_deviation(
        "meter:main", window="1h", baseline_periods=12
    )
    anomalies = deviation[deviation["is_anomaly"]]
    print(f"\nBaseline deviation anomalies: {len(anomalies)} out of {len(deviation)} windows")

    # Energy per unit
    epu = ec.energy_per_unit("meter:main", "counter:line1", window="1h")
    print(f"\nEnergy per unit (hourly, first 5 rows):")
    print(epu[["start", "energy", "units_produced", "energy_per_unit"]].head())

    # -----------------------------------------------------------------------
    # 2. Efficiency Tracking
    # -----------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("2. ENERGY EFFICIENCY TRACKING")
    print("-" * 70)

    ee = EnergyEfficiencyEvents(df)

    # Efficiency trend
    trend = ee.efficiency_trend(
        "meter:main", "counter:line1", window="1h", trend_window=12
    )
    print(f"\nEfficiency trend (first 5 rows):")
    print(trend[["start", "efficiency", "rolling_avg_efficiency", "trend_direction"]].head())

    # Idle energy waste
    waste = ee.idle_energy_waste("meter:main", "state:machine1", window="1h")
    idle_waste = waste[waste["is_idle_waste"]]
    total_waste = idle_waste["waste_energy"].sum()
    print(f"\nIdle energy waste: {len(idle_waste)} windows, total waste = {total_waste:.1f}")

    # Specific energy consumption
    sec = ee.specific_energy_consumption("meter:main", "counter:line1", window="1D")
    print(f"\nDaily specific energy consumption:")
    print(sec[["start", "total_energy", "total_output", "sec", "sec_trend"]])

    # Shift efficiency comparison
    comparison = ee.efficiency_comparison("meter:main", "counter:line1")
    print(f"\nShift efficiency comparison:")
    print(comparison)

    print("\n" + "=" * 70)
    print("Demo complete.")
    print("=" * 70)
