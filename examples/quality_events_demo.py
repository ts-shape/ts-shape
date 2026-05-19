#!/usr/bin/env python3
"""
Quality Events Demo for ts-shape
=================================

Demonstrates quality event detection classes for manufacturing measurement data:
1. OutlierDetectionEvents  - Detect outliers via Z-score, IQR, MAD, and Isolation Forest
2. StatisticalProcessControlRuleBased - Western Electric SPC rules and CUSUM analysis
3. ToleranceDeviationEvents - Tolerance deviation detection with severity classification

Scenario: A CNC milling machine produces parts with a target bore diameter of 25.000 mm.
Sensor readings are collected every 30 seconds over an 8-hour shift.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from ts_shape.events.quality.outlier_detection import OutlierDetectionEvents
from ts_shape.events.quality.statistical_process_control import StatisticalProcessControlRuleBased
from ts_shape.events.quality.tolerance_deviation import ToleranceDeviationEvents


# ---------------------------------------------------------------------------
# Helper: build a realistic manufacturing timeseries DataFrame
# ---------------------------------------------------------------------------

def create_bore_diameter_data(n_points: int = 960) -> pd.DataFrame:
    """
    Simulate 8 hours of bore-diameter measurements (one every 30 s).

    The signal has:
      - a stable baseline around 25.000 mm with normal noise (sigma ~0.003)
      - a slow upward drift starting at sample 600 (simulating tool wear)
      - a handful of spike outliers injected at known positions
    """
    np.random.seed(42)
    start = datetime(2025, 6, 10, 6, 0, 0)
    times = [start + timedelta(seconds=30 * i) for i in range(n_points)]

    # Baseline + noise
    values = 25.000 + np.random.normal(0, 0.003, n_points)

    # Gradual tool-wear drift after sample 600
    drift = np.zeros(n_points)
    drift[600:] = np.linspace(0, 0.012, n_points - 600)
    values += drift

    # Inject obvious outlier spikes
    spike_indices = [120, 350, 351, 780]
    for idx in spike_indices:
        values[idx] += np.random.choice([-1, 1]) * np.random.uniform(0.025, 0.040)

    df = pd.DataFrame({
        "systime": pd.to_datetime(times),
        "uuid": "bore_diameter_sensor",
        "value_bool": None,
        "value_integer": None,
        "value_double": np.round(values, 6),
        "value_string": None,
        "is_delta": True,
    })
    return df


def create_spc_data() -> pd.DataFrame:
    """
    Build a DataFrame suitable for SPC analysis.

    Contains two uuid groups:
      - 'bore_tolerance': historical tolerance / reference measurements
      - 'bore_actual':    production measurements (some with rule violations)
    """
    np.random.seed(99)
    start = datetime(2025, 6, 10, 6, 0, 0)

    # Tolerance baseline (100 samples, tightly controlled)
    n_tol = 100
    tol_times = [start + timedelta(seconds=30 * i) for i in range(n_tol)]
    tol_values = 25.000 + np.random.normal(0, 0.002, n_tol)

    tol_df = pd.DataFrame({
        "systime": pd.to_datetime(tol_times),
        "uuid": "bore_tolerance",
        "value_bool": None,
        "value_integer": None,
        "value_double": np.round(tol_values, 6),
        "value_string": None,
        "is_delta": True,
    })

    # Actual production measurements (500 samples)
    n_act = 500
    act_times = [start + timedelta(seconds=30 * (n_tol + i)) for i in range(n_act)]
    act_values = 25.000 + np.random.normal(0, 0.003, n_act)

    # Inject a mean shift (rule 2) around samples 200-220
    act_values[200:220] += 0.007

    # Inject a steady upward trend (rule 3) around samples 350-360
    act_values[350:360] += np.linspace(0, 0.015, 10)

    # Inject a point beyond 3-sigma (rule 1)
    act_values[450] = 25.035

    act_df = pd.DataFrame({
        "systime": pd.to_datetime(act_times),
        "uuid": "bore_actual",
        "value_bool": None,
        "value_integer": None,
        "value_double": np.round(act_values, 6),
        "value_string": None,
        "is_delta": True,
    })

    return pd.concat([tol_df, act_df], ignore_index=True)


def create_tolerance_data() -> pd.DataFrame:
    """
    Build a DataFrame for tolerance deviation analysis.

    Contains three uuid groups:
      - 'upper_spec_limit': upper tolerance (25.010 mm)
      - 'lower_spec_limit': lower tolerance (24.990 mm)
      - 'bore_measurement': actual bore measurements with some out-of-spec values
    """
    np.random.seed(7)
    start = datetime(2025, 6, 10, 6, 0, 0)

    rows = []

    # Upper tolerance setting event
    rows.append({
        "systime": start,
        "uuid": "upper_spec_limit",
        "value_bool": None,
        "value_integer": None,
        "value_double": 25.010,
        "value_string": None,
        "is_delta": True,
    })

    # Lower tolerance setting event
    rows.append({
        "systime": start,
        "uuid": "lower_spec_limit",
        "value_bool": None,
        "value_integer": None,
        "value_double": 24.990,
        "value_string": None,
        "is_delta": True,
    })

    # Actual measurements (300 samples, some outside tolerance)
    n_meas = 300
    for i in range(n_meas):
        t = start + timedelta(seconds=30 * (i + 1))
        val = 25.000 + np.random.normal(0, 0.005)
        # Force some consecutive out-of-spec values around index 100-104
        if 100 <= i <= 104:
            val = 25.015 + np.random.uniform(0, 0.005)
        # Another group around 250-252
        if 250 <= i <= 252:
            val = 24.983 - np.random.uniform(0, 0.003)
        rows.append({
            "systime": t,
            "uuid": "bore_measurement",
            "value_bool": None,
            "value_integer": None,
            "value_double": round(val, 6),
            "value_string": None,
            "is_delta": True,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Demo functions
# ---------------------------------------------------------------------------

def demo_outlier_detection():
    """Demonstrate all outlier detection methods."""
    print("=" * 72)
    print("1. OUTLIER DETECTION EVENTS")
    print("=" * 72)

    df = create_bore_diameter_data()
    print(f"\nInput data: {len(df)} bore-diameter readings over 8 hours")
    print(f"Value range: {df['value_double'].min():.4f} - {df['value_double'].max():.4f} mm\n")

    detector = OutlierDetectionEvents(
        dataframe=df,
        value_column="value_double",
        event_uuid="outlier_event",
        time_threshold="5min",
    )

    # --- Z-score method ---
    print("--- Z-score method (threshold=3.0) ---")
    zscore_events = detector.detect_outliers_zscore(threshold=3.0)
    print(f"  Detected {len(zscore_events)} outlier event rows")
    if not zscore_events.empty:
        print(zscore_events[["systime", "value_double", "severity"]].to_string(index=False))
    print()

    # --- IQR method ---
    print("--- IQR method (threshold=(1.5, 1.5)) ---")
    iqr_events = detector.detect_outliers_iqr(threshold=(1.5, 1.5))
    print(f"  Detected {len(iqr_events)} outlier event rows")
    if not iqr_events.empty:
        print(iqr_events[["systime", "value_double", "severity"]].head(10).to_string(index=False))
    print()

    # --- MAD method ---
    print("--- MAD method (threshold=3.5) ---")
    mad_events = detector.detect_outliers_mad(threshold=3.5)
    print(f"  Detected {len(mad_events)} outlier event rows")
    if not mad_events.empty:
        print(mad_events[["systime", "value_double", "severity"]].head(10).to_string(index=False))
    print()

    # --- Isolation Forest method ---
    print("--- Isolation Forest method (contamination=0.05) ---")
    try:
        iforest_events = detector.detect_outliers_isolation_forest(contamination=0.05)
        print(f"  Detected {len(iforest_events)} outlier event rows")
        if not iforest_events.empty:
            print(iforest_events[["systime", "value_double", "severity"]].head(10).to_string(index=False))
    except ImportError:
        print("  scikit-learn not installed -- skipping Isolation Forest demo")
    print()


def demo_spc():
    """Demonstrate Statistical Process Control (Western Electric Rules + CUSUM)."""
    print("=" * 72)
    print("2. STATISTICAL PROCESS CONTROL (SPC)")
    print("=" * 72)

    df = create_spc_data()
    print(f"\nInput data: {len(df)} rows ({df['uuid'].value_counts().to_dict()})")

    spc = StatisticalProcessControlRuleBased(
        dataframe=df,
        value_column="value_double",
        tolerance_uuid="bore_tolerance",
        actual_uuid="bore_actual",
        event_uuid="spc_violation_event",
    )

    # --- Control limits ---
    print("\n--- Static control limits (from tolerance data) ---")
    limits = spc.calculate_control_limits()
    for col in limits.columns:
        print(f"  {col}: {limits[col].values[0]:.6f}")

    # --- Dynamic control limits ---
    print("\n--- Dynamic control limits (EWMA, window=20) ---")
    dynamic_limits = spc.calculate_dynamic_control_limits(method="ewma", window=20)
    print(f"  Generated {len(dynamic_limits)} rows of dynamic limits")
    print(dynamic_limits.tail(5).to_string(index=False))

    # --- Apply all rules ---
    print("\n--- Apply all Western Electric rules ---")
    violations = spc.process(include_severity=True)
    if not violations.empty:
        print(f"  Found {len(violations)} violations")
        print(violations.head(10).to_string(index=False))
    else:
        print("  No violations detected (data is well-behaved)")

    # --- Vectorized rule application (selected rules) ---
    print("\n--- Vectorized rules: rule_1, rule_2, rule_3 ---")
    vec_violations = spc.apply_rules_vectorized(selected_rules=["rule_1", "rule_2", "rule_3"])
    if not vec_violations.empty:
        print(f"  Found {len(vec_violations)} violations")
        print(vec_violations.head(10).to_string(index=False))
    else:
        print("  No violations from selected rules")

    # --- Interpret violations ---
    if not vec_violations.empty:
        print("\n--- Interpret violations ---")
        interpreted = spc.interpret_violations(vec_violations)
        for _, row in interpreted.head(3).iterrows():
            print(f"  Rule: {row['rule']}")
            print(f"    Severity:       {row['severity']}")
            print(f"    Interpretation: {row['interpretation']}")
            print(f"    Recommendation: {row['recommendation']}")
            print()

    # --- CUSUM shift detection ---
    print("--- CUSUM shift detection ---")
    cusum_shifts = spc.detect_cusum_shifts(k=0.5, h=5.0)
    if not cusum_shifts.empty:
        print(f"  Detected {len(cusum_shifts)} CUSUM shift points")
        print(cusum_shifts.head(10).to_string(index=False))
    else:
        print("  No CUSUM shifts detected")
    print()


def demo_tolerance_deviation():
    """Demonstrate tolerance deviation detection with separate upper/lower limits."""
    print("=" * 72)
    print("3. TOLERANCE DEVIATION EVENTS")
    print("=" * 72)

    df = create_tolerance_data()
    print(f"\nInput data: {len(df)} rows")
    print(f"  Upper spec: 25.010 mm, Lower spec: 24.990 mm")

    tol = ToleranceDeviationEvents(
        dataframe=df,
        tolerance_column="value_double",
        actual_column="value_double",
        actual_uuid="bore_measurement",
        event_uuid="tolerance_deviation_event",
        upper_tolerance_uuid="upper_spec_limit",
        lower_tolerance_uuid="lower_spec_limit",
        warning_threshold=0.8,
        time_threshold="5min",
    )

    # --- Process and group tolerance events ---
    print("\n--- Process tolerance deviation events ---")
    events = tol.process_and_group_data_with_events()
    if not events.empty:
        print(f"  Generated {len(events)} event rows")
        display_cols = [c for c in ["systime", "uuid", "deviation_abs", "deviation_pct", "severity"] if c in events.columns]
        if display_cols:
            print(events[display_cols].head(10).to_string(index=False))
    else:
        print("  No grouped tolerance deviation events (individual deviations may exist)")

    # --- Process capability indices ---
    print("\n--- Process capability indices ---")
    try:
        capability = tol.compute_capability_indices()
        print(f"  Cp  = {capability['Cp']:.4f}  (potential capability)")
        print(f"  Cpk = {capability['Cpk']:.4f}  (actual capability)")
        print(f"  Pp  = {capability['Pp']:.4f}  (process performance)")
        print(f"  Ppk = {capability['Ppk']:.4f}  (process performance index)")
        print(f"  USL = {capability['usl']:.4f} mm")
        print(f"  LSL = {capability['lsl']:.4f} mm")
        print(f"  Process mean = {capability['mean']:.4f} mm")
        print(f"  Process std  = {capability['std']:.6f} mm")
        if capability['Cpk'] >= 1.33:
            print("  --> Process is CAPABLE (Cpk >= 1.33)")
        else:
            print("  --> Process is NOT capable (Cpk < 1.33) -- investigate variation sources")
    except Exception as e:
        print(f"  Could not compute capability indices: {e}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo_outlier_detection()
    demo_spc()
    demo_tolerance_deviation()

    print("=" * 72)
    print("Quality events demo complete.")
    print("=" * 72)
