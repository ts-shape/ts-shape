#!/usr/bin/env python3
"""
Comprehensive demonstration of maintenance event detection in ts-shape.

Simulates an industrial facility with multiple sensor types:
  - Bearing temperature sensor   (bearing_temp_01)   -- gradual thermal degradation
  - Hydraulic pressure sensor    (hydraulic_psi_01)   -- level shifts from seal wear
  - Motor vibration accelerometer (accel_motor_x)     -- growing vibration amplitude
  - Coolant flow sensor          (coolant_flow_01)    -- declining flow rate

Classes demonstrated:
  1. DegradationDetectionEvents  -- trend degradation, variance increase, level shift, health score
  2. FailurePredictionEvents     -- remaining useful life, exceedance patterns, time to threshold
  3. VibrationAnalysisEvents     -- RMS exceedance, amplitude growth, bearing health indicators
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# ---------------------------------------------------------------------------
# Allow import from the local source tree when running as a standalone script
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ts_shape.events.maintenance.degradation_detection import DegradationDetectionEvents
from ts_shape.events.maintenance.failure_prediction import FailurePredictionEvents
from ts_shape.events.maintenance.vibration_analysis import VibrationAnalysisEvents


# ============================================================================
# DATA GENERATION HELPERS
# ============================================================================

def _make_rows(timestamps, uuid, values):
    """Build a list of dicts in the standard ts-shape row format."""
    return [
        {
            'systime': ts,
            'uuid': uuid,
            'value_bool': None,
            'value_integer': None,
            'value_double': float(val),
            'value_string': None,
            'is_delta': True,
        }
        for ts, val in zip(timestamps, values)
    ]


def create_bearing_temperature_data(start: datetime, n_points: int = 500):
    """
    Bearing temperature sensor (bearing_temp_01).

    Profile:
      0-200   Stable at ~85 C with low noise  (healthy baseline)
      200-350 Gradual rise ~0.03 C per sample  (early degradation)
      350-500 Faster rise ~0.07 C per sample + increased noise (advanced degradation)
    A level shift of +4 C is injected at sample 160 to simulate a sudden
    change in operating conditions.
    """
    np.random.seed(42)
    timestamps = [start + timedelta(minutes=i * 5) for i in range(n_points)]
    values = np.empty(n_points)

    for i in range(n_points):
        noise = np.random.normal(0, 0.4)
        if i < 200:
            values[i] = 85.0 + noise
        elif i < 350:
            values[i] = 85.0 + (i - 200) * 0.03 + noise
        else:
            values[i] = 85.0 + (350 - 200) * 0.03 + (i - 350) * 0.07 + np.random.normal(0, 1.5)

    # Inject a level shift at sample 160
    values[160:] += 4.0

    return pd.DataFrame(_make_rows(timestamps, 'bearing_temp_01', values))


def create_hydraulic_pressure_data(start: datetime, n_points: int = 400):
    """
    Hydraulic pressure sensor (hydraulic_psi_01).

    Profile:
      0-120   Stable at ~3000 PSI
      120-260 Gradual decline as seals wear     (decreasing trend)
      260+    Two abrupt drops from seal failures (level shifts)
    """
    np.random.seed(77)
    timestamps = [start + timedelta(minutes=i * 3) for i in range(n_points)]
    values = np.empty(n_points)

    for i in range(n_points):
        noise = np.random.normal(0, 8.0)
        if i < 120:
            values[i] = 3000.0 + noise
        elif i < 260:
            values[i] = 3000.0 - (i - 120) * 0.5 + noise
        else:
            values[i] = 3000.0 - (260 - 120) * 0.5 - (i - 260) * 0.8 + noise

    # Abrupt seal-failure drops
    values[260:] -= 40.0
    values[340:] -= 30.0

    return pd.DataFrame(_make_rows(timestamps, 'hydraulic_psi_01', values))


def create_motor_vibration_data(start: datetime, n_points: int = 720):
    """
    Motor vibration accelerometer (accel_motor_x).

    A sinusoidal vibration whose amplitude grows from 1.0 g to 5.0 g over
    the observation period, with random impulse spikes added in the second
    half to simulate developing bearing defects.
    """
    np.random.seed(55)
    timestamps = [start + timedelta(seconds=i * 10) for i in range(n_points)]
    values = np.empty(n_points)

    for i in range(n_points):
        amplitude = 1.0 + (i / n_points) * 4.0
        base = amplitude * np.sin(2 * np.pi * 0.1 * i * 10)
        noise = np.random.normal(0, 0.2)

        # Add occasional sharp spikes in the second half (bearing defect impulses)
        spike = 0.0
        if i > n_points // 2 and np.random.random() > 0.92:
            spike = np.random.uniform(3.0, 6.0)

        values[i] = base + noise + spike

    return pd.DataFrame(_make_rows(timestamps, 'accel_motor_x', values))


def create_coolant_flow_data(start: datetime, n_points: int = 300):
    """
    Coolant flow sensor (coolant_flow_01).

    Signal starts at ~50 L/min and drops toward a critical low of 30 L/min,
    used to demonstrate failure prediction and time-to-threshold estimation.
    Warning exceedances become more frequent as the flow declines.
    """
    np.random.seed(99)
    timestamps = [start + timedelta(minutes=i * 10) for i in range(n_points)]
    values = np.empty(n_points)

    for i in range(n_points):
        base = 50.0 - i * 0.06
        noise = np.random.normal(0, 1.0)
        # Extra dips after halfway mark
        if i > 150 and np.random.random() > 0.88:
            noise -= np.random.uniform(3.0, 6.0)
        values[i] = base + noise

    return pd.DataFrame(_make_rows(timestamps, 'coolant_flow_01', values))


def build_combined_dataframe():
    """
    Assemble all four sensor streams into a single DataFrame, mirroring
    a real scenario where multiple signals share the same datastore.
    """
    start = datetime(2024, 6, 1, 0, 0, 0)
    frames = [
        create_bearing_temperature_data(start),
        create_hydraulic_pressure_data(start),
        create_motor_vibration_data(start),
        create_coolant_flow_data(start),
    ]
    df = pd.concat(frames, ignore_index=True)
    print(f"Combined dataset: {len(df)} rows, "
          f"{df['uuid'].nunique()} sensor UUIDs")
    print(f"  UUIDs: {sorted(df['uuid'].unique().tolist())}")
    print(f"  Time range: {df['systime'].min()} -> {df['systime'].max()}")
    return df


# ============================================================================
# DEMO 1 -- DEGRADATION DETECTION
# ============================================================================

def demo_degradation_detection(df: pd.DataFrame):
    """
    Demonstrate DegradationDetectionEvents using the bearing temperature
    and hydraulic pressure signals.
    """
    print("\n" + "=" * 72)
    print("  DEMO 1: DegradationDetectionEvents")
    print("=" * 72)

    # ---- 1a. Bearing temperature -- increasing trend degradation ----------
    print("\n--- 1a. Trend Degradation: bearing_temp_01 (increasing direction) ---")
    det_temp = DegradationDetectionEvents(
        dataframe=df,
        signal_uuid='bearing_temp_01',
        event_uuid='evt:bearing_temp_trend',
        value_column='value_double',
    )

    trends = det_temp.detect_trend_degradation(
        window='2h',
        min_slope=0.00001,
        direction='increasing',
    )
    print(f"  Degradation intervals found: {len(trends)}")
    if not trends.empty:
        for _, row in trends.iterrows():
            print(f"    {row['start']}  ->  {row['end']}  |  "
                  f"avg_slope={row['avg_slope']:.6f}  "
                  f"total_change={row['total_change']:+.2f}  "
                  f"duration={row['duration_seconds']:.0f}s")

    # ---- 1b. Variance increase on bearing temperature ---------------------
    print("\n--- 1b. Variance Increase: bearing_temp_01 (threshold x3) ---")
    var_events = det_temp.detect_variance_increase(
        window='3h',
        threshold_factor=3.0,
    )
    print(f"  Variance increase intervals: {len(var_events)}")
    if not var_events.empty:
        print(var_events[['start', 'end', 'baseline_variance',
                          'current_variance', 'ratio']].to_string(index=False))

    # ---- 1c. Level shift on bearing temperature ---------------------------
    print("\n--- 1c. Level Shift: bearing_temp_01 (min_shift=3.0 C) ---")
    shifts = det_temp.detect_level_shift(min_shift=3.0, hold='15m')
    print(f"  Level shifts detected: {len(shifts)}")
    if not shifts.empty:
        print(shifts[['systime', 'shift_magnitude', 'prev_mean',
                       'new_mean']].to_string(index=False))

    # ---- 1d. Health score on bearing temperature --------------------------
    print("\n--- 1d. Health Score: bearing_temp_01 (window=2h, baseline=8h) ---")
    health = det_temp.health_score(window='2h', baseline_window='8h')
    if not health.empty:
        print(f"  Health observations: {len(health)}")
        early = health['health_score'].iloc[:20].mean()
        late = health['health_score'].iloc[-20:].mean()
        print(f"  Mean health (first 20 points):  {early:.1f}")
        print(f"  Mean health (last 20 points):   {late:.1f}")
        print(f"  Minimum health score:           {health['health_score'].min():.1f}")
        print("\n  Sample health readings (every 100th point):")
        sample = health.iloc[::100][['systime', 'health_score',
                                      'mean_drift_pct', 'variance_ratio',
                                      'trend_slope']]
        print(sample.to_string(index=False))

    # ---- 1e. Hydraulic pressure -- decreasing trend degradation -----------
    print("\n--- 1e. Trend Degradation: hydraulic_psi_01 (decreasing) ---")
    det_hyd = DegradationDetectionEvents(
        dataframe=df,
        signal_uuid='hydraulic_psi_01',
        event_uuid='evt:hydraulic_trend',
        value_column='value_double',
    )
    hyd_trends = det_hyd.detect_trend_degradation(
        window='1h',
        min_slope=0.0001,
        direction='decreasing',
    )
    print(f"  Degradation intervals found: {len(hyd_trends)}")
    if not hyd_trends.empty:
        print(hyd_trends[['start', 'end', 'avg_slope',
                           'total_change']].head(5).to_string(index=False))

    # ---- 1f. Level shift on hydraulic pressure ----------------------------
    print("\n--- 1f. Level Shift: hydraulic_psi_01 (min_shift=20 PSI) ---")
    hyd_shifts = det_hyd.detect_level_shift(min_shift=20.0, hold='10m')
    print(f"  Level shifts detected: {len(hyd_shifts)}")
    if not hyd_shifts.empty:
        print(hyd_shifts[['systime', 'shift_magnitude', 'prev_mean',
                           'new_mean']].to_string(index=False))


# ============================================================================
# DEMO 2 -- FAILURE PREDICTION
# ============================================================================

def demo_failure_prediction(df: pd.DataFrame):
    """
    Demonstrate FailurePredictionEvents using the coolant flow sensor
    (decreasing toward a critical threshold) and the bearing temperature
    sensor (increasing toward an overheat threshold).
    """
    print("\n" + "=" * 72)
    print("  DEMO 2: FailurePredictionEvents")
    print("=" * 72)

    # ---- 2a. Remaining Useful Life: coolant flow --------------------------
    print("\n--- 2a. Remaining Useful Life: coolant_flow_01 ---")
    print("  Failure threshold: 30 L/min (critically low flow)")
    fp_cool = FailurePredictionEvents(
        dataframe=df,
        signal_uuid='coolant_flow_01',
        event_uuid='evt:coolant_rul',
        value_column='value_double',
    )

    rul = fp_cool.remaining_useful_life(
        degradation_rate=-0.0001,    # fallback: declining ~0.0001 L/min/s
        failure_threshold=30.0,
    )
    if not rul.empty:
        print(f"  RUL estimates computed: {len(rul)}")
        # Show a few snapshots across the timeline
        indices = [0, len(rul) // 4, len(rul) // 2,
                   3 * len(rul) // 4, len(rul) - 1]
        sample = rul.iloc[indices][['systime', 'current_value',
                                     'rul_hours', 'confidence']]
        print(sample.to_string(index=False))

        first_rul = rul['rul_hours'].iloc[0]
        last_valid = rul[rul['rul_hours'].notna()]
        last_rul = last_valid['rul_hours'].iloc[-1] if not last_valid.empty else None
        print(f"\n  Initial RUL estimate: {first_rul:.1f} hours"
              if first_rul is not None else "\n  Initial RUL: N/A")
        if last_rul is not None:
            print(f"  Final RUL estimate:   {last_rul:.1f} hours")

    # ---- 2b. Exceedance Pattern: bearing temperature ----------------------
    print("\n--- 2b. Exceedance Pattern: bearing_temp_01 ---")
    print("  Warning: 90 C  |  Critical: 95 C")
    fp_temp = FailurePredictionEvents(
        dataframe=df,
        signal_uuid='bearing_temp_01',
        event_uuid='evt:bearing_exceedance',
        value_column='value_double',
    )

    exceed = fp_temp.detect_exceedance_pattern(
        warning_threshold=90.0,
        critical_threshold=95.0,
        window='4h',
    )
    if not exceed.empty:
        print(f"  Windows analyzed: {len(exceed)}")
        escalating = exceed[exceed['escalation_detected']]
        print(f"  Escalation windows: {len(escalating)}")
        print(exceed[['start', 'warning_count', 'critical_count',
                       'escalation_detected']].to_string(index=False))
    else:
        print("  No exceedances detected in any window.")

    # ---- 2c. Time to Threshold: bearing temperature -----------------------
    print("\n--- 2c. Time to Threshold: bearing_temp_01 (threshold=100 C) ---")
    ttt = fp_temp.time_to_threshold(threshold=100.0, direction='increasing')
    if not ttt.empty:
        valid = ttt[ttt['estimated_time_seconds'].notna()]
        if not valid.empty:
            # Show a few estimates spread over time
            step = max(1, len(valid) // 5)
            sample = valid.iloc[::step][['systime', 'current_value',
                                          'rate_of_change',
                                          'estimated_time_seconds']].head(6)
            print(sample.to_string(index=False))

            last = valid.iloc[-1]
            est_hrs = last['estimated_time_seconds'] / 3600.0
            print(f"\n  Latest estimate: {est_hrs:.1f} hours to reach 100 C "
                  f"(current: {last['current_value']:.1f} C, "
                  f"rate: {last['rate_of_change']:.6f} C/s)")
        else:
            print("  No valid time-to-threshold estimates (signal may not be "
                  "trending in the expected direction yet).")

    # ---- 2d. Time to Threshold: coolant flow (decreasing) -----------------
    print("\n--- 2d. Time to Threshold: coolant_flow_01 (threshold=35 L/min, decreasing) ---")
    ttt_cool = fp_cool.time_to_threshold(threshold=35.0, direction='decreasing')
    if not ttt_cool.empty:
        valid_cool = ttt_cool[ttt_cool['estimated_time_seconds'].notna()]
        if not valid_cool.empty:
            last_c = valid_cool.iloc[-1]
            est_hrs_c = last_c['estimated_time_seconds'] / 3600.0
            print(f"  Latest estimate: {est_hrs_c:.1f} hours to reach 35 L/min "
                  f"(current: {last_c['current_value']:.1f} L/min)")
        else:
            print("  No valid estimates available.")


# ============================================================================
# DEMO 3 -- VIBRATION ANALYSIS
# ============================================================================

def demo_vibration_analysis(df: pd.DataFrame):
    """
    Demonstrate VibrationAnalysisEvents using the motor accelerometer signal.
    """
    print("\n" + "=" * 72)
    print("  DEMO 3: VibrationAnalysisEvents")
    print("=" * 72)

    analyzer = VibrationAnalysisEvents(
        dataframe=df,
        signal_uuid='accel_motor_x',
        event_uuid='evt:motor_vibration',
        value_column='value_double',
    )

    # ---- 3a. RMS Exceedance -----------------------------------------------
    print("\n--- 3a. RMS Exceedance: accel_motor_x ---")
    print("  Baseline RMS: 1.0 g  |  Alarm factor: 2.0x")
    rms_events = analyzer.detect_rms_exceedance(
        baseline_rms=1.0,
        threshold_factor=2.0,
        window='2min',
    )
    print(f"  Exceedance intervals: {len(rms_events)}")
    if not rms_events.empty:
        print(rms_events[['start', 'end', 'rms_value', 'ratio',
                           'duration_seconds']].head(10).to_string(index=False))
        total_alarm_sec = rms_events['duration_seconds'].sum()
        print(f"\n  Total alarm time: {total_alarm_sec:.0f} seconds "
              f"({total_alarm_sec / 60:.1f} minutes)")

    # ---- 3b. Amplitude Growth ---------------------------------------------
    print("\n--- 3b. Amplitude Growth: accel_motor_x (10-min windows, >25% growth) ---")
    amp = analyzer.detect_amplitude_growth(
        window='10min',
        growth_threshold=0.25,
    )
    if not amp.empty:
        print(f"  Windows analyzed: {len(amp)}")
        growing = amp[amp['growth_pct'] > 0.25]
        print(f"  Windows exceeding 25% growth: {len(growing)}")
        print("\n  First and last 4 windows:")
        combined = pd.concat([amp.head(4), amp.tail(4)]).drop_duplicates()
        print(combined[['start', 'amplitude',
                         'baseline_amplitude', 'growth_pct']].to_string(index=False))

    # ---- 3c. Bearing Health Indicators ------------------------------------
    print("\n--- 3c. Bearing Health Indicators: accel_motor_x (5-min windows) ---")
    bhi = analyzer.bearing_health_indicators(window='5min')
    if not bhi.empty:
        print(f"  Windows: {len(bhi)}")
        print(bhi[['start', 'rms', 'peak', 'crest_factor',
                     'kurtosis']].to_string(index=False))

        print("\n  Summary across windows:")
        print(f"    RMS    : {bhi['rms'].iloc[0]:.4f} -> {bhi['rms'].iloc[-1]:.4f}  "
              f"(+{(bhi['rms'].iloc[-1] / bhi['rms'].iloc[0] - 1) * 100:.0f}%)")
        print(f"    Peak   : {bhi['peak'].iloc[0]:.4f} -> {bhi['peak'].iloc[-1]:.4f}")
        cf_first = bhi['crest_factor'].iloc[0]
        cf_last = bhi['crest_factor'].iloc[-1]
        if cf_first is not None and cf_last is not None:
            print(f"    Crest  : {cf_first:.4f} -> {cf_last:.4f}")
        k_first = bhi['kurtosis'].iloc[0]
        k_last = bhi['kurtosis'].iloc[-1]
        print(f"    Kurtosis: {k_first:.4f} -> {k_last:.4f}")
        print("    (Kurtosis > 4 suggests impulsive bearing defects developing)")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all maintenance event demonstrations."""
    print("\n" + "#" * 72)
    print("#  ts-shape  --  Maintenance Events Demonstration")
    print("#")
    print("#  Simulated sensors:")
    print("#    bearing_temp_01   : bearing temperature (C)")
    print("#    hydraulic_psi_01  : hydraulic line pressure (PSI)")
    print("#    accel_motor_x     : motor vibration accelerometer (g)")
    print("#    coolant_flow_01   : coolant flow rate (L/min)")
    print("#" * 72)

    try:
        df = build_combined_dataframe()

        demo_degradation_detection(df)
        demo_failure_prediction(df)
        demo_vibration_analysis(df)

        print("\n" + "=" * 72)
        print("  All demonstrations completed successfully.")
        print("=" * 72 + "\n")

    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
