"""Correlation Events Demo

Demonstrates cross-signal correlation analysis and anomaly correlation
for detecting related process issues in manufacturing.

Run: python examples/correlation_events_demo.py
"""

import pandas as pd
import numpy as np

from ts_shape.events.correlation.signal_correlation import SignalCorrelationEvents
from ts_shape.events.correlation.anomaly_correlation import AnomalyCorrelationEvents


def create_correlated_process_data(n: int = 1000) -> pd.DataFrame:
    """Create realistic multi-signal process data with correlations.

    Simulates a process with:
    - Temperature and pressure (normally correlated)
    - Vibration that increases when temperature spikes
    - Flow rate that drops when pressure is abnormal
    """
    np.random.seed(42)
    base = pd.Timestamp("2024-01-01")
    rows = []

    # Base temperature signal with gradual drift
    temp = np.cumsum(np.random.normal(0, 0.1, n)) + 85.0

    # Pressure correlated with temperature (~0.8 correlation)
    pressure = temp * 0.12 + 2.5 + np.random.normal(0, 0.3, n)

    # Inject process upset at t=400-450: temperature spike
    temp[400:450] += np.linspace(0, 15, 50)
    # Pressure diverges during upset
    pressure[410:460] -= np.linspace(0, 3, 50)

    # Vibration: normally low, spikes after temperature issues
    vibration = np.abs(np.random.normal(2.0, 0.5, n))
    vibration[420:470] += np.linspace(0, 8, 50)  # delayed reaction to temp

    # Flow rate: drops when pressure drops
    flow = 100 + pressure * 5 + np.random.normal(0, 1, n)

    # Inject another anomaly at t=700
    temp[700] += 25  # sudden spike
    vibration[705] += 12  # delayed vibration response
    pressure[710] -= 5  # delayed pressure response

    for i in range(n):
        t = base + pd.Timedelta(minutes=i)
        for uuid, val in [
            ("proc:temperature", temp[i]),
            ("proc:pressure", pressure[i]),
            ("proc:vibration", vibration[i]),
            ("proc:flow_rate", flow[i]),
        ]:
            rows.append({
                "systime": t, "uuid": uuid,
                "value_double": val, "is_delta": True,
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("=" * 70)
    print("CORRELATION EVENTS DEMO")
    print("=" * 70)

    df = create_correlated_process_data()
    print(f"\nCreated dataset: {len(df)} rows, {df['uuid'].nunique()} signals")

    # -----------------------------------------------------------------------
    # 1. Signal Correlation Analysis
    # -----------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("1. ROLLING CORRELATION: Temperature vs Pressure")
    print("-" * 70)

    sc = SignalCorrelationEvents(df)

    corr = sc.rolling_correlation(
        "proc:temperature", "proc:pressure", resample="1min", window=30
    )
    print(f"\nRolling correlation points: {len(corr)}")
    print(f"Mean correlation: {corr['correlation'].mean():.3f}")
    print(f"Min correlation:  {corr['correlation'].min():.3f}")
    print(f"Max correlation:  {corr['correlation'].max():.3f}")

    # -----------------------------------------------------------------------
    # 2. Correlation Breakdown Detection
    # -----------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("2. CORRELATION BREAKDOWN: Temperature vs Pressure")
    print("-" * 70)

    breakdowns = sc.correlation_breakdown(
        "proc:temperature", "proc:pressure",
        resample="1min", window=30, threshold=0.3,
    )
    print(f"\nCorrelation breakdowns detected: {len(breakdowns)}")
    if not breakdowns.empty:
        for _, row in breakdowns.iterrows():
            print(f"  {row['start']} to {row['end']} "
                  f"(min_corr={row['min_correlation']:.3f}, "
                  f"duration={row['duration_seconds']:.0f}s)")

    # -----------------------------------------------------------------------
    # 3. Lag Correlation Analysis
    # -----------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("3. LAG CORRELATION: Temperature -> Vibration")
    print("-" * 70)

    lag = sc.lag_correlation(
        "proc:temperature", "proc:vibration", resample="1min", max_lag=20
    )
    best = lag[lag["is_best_lag"]].iloc[0]
    print(f"\nBest lag: {best['lag_periods']} periods "
          f"(correlation = {best['correlation']:.3f})")
    print(f"Top 5 lag correlations:")
    top5 = lag.nlargest(5, "correlation")
    for _, row in top5.iterrows():
        marker = " <-- best" if row["is_best_lag"] else ""
        print(f"  Lag {row['lag_periods']:+3d}: r = {row['correlation']:.3f}{marker}")

    # -----------------------------------------------------------------------
    # 4. Anomaly Correlation
    # -----------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("4. COINCIDENT ANOMALIES across all signals")
    print("-" * 70)

    ac = AnomalyCorrelationEvents(df)

    signals = ["proc:temperature", "proc:pressure", "proc:vibration", "proc:flow_rate"]
    coincident = ac.coincident_anomalies(
        signals, z_threshold=2.5, coincidence_window="15min", min_signals=2
    )
    print(f"\nCoincident anomaly windows: {len(coincident)}")
    if not coincident.empty:
        for _, row in coincident.iterrows():
            print(f"  {row['start']}: "
                  f"{row['anomaly_count']} anomalies across "
                  f"[{row['signal_uuids_involved']}]")

    # -----------------------------------------------------------------------
    # 5. Cascade Detection
    # -----------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("5. CASCADE DETECTION: Temperature -> Vibration")
    print("-" * 70)

    cascades = ac.cascade_detection(
        "proc:temperature", "proc:vibration",
        z_threshold=2.5, max_delay="15min",
    )
    print(f"\nCascade events detected: {len(cascades)}")
    if not cascades.empty:
        for _, row in cascades.iterrows():
            print(f"  Leader: {row['leader_time']} -> "
                  f"Follower: {row['follower_time']} "
                  f"(delay={row['delay_seconds']:.0f}s)")

    # -----------------------------------------------------------------------
    # 6. Root Cause Ranking
    # -----------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("6. ROOT CAUSE RANKING")
    print("-" * 70)

    ranking = ac.root_cause_ranking(signals, z_threshold=2.5, max_delay="15min")
    print(f"\nSignal ranking by anomaly leadership:")
    for _, row in ranking.iterrows():
        print(f"  #{int(row['rank'])} {row['signal_uuid']}: "
              f"leader_ratio={row['leader_ratio']:.2f} "
              f"(leads={int(row['leader_count'])}, follows={int(row['follower_count'])})")

    print("\n" + "=" * 70)
    print("Demo complete.")
    print("=" * 70)
