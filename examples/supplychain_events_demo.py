#!/usr/bin/env python3
"""
Comprehensive demonstration of all supply chain event classes in ts-shape.

This script demonstrates every supply chain event class and method:

  1. InventoryMonitoringEvents
     - detect_low_stock()        -- flag intervals where stock stays below a threshold
     - consumption_rate()        -- rolling consumption rate per time window
     - reorder_point_breach()    -- detect crossings below reorder point / safety stock
     - stockout_prediction()     -- estimate hours until stockout at current burn rate

  2. LeadTimeAnalysisEvents
     - calculate_lead_times()    -- pair orders to deliveries and compute lead times
     - lead_time_statistics()    -- summary stats (mean, std, min, max, p95)
     - detect_lead_time_anomalies() -- flag lead times exceeding mean + N*std

  3. DemandPatternEvents
     - demand_by_period()        -- aggregate demand into daily/hourly buckets
     - detect_demand_spikes()    -- flag periods where demand exceeds baseline + N*std
     - seasonality_summary()     -- day-of-week or hour-of-day demand patterns

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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ts_shape.events.supplychain.inventory_monitoring import InventoryMonitoringEvents
from ts_shape.events.supplychain.lead_time_analysis import LeadTimeAnalysisEvents
from ts_shape.events.supplychain.demand_pattern import DemandPatternEvents


# ===========================================================================
# Helper: pretty section / subsection printers
# ===========================================================================

def section(title: str) -> None:
    print(f"\n{'=' * 74}")
    print(f"  {title}")
    print(f"{'=' * 74}")


def subsection(title: str) -> None:
    print(f"\n  --- {title} ---")


# ===========================================================================
# 1. Build realistic inventory data for TWO warehouses
# ===========================================================================

def create_inventory_data() -> pd.DataFrame:
    """Simulate inventory levels for two warehouses over ~10 days (hourly).

    Warehouse A:
        Starts at 500 units, steady consumption of 2-8 units/hour.
        Replenished at hours 60, 120, and 170 (+300 each).
        Drops below reorder point and safety stock before replenishments.

    Warehouse B:
        Starts at 300 units, lower consumption of 1-4 units/hour.
        Single replenishment at hour 80 (+250).
        Experiences a near-stockout toward the end.
    """
    np.random.seed(42)
    base = datetime(2024, 1, 1, 0, 0, 0)
    rows = []

    # --- Warehouse A ---
    level_a = 500.0
    for i in range(200):
        rows.append({
            'systime': base + timedelta(hours=i),
            'uuid': 'warehouse_a_level',
            'value_bool': None,
            'value_integer': None,
            'value_double': round(level_a, 2),
            'value_string': None,
            'is_delta': True,
        })
        level_a -= np.random.uniform(2, 8)
        if i in [60, 120, 170]:
            level_a += 300.0
        level_a = max(0.0, level_a)

    # --- Warehouse B ---
    level_b = 300.0
    for i in range(200):
        rows.append({
            'systime': base + timedelta(hours=i),
            'uuid': 'warehouse_b_level',
            'value_bool': None,
            'value_integer': None,
            'value_double': round(level_b, 2),
            'value_string': None,
            'is_delta': True,
        })
        level_b -= np.random.uniform(1, 4)
        if i == 80:
            level_b += 250.0
        level_b = max(0.0, level_b)

    return pd.DataFrame(rows)


# ===========================================================================
# 2. Build order / delivery event data (for lead-time analysis)
# ===========================================================================

def create_lead_time_data() -> pd.DataFrame:
    """Create 20 purchase orders with matched delivery events.

    Most orders have a lead time of ~60 hours (std ~8 h).
    Orders #7 and #15 are deliberately anomalous (120-140 h).
    """
    np.random.seed(77)
    base = datetime(2024, 1, 1, 8, 0, 0)
    rows = []
    t = base

    for i in range(20):
        order_id = f"PO-{1000 + i}"

        # Order placement event
        rows.append({
            'systime': t,
            'uuid': 'order_placed',
            'value_bool': None,
            'value_integer': i,
            'value_double': None,
            'value_string': order_id,
            'is_delta': True,
        })

        # Delivery event (normally ~60 h, anomalous for indices 7 & 15)
        lead_hours = np.random.normal(60, 8)
        if i in [7, 15]:
            lead_hours = 120 + np.random.uniform(0, 20)
        lead_hours = max(24.0, lead_hours)

        rows.append({
            'systime': t + timedelta(hours=lead_hours),
            'uuid': 'delivery_received',
            'value_bool': None,
            'value_integer': i,
            'value_double': None,
            'value_string': order_id,
            'is_delta': True,
        })

        # Next order placed 1-3 days later
        t += timedelta(hours=np.random.uniform(24, 72))

    return pd.DataFrame(rows)


# ===========================================================================
# 3. Build demand signal data with weekly seasonality and spikes
# ===========================================================================

def create_demand_data() -> pd.DataFrame:
    """Simulate 60 days of hourly demand for a single product.

    Weekday base demand ~ N(100, 15); weekend ~ N(40, 10).
    Spikes injected on days 10, 35, 50 (+80..120 units).
    Demand is generated for business hours 08:00-19:00 each day.
    """
    np.random.seed(33)
    base = datetime(2024, 1, 1, 0, 0, 0)
    rows = []

    for day_offset in range(60):
        day_start = base + timedelta(days=day_offset)
        weekday = day_start.weekday()

        # Daily baseline
        if weekday < 5:
            day_base = np.random.normal(100, 15)
        else:
            day_base = np.random.normal(40, 10)

        # Inject spikes on specific days
        if day_offset in [10, 35, 50]:
            day_base += np.random.uniform(80, 120)

        # Hourly demand during business hours (12 hours)
        for hour in range(8, 20):
            hourly = max(0.0, day_base / 12 + np.random.normal(0, 2))
            rows.append({
                'systime': day_start + timedelta(hours=hour),
                'uuid': 'product_demand',
                'value_bool': None,
                'value_integer': None,
                'value_double': round(hourly, 2),
                'value_string': None,
                'is_delta': True,
            })

    return pd.DataFrame(rows)


# ===========================================================================
# DEMO 1 -- InventoryMonitoringEvents
# ===========================================================================

def demo_inventory_monitoring() -> None:
    section("DEMO 1: InventoryMonitoringEvents")

    df = create_inventory_data()
    n_a = len(df[df['uuid'] == 'warehouse_a_level'])
    n_b = len(df[df['uuid'] == 'warehouse_b_level'])
    print(f"\n  Created {len(df)} inventory rows  "
          f"(warehouse_a: {n_a}, warehouse_b: {n_b})")

    # -----------------------------------------------------------------------
    # Warehouse A
    # -----------------------------------------------------------------------
    print(f"\n  ** Warehouse A **")

    monitor_a = InventoryMonitoringEvents(
        dataframe=df,
        level_uuid='warehouse_a_level',
        event_uuid='sc:inv:wh_a',
        value_column='value_double',
    )

    # 1a. Low stock detection
    subsection("detect_low_stock(min_level=100, hold='3h')")
    low_a = monitor_a.detect_low_stock(min_level=100, hold='3h')
    print(f"  Low-stock intervals found: {len(low_a)}")
    if not low_a.empty:
        for _, r in low_a.iterrows():
            print(f"    {r['start']}  ->  {r['end']}  "
                  f"min={r['min_value']:.1f}  avg={r['avg_value']:.1f}  "
                  f"dur={r['duration_seconds']:.0f}s")

    # 1b. Consumption rate
    subsection("consumption_rate(window='4h')")
    rate_a = monitor_a.consumption_rate(window='4h')
    if not rate_a.empty:
        positive = rate_a[rate_a['consumption_rate'] > 0]
        print(f"  Windows analyzed: {len(rate_a)}  |  "
              f"with positive consumption: {len(positive)}")
        if not positive.empty:
            print(f"  Avg consumption rate : {positive['consumption_rate'].mean():.2f} units/h")
            print(f"  Max consumption rate : {positive['consumption_rate'].max():.2f} units/h")
        print(f"\n  First 8 windows:")
        print(rate_a[['start', 'consumption_rate',
                       'level_start', 'level_end']].head(8).to_string(index=False))

    # 1c. Reorder point breach
    subsection("reorder_point_breach(reorder_level=200, safety_stock=50)")
    breaches_a = monitor_a.reorder_point_breach(reorder_level=200, safety_stock=50)
    print(f"  Breach events: {len(breaches_a)}")
    if not breaches_a.empty:
        for _, r in breaches_a.iterrows():
            print(f"    {r['systime']}  level={r['current_level']:.1f}  "
                  f"type={r['breach_type']}  deficit={r['deficit']:.1f}")

    # 1d. Stockout prediction
    subsection("stockout_prediction(consumption_rate_window='8h')")
    pred_a = monitor_a.stockout_prediction(consumption_rate_window='8h')
    if not pred_a.empty:
        finite = pred_a[pred_a['estimated_stockout_time_hours'] < np.inf]
        print(f"  Total predictions: {len(pred_a)}  |  finite: {len(finite)}")
        if not finite.empty:
            print(f"  Min time-to-stockout : {finite['estimated_stockout_time_hours'].min():.1f} h")
            print(f"  Max time-to-stockout : {finite['estimated_stockout_time_hours'].max():.1f} h")
            print(f"\n  5 closest to stockout:")
            top5 = finite.nsmallest(5, 'estimated_stockout_time_hours')
            print(top5[['systime', 'current_level', 'consumption_rate',
                         'estimated_stockout_time_hours']].to_string(index=False))

    # -----------------------------------------------------------------------
    # Warehouse B -- demonstrate with different thresholds
    # -----------------------------------------------------------------------
    print(f"\n\n  ** Warehouse B **")

    monitor_b = InventoryMonitoringEvents(
        dataframe=df,
        level_uuid='warehouse_b_level',
        event_uuid='sc:inv:wh_b',
        value_column='value_double',
    )

    subsection("detect_low_stock(min_level=80, hold='5h')")
    low_b = monitor_b.detect_low_stock(min_level=80, hold='5h')
    print(f"  Low-stock intervals found: {len(low_b)}")
    if not low_b.empty:
        for _, r in low_b.iterrows():
            print(f"    {r['start']}  ->  {r['end']}  "
                  f"min={r['min_value']:.1f}  avg={r['avg_value']:.1f}  "
                  f"dur={r['duration_seconds']:.0f}s")

    subsection("reorder_point_breach(reorder_level=120, safety_stock=30)")
    breaches_b = monitor_b.reorder_point_breach(reorder_level=120, safety_stock=30)
    print(f"  Breach events: {len(breaches_b)}")
    if not breaches_b.empty:
        for _, r in breaches_b.iterrows():
            print(f"    {r['systime']}  level={r['current_level']:.1f}  "
                  f"type={r['breach_type']}  deficit={r['deficit']:.1f}")

    subsection("stockout_prediction(consumption_rate_window='6h')")
    pred_b = monitor_b.stockout_prediction(consumption_rate_window='6h')
    if not pred_b.empty:
        finite_b = pred_b[pred_b['estimated_stockout_time_hours'] < np.inf]
        print(f"  Total predictions: {len(pred_b)}  |  finite: {len(finite_b)}")
        if not finite_b.empty:
            print(f"  Min time-to-stockout : {finite_b['estimated_stockout_time_hours'].min():.1f} h")
            print(f"\n  5 closest to stockout:")
            top5b = finite_b.nsmallest(5, 'estimated_stockout_time_hours')
            print(top5b[['systime', 'current_level', 'consumption_rate',
                          'estimated_stockout_time_hours']].to_string(index=False))


# ===========================================================================
# DEMO 2 -- LeadTimeAnalysisEvents
# ===========================================================================

def demo_lead_time_analysis() -> None:
    section("DEMO 2: LeadTimeAnalysisEvents")

    df = create_lead_time_data()
    n_orders = len(df[df['uuid'] == 'order_placed'])
    n_deliveries = len(df[df['uuid'] == 'delivery_received'])
    print(f"\n  Created {len(df)} rows  "
          f"(orders: {n_orders}, deliveries: {n_deliveries})")

    analyzer = LeadTimeAnalysisEvents(
        dataframe=df,
        event_uuid='sc:lead_time',
    )

    # 2a. Calculate lead times
    subsection("calculate_lead_times(order_uuid, delivery_uuid)")
    lead_times = analyzer.calculate_lead_times(
        order_uuid='order_placed',
        delivery_uuid='delivery_received',
        value_column='value_string',
    )
    print(f"  Paired orders: {len(lead_times)}")
    if not lead_times.empty:
        print(lead_times[['order_time', 'delivery_time',
                           'lead_time_hours', 'order_id']].to_string(index=False))

    # 2b. Lead time statistics
    subsection("lead_time_statistics()")
    stats = analyzer.lead_time_statistics(
        order_uuid='order_placed',
        delivery_uuid='delivery_received',
    )
    if not stats.empty:
        print(f"  mean  = {stats['mean_hours'].iloc[0]:.2f} h")
        print(f"  std   = {stats['std_hours'].iloc[0]:.2f} h")
        print(f"  min   = {stats['min_hours'].iloc[0]:.2f} h")
        print(f"  max   = {stats['max_hours'].iloc[0]:.2f} h")
        print(f"  p95   = {stats['p95_hours'].iloc[0]:.2f} h")
        print(f"  count = {stats['count'].iloc[0]}")

    # 2c. Detect anomalies with default threshold (2.0)
    subsection("detect_lead_time_anomalies(threshold_factor=2.0)")
    anomalies_strict = analyzer.detect_lead_time_anomalies(
        order_uuid='order_placed',
        delivery_uuid='delivery_received',
        threshold_factor=2.0,
    )
    print(f"  Anomalies (>mean+2*std): {len(anomalies_strict)}")
    if not anomalies_strict.empty:
        print(anomalies_strict[['order_time', 'delivery_time',
                                 'lead_time_hours', 'z_score']].to_string(index=False))

    # 2d. Detect anomalies with a looser threshold (1.5)
    subsection("detect_lead_time_anomalies(threshold_factor=1.5)")
    anomalies_loose = analyzer.detect_lead_time_anomalies(
        order_uuid='order_placed',
        delivery_uuid='delivery_received',
        threshold_factor=1.5,
    )
    print(f"  Anomalies (>mean+1.5*std): {len(anomalies_loose)}")
    if not anomalies_loose.empty:
        print(anomalies_loose[['order_time', 'delivery_time',
                                'lead_time_hours', 'z_score']].to_string(index=False))


# ===========================================================================
# DEMO 3 -- DemandPatternEvents
# ===========================================================================

def demo_demand_patterns() -> None:
    section("DEMO 3: DemandPatternEvents")

    df = create_demand_data()
    print(f"\n  Created {len(df)} demand rows over 60 days")

    analyzer = DemandPatternEvents(
        dataframe=df,
        demand_uuid='product_demand',
        event_uuid='sc:demand',
        value_column='value_double',
    )

    # 3a. Demand by period -- daily
    subsection("demand_by_period(period='1D')  [daily]")
    daily = analyzer.demand_by_period(period='1D')
    if not daily.empty:
        print(f"  Days with data   : {len(daily)}")
        print(f"  Avg daily demand : {daily['total_demand'].mean():.1f}")
        print(f"  Max daily demand : {daily['total_demand'].max():.1f}")
        print(f"\n  First 14 days:")
        print(daily[['start', 'total_demand',
                      'avg_demand', 'peak_demand']].head(14).to_string(index=False))

    # 3b. Demand by period -- hourly
    subsection("demand_by_period(period='1h')  [hourly]")
    hourly_agg = analyzer.demand_by_period(period='1h')
    if not hourly_agg.empty:
        print(f"  Hourly buckets   : {len(hourly_agg)}")
        print(f"  Avg hourly demand: {hourly_agg['total_demand'].mean():.2f}")
        print(f"\n  Sample (first 12 hourly buckets):")
        print(hourly_agg[['start', 'total_demand',
                           'avg_demand', 'peak_demand']].head(12).to_string(index=False))

    # 3c. Demand spike detection -- daily
    subsection("detect_demand_spikes(threshold_factor=1.5, window='1D')")
    spikes = analyzer.detect_demand_spikes(threshold_factor=1.5, window='1D')
    print(f"  Spikes detected: {len(spikes)}")
    if not spikes.empty:
        print(spikes[['start', 'demand',
                       'baseline_mean', 'spike_magnitude']].to_string(index=False))

    # 3d. Demand spike detection -- tighter threshold
    subsection("detect_demand_spikes(threshold_factor=2.0, window='1D')")
    spikes_tight = analyzer.detect_demand_spikes(threshold_factor=2.0, window='1D')
    print(f"  Spikes detected (stricter): {len(spikes_tight)}")
    if not spikes_tight.empty:
        print(spikes_tight[['start', 'demand',
                             'baseline_mean', 'spike_magnitude']].to_string(index=False))

    # 3e. Seasonality -- day-of-week
    subsection("seasonality_summary(period='1D')  [day-of-week]")
    dow = analyzer.seasonality_summary(period='1D')
    if not dow.empty:
        print(dow.to_string(index=False))

    # 3f. Seasonality -- hour-of-day
    subsection("seasonality_summary(period='1h')  [hour-of-day]")
    hod = analyzer.seasonality_summary(period='1h')
    if not hod.empty:
        print(hod.to_string(index=False))


# ===========================================================================
# DEMO 4 -- Cross-signal scenario: combining all three classes
# ===========================================================================

def demo_combined_scenario() -> None:
    """Show how the three classes can operate on a single merged DataFrame.

    In practice a supply chain system might store inventory levels, order/delivery
    events, and demand signals in the same timeseries table, differentiated only
    by uuid.  This demo merges all three datasets and runs analyses on the
    combined frame.
    """
    section("DEMO 4: Combined Cross-Signal Scenario")

    inv_df = create_inventory_data()
    lt_df = create_lead_time_data()
    dem_df = create_demand_data()

    # Merge into a single DataFrame (as would be stored in a real system)
    combined = pd.concat([inv_df, lt_df, dem_df], ignore_index=True)
    combined = combined.sort_values('systime').reset_index(drop=True)

    uuids = combined['uuid'].unique()
    print(f"\n  Combined DataFrame: {len(combined)} rows, "
          f"{len(uuids)} unique UUIDs")
    print(f"  UUIDs: {list(uuids)}")

    # Inventory monitoring on the combined frame -- just pass it the right uuid
    monitor = InventoryMonitoringEvents(
        dataframe=combined,
        level_uuid='warehouse_a_level',
    )
    subsection("Inventory: low stock from combined frame")
    low = monitor.detect_low_stock(min_level=150, hold='2h')
    print(f"  Low-stock intervals (warehouse_a, <150): {len(low)}")

    # Lead time analysis on the combined frame
    lta = LeadTimeAnalysisEvents(dataframe=combined)
    subsection("Lead times from combined frame")
    lt = lta.calculate_lead_times('order_placed', 'delivery_received')
    print(f"  Paired orders: {len(lt)}")
    if not lt.empty:
        print(f"  Mean lead time: {lt['lead_time_hours'].mean():.1f} h")

    # Demand analysis on the combined frame
    dpa = DemandPatternEvents(dataframe=combined, demand_uuid='product_demand')
    subsection("Daily demand from combined frame")
    daily = dpa.demand_by_period(period='1D')
    print(f"  Days with demand data: {len(daily)}")
    if not daily.empty:
        print(f"  Avg daily demand: {daily['total_demand'].mean():.1f}")

    print("\n  All three analyses ran successfully on a single merged DataFrame.")


# ===========================================================================
# Main entry point
# ===========================================================================

def main() -> int:
    print("\n" + "#" * 74)
    print("#  ts-shape Supply Chain Events -- Comprehensive Demo")
    print("#" * 74)

    try:
        demo_inventory_monitoring()
        demo_lead_time_analysis()
        demo_demand_patterns()
        demo_combined_scenario()

        section("ALL DEMOS COMPLETED SUCCESSFULLY")
        return 0

    except Exception as exc:
        print(f"\n  ERROR during demo: {exc}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
