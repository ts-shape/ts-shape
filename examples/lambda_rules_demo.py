#!/usr/bin/env python3
"""
Lambda Rules Demo for ts-shape
==============================

Demonstrates user-authored detection rules — declared in YAML, no Python
class on disk — that flow through the same canonical-event-log plumbing
as the 290 built-in detector methods.

This example covers both shapes that the MVP supports:

1. **Case 1 — point/threshold:** "high torque on a running tool" — fires
   per row that crosses a threshold. Severity is derived from a numeric
   ``severity_score`` column the rule binds via ``severity_field``.
2. **Case 2 — interval / hysteresis / group-by:** "sustained hot bearing
   window per asset" — coalesces consecutive True rows per ``source_uuid``
   into a single interval event, then drops anything shorter than
   ``min_duration_s=30``.

Run it:  ``python examples/lambda_rules_demo.py``
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ts_shape.eventlog import (
    concat,
    load_yaml,
    run_backtest,
    to_event_log_ocel,
    to_event_log_xes,
    unregister_lambda_rule,
)


def synth_inputs() -> pd.DataFrame:
    """Synthesize one DataFrame that covers both demo rules.

    Columns: systime, torque, state, severity_score, bearing_temp_c,
             source_uuid. 30 minutes at 30-second resolution; 2 assets.
    """
    start = datetime(2026, 5, 7, 8, 0, 0)
    ts = pd.date_range(start, periods=60, freq="30s", tz="UTC")
    rng = np.random.default_rng(42)

    asset_a = pd.DataFrame({
        "systime": ts,
        "source_uuid": ["asset-A"] * 60,
        "state": ["run"] * 50 + ["idle"] * 10,
        # Baseline torque ~ 42, three sustained spikes for the threshold rule.
        "torque": rng.normal(loc=42.0, scale=1.5, size=60),
        # Baseline bearing temp ~ 80, one long hot window (10 contiguous rows = 270s)
        # at rows 20..29, plus one isolated spike at row 45.
        "bearing_temp_c": rng.normal(loc=80.0, scale=0.7, size=60),
    })
    asset_a.loc[10, "torque"] = 79.5
    asset_a.loc[15, "torque"] = 86.0
    asset_a.loc[40, "torque"] = 95.0
    asset_a.loc[10, "severity_score"] = 3.2
    asset_a.loc[15, "severity_score"] = 4.6
    asset_a.loc[40, "severity_score"] = 4.9
    asset_a.loc[20:29, "bearing_temp_c"] = [86, 87, 88, 87, 89, 90, 88, 86, 87, 86]
    asset_a.loc[45, "bearing_temp_c"] = 91  # isolated spike, < min_duration

    asset_b = pd.DataFrame({
        "systime": ts,
        "source_uuid": ["asset-B"] * 60,
        "state": ["run"] * 60,
        "torque": rng.normal(loc=40.0, scale=1.2, size=60),
        # asset-B has a 90s hot window at rows 5..8 (>= 30s threshold).
        "bearing_temp_c": rng.normal(loc=78.0, scale=0.6, size=60),
    })
    asset_b.loc[5:8, "bearing_temp_c"] = [88, 89, 90, 86]

    df = pd.concat([asset_a, asset_b], ignore_index=True)
    df["severity_score"] = df["severity_score"].fillna(1.0)
    return df


def main() -> None:
    df = synth_inputs()
    yaml_path = Path(__file__).parent / "lambda_rules_demo.yaml"

    # ---------------------------------------------------------------------
    # 1. Load + register the two YAML rules.
    # ---------------------------------------------------------------------
    detectors = load_yaml(yaml_path)
    try:
        torque_det, bearing_det = detectors
        print("=" * 70)
        print("Registered lambda rules")
        print("=" * 70)
        for d in detectors:
            print(f"  {d.detector_name:48s}  shape={d.spec.shape}")
        print()

        # -----------------------------------------------------------------
        # 2. Run each detector → EventLog and inspect.
        # -----------------------------------------------------------------
        torque_log = torque_det.to_event_log(df)
        bearing_log = bearing_det.to_event_log(df)
        log = concat(torque_log, bearing_log)

        print("=" * 70)
        print("EventLog summary")
        print("=" * 70)
        print(log)
        print()

        cols = [
            "ocel:eid", "ocel:activity", "ocel:timestamp",
            "ts_shape:start_timestamp", "ts_shape:duration_s",
            "ts_shape:severity", "ts_shape:value",
        ]
        print("--- events ---")
        print(log.events[cols].to_string(index=False))
        print()

        # -----------------------------------------------------------------
        # 3. XES and OCEL 2.0 round-trip.
        # -----------------------------------------------------------------
        xes = to_event_log_xes(log, case_object_type="asset")
        print("=" * 70)
        print("Flat XES export — case = asset (first 10 rows)")
        print("=" * 70)
        print(xes[
            ["case:concept:name", "concept:name", "time:timestamp"]
        ].head(10).to_string(index=False))
        print()

        events_df, objects_df, relations_df = to_event_log_ocel(log)
        print("=" * 70)
        print("OCEL 2.0 tables")
        print("=" * 70)
        print(f"events:    {events_df.shape}")
        print(f"objects:   {objects_df.shape}")
        print(f"relations: {relations_df.shape}")
        print()

        # -----------------------------------------------------------------
        # 4. Backtest the threshold rule.
        # -----------------------------------------------------------------
        print("=" * 70)
        print("Backtest — torque threshold rule")
        print("=" * 70)
        result = run_backtest(torque_det, df)
        print(result)
        print()
        print("  by_severity:", result.by_severity)
        print("  by_asset:   ", result.by_asset)
    finally:
        # Keep the global REGISTRY clean so re-running the demo is idempotent
        # and so coverage tests run after the demo still pass.
        for d in detectors:
            unregister_lambda_rule(d.spec.class_name, d.spec.method_name)


if __name__ == "__main__":
    main()
