#!/usr/bin/env python3
"""
EventLog Demo for ts-shape
==========================

Demonstrates the canonical OCEL 2.0 / XES-shaped event log produced by the
``ts_shape.eventlog`` package:

1. Run two detectors on a synthetic timeseries.
2. Normalize each detector's legacy DataFrame into an :class:`EventLog`.
3. Concatenate the logs and inspect the events / objects / relations tables.
4. Export to a flat XES-style DataFrame (one row per event, with
   ``case:concept:name``, ``concept:name``, ``time:timestamp``, ...).
5. Export to OCEL 2.0 tables — ready for ``pm4py.write_ocel2_json``.

Scenario: a packaging line asset (``asset-A``) cycles between run/idle and
produces occasional outlier readings on its torque sensor. We want a single
event log spanning both quality and production events, keyed by asset.

Run it:  ``python examples/eventlog_demo.py``
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from ts_shape.events.production.machine_state import MachineStateEvents
from ts_shape.events.quality.outlier_detection import OutlierDetectionEvents
from ts_shape.eventlog import (
    concat,
    to_event_log,
    to_flat_df,
    to_ocel_tables,
)


# ---------------------------------------------------------------------------
# 1. Synthesize a 30-minute timeseries: machine state + torque readings
# ---------------------------------------------------------------------------

def synth_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return two legacy DataFrames in the format ts-shape detectors expect."""
    start = datetime(2026, 5, 7, 8, 0, 0)
    rng = np.random.default_rng(42)

    # Run/idle bool stream sampled every 30s, alternating in 5-minute blocks.
    state_ts = pd.date_range(start, periods=60, freq="30s", tz="UTC")
    state_vals = ([True] * 10 + [False] * 5) * 4  # 5min run / 2.5min idle
    state_df = pd.DataFrame({
        "systime": state_ts,
        "value_bool": state_vals,
        "uuid": ["asset-A"] * 60,
        "is_delta": [(i == 0 or state_vals[i] != state_vals[i - 1])
                     for i in range(60)],
    })

    # Torque readings every 1 minute with two injected outliers.
    torque_ts = pd.date_range(start, periods=30, freq="1min", tz="UTC")
    torque = rng.normal(loc=42.0, scale=0.8, size=30)
    torque[12] = 80.0   # spike up
    torque[24] = 5.0    # spike down
    torque_df = pd.DataFrame({
        "systime": torque_ts,
        "value_double": torque,
        "uuid": ["torque_sensor"] * 30,
        "is_delta": [False] * 30,
        "source_uuid": ["asset-A"] * 30,
        "batch_id": ["B-2026-117"] * 15 + ["B-2026-118"] * 15,
    })
    return state_df, torque_df


# ---------------------------------------------------------------------------
# 2. Run detectors and normalize their output
# ---------------------------------------------------------------------------

def main() -> None:
    state_df, torque_df = synth_inputs()

    # --- machine state: interval events ------------------------------------
    state_legacy = MachineStateEvents(
        state_df, run_state_uuid="asset-A"
    ).detect_run_idle()
    # The legacy DataFrame already carries ``source_uuid``; the adapter
    # auto-binds it to the OCEL ``asset`` object type.
    state_log = to_event_log(
        state_legacy,
        detector="MachineStateEvents.detect_run_idle",
    )

    # --- outlier: point events with severity -------------------------------
    outlier_legacy = OutlierDetectionEvents(
        torque_df, value_column="value_double"
    ).detect_outliers_zscore()
    # Bind the ``batch_id`` column to the ``batch`` object type as well, so
    # we can later flatten the log per-asset OR per-batch.
    outlier_log = to_event_log(
        outlier_legacy,
        detector="OutlierDetectionEvents.detect_outliers_zscore",
        objects={"batch": "batch_id"},
        qualifiers={"asset": "produced_on", "batch": "during_batch"},
    )

    # --- concat into a single log ------------------------------------------
    log = concat(state_log, outlier_log)

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
    print("--- events (head) ---")
    print(log.events[cols].to_string(index=False))
    print()
    print("--- objects ---")
    print(log.objects.to_string(index=False))
    print()
    print("--- relations (head) ---")
    print(log.relations.head(10).to_string(index=False))
    print()

    # ---------------------------------------------------------------------
    # 3. Flat XES-style export
    # ---------------------------------------------------------------------

    print("=" * 70)
    print("Flat XES export — case = asset")
    print("=" * 70)
    xes_asset = to_flat_df(log, case_object_type="asset", lifecycle="single")
    print(xes_asset[
        ["case:concept:name", "concept:name", "time:timestamp",
         "lifecycle:transition"]
    ].to_string(index=False))
    print()

    print("=" * 70)
    print("Flat XES export — case = batch (only outlier events have batches)")
    print("=" * 70)
    xes_batch = to_flat_df(log, case_object_type="batch")
    print(xes_batch[
        ["case:concept:name", "concept:name", "time:timestamp"]
    ].to_string(index=False))
    print()

    # ---------------------------------------------------------------------
    # 4. OCEL 2.0 export (column names match the spec verbatim)
    # ---------------------------------------------------------------------

    events_df, objects_df, relations_df = to_ocel_tables(log)
    print("=" * 70)
    print("OCEL 2.0 tables")
    print("=" * 70)
    print(f"events:    {events_df.shape}")
    print(f"objects:   {objects_df.shape}")
    print(f"relations: {relations_df.shape}")
    print()
    print("These three frames can be passed directly to e.g.:")
    print("  pm4py.write_ocel2_json(...)")
    print("  pm4py.format_dataframe(xes_asset, ...)")
    print("ts-shape itself imports neither — column names match the specs.")


if __name__ == "__main__":
    main()
