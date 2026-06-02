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
    EventLog,
    concat,
    register_adapter,
    to_event_log,
    to_event_log_ocel,
    to_event_log_xes,
)
from ts_shape.eventlog import schema as S
from ts_shape.eventlog import taxonomy
from ts_shape.eventlog.taxonomy import REGISTRY, LabelRule


def _print_label_rule(class_name: str, method_name: str) -> None:
    """Pretty-print the LabelRule that drives a detector's adapter."""
    rule = taxonomy.get(class_name, method_name)
    if rule is None:
        print(f"  (no LabelRule for {class_name}.{method_name})")
        return
    print(f"--- LabelRule for {class_name}.{method_name} ---")
    print(f"  template:           {rule.template}")
    print(f"  pack:               {rule.pack}")
    print(f"  shape:              {rule.shape}")
    print(f"  produces_objects:   {rule.produces_objects}")
    print(f"  severity_field:     {rule.severity_field}")
    print(f"  value_field:        {rule.value_field}")
    print(f"  drop_fields:        {rule.drop_fields}")


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

    # ---------------------------------------------------------------------
    # 2a. Inspect the LabelRule that drives each adapter.
    # ---------------------------------------------------------------------
    # Each detector method is registered with a LabelRule in
    # ts_shape.eventlog.taxonomy.REGISTRY. The generic adapter consults
    # this entry to know the activity name template, shape, severity
    # source column, etc.
    print("=" * 70)
    print("Adapter inputs — registry entries that drive normalization")
    print("=" * 70)
    _print_label_rule("MachineStateEvents", "detect_run_idle")
    print()
    _print_label_rule("OutlierDetectionEvents", "detect_outliers_zscore")
    print()

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

    # ---------------------------------------------------------------------
    # 2b. Show the column-by-column mapping for one row.
    # ---------------------------------------------------------------------
    # This is what the "Concrete walkthrough" table in the guide describes,
    # made concrete on actual data.
    print("=" * 70)
    print("Adapter output — column mapping for the first run/idle row")
    print("=" * 70)
    legacy_row = state_legacy.iloc[0]
    canonical_row = state_log.events.iloc[0]

    print("legacy DataFrame row:")
    for col, val in legacy_row.items():
        print(f"  {col:24s} = {val!r}")
    print()
    print("canonical EventLog row (lands in the events table):")
    for col, val in canonical_row.items():
        print(f"  {col:32s} = {val!r}")
    print()
    print("relations row (link to objects table):")
    rel_row = state_log.relations.iloc[0]
    for col, val in rel_row.items():
        print(f"  {col:24s} = {val!r}")
    print()

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

    print("--- standard attribute extension columns ---")
    std_cols = [c for c in log.events.columns
                if c.startswith("ts_shape:") and c not in {
                    "ts_shape:start_timestamp", "ts_shape:duration_s",
                    "ts_shape:detector", "ts_shape:pack",
                    "ts_shape:severity", "ts_shape:value",
                }]
    if std_cols:
        print(log.events[["ocel:activity"] + std_cols].to_string(index=False))
    else:
        print("(no standard attrs populated)")
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
    xes_asset = to_event_log_xes(log, case_object_type="asset", lifecycle="single")
    print(xes_asset[
        ["case:concept:name", "concept:name", "time:timestamp",
         "lifecycle:transition"]
    ].to_string(index=False))
    print()

    print("=" * 70)
    print("Flat XES export — case = batch (only outlier events have batches)")
    print("=" * 70)
    xes_batch = to_event_log_xes(log, case_object_type="batch")
    print(xes_batch[
        ["case:concept:name", "concept:name", "time:timestamp"]
    ].to_string(index=False))
    print()

    # ---------------------------------------------------------------------
    # 3b. Custom adapter override — when the generic shape adapter is not
    #     enough. This is purely illustrative; real detectors register the
    #     LabelRule in src/ts_shape/eventlog/taxonomy.py.
    # ---------------------------------------------------------------------

    print("=" * 70)
    print("Custom adapter — emit two events per legacy row")
    print("=" * 70)

    # 1. Make the registry aware of the (otherwise unknown) method.
    REGISTRY[("MyDetector", "alarm_pair")] = LabelRule(
        template="production.custom.{kind}",
        pack="production",
        shape="point",
        produces_objects=("asset",),
    )

    # 2. Register an override that produces *two* events per legacy row.
    @register_adapter("MyDetector", "alarm_pair")
    def _expand_pairs(legacy_df, *, rule, detector, objects, qualifiers):
        rows: list[dict] = []
        rels: list[dict] = []
        for i, row in legacy_df.iterrows():
            for kind in ("raised", "cleared"):
                eid = f"e-MyDetector-{i}-{kind}"
                rows.append({
                    S.OCEL_EID: eid,
                    S.OCEL_ACTIVITY: f"production.custom.{kind}",
                    S.OCEL_TIMESTAMP: pd.Timestamp(row[f"{kind}_at"], tz="UTC"),
                    S.TS_DETECTOR: detector,
                    S.TS_PACK: rule.pack,
                })
                rels.append({
                    S.OCEL_EID: eid,
                    S.OCEL_OID: row["asset_id"],
                    S.OCEL_TYPE: "asset",
                    S.OCEL_QUALIFIER: "produced_on",
                })
        events = pd.concat(
            [S.empty_events(), pd.DataFrame(rows)], ignore_index=True
        )
        relations = pd.concat(
            [S.empty_relations(), pd.DataFrame(rels)], ignore_index=True
        )
        objects = pd.DataFrame({
            S.OCEL_OID: legacy_df["asset_id"].astype("string").unique(),
            S.OCEL_TYPE: "asset",
        })
        return EventLog(events=events, objects=objects, relations=relations)

    # 3. Pretend a detector returned this two-row legacy DataFrame.
    pair_legacy = pd.DataFrame({
        "asset_id":   ["asset-A", "asset-A"],
        "raised_at":  ["2026-05-07T08:10:00Z", "2026-05-07T08:25:00Z"],
        "cleared_at": ["2026-05-07T08:11:30Z", "2026-05-07T08:26:15Z"],
    })

    # 4. Same to_event_log() entry point — the override is dispatched
    #    automatically based on the detector name.
    pair_log = to_event_log(pair_legacy, detector="MyDetector.alarm_pair")
    print("two legacy rows in →", len(pair_log.events), "events out:")
    print(pair_log.events[
        ["ocel:eid", "ocel:activity", "ocel:timestamp"]
    ].to_string(index=False))
    print()

    # ---------------------------------------------------------------------
    # 4. OCEL 2.0 export (column names match the spec verbatim)
    # ---------------------------------------------------------------------

    events_df, objects_df, relations_df = to_event_log_ocel(log)
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
