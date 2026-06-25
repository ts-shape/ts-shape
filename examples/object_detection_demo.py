"""Object detection demo — extract OCEL 2.0 objects from id-bearing signals.

Runs as-is, no data files::

    python examples/object_detection_demo.py

Shows the three-step methodology:
  1. detect objects (batch / serial / coil) from identifier signals,
  2. infer object-to-object relations (serial part_of batch) from overlap,
  3. attach the detected objects to a real event detector's output by time —
     so events gain rich object references with no per-detector code.
"""

from __future__ import annotations

import pandas as pd

import ts_shape
from ts_shape.eventlog import (
    ObjectSpec,
    attach_objects,
    detect_objects,
    to_event_log,
    to_event_log_ocel,
)
from ts_shape.events.production.machine_state import MachineStateEvents


def main() -> None:
    # 1. Synthesize identifier signals on one production line.
    batch = ts_shape.make_id_signal(
        "sig:batch", ["B1", "B2"], hold=30, source_uuid="line-1"
    )
    serial = ts_shape.make_id_signal(
        "sig:serial",
        ["S1", "S2", "S3", "S4", "S5", "S6"],
        hold=10,
        source_uuid="line-1",
    )
    coil = ts_shape.make_id_signal(
        "sig:coil", ["C1", "C2"], hold=30, source_uuid="line-1"
    )
    df = pd.concat([batch, serial, coil], ignore_index=True)

    specs = [
        ObjectSpec("sig:batch", "batch", id_template="{type}:{value}"),
        ObjectSpec("sig:serial", "serial", id_template="{type}:{value}"),
        # 'coil' is not a standard type — auto-registered, no code needed.
        ObjectSpec("sig:coil", "coil", id_template="{type}:{value}"),
    ]

    # 2. Detect objects + object-to-object relations. `part_of` is asserted only
    #    along a declared hierarchy (a serial belongs to a batch); pure temporal
    #    overlap is reported honestly as `co_occurs`, never guessed as part_of.
    obj_log = detect_objects(df, specs, hierarchy={"serial": "batch"})
    print("=" * 70)
    print("Detected objects")
    print("=" * 70)
    print(obj_log.objects.to_string(index=False))
    print("\nObject-to-object relations (declared hierarchy → part_of):")
    print(obj_log.o2o.to_string(index=False))

    # 3. Attach detected objects to a real event detector's output by time.
    state = ts_shape.make_id_signal(
        "sig:state", ["run", "idle", "run", "idle"], hold=15, source_uuid="line-1"
    )
    state["value_bool"] = state["value_string"].isin(["run"])
    legacy = MachineStateEvents(state, run_state_uuid="sig:state").detect_run_idle()
    legacy["source_uuid"] = "line-1"
    ev_log = to_event_log(legacy, detector="MachineStateEvents.detect_run_idle")

    enriched = attach_objects(
        ev_log,
        df,
        specs,
        hierarchy={"serial": "batch"},
        qualifiers={
            "batch": "during_batch",
            "serial": "identified_by",
            "coil": "made_of",
        },
    )

    print("\n" + "=" * 70)
    print("Events linked to detected objects (E2O relations)")
    print("=" * 70)
    print(enriched.relations.to_string(index=False))

    tables = to_event_log_ocel(enriched)
    print("\nUnified OCEL 2.0 log:")
    print(f"  events:         {tables.events.shape}")
    print(f"  objects:        {tables.objects.shape}")
    print(f"  relations:      {tables.relations.shape}  (event-to-object)")
    print(f"  o2o:            {tables.o2o.shape}  (object-to-object)")
    print(f"  object_changes: {tables.object_changes.shape}  (lifecycle/attrs)")
    print("\nNew object types are added by data/config — never by writing detectors.")


if __name__ == "__main__":
    main()
