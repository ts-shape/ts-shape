"""ts-shape canonical event-log package.

A lightweight, pm4py-compatible (XES + OCEL 2.0) representation of detector
output. ts-shape itself adds no process-mining dependencies — the columns
match the specs verbatim so users can hand the resulting DataFrames to
pm4py / Disco / Celonis / OCEL viewers directly.

Typical use::

    from ts_shape.eventlog import to_event_log, concat, to_flat_df, to_ocel_tables
    from ts_shape.events.production.machine_state import MachineStateEvents
    from ts_shape.events.quality.outlier_detection import OutlierDetectionEvents

    state_log = to_event_log(
        MachineStateEvents(df, run_state_uuid="m").detect_run_idle(),
        detector="MachineStateEvents.detect_run_idle",
    )
    outlier_log = to_event_log(
        OutlierDetectionEvents(df, value_column="value_double").detect_outliers_zscore(),
        detector="OutlierDetectionEvents.detect_outliers_zscore",
    )
    log = concat(state_log, outlier_log)
    xes_df = to_flat_df(log, case_object_type="asset")
    events_df, objects_df, relations_df = to_ocel_tables(log)
"""
from .adapters import register_adapter
from .concat import concat
from .flat import to_flat_df
from .model import EventLog
from .normalizer import to_event_log
from .ocel import to_ocel_tables
from .schema import (
    OCEL_ACTIVITY,
    OCEL_EID,
    OCEL_OID,
    OCEL_QUALIFIER,
    OCEL_TIMESTAMP,
    OCEL_TYPE,
    TS_DETECTOR,
    TS_DURATION_S,
    TS_PACK,
    TS_SEVERITY,
    TS_START_TIMESTAMP,
    TS_VALUE,
    XES_ACTIVITY,
    XES_CASE,
    XES_LIFECYCLE,
    XES_RESOURCE,
    XES_TIMESTAMP,
    register_object_type,
    validate,
)
from .taxonomy import REGISTRY, LabelRule

__all__ = [
    "EventLog",
    "LabelRule",
    "REGISTRY",
    "concat",
    "register_adapter",
    "register_object_type",
    "to_event_log",
    "to_flat_df",
    "to_ocel_tables",
    "validate",
    "OCEL_ACTIVITY",
    "OCEL_EID",
    "OCEL_OID",
    "OCEL_QUALIFIER",
    "OCEL_TIMESTAMP",
    "OCEL_TYPE",
    "TS_DETECTOR",
    "TS_DURATION_S",
    "TS_PACK",
    "TS_SEVERITY",
    "TS_START_TIMESTAMP",
    "TS_VALUE",
    "XES_ACTIVITY",
    "XES_CASE",
    "XES_LIFECYCLE",
    "XES_RESOURCE",
    "XES_TIMESTAMP",
]
