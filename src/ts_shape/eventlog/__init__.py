"""ts-shape canonical event-log package.

A lightweight, pm4py-compatible (XES + OCEL 2.0) representation of detector
output. ts-shape itself adds no process-mining dependencies — the columns
match the specs verbatim so users can hand the resulting DataFrames to
pm4py / Disco / Celonis / OCEL viewers directly.

Typical use::

    from ts_shape.eventlog import (
        to_event_log, concat, to_event_log_xes, to_event_log_ocel,
    )
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
    xes_df = to_event_log_xes(log, case_object_type="asset")
    tables = to_event_log_ocel(log)  # OCEL2Tables(events, objects, relations, ...)
"""

from .adapters import register_adapter
from .align import align_columns
from .concat import concat
from .flat import to_event_log_xes
from .lambda_rules import (
    BacktestResult,
    LambdaDetector,
    RuleSpec,
    TriggerSpec,
    UnsafeExpression,
    compile_expression,
    load_dicts,
    load_yaml,
    register_lambda_rule,
    run_backtest,
    unregister_lambda_rule,
)
from .model import EventLog
from .normalizer import to_event_log
from .objects import (
    ObjectSpec,
    attach_objects,
    detect_objects,
    object_intervals,
    object_specs_from_metadata,
)
from .ocel import OCEL2Tables, to_event_log_ocel
from .schema import (
    OCEL_ACTIVITY,
    OCEL_EID,
    OCEL_FIELD,
    OCEL_OID,
    OCEL_OID2,
    OCEL_QUALIFIER,
    OCEL_TIMESTAMP,
    OCEL_TYPE,
    OCEL_VALUE,
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
    empty_object_changes,
    empty_o2o,
    register_object_type,
    validate,
)
from .taxonomy import REGISTRY, LabelRule

__all__ = [
    "BacktestResult",
    "EventLog",
    "OCEL2Tables",
    "ObjectSpec",
    "align_columns",
    "attach_objects",
    "detect_objects",
    "object_intervals",
    "object_specs_from_metadata",
    "LabelRule",
    "LambdaDetector",
    "REGISTRY",
    "RuleSpec",
    "TriggerSpec",
    "UnsafeExpression",
    "compile_expression",
    "concat",
    "empty_object_changes",
    "empty_o2o",
    "load_dicts",
    "load_yaml",
    "register_adapter",
    "register_lambda_rule",
    "register_object_type",
    "run_backtest",
    "to_event_log",
    "to_event_log_ocel",
    "to_event_log_xes",
    "unregister_lambda_rule",
    "validate",
    "OCEL_ACTIVITY",
    "OCEL_EID",
    "OCEL_FIELD",
    "OCEL_OID",
    "OCEL_OID2",
    "OCEL_QUALIFIER",
    "OCEL_TIMESTAMP",
    "OCEL_TYPE",
    "OCEL_VALUE",
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
