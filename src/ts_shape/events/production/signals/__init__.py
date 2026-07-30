"""Signal normalization for global production reporting.

The analytics in :mod:`ts_shape.events.production` all assume a specific signal
encoding -- a boolean run flag, a monotonic counter, a string part number. That
assumption holds on one machine and breaks across a plant, let alone a fleet:
the same logical quantity is encoded differently on nearly every controller.

This subpackage is the layer that removes the assumption. Point it at whatever
a machine actually publishes and it emits canonical production events:

===========================  ====================================================
:mod:`.spec`                 Declarative, JSON-round-trippable signal mapping
                             (:class:`~.spec.MachineSpec`, :class:`~.spec.FleetSpec`).
:mod:`.values`               Value-column resolution and dtype coercion.
:mod:`.counters`             Any part-counter encoding -> OK/NOK/total quantities.
:mod:`.status`               Any status encoding -> canonical state intervals.
:mod:`.cycles`               Any cycle-time source -> per-cycle durations.
:mod:`.context`              Any identifier signal -> production context columns.
:mod:`.report`               Fleet-scale roll-up over hundreds of machines.
===========================  ====================================================

Quick start -- one machine, whatever it publishes::

    from ts_shape.events.production.signals import (
        CounterKind, CounterMode, CounterSpec, ProductionCounterNormalizer,
    )

    counts = ProductionCounterNormalizer(
        df,
        counters=[
            CounterSpec(uuid="m1.parts", kind=CounterKind.TOTAL,
                        mode=CounterMode.CUMULATIVE, rollover_at=65536),
            CounterSpec(uuid="m1.reject", kind=CounterKind.NOK,
                        mode=CounterMode.PULSE),
        ],
        machine_id="M1",
    )
    counts.counts_by_window("1h")

Fleet scale -- the same report over hundreds of machines::

    from ts_shape.events.production.signals import FleetProductionReport, FleetSpec

    fleet = FleetSpec.from_records(config_table.to_dict("records"))
    report = FleetProductionReport(fleet, df)
    report.production_by_window("1h")     # tidy: machine x window
    report.oee("1D")                      # availability / performance / quality
    report.signal_coverage()              # which machines are mis-mapped
"""

from ts_shape.events.production.signals.context import ProductionContextResolver
from ts_shape.events.production.signals.counters import ProductionCounterNormalizer
from ts_shape.events.production.signals.cycles import CycleTimeNormalizer
from ts_shape.events.production.signals.report import FleetProductionReport
from ts_shape.events.production.signals.spec import (
    CONTEXT_DIMENSIONS,
    PACKML_STATE_MAP,
    STACKLIGHT_STATE_MAP,
    ContextBinding,
    ContextSpec,
    CounterKind,
    CounterMode,
    CounterSpec,
    CycleMode,
    CycleSpec,
    FleetSpec,
    MachineSpec,
    MachineState,
    ResetPolicy,
    Rollover,
    StatusMode,
    StatusSpec,
    TimeUnit,
)
from ts_shape.events.production.signals.status import MachineStatusNormalizer
from ts_shape.events.production.signals.values import (
    DEFAULT_SENTINELS,
    VALUE_COLUMNS,
    coerce_bool,
    coerce_label,
    coerce_numeric,
    resolve_value_column,
    signal_frame,
)

__all__ = [
    "CONTEXT_DIMENSIONS",
    "DEFAULT_SENTINELS",
    "PACKML_STATE_MAP",
    "STACKLIGHT_STATE_MAP",
    "VALUE_COLUMNS",
    "ContextBinding",
    "ContextSpec",
    "CounterKind",
    "CounterMode",
    "CounterSpec",
    "CycleMode",
    "CycleSpec",
    "CycleTimeNormalizer",
    "FleetProductionReport",
    "FleetSpec",
    "MachineSpec",
    "MachineState",
    "MachineStatusNormalizer",
    "ProductionContextResolver",
    "ProductionCounterNormalizer",
    "ResetPolicy",
    "Rollover",
    "StatusMode",
    "StatusSpec",
    "TimeUnit",
    "coerce_bool",
    "coerce_label",
    "coerce_numeric",
    "resolve_value_column",
    "signal_frame",
]
