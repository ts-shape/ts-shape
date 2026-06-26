"""End-to-end handover guarantee: every detector method registered in the
taxonomy must be able to hand its output to the event layer.

This is the methodology that keeps the library working end to end — a detector
is only useful if ``to_event_log`` can normalize its output into a canonical,
schema-valid :class:`EventLog`. We synthesize the canonical output shape each
method declares (``point`` / ``interval`` / ``summary`` / ``static`` — the same
shapes ``ts_shape.events._output`` finalizers emit) and assert the full handover
path succeeds and round-trips through the OCEL 2.0 exporter.

Complements ``test_adapter_coverage.py`` (which checks a rule *exists* for every
method) by exercising the rule against real input for all of them.
"""

from __future__ import annotations

import re

import pandas as pd
import pytest

from ts_shape.eventlog import (
    OCEL2Tables,
    to_event_log,
    to_event_log_ocel,
    to_event_log_xes,
    validate,
)
from ts_shape.eventlog.taxonomy import REGISTRY, LabelRule
from ts_shape.events._output import COL_END, COL_START, COL_SYSTIME

_N = 3


def _synthetic_output(rule: LabelRule) -> pd.DataFrame:
    """Build a minimal frame in the canonical shape ``rule`` declares.

    Mirrors what a real detector emits: ``systime`` for point shapes,
    ``start``/``end`` for interval/summary, no time column for static. Adds the
    columns the activity template and severity/value fields reference so the
    full adapter path (templating, severity bucketing, object extraction) runs.
    """
    base = pd.date_range("2026-05-07", periods=_N, freq="1min", tz="UTC")
    df = pd.DataFrame({"uuid": ["sig-1"] * _N, "source_uuid": ["asset-1"] * _N})

    if rule.shape in ("interval", "summary"):
        df[COL_START] = base
        df[COL_END] = base + pd.Timedelta("30s")
    elif rule.shape == "point":
        df[COL_SYSTIME] = base
    # static: no natural time column — the adapter broadcasts a fixed now.

    # Templated activity segments (e.g. production.machine_state.{state}).
    for placeholder in re.findall(r"\{(\w+)\}", rule.template):
        df[placeholder] = "x"

    if rule.severity_field:
        df[rule.severity_field] = [1.0, 3.5, 5.0][:_N]
    if rule.value_field:
        df[rule.value_field] = [1.0] * _N

    return df


@pytest.mark.parametrize("key", sorted(REGISTRY), ids=lambda k: f"{k[0]}.{k[1]}")
def test_detector_output_hands_over_to_event_layer(key: tuple[str, str]) -> None:
    class_name, method_name = key
    rule = REGISTRY[key]
    legacy = _synthetic_output(rule)

    # to_event_log validates internally; an invalid handover raises here.
    log = to_event_log(legacy, detector=f"{class_name}.{method_name}")
    validate(log)
    assert len(log.events) == _N

    # The canonical OCEL 2.0 export must always succeed.
    tables = to_event_log_ocel(log)
    assert isinstance(tables, OCEL2Tables)
    assert len(tables.events) == _N

    # When the handover produced objects, the XES flattening must work too.
    if log.has_objects:
        flat = to_event_log_xes(log, case_object_type="asset")
        assert not flat.empty


def test_every_registered_method_is_covered() -> None:
    """Guard: the parametrization above actually spans the whole registry."""
    assert len(REGISTRY) > 250, "registry unexpectedly small — handover sweep is thin"
