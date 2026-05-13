"""Declarative rule spec for the lambda-rule subsystem.

A :class:`RuleSpec` is the user-facing description of one detection rule.
It carries everything the loader needs to (a) compile the trigger
expression, (b) build a :class:`~ts_shape.eventlog.taxonomy.LabelRule`,
and (c) classify the rule into an archetype that
``tests/eventlog/test_adapter_coverage.py`` already validates.

The shape mirrors the built-in detector taxonomy: every lambda rule
ultimately becomes a ``(class_name, method_name)`` REGISTRY entry that
the canonical :func:`~ts_shape.eventlog.to_event_log` dispatcher
understands without any special-case code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

_VALID_SHAPES: frozenset[str] = frozenset({"point", "interval", "summary", "static"})
_VALID_PACKS: frozenset[str] = frozenset(
    {
        "quality",
        "production",
        "engineering",
        "maintenance",
        "supplychain",
        "energy",
        "correlation",
    }
)
_VALID_ARCHETYPES: frozenset[str] = frozenset(
    {
        "threshold",
        "interval",
        "aggregate",
        "outcome",
        "static",
        "trace",
        "forecast",
        "correlation",
    }
)


@dataclass(frozen=True)
class TriggerSpec:
    """When does the rule fire?

    ``expression`` is evaluated against the input DataFrame by the
    AST-restricted compiler in :mod:`.expression`. It must produce a
    boolean Series aligned with the input rows.

    ``min_duration_s`` and ``group_by`` are only meaningful for
    ``shape="interval"`` rules: consecutive True rows are coalesced
    (per group) and the resulting interval is dropped if shorter than
    ``min_duration_s`` seconds.
    """

    expression: str
    min_duration_s: float | None = None
    group_by: tuple[str, ...] = ()


@dataclass(frozen=True)
class RuleSpec:
    """A complete lambda-rule definition.

    Required fields mirror the built-in REGISTRY contract — ``class_name``
    is synthesized (must start with ``Lambda``) so the rule slots into the
    same ``(class, method) -> LabelRule`` table as the 290 built-ins.
    """

    id: str
    class_name: str
    method_name: str
    pack: str
    shape: str
    archetype: str
    template: str
    trigger: TriggerSpec
    produces_objects: tuple[str, ...] = ("asset",)
    severity_field: str | None = None
    value_field: str | None = None
    standard_attrs: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.class_name.startswith("Lambda"):
            raise ValueError(
                f"class_name must start with 'Lambda', got {self.class_name!r}"
            )
        if self.shape not in _VALID_SHAPES:
            raise ValueError(
                f"shape must be one of {sorted(_VALID_SHAPES)}, got {self.shape!r}"
            )
        if self.pack not in _VALID_PACKS:
            raise ValueError(
                f"pack must be one of {sorted(_VALID_PACKS)}, got {self.pack!r}"
            )
        if self.archetype not in _VALID_ARCHETYPES:
            raise ValueError(
                f"archetype must be one of {sorted(_VALID_ARCHETYPES)}, "
                f"got {self.archetype!r}"
            )
        if self.shape == "interval" and self.archetype != "interval":
            raise ValueError(
                "shape='interval' requires archetype='interval' "
                f"(got archetype={self.archetype!r})"
            )
        if self.trigger.min_duration_s is not None and self.shape != "interval":
            raise ValueError(
                "trigger.min_duration_s is only valid for shape='interval'"
            )
