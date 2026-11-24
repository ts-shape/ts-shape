"""Event-log rules of the mapping-free unit profilers, in one place.

Every profiler (:mod:`ts_shape.events.production._unit_profile`) has the
same kinds of tables: point records per unit, registers, evidence tables and
an ``evaluation_window``. This module declares them compactly; the taxonomy
(:data:`ts_shape.eventlog.taxonomy.REGISTRY`) and the archetype map
(:data:`ts_shape.eventlog.archetypes.ARCHETYPE_BY_METHOD`) are filled from
it. ``evaluation_window`` gets :data:`EVALUATION_RULE` unless a class names
its own.

Pure data (no imports from the taxonomy), so both modules can read it.

Per class: ``pack``, ``slug`` (the template is ``<pack>.<slug>.<name>``) and
``tables``: method -> ``name``, ``shape`` (point / interval / summary /
static), ``archetype`` and the other :class:`~ts_shape.eventlog.taxonomy.
LabelRule` fields (``objs`` for ``produces_objects``).
"""

from __future__ import annotations

from typing import Any

__all__ = ["EVALUATION_RULE", "PROFILER_RULES"]

#: The rule of every profiler's ``evaluation_window()``.
EVALUATION_RULE: dict[str, Any] = {
    "name": "evaluation_window",
    "shape": "static",
    "archetype": "aggregate",
    "standard_attrs": {"ts_shape:sample_count": "unit_count"},
}


def _records(name: str, unit: str | None, value: str, outcome: str) -> dict[str, Any]:
    """A point record per unit (outcome archetype)."""
    rule: dict[str, Any] = {
        "name": name,
        "shape": "point",
        "archetype": "outcome",
        "value_field": value,
        "standard_attrs": {"ts_shape:outcome": outcome},
    }
    if unit is not None:
        rule["objs"] = ("asset", unit)
        rule["object_columns"] = {unit: "unit_uuid"}
    return rule


def _table(name: str, archetype: str, **attrs: Any) -> dict[str, Any]:
    """A static table (register, evidence, view)."""
    return {
        "name": name,
        "shape": "static",
        "archetype": archetype,
        "standard_attrs": {f"ts_shape:{k}": v for k, v in attrs.items()},
    }


_COUNTER_ROLES = _table("counter_roles", "static", method="sum_identity")

PROFILER_RULES: dict[str, dict[str, Any]] = {
    "StopProfileEvents": {
        "pack": "production",
        "slug": "stop_profile",
        "tables": {
            "stop_records": _records("stop", "stop", "lost_seconds", "signature"),
            "stop_signatures": _table(
                "signature_register", "aggregate", sample_count="stop_count"
            ),
            "signature_evidence": _table(
                "signature_evidence",
                "static",
                method="fisher_enrichment",
                sample_count="unit_count",
            ),
        },
    },
    "ChangeoverProfileEvents": {
        "pack": "production",
        "slug": "changeover_profile",
        "tables": {
            "changeover_records": _records(
                "changeover", "changeover", "duration_seconds", "to_variant"
            ),
            "changeover_view": _table(
                "standard_steps", "static", method="presence_ratio"
            ),
            "changeover_register": _table(
                "register", "aggregate", sample_count="unit_count"
            ),
            "changeover_matrix": _table(
                "matrix", "aggregate", sample_count="changeover_count"
            ),
        },
    },
    "StartupProfileEvents": {
        "pack": "production",
        "slug": "startup_profile",
        "tables": {
            "startup_records": _records(
                "startup", "startup", "duration_seconds", "trigger"
            ),
            "startup_signals": _table(
                "signals", "aggregate", sample_count="startup_count"
            ),
            "startup_register": _table(
                "register", "aggregate", sample_count="unit_count"
            ),
        },
    },
    "OrderProfileEvents": {
        "pack": "production",
        "slug": "order_profile",
        "tables": {
            "order_records": _records("order", "order", "loss_seconds", "variant"),
            "order_register": _table(
                "register", "aggregate", sample_count="unit_count"
            ),
            "counter_roles": _COUNTER_ROLES,
        },
    },
    "ScrapProfileEvents": {
        "pack": "production",
        "slug": "scrap_profile",
        "tables": {
            "nok_records": _records(
                "nok_cycle", "cycle", "bad_count", "assigned_context"
            ),
            "scrap_pareto": _table(
                "pareto",
                "static",
                method="fisher_enrichment",
                sample_count="nok_count",
            ),
            "counter_roles": _COUNTER_ROLES,
        },
    },
    "HandshakeProfileEvents": {
        "pack": "production",
        "slug": "handshake_profile",
        "tables": {
            "handshake_pairs": _table(
                "pairs",
                "static",
                method="edge_succession",
                sample_count="occurrence_count",
            ),
            "handshake_records": _records("wait", None, "duration_seconds", "pair"),
            "handshake_register": _table(
                "register", "aggregate", sample_count="unit_count"
            ),
        },
    },
    "CurveSignatureEvents": {
        "pack": "production",
        "slug": "curve_signature",
        "tables": {
            "curve_golden": _table(
                "golden_curve", "static", method="milestone_warped_band"
            ),
            "curve_cycle_scores": {
                **_records("cycle_score", "cycle", "deviation_score", "pattern_id"),
                "object_columns": {"cycle": "cycle_uuid"},
            },
            "curve_patterns": _table(
                "patterns", "aggregate", sample_count="cycle_count"
            ),
        },
    },
    "SettingImpactEvents": {
        "pack": "production",
        "slug": "setting_impact",
        "tables": {
            "setting_changes": _records(
                "change", None, "after_value", "recipe_parameter"
            ),
            "setting_impact": _table(
                "before_after", "aggregate", sample_count="after_count"
            ),
            "setting_register": _table(
                "register", "aggregate", sample_count="change_count"
            ),
        },
    },
    "ControlLoopProfileEvents": {
        "pack": "engineering",
        "slug": "control_loop_profile",
        "tables": {
            "loop_pairs": _table("pairs", "static", method="step_follow_pairing"),
            "loop_steps": {
                "name": "step_response",
                "shape": "summary",
                "archetype": "aggregate",
                "value_field": "t_settle_seconds",
                "standard_attrs": {"ts_shape:sample_count": None},
            },
            "loop_register": _table(
                "register",
                "aggregate",
                method="percentile_rank_health",
                sample_count="step_count",
            ),
        },
    },
}


def profiler_rule_specs() -> dict[tuple[str, str], dict[str, Any]]:
    """``(class, method) -> spec`` including every ``evaluation_window``."""
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for cls, spec in PROFILER_RULES.items():
        tables = {"evaluation_window": EVALUATION_RULE, **spec["tables"]}
        for method, rule in tables.items():
            out[(cls, method)] = {
                **rule,
                "template": f"{spec['pack']}.{spec['slug']}.{rule['name']}",
                "pack": spec["pack"],
            }
    return out
