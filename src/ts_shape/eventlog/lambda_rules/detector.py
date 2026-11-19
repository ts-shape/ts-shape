"""``LambdaDetector`` — the runtime object produced by ``register_lambda_rule``.

A LambdaDetector is the lambda-rule counterpart of the 290 built-in
detector classes under ``ts_shape.events.*``. It does not subclass any
of them — instead it implements the same conceptual contract:

1. ``evaluate(df)`` returns a detector frame (the layout a built-in
   detector returns).
2. ``to_event_log(df, ...)`` runs the evaluator and hands the result to
   the canonical :func:`~ts_shape.eventlog.to_event_log` dispatcher.

Because the spec is registered in :data:`ts_shape.eventlog.taxonomy.REGISTRY`
at registration time, ``to_event_log`` finds it without any new code path.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Mapping

import pandas as pd

from ts_shape.utils.deprecation import renamed_kwargs

from ..model import EventLog
from ..normalizer import to_event_log
from .evaluator import evaluate
from .spec import RuleSpec


@contextlib.contextmanager
def _label_rule(spec: RuleSpec) -> Iterator[None]:
    """Make ``spec``'s label rule known to the taxonomy for one call.

    A registered rule is used as it is; otherwise a :class:`LabelRule` built
    from ``spec`` is added and removed again afterwards.
    """
    from ..archetypes import ARCHETYPE_BY_METHOD
    from ..taxonomy import REGISTRY, LabelRule

    key = (spec.class_name, spec.method_name)
    if key in REGISTRY:
        yield
        return
    REGISTRY[key] = LabelRule(
        template=spec.template,
        pack=spec.pack,
        shape=spec.shape,
        produces_objects=spec.produces_objects,
        severity_field=spec.severity_field,
        value_field=spec.value_field,
        standard_attrs=dict(spec.standard_attrs),
    )
    had_archetype = key in ARCHETYPE_BY_METHOD
    previous = ARCHETYPE_BY_METHOD.get(key)
    ARCHETYPE_BY_METHOD[key] = spec.archetype
    try:
        yield
    finally:
        REGISTRY.pop(key, None)
        if had_archetype:
            ARCHETYPE_BY_METHOD[key] = str(previous)
        else:
            ARCHETYPE_BY_METHOD.pop(key, None)


class LambdaDetector:
    """Runnable form of a :class:`RuleSpec`."""

    def __init__(self, spec: RuleSpec) -> None:
        """Initialize the detector from a rule spec.

        Args:
            spec: Rule spec whose trigger expression the detector evaluates.
        """
        self.spec = spec
        self.detector_name = f"{spec.class_name}.{spec.method_name}"

    @renamed_kwargs(df="dataframe")
    def evaluate(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Apply the rule to ``dataframe`` and return a detector frame.

        Args:
            dataframe: Input DataFrame the rule's trigger expression is evaluated on.

        Returns:
            Detector frame whose layout depends on the rule's shape
            (see :func:`~ts_shape.eventlog.lambda_rules.evaluator.evaluate`).
        """
        return evaluate(self.spec, dataframe)

    @renamed_kwargs(df="dataframe")
    def to_event_log(
        self,
        dataframe: pd.DataFrame,
        *,
        objects: Mapping[str, object] | None = None,
        qualifiers: Mapping[str, str] | None = None,
    ) -> EventLog:
        """Evaluate the rule on ``dataframe`` and normalize the result to an event log.

        Args:
            dataframe: Input DataFrame the rule is evaluated on.
            objects: Optional OCEL object bindings, mapping an object type to a
                column name, callable, Series or scalar; passed through to
                :func:`~ts_shape.eventlog.to_event_log`. None uses only the
                auto-extracted objects.
            qualifiers: Optional mapping of object type to the relation
                qualifier written for that type. None leaves qualifiers empty.

        A rule that is not registered (built directly from a :class:`RuleSpec`
        rather than with :func:`register_lambda_rule`) is labelled from its
        own spec for this call; the global taxonomy is left as it is.

        Returns:
            The canonical :class:`~ts_shape.eventlog.model.EventLog`.
        """
        frame = self.evaluate(dataframe)
        with _label_rule(self.spec):
            return to_event_log(
                frame,
                detector=self.detector_name,
                objects=objects,
                qualifiers=qualifiers,
            )

    def __repr__(self) -> str:
        return f"LambdaDetector({self.detector_name!r}, shape={self.spec.shape!r})"
