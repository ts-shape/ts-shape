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

from collections.abc import Mapping

import pandas as pd

from ..model import EventLog
from ..normalizer import to_event_log
from .evaluator import evaluate
from .spec import RuleSpec


class LambdaDetector:
    """Runnable form of a :class:`RuleSpec`."""

    def __init__(self, spec: RuleSpec) -> None:
        self.spec = spec
        self.detector_name = f"{spec.class_name}.{spec.method_name}"

    def evaluate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the rule to ``df`` and return a detector frame.

        Args:
            df: Input DataFrame the rule's trigger expression is evaluated on.

        Returns:
            Detector frame whose layout depends on the rule's shape
            (see :func:`~ts_shape.eventlog.lambda_rules.evaluator.evaluate`).
        """
        return evaluate(self.spec, df)

    def to_event_log(
        self,
        df: pd.DataFrame,
        *,
        objects: Mapping[str, object] | None = None,
        qualifiers: Mapping[str, str] | None = None,
    ) -> EventLog:
        """Evaluate the rule on ``df`` and normalize the result to an event log.

        Args:
            df: Input DataFrame the rule is evaluated on.
            objects: Optional OCEL object bindings, mapping an object type to a
                column name, callable, Series or scalar; passed through to
                :func:`~ts_shape.eventlog.to_event_log`. None uses only the
                auto-extracted objects.
            qualifiers: Optional mapping of object type to the relation
                qualifier written for that type. None leaves qualifiers empty.

        Returns:
            The canonical :class:`~ts_shape.eventlog.model.EventLog`.
        """
        frame = self.evaluate(df)
        return to_event_log(
            frame,
            detector=self.detector_name,
            objects=objects,
            qualifiers=qualifiers,
        )

    def __repr__(self) -> str:
        return f"LambdaDetector({self.detector_name!r}, shape={self.spec.shape!r})"
