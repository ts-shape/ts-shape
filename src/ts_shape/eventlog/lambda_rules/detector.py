"""``LambdaDetector`` — the runtime object produced by ``register_lambda_rule``.

A LambdaDetector is the lambda-rule counterpart of the 290 built-in
detector classes under ``ts_shape.events.*``. It does not subclass any
of them — instead it implements the same conceptual contract:

1. ``evaluate(df)`` returns a legacy-shaped DataFrame.
2. ``to_event_log(df, ...)`` runs the evaluator and hands the result to
   the canonical :func:`~ts_shape.eventlog.to_event_log` dispatcher.

Because the spec is registered in :data:`ts_shape.eventlog.taxonomy.REGISTRY`
at registration time, ``to_event_log`` finds it without any new code path.
"""

from __future__ import annotations

from typing import Mapping

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
        """Apply the rule to ``df`` and return a legacy-shaped DataFrame."""
        return evaluate(self.spec, df)

    def to_event_log(
        self,
        df: pd.DataFrame,
        *,
        objects: Mapping[str, object] | None = None,
        qualifiers: Mapping[str, str] | None = None,
    ) -> EventLog:
        legacy = self.evaluate(df)
        return to_event_log(
            legacy,
            detector=self.detector_name,
            objects=objects,
            qualifiers=qualifiers,
        )

    def __repr__(self) -> str:
        return f"LambdaDetector({self.detector_name!r}, shape={self.spec.shape!r})"
