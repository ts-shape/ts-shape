"""Declarative pipeline orchestrator for ts-shape.

A :class:`Pipeline` chains transforms and event detectors into a single,
reusable processing definition. It is **linear single-channel**:

* a **transform** step takes the working DataFrame and returns a new one that
  *replaces* it (the signal flows on);
* a **detect** step runs against the current working DataFrame, stores its
  event output under a name, and leaves the working DataFrame *unchanged*
  (detectors branch off).

Whether a step is a transform or a detector is **declared by the caller** via
``.transform()`` vs ``.detect()`` -- it is never inferred.

Example::

    from ts_shape import Pipeline
    from ts_shape.transform.calculator.numeric_calc import IntegerCalc
    from ts_shape.events.quality.outlier_detection import OutlierDetectionEvents

    pipe = (
        Pipeline(name="sensor-quality")
        .transform(IntegerCalc, "scale_column",
                   column_name="value_double", factor=0.1)
        .detect(OutlierDetectionEvents, "detect_outliers_zscore",
                name="outliers", value_column="value_double", threshold=3.0)
    )
    result = pipe.run(dataframe)
    result.data                 # final transformed DataFrame
    result.events["outliers"]   # detector output
    result.to_event_log()       # normalized, combined EventLog

A transform step must preserve the long ``systime`` / ``uuid`` / ``value_*``
schema that downstream detectors expect. Reshaping operations (e.g.
``DataHarmonizer.pivot_to_wide``) are terminal and do not belong mid-chain.

See also
--------
``ts_shape.features.segment_analysis.feature_pipeline.FeaturePipeline`` is a
related but distinct tool. ``FeaturePipeline`` chains transforms and feature
classes for **feature extraction**, binds its DataFrame at construction, and
returns a single DataFrame. Use ``Pipeline`` (this module) when the goal is to
run **detectors**: it adds branch-off named detector results, reuse of one
definition across many DataFrames (``run(df)``), and OCEL event-log export.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd  # type: ignore

logger = logging.getLogger(__name__)

_StepFunc = Callable[[pd.DataFrame], pd.DataFrame]


def _param_names(func: Any) -> set[str]:
    """Return the keyword-accepting parameter names of ``func``.

    Excludes ``self`` / ``cls`` and ``*args`` / ``**kwargs`` markers.
    """
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return set()
    names: set[str] = set()
    for param in sig.parameters.values():
        if param.name in ("self", "cls"):
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        names.add(param.name)
    return names


def _resolve(
    target: Any, method: Optional[str], kwargs: Dict[str, Any]
) -> Tuple[_StepFunc, Optional[str]]:
    """Build a ``df -> df`` callable for one step.

    Returns ``(func, detector_id)`` where ``detector_id`` is the
    ``"ClassName.method"`` string used by :func:`ts_shape.eventlog.to_event_log`
    (``None`` for plain-callable steps).

    Two forms:

    * **callable** -- ``method is None`` and ``target`` is callable: used as-is.
    * **class + method** -- ``target`` is a class; a ``classmethod`` /
      ``staticmethod`` is called directly on the class, a plain (instance)
      method instantiates ``target(df, **init_kwargs)`` first, with ``kwargs``
      routed between the constructor and the method by parameter name.
    """
    if method is None:
        if not callable(target):
            raise TypeError(
                f"step target must be callable when no method name is given, "
                f"got {type(target).__name__}"
            )
        return target, None

    if not isinstance(target, type):
        raise TypeError(
            f"when a method name is given, target must be a class, "
            f"got {type(target).__name__}"
        )

    try:
        static_attr = inspect.getattr_static(target, method)
    except AttributeError:
        raise AttributeError(f"{target.__name__!r} has no method {method!r}") from None

    # Stateless transform: classmethod / staticmethod -- call on the class.
    if isinstance(static_attr, (classmethod, staticmethod)):
        bound = getattr(target, method)

        def _call_static(df: pd.DataFrame) -> pd.DataFrame:
            return bound(df, **kwargs)

        return _call_static, None

    # Stateful class: instantiate, then call the instance method.
    init_params = _param_names(target.__init__)
    method_params = _param_names(getattr(target, method))
    init_kwargs: Dict[str, Any] = {}
    method_kwargs: Dict[str, Any] = {}
    unknown: List[str] = []
    for key, value in kwargs.items():
        matched = False
        if key in init_params:
            init_kwargs[key] = value
            matched = True
        if key in method_params:
            method_kwargs[key] = value
            matched = True
        if not matched:
            unknown.append(key)
    if unknown:
        raise ValueError(
            f"unknown argument(s) {unknown} for {target.__name__}.{method} -- "
            f"accepted: constructor {sorted(init_params)}, "
            f"method {sorted(method_params)}"
        )

    def _call_instance(df: pd.DataFrame) -> pd.DataFrame:
        instance = target(df, **init_kwargs)
        return getattr(instance, method)(**method_kwargs)

    return _call_instance, f"{target.__name__}.{method}"


def _default_name(target: Any, method: Optional[str]) -> str:
    """Pick a readable step name when the caller did not supply one."""
    if method is not None:
        return method
    return getattr(target, "__name__", "step")


@dataclass(frozen=True)
class _Step:
    kind: str  # "transform" | "detect"
    name: str
    func: _StepFunc
    detector_id: Optional[str]


@dataclass(frozen=True)
class PipelineResult:
    """Outcome of :meth:`Pipeline.run`.

    Attributes:
        name: The pipeline's name.
        data: The working DataFrame after the final transform step.
        events: Detector outputs keyed by step name.
    """

    name: str
    data: pd.DataFrame
    events: Dict[str, pd.DataFrame]
    _detector_ids: Dict[str, Optional[str]] = field(default_factory=dict)

    def to_event_log(self, *, concat: bool = True) -> Any:
        """Normalize detector outputs into an OCEL event log.

        Each detect step added via the *class + method* form is run through
        :func:`ts_shape.eventlog.to_event_log` (the pipeline already knows the
        ``"ClassName.method"`` identifier). Plain-callable detect steps have no
        detector identity and are skipped.

        Args:
            concat: When True (default) merge all logs into one ``EventLog``;
                when False return a ``dict`` keyed by step name.

        Raises:
            ValueError: If no detector step has a known identity to normalize.
        """
        from ts_shape.eventlog import concat as _concat
        from ts_shape.eventlog import to_event_log as _to_event_log

        logs: Dict[str, Any] = {}
        for step_name, events_df in self.events.items():
            detector_id = self._detector_ids.get(step_name)
            if detector_id is None:
                logger.info(
                    "Skipping event-log normalization for step %r: "
                    "no detector identity (plain-callable step).",
                    step_name,
                )
                continue
            logs[step_name] = _to_event_log(events_df, detector=detector_id)

        if not logs:
            raise ValueError(
                "no detector step with a known identity to normalize -- "
                "add detect steps via the (class, method) form"
            )
        if not concat:
            return logs
        return _concat(*logs.values())

    def __repr__(self) -> str:
        return (
            f"PipelineResult(name={self.name!r}, "
            f"data={len(self.data)} rows, "
            f"events={sorted(self.events)})"
        )


class Pipeline:
    """A reusable, declarative chain of transform and detector steps.

    Build with the fluent :meth:`transform` and :meth:`detect` methods, then
    call :meth:`run` -- on as many DataFrames as you like.
    """

    def __init__(self, name: str = "pipeline") -> None:
        """Initialize an empty pipeline.

        Args:
            name: A label for the pipeline, surfaced in ``repr`` and results.
        """
        self.name = name
        self._steps: List[_Step] = []

    def transform(
        self,
        target: Any,
        method: Optional[str] = None,
        *,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> "Pipeline":
        """Add a transform step -- its output replaces the working DataFrame.

        Args:
            target: A callable ``df -> df``, or a transform class.
            method: Method name to call on ``target`` (omit for the callable
                form).
            name: Optional step label (defaults to the method/callable name).
            **kwargs: Forwarded to the transform (routed between constructor
                and method for stateful classes).

        Returns:
            ``self``, for chaining.
        """
        func, _ = _resolve(target, method, kwargs)
        self._steps.append(
            _Step("transform", name or _default_name(target, method), func, None)
        )
        return self

    def detect(
        self,
        target: Any,
        method: Optional[str] = None,
        *,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> "Pipeline":
        """Add a detector step -- its output is stored, the signal is unchanged.

        Args:
            target: A callable ``df -> events_df``, or a detector class.
            method: Method name to call on ``target`` (omit for the callable
                form).
            name: Optional result key (defaults to the method/callable name).
            **kwargs: Forwarded to the detector (routed between constructor and
                method).

        Returns:
            ``self``, for chaining.

        Raises:
            ValueError: If the resolved step name collides with an existing
                detector step.
        """
        func, detector_id = _resolve(target, method, kwargs)
        step_name = name or _default_name(target, method)
        existing = {s.name for s in self._steps if s.kind == "detect"}
        if step_name in existing:
            raise ValueError(
                f"duplicate detector step name {step_name!r}; " f"pass a unique name="
            )
        self._steps.append(_Step("detect", step_name, func, detector_id))
        return self

    @property
    def steps(self) -> List[Tuple[str, str]]:
        """The ordered ``(kind, name)`` pairs of the configured steps."""
        return [(s.kind, s.name) for s in self._steps]

    def run(self, dataframe: pd.DataFrame) -> PipelineResult:
        """Execute every step against ``dataframe``.

        Args:
            dataframe: The input timeseries DataFrame.

        Returns:
            A :class:`PipelineResult` with the final signal and detector
            outputs.

        Raises:
            TypeError: If ``dataframe`` is not a DataFrame, or a step returns a
                non-DataFrame.
            RuntimeError: If a step raises; the message names the step.
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                f"Pipeline.run expects a pandas DataFrame, "
                f"got {type(dataframe).__name__}"
            )

        data = dataframe
        events: Dict[str, pd.DataFrame] = {}
        detector_ids: Dict[str, Optional[str]] = {}

        for index, step in enumerate(self._steps):
            try:
                result = step.func(data)
            except Exception as exc:  # noqa: BLE001 -- re-raised with context
                raise RuntimeError(
                    f"pipeline {self.name!r}: step {index} "
                    f"({step.kind} {step.name!r}) failed: {exc}"
                ) from exc
            if not isinstance(result, pd.DataFrame):
                raise TypeError(
                    f"pipeline {self.name!r}: step {index} "
                    f"({step.kind} {step.name!r}) returned "
                    f"{type(result).__name__}, expected a DataFrame"
                )
            if step.kind == "transform":
                data = result
            else:
                events[step.name] = result
                detector_ids[step.name] = step.detector_id

        return PipelineResult(
            name=self.name,
            data=data,
            events=events,
            _detector_ids=detector_ids,
        )

    def __repr__(self) -> str:
        if not self._steps:
            return f"Pipeline(name={self.name!r}, steps=[])"
        lines = [f"Pipeline(name={self.name!r}, {len(self._steps)} steps):"]
        for index, step in enumerate(self._steps):
            lines.append(f"  {index}. [{step.kind}] {step.name}")
        return "\n".join(lines)
