"""Declarative pipeline orchestrator for ts-shape.

A :class:`Pipeline` is the single way to chain ts-shape processing steps --
transforms and event detectors -- into one reusable definition. It is
**linear single-channel**:

* a **transform** step takes the working DataFrame and returns a new one that
  *replaces* it (the signal flows on);
* a **detect** step runs against the current working DataFrame, stores its
  event output under a name, and leaves the working DataFrame *unchanged*
  (detectors branch off).

Whether a step is a transform or a detector is **declared by the caller** via
``.transform()`` vs ``.detect()`` -- it is never inferred.

A step's target may be:

* a plain callable ``df -> df``;
* a ``(class, "method")`` pair -- a ``classmethod`` / ``staticmethod`` is
  called on the class, an instance method instantiates the class first.
  Keyword arguments are routed between the constructor and the method by
  parameter name automatically.

Keyword-argument values may use two **sentinels**, resolved at run time:

* ``"$input"`` -- the DataFrame originally passed to :meth:`Pipeline.run`;
* ``"$prev"``  -- the current working DataFrame.

These wire steps that need a second DataFrame (e.g.
``SegmentProcessor.apply_ranges``). If a kwarg names the step's own
DataFrame parameter, that value is used instead of the auto-injected one.

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

Debugging::

    print(pipe.describe())          # preview steps without running
    steps = pipe.run_steps(df)      # dict of every intermediate DataFrame

A transform step must preserve the long ``systime`` / ``uuid`` / ``value_*``
schema that downstream detectors expect. Reshaping operations (e.g.
``DataHarmonizer.pivot_to_wide``) are terminal and belong at the end.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd  # type: ignore

logger = logging.getLogger(__name__)

_Executor = Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]
_SENTINELS = frozenset({"$input", "$prev"})


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


def _first_param_name(func: Any) -> Optional[str]:
    """Return the first non-``self`` / ``cls`` parameter name of ``func``."""
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return None
    for param in sig.parameters.values():
        if param.name in ("self", "cls"):
            continue
        return param.name
    return None


def _subst(value: Any, working_df: pd.DataFrame, input_df: pd.DataFrame) -> Any:
    """Resolve a ``$input`` / ``$prev`` sentinel to its DataFrame."""
    if value == "$input":
        return input_df
    if value == "$prev":
        return working_df
    return value


def _validate_sentinels(kwargs: Dict[str, Any]) -> None:
    """Reject unknown ``$``-prefixed sentinel strings at registration time."""
    for key, value in kwargs.items():
        if isinstance(value, str) and value.startswith("$") and value not in _SENTINELS:
            raise ValueError(
                f"unknown sentinel {value!r} for argument {key!r}; "
                f"valid sentinels are $input and $prev"
            )


def _resolve(
    target: Any, method: Optional[str], kwargs: Dict[str, Any]
) -> Tuple[_Executor, Optional[str]]:
    """Build a ``(working_df, input_df) -> df`` executor for one step.

    Returns ``(executor, detector_id)``; ``detector_id`` is the
    ``"ClassName.method"`` string used by :func:`ts_shape.eventlog.to_event_log`
    (``None`` for plain-callable steps). Signature inspection happens once,
    here; only sentinel values are resolved per run.
    """
    _validate_sentinels(kwargs)

    # -- callable form -----------------------------------------------------
    if method is None:
        if not callable(target):
            raise TypeError(
                f"step target must be callable when no method name is given, "
                f"got {type(target).__name__}"
            )

        def _call(working_df: pd.DataFrame, input_df: pd.DataFrame) -> pd.DataFrame:
            resolved = {k: _subst(v, working_df, input_df) for k, v in kwargs.items()}
            return target(working_df, **resolved)

        return _call, None

    if not isinstance(target, type):
        raise TypeError(
            f"when a method name is given, target must be a class, "
            f"got {type(target).__name__}"
        )
    try:
        static_attr = inspect.getattr_static(target, method)
    except AttributeError:
        raise AttributeError(f"{target.__name__!r} has no method {method!r}") from None

    # -- classmethod / staticmethod: call on the class ---------------------
    if isinstance(static_attr, (classmethod, staticmethod)):
        bound = getattr(target, method)
        df_param = _first_param_name(bound)

        def _call_static(
            working_df: pd.DataFrame, input_df: pd.DataFrame
        ) -> pd.DataFrame:
            resolved = {k: _subst(v, working_df, input_df) for k, v in kwargs.items()}
            if df_param is not None and df_param in resolved:
                return bound(**resolved)
            return bound(working_df, **resolved)

        return _call_static, None

    # -- instance method: instantiate, then call ---------------------------
    init_params = _param_names(target.__init__)
    method_params = _param_names(getattr(target, method))
    df_param = _first_param_name(target.__init__)
    init_keys: set[str] = set()
    method_keys: set[str] = set()
    unknown: List[str] = []
    for key in kwargs:
        in_init = key in init_params
        in_method = key in method_params
        if in_init:
            init_keys.add(key)
        if in_method:
            method_keys.add(key)
        if not in_init and not in_method:
            unknown.append(key)
    if unknown:
        raise ValueError(
            f"unknown argument(s) {unknown} for {target.__name__}.{method} -- "
            f"accepted: constructor {sorted(init_params)}, "
            f"method {sorted(method_params)}"
        )

    def _call_instance(
        working_df: pd.DataFrame, input_df: pd.DataFrame
    ) -> pd.DataFrame:
        resolved = {k: _subst(v, working_df, input_df) for k, v in kwargs.items()}
        init_kwargs = {k: resolved[k] for k in init_keys}
        method_kwargs = {k: resolved[k] for k in method_keys}
        if df_param is not None and df_param in init_kwargs:
            instance = target(**init_kwargs)
        else:
            instance = target(working_df, **init_kwargs)
        return getattr(instance, method)(**method_kwargs)

    return _call_instance, f"{target.__name__}.{method}"


def _default_name(target: Any, method: Optional[str]) -> str:
    """Pick a readable step name when the caller did not supply one."""
    if method is not None:
        return method
    return getattr(target, "__name__", "step")


def _format_kwargs(kwargs: Dict[str, Any]) -> str:
    """Render kwargs for :meth:`Pipeline.describe`."""
    return ", ".join(f"{k}={v!r}" for k, v in kwargs.items())


@dataclass(frozen=True)
class _Step:
    kind: str  # "transform" | "detect"
    name: str
    executor: _Executor
    detector_id: Optional[str]
    kwargs: Dict[str, Any]


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

        Each detect step added via the *(class, method)* form is run through
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
                and method for stateful classes). Values may be the ``$input``
                / ``$prev`` sentinels.

        Returns:
            ``self``, for chaining.
        """
        executor, _ = _resolve(target, method, kwargs)
        self._steps.append(
            _Step(
                "transform",
                name or _default_name(target, method),
                executor,
                None,
                kwargs,
            )
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
                method). Values may be the ``$input`` / ``$prev`` sentinels.

        Returns:
            ``self``, for chaining.

        Raises:
            ValueError: If the resolved step name collides with an existing
                detector step.
        """
        executor, detector_id = _resolve(target, method, kwargs)
        step_name = name or _default_name(target, method)
        existing = {s.name for s in self._steps if s.kind == "detect"}
        if step_name in existing:
            raise ValueError(
                f"duplicate detector step name {step_name!r}; pass a unique name="
            )
        self._steps.append(_Step("detect", step_name, executor, detector_id, kwargs))
        return self

    @property
    def steps(self) -> List[Tuple[str, str]]:
        """The ordered ``(kind, name)`` pairs of the configured steps."""
        return [(s.kind, s.name) for s in self._steps]

    def describe(self) -> str:
        """Return a human-readable summary of the pipeline without running it."""
        if not self._steps:
            return f"Pipeline {self.name!r} (no steps)"
        lines = [f"Pipeline {self.name!r} ({len(self._steps)} steps):"]
        for index, step in enumerate(self._steps):
            params = _format_kwargs(step.kwargs)
            suffix = f"  {params}" if params else ""
            lines.append(f"  {index}. [{step.kind:<9s}] {step.name}{suffix}")
        return "\n".join(lines)

    def _execute(self, dataframe: pd.DataFrame, *, capture: bool) -> Tuple[
        pd.DataFrame,
        Dict[str, pd.DataFrame],
        Dict[str, Optional[str]],
        Dict[str, pd.DataFrame],
    ]:
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                f"Pipeline.run expects a pandas DataFrame, "
                f"got {type(dataframe).__name__}"
            )
        input_df = dataframe
        working = dataframe
        events: Dict[str, pd.DataFrame] = {}
        detector_ids: Dict[str, Optional[str]] = {}
        intermediates: Dict[str, pd.DataFrame] = {}
        if capture:
            intermediates["input"] = dataframe

        for index, step in enumerate(self._steps):
            try:
                result = step.executor(working, input_df)
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
                working = result
            else:
                events[step.name] = result
                detector_ids[step.name] = step.detector_id
            if capture:
                intermediates[step.name] = result

        return working, events, detector_ids, intermediates

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
        working, events, detector_ids, _ = self._execute(dataframe, capture=False)
        return PipelineResult(
            name=self.name,
            data=working,
            events=events,
            _detector_ids=detector_ids,
        )

    def run_steps(self, dataframe: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Execute every step and return *all* intermediate DataFrames.

        Useful for debugging which step changes the data unexpectedly.

        Args:
            dataframe: The input timeseries DataFrame.

        Returns:
            A dict keyed by step name; key ``"input"`` holds the original
            DataFrame. Transform steps store the transformed signal; detect
            steps store their event output.
        """
        _, _, _, intermediates = self._execute(dataframe, capture=True)
        return intermediates

    def __repr__(self) -> str:
        return self.describe()
