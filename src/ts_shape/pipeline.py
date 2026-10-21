"""Declarative pipeline orchestrator for ts-shape.

A :class:`Pipeline` is the single way to chain ts-shape processing steps --
transforms and event detectors -- into one reusable definition. It is
**linear single-channel**:

* an optional **source** step (always first, at most one) calls a ts-shape
  loader and *produces* the pipeline's first DataFrame;
* a **transform** step takes the working DataFrame and returns a new one that
  *replaces* it (the signal flows on);
* a **detect** step runs against the current working DataFrame, stores its
  event output under a name, and leaves the working DataFrame *unchanged*
  (detectors branch off).

Whether a step is a source, transform or detector is **declared by the
caller** via ``.source()`` / ``.transform()`` / ``.detect()`` -- it is never
inferred.

A pipeline with a source step is **source-bound**: call :meth:`run` with no
argument and the source produces the data. A pipeline without one is
**DataFrame-driven**: pass the input DataFrame to :meth:`run`, as before.

A step's target may be:

* a plain callable ``df -> df``;
* a ``(class, "method")`` pair -- a ``classmethod`` / ``staticmethod`` is
  called on the class, an instance method instantiates the class first.
  Keyword arguments are routed between the constructor and the method by
  parameter name automatically; a name both accept goes to **both**. Write
  ``init__<name>=`` to send it to the constructor only and
  ``method__<name>=`` to send it to the method only (see
  :meth:`Pipeline.detect`).

Keyword-argument values may use two **sentinels**, resolved at run time:

* ``"$input"`` -- the DataFrame originally passed to :meth:`Pipeline.run`;
* ``"$prev"``  -- the current working DataFrame.

These wire steps that need a second DataFrame (e.g.
``SegmentProcessor.apply_ranges``). If a kwarg names the step's own
DataFrame parameter, that value is used instead of the auto-injected one.
Only string values are compared with the sentinels; write ``"$$..."`` for a
literal string starting with ``$`` (``"$$input"`` passes ``"$input"``).

Each transform and detect step works on its own copy of the working
DataFrame, and every ``$input`` use is a fresh copy, so a step that changes
its input in place never changes the caller's DataFrame or a later step's
input. Step names
are unique: they key :meth:`Pipeline.run_steps` and the detector results,
and ``"input"`` is reserved for the input frame.

Example::

    from ts_shape import Pipeline
    from ts_shape.transform.calculator.numeric_calc import IntegerCalc
    from ts_shape.events.quality.outlier_detection import OutlierDetectionEvents

    pipe = (
        Pipeline(name="sensor-quality")
        .transform(IntegerCalc, "scale_column",
                   value_column="value_double", factor=0.1)
        .detect(OutlierDetectionEvents, "detect_outliers_zscore",
                name="outliers", value_column="value_double", n_sigma=3.0)
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
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ts_shape.utils.base import _copy_on_write

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
    # Deprecated keyword names (``renamed_kwargs``) are accepted too; the
    # call forwards them with a DeprecationWarning.
    obj = getattr(func, "__init__") if isinstance(func, type) else func
    seen: set[int] = set()
    while obj is not None and id(obj) not in seen:
        seen.add(id(obj))
        names.update(getattr(obj, "__deprecated_kwargs__", {}))
        obj = getattr(obj, "__wrapped__", None)
    return names


def _first_param_name(func: Any) -> str | None:
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


def _private_copy(dataframe: pd.DataFrame) -> pd.DataFrame:
    """A copy that a step may change in place without affecting ``dataframe``.

    Under copy-on-write (always on in pandas 3) a shallow copy suffices: the
    first write to a column copies that column. Without it (pandas 2
    default) the frame is copied deeply.
    """
    return dataframe.copy(deep=not _copy_on_write())


def _subst(value: Any, working_df: pd.DataFrame, input_df: pd.DataFrame) -> Any:
    """Resolve a ``$input`` / ``$prev`` sentinel to its DataFrame.

    Only strings are compared, so arrays, Series and DataFrames pass through
    unchanged. ``"$$..."`` is the escaped literal ``"$..."``.
    """
    if not isinstance(value, str):
        return value
    if value == "$input":
        # A fresh copy per use: a step that changes its ``$input`` argument
        # in place cannot reach later steps.
        return _private_copy(input_df)
    if value == "$prev":
        return working_df
    if value.startswith("$$"):
        return value[1:]
    return value


def _validate_sentinels(kwargs: dict[str, Any]) -> None:
    """Reject unknown ``$``-prefixed sentinel strings at registration time."""
    for key, value in kwargs.items():
        if (
            isinstance(value, str)
            and value.startswith("$")
            and not value.startswith("$$")
            and value not in _SENTINELS
        ):
            raise ValueError(
                f"unknown sentinel {value!r} for argument {key!r}; "
                f"valid sentinels are $input and $prev (write $$ for a "
                f"literal $, e.g. '$${value[1:]}')"
            )


_INIT_PREFIX = "init__"
_METHOD_PREFIX = "method__"


def _accepts_var_keyword(func: Any) -> bool:
    """Whether ``func`` takes ``**kwargs``."""
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return False
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())


def _plain_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """``kwargs`` with the ``init__`` / ``method__`` prefixes removed.

    A prefixed name wins over the same unprefixed name. Used where only the
    effective parameter values matter (e.g. :mod:`ts_shape.stream`'s plan).
    """
    out: dict[str, Any] = {}
    prefixed: dict[str, Any] = {}
    for key, value in kwargs.items():
        for prefix in (_INIT_PREFIX, _METHOD_PREFIX):
            if key.startswith(prefix):
                prefixed[key[len(prefix) :]] = value
                break
        else:
            out[key] = value
    out.update(prefixed)
    return out


def _split_kwargs(
    target: type, method: str, kwargs: dict[str, Any]
) -> tuple[dict[str, str], dict[str, str]]:
    """Route kwargs between ``target.__init__`` and ``target.method``.

    Returns ``(init_map, method_map)``, each mapping a parameter name to the
    kwarg key whose value it receives. An unprefixed kwarg goes to **both**
    when both accept the name. ``init__<name>=`` goes to the constructor
    only and ``method__<name>=`` to the method only; a prefixed kwarg takes
    precedence over the unprefixed one on its side. Raises ``ValueError`` if
    a kwarg matches neither side (or a prefixed one does not match its side).
    """
    init_params = _param_names(target)
    method_func = getattr(target, method)
    method_params = _param_names(method_func)
    init_map: dict[str, str] = {}
    method_map: dict[str, str] = {}
    unknown: list[str] = []
    plain: list[str] = []
    for key in kwargs:
        if key.startswith(_INIT_PREFIX):
            name = key[len(_INIT_PREFIX) :]
            if name not in init_params:
                raise ValueError(
                    f"{key!r}: {target.__name__}() has no parameter {name!r} -- "
                    f"constructor accepts {sorted(init_params)}"
                )
            init_map[name] = key
        elif key.startswith(_METHOD_PREFIX):
            name = key[len(_METHOD_PREFIX) :]
            if name not in method_params:
                raise ValueError(
                    f"{key!r}: {target.__name__}.{method} has no parameter "
                    f"{name!r} -- method accepts {sorted(method_params)}"
                )
            method_map[name] = key
        else:
            plain.append(key)
    for key in plain:
        in_init = key in init_params
        in_method = key in method_params
        if not in_init and not in_method:
            unknown.append(key)
            continue
        routed = False
        if in_init and key not in init_map:
            init_map[key] = key
            routed = True
        if in_method and key not in method_map:
            method_map[key] = key
            routed = True
        if not routed:
            raise ValueError(
                f"argument {key!r} for {target.__name__}.{method} is overridden "
                f"on every side it applies to by init__{key} / method__{key}; "
                f"drop it or pass only the prefixed forms"
            )
    if unknown:
        raise ValueError(
            f"unknown argument(s) {unknown} for {target.__name__}.{method} -- "
            f"accepted: constructor {sorted(init_params)}, "
            f"method {sorted(method_params)}"
        )
    return init_map, method_map


def _check_static_kwargs(
    target: type, method: str, func: Any, kwargs: dict[str, Any]
) -> dict[str, str]:
    """Map parameter names of a class/static method to kwarg keys.

    ``method__<name>=`` is accepted (there is no constructor, so
    ``init__<name>=`` is not). Raises ``ValueError`` for a name the bound
    signature does not accept.
    """
    params = _param_names(func)
    accepts_any = _accepts_var_keyword(func)
    mapping: dict[str, str] = {}
    unknown: list[str] = []
    for key in kwargs:
        if key.startswith(_INIT_PREFIX):
            raise ValueError(
                f"{key!r}: {target.__name__}.{method} is a class/static method "
                f"and has no constructor arguments"
            )
        name = key[len(_METHOD_PREFIX) :] if key.startswith(_METHOD_PREFIX) else key
        if not (name in params or accepts_any):
            unknown.append(key)
            continue
        if key.startswith(_METHOD_PREFIX) or name not in mapping:
            mapping[name] = key
    if unknown:
        raise ValueError(
            f"unknown argument(s) {unknown} for {target.__name__}.{method} -- "
            f"accepted: {sorted(params)}"
        )
    return mapping


def _resolve(
    target: Any, method: str | None, kwargs: dict[str, Any]
) -> tuple[_Executor, str | None]:
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

        call_df_param = _first_param_name(target)

        def _call(working_df: pd.DataFrame, input_df: pd.DataFrame) -> pd.DataFrame:
            resolved = {k: _subst(v, working_df, input_df) for k, v in kwargs.items()}
            if call_df_param is not None and call_df_param in resolved:
                return target(**resolved)
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
        static_map = _check_static_kwargs(target, method, bound, kwargs)

        def _call_static(
            working_df: pd.DataFrame, input_df: pd.DataFrame
        ) -> pd.DataFrame:
            resolved = {
                name: _subst(kwargs[key], working_df, input_df)
                for name, key in static_map.items()
            }
            if df_param is not None and df_param in resolved:
                return bound(**resolved)
            return bound(working_df, **resolved)

        return _call_static, f"{target.__name__}.{method}"

    # -- instance method: instantiate, then call ---------------------------
    df_param = _first_param_name(target)
    init_map, method_map = _split_kwargs(target, method, kwargs)

    def _call_instance(
        working_df: pd.DataFrame, input_df: pd.DataFrame
    ) -> pd.DataFrame:
        # Each side resolves its own sentinels, so a kwarg shared by the
        # constructor and the method gets two separate ``$input`` copies.
        init_kwargs = {
            name: _subst(kwargs[key], working_df, input_df)
            for name, key in init_map.items()
        }
        method_kwargs = {
            name: _subst(kwargs[key], working_df, input_df)
            for name, key in method_map.items()
        }
        if df_param is not None and df_param in init_kwargs:
            instance = target(**init_kwargs)
        else:
            instance = target(working_df, **init_kwargs)
        return getattr(instance, method)(**method_kwargs)

    return _call_instance, f"{target.__name__}.{method}"


def _resolve_source(
    target: Any, method: str | None, kwargs: dict[str, Any]
) -> Callable[[], pd.DataFrame]:
    """Build a ``() -> DataFrame`` executor for a source (loader) step.

    Unlike :func:`_resolve`, no working/input DataFrame is injected -- a source
    *produces* the pipeline's first frame from its kwargs alone. The ``$input``
    / ``$prev`` sentinels are rejected: there is no prior data to reference.
    """
    for key, value in kwargs.items():
        if isinstance(value, str) and value in _SENTINELS:
            raise ValueError(
                f"a source step cannot use the {value!r} sentinel for "
                f"argument {key!r}; it produces the pipeline's first frame"
            )
    _validate_sentinels(kwargs)

    # -- callable form: call directly --------------------------------------
    if method is None:
        if not callable(target):
            raise TypeError(
                f"source target must be callable when no method name is "
                f"given, got {type(target).__name__}"
            )

        def _load_callable() -> pd.DataFrame:
            return target(**_literal(kwargs))

        return _load_callable

    if not isinstance(target, type):
        raise TypeError(
            f"when a method name is given, source target must be a class, "
            f"got {type(target).__name__}"
        )
    try:
        static_attr = inspect.getattr_static(target, method)
    except AttributeError:
        raise AttributeError(f"{target.__name__!r} has no method {method!r}") from None

    # -- classmethod / staticmethod: call on the class ---------------------
    if isinstance(static_attr, (classmethod, staticmethod)):
        bound = getattr(target, method)

        def _load_static() -> pd.DataFrame:
            return bound(**_literal(kwargs))

        return _load_static

    # -- instance method: instantiate from config, then call ---------------
    init_map, method_map = _split_kwargs(target, method, kwargs)

    def _load_instance() -> pd.DataFrame:
        resolved = _literal(kwargs)
        init_kwargs = {name: resolved[key] for name, key in init_map.items()}
        method_kwargs = {name: resolved[key] for name, key in method_map.items()}
        instance = target(**init_kwargs)
        return getattr(instance, method)(**method_kwargs)

    return _load_instance


def _literal(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Unescape ``"$$..."`` literals (sources have no sentinels)."""
    return {
        k: v[1:] if isinstance(v, str) and v.startswith("$$") else v
        for k, v in kwargs.items()
    }


def _default_name(target: Any, method: str | None) -> str:
    """Pick a readable step name when the caller did not supply one."""
    if method is not None:
        return method
    return getattr(target, "__name__", "step")


def _format_kwargs(kwargs: dict[str, Any]) -> str:
    """Render kwargs for :meth:`Pipeline.describe`."""
    return ", ".join(f"{k}={v!r}" for k, v in kwargs.items())


@dataclass(frozen=True)
class _Step:
    kind: str  # "source" | "transform" | "detect"
    name: str
    # source executors take no args; transform/detect take (working, input).
    executor: Callable[..., pd.DataFrame]
    detector_id: str | None
    kwargs: dict[str, Any]
    # The step's target and method name, for Pipeline.run_stream.
    target: Any = None
    method: str | None = None


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
    events: dict[str, pd.DataFrame]
    _detector_ids: dict[str, str | None] = field(default_factory=dict)

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

        logs: dict[str, Any] = {}
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

    Example:
        >>> import ts_shape
        >>> from ts_shape import OutlierDetectionEvents, Pipeline
        >>> pipe = Pipeline(name="sensor-quality").detect(
        ...     OutlierDetectionEvents, "detect_outliers_zscore",
        ...     name="outliers", value_column="value_double")
        >>> print(pipe.describe())
        Pipeline 'sensor-quality' (1 steps):
          0. [detect   ] outliers  value_column='value_double'
        >>> result = pipe.run(ts_shape.make_timeseries(["temp"], n_outliers=3))
        >>> result
        PipelineResult(name='sensor-quality', data=1000 rows, events=['outliers'])
        >>> len(result.events["outliers"])
        4
    """

    def __init__(self, name: str = "pipeline") -> None:
        """Initialize an empty pipeline.

        Args:
            name: A label for the pipeline, surfaced in ``repr`` and results.
        """
        self.name = name
        self._steps: list[_Step] = []

    def source(
        self,
        target: Any,
        method: str | None = None,
        *,
        name: str | None = None,
        **kwargs: Any,
    ) -> Pipeline:
        """Add a source step -- a loader that produces the pipeline's first frame.

        A source step must be the **first** step, and a pipeline may have at
        most one. With a source step, :meth:`run` is called with no DataFrame;
        without one, :meth:`run` requires a DataFrame as before.

        Args:
            target: A callable returning a DataFrame, or a loader class.
            method: Method name to call on ``target`` (omit for the callable
                form). A ``classmethod`` / ``staticmethod`` is called on the
                class; an instance method instantiates the class first, with
                kwargs routed between constructor and method by name.
            name: Optional step label (defaults to the method/callable name).
            **kwargs: Forwarded to the loader. The ``$input`` / ``$prev``
                sentinels are not allowed -- a source has no prior data.

        Returns:
            ``self``, for chaining.

        Raises:
            ValueError: If the pipeline already has steps -- a source must be
                the first step, and only one is allowed.
        """
        if self._steps:
            raise ValueError(
                f"a source step must be the first step; pipeline {self.name!r} "
                f"already has {len(self._steps)} step(s)"
            )
        executor = _resolve_source(target, method, kwargs)
        self._steps.append(
            _Step(
                "source",
                name or _default_name(target, method),
                executor,
                None,
                kwargs,
            )
        )
        return self

    def transform(
        self,
        target: Any,
        method: str | None = None,
        *,
        name: str | None = None,
        **kwargs: Any,
    ) -> Pipeline:
        """Add a transform step -- its output replaces the working DataFrame.

        Args:
            target: A callable ``df -> df``, or a transform class.
            method: Method name to call on ``target`` (omit for the callable
                form).
            name: Optional step label (defaults to the method/callable name,
                with a ``_2``, ``_3`` suffix when that name is taken).
            **kwargs: Forwarded to the transform (routed between constructor
                and method for stateful classes). Values may be the ``$input``
                / ``$prev`` sentinels.

        Returns:
            ``self``, for chaining.

        Raises:
            ValueError: If an explicit ``name`` is taken or is ``"input"``.
        """
        executor, _ = _resolve(target, method, kwargs)
        self._steps.append(
            _Step(
                "transform",
                self._step_name(name, target, method, kind="transform"),
                executor,
                None,
                kwargs,
                target,
                method,
            )
        )
        return self

    def _step_name(
        self, name: str | None, target: Any, method: str | None, *, kind: str
    ) -> str:
        """A unique step name; a default transform name gets a suffix."""
        taken = {s.name for s in self._steps}
        if name is None and kind == "transform":
            base = _default_name(target, method)
            candidate, n = base, 1
            while candidate in taken or candidate == "input":
                n += 1
                candidate = f"{base}_{n}"
            return candidate
        step_name = name or _default_name(target, method)
        if step_name == "input":
            raise ValueError(
                "the step name 'input' is reserved for the input DataFrame "
                "in run_steps(); pass another name="
            )
        if step_name in taken:
            label = "detector step" if kind == "detect" else "step"
            raise ValueError(
                f"duplicate {label} name {step_name!r}; pass a unique name="
            )
        return step_name

    def detect(
        self,
        target: Any,
        method: str | None = None,
        *,
        name: str | None = None,
        **kwargs: Any,
    ) -> Pipeline:
        """Add a detector step -- its output is stored, the signal is unchanged.

        Args:
            target: A callable ``df -> events_df``, or a detector class.
            method: Method name to call on ``target`` (omit for the callable
                form).
            name: Optional result key (defaults to the method/callable name).
            **kwargs: Forwarded to the detector (routed between constructor and
                method by parameter name). Values may be the ``$input`` /
                ``$prev`` sentinels. A name that **both** the constructor
                and the method accept is passed to both. That is a trap
                when the two mean different things: in
                ``LongDowntimeEvents.count_events_between_gaps`` the
                constructor's ``value_column`` is the run-state column and
                the method's ``value_column`` is the production counter's,
                and in ``ContinuousProcessAlignmentEvents.align_to_reference``
                the constructor's ``value_column`` is the line-speed column
                while the method's is the station value carried through.
                Prefix the name to
                pick one side: ``init__<name>=`` goes to the constructor
                only, ``method__<name>=`` to the method only (a prefixed
                kwarg wins over the unprefixed one on its side). A
                class/static method takes ``method__<name>=`` too.

        Returns:
            ``self``, for chaining.

        Raises:
            ValueError: If the resolved step name collides with an existing
                step or is ``"input"``, or a kwarg (prefixed or not) matches
                no parameter of the constructor / method it is routed to.

        Example:
            Count production between long stops: the run state is a bool
            column, the counter an integer one::

                Pipeline("between-stops").detect(
                    LongDowntimeEvents, "count_events_between_gaps",
                    name="windows", run_state_uuid="m:run",
                    production_df="$input", production_uuid="m:count",
                    aggregation="sum",
                    method__value_column="value_integer",
                )
        """
        executor, detector_id = _resolve(target, method, kwargs)
        step_name = self._step_name(name, target, method, kind="detect")
        self._steps.append(
            _Step("detect", step_name, executor, detector_id, kwargs, target, method)
        )
        return self

    @property
    def steps(self) -> list[tuple[str, str]]:
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

    def _execute(self, dataframe: pd.DataFrame | None, *, capture: bool) -> tuple[
        pd.DataFrame,
        dict[str, pd.DataFrame],
        dict[str, str | None],
        dict[str, pd.DataFrame],
    ]:
        events: dict[str, pd.DataFrame] = {}
        detector_ids: dict[str, str | None] = {}
        intermediates: dict[str, pd.DataFrame] = {}
        has_source = bool(self._steps) and self._steps[0].kind == "source"

        if has_source:
            if dataframe is not None:
                raise TypeError(
                    f"pipeline {self.name!r} defines a source step; "
                    f"call run() without a DataFrame"
                )
            source_step = self._steps[0]
            try:
                dataframe = source_step.executor()
            except Exception as exc:
                raise RuntimeError(
                    f"pipeline {self.name!r}: step 0 "
                    f"(source {source_step.name!r}) failed: {exc}"
                ) from exc
            if not isinstance(dataframe, pd.DataFrame):
                raise TypeError(
                    f"pipeline {self.name!r}: step 0 "
                    f"(source {source_step.name!r}) returned "
                    f"{type(dataframe).__name__}, expected a DataFrame"
                )
            remaining = self._steps[1:]
            if capture:
                intermediates[source_step.name] = dataframe
        else:
            if dataframe is None:
                raise TypeError(
                    f"pipeline {self.name!r} has no source step; "
                    f"run() requires a DataFrame"
                )
            if not isinstance(dataframe, pd.DataFrame):
                raise TypeError(
                    f"Pipeline.run expects a pandas DataFrame, "
                    f"got {type(dataframe).__name__}"
                )
            remaining = self._steps
            if capture:
                intermediates["input"] = dataframe

        # The caller's frame is never handed to a step: every step gets its
        # own copy of the working frame and every ``$input`` use is a fresh
        # copy, so a step that changes its input in place cannot reach the
        # caller or any later step.
        input_df = dataframe.copy()
        working = input_df

        for index, step in enumerate(remaining, start=1 if has_source else 0):
            try:
                result = step.executor(_private_copy(working), input_df)
            except Exception as exc:
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

    def run(self, dataframe: pd.DataFrame | None = None) -> PipelineResult:
        """Execute every step.

        Args:
            dataframe: The input timeseries DataFrame. Omit it when the
                pipeline has a ``.source()`` step (the source produces it);
                supply it otherwise.

        Returns:
            A :class:`PipelineResult` with the final signal and detector
            outputs.

        Raises:
            TypeError: If ``dataframe`` is passed to a source-bound pipeline,
                omitted from a sourceless one, is not a DataFrame, or a step
                returns a non-DataFrame.
            RuntimeError: If a step raises; the message names the step.
        """
        working, events, detector_ids, _ = self._execute(dataframe, capture=False)
        return PipelineResult(
            name=self.name,
            data=working,
            events=events,
            _detector_ids=detector_ids,
        )

    def run_steps(
        self, dataframe: pd.DataFrame | None = None
    ) -> dict[str, pd.DataFrame]:
        """Execute every step and return *all* intermediate DataFrames.

        Useful for debugging which step changes the data unexpectedly.

        Args:
            dataframe: The input timeseries DataFrame. Omit it when the
                pipeline has a ``.source()`` step; supply it otherwise.

        Returns:
            A dict keyed by step name. For a DataFrame-driven pipeline, key
            ``"input"`` holds the original DataFrame; for a source-bound
            pipeline the loaded frame is keyed by the source step's name.
            Transform steps store the transformed signal; detect steps store
            their event output.
        """
        _, _, _, intermediates = self._execute(dataframe, capture=True)
        return intermediates

    def run_stream(
        self,
        source: Any,
        *,
        chunk: Any = "1D",
        overlap: Any = None,
        time_column: str = "systime",
        max_carry: Any = None,
        keep_data: bool = False,
    ) -> PipelineResult:
        """Execute the pipeline slice by slice on data larger than memory.

        Every detect step runs through :func:`ts_shape.run_chunked`'s engine
        on each slice as transformed by the transform steps **before it** in
        the pipeline (as :meth:`run` does), so each step's events equal those
        of :meth:`run` on the concatenated data. Transforms run on each slice
        separately: they must be **row-local** (each output row depends only
        on its input row, as unit conversions, scaling and value mappings
        do). The ``$input`` / ``$prev`` sentinels refer to the current
        slice.

        Args:
            source: Frames or ``(name, frame)`` pairs in time order, or a
                :func:`ts_shape.time_range_source`.
            chunk: Slice length (default ``"1D"``), or a calendar period
                (``"W"``, ``"MS"``) as in :func:`ts_shape.run_chunked`.
            overlap: History before each slice; default per detector.
            time_column: Time column of the source frames.
            max_carry: Longest an event may stay open across slices
                (default 7 days).
            keep_data: When True, ``result.data`` is the transformed data of
                every slice concatenated (it then holds all rows in memory);
                by default it is an empty frame with the transformed columns.

        Returns:
            A :class:`PipelineResult`.

        Raises:
            NotChunkableError: If a detect step's method cannot run chunked
                (see :data:`ts_shape.stream.SUPPORT` and
                :data:`ts_shape.stream.CONDITIONAL`), or is a plain callable.
            TypeError: If the pipeline has a ``.source()`` step.
        """
        from ts_shape.stream import _run_pipeline

        return _run_pipeline(
            self,
            source,
            chunk=chunk,
            overlap=overlap,
            time_column=time_column,
            max_carry=max_carry,
            keep_data=keep_data,
        )

    def __repr__(self) -> str:
        return self.describe()
