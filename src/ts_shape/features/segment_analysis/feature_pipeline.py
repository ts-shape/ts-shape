import logging
import time
import inspect
import pandas as pd  # type: ignore
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

logger = logging.getLogger(__name__)

# Valid sentinel values that can be used in keyword arguments
_VALID_SENTINELS = frozenset({"$prev", "$input"})


class FeaturePipeline:
    """Flexible pipeline builder for chaining any ts-shape class.

    ts-shape has two class patterns.  Choosing the wrong ``add_*`` method is
    the most common mistake — the pipeline validates your choice at
    registration time and tells you which method to use instead.

    **Pattern 1 — Stateless classmethods** (``add_step``)

    The vast majority of ts-shape classes.  The class is a namespace; every
    method is a ``@classmethod`` whose first argument is a DataFrame::

        # No object creation — call directly on the class:
        result = DoubleFilter.filter_nan_value_double(df)

    Classes that follow this pattern:

    - **Filters:** ``DoubleFilter``, ``IntegerFilter``, ``StringFilter``,
      ``BooleanFilter``, ``IsDeltaFilter``, ``DateTimeFilter``, ``CustomFilter``
    - **Calculators:** ``IntegerCalc``
    - **Functions:** ``LambdaProcessor``
    - **Time:** ``TimestampConverter``, ``TimezoneShift``
    - **Segment analysis:** ``SegmentExtractor``, ``SegmentProcessor``,
      ``TimeWindowedFeatureTable``, ``ProfileComparison``
    - **Pattern recognition:** ``PatternRecognition``
    - **Statistics:** ``NumericStatistics``, ``BooleanStatistics``,
      ``StringStatistics``, ``TimestampStatistics``, ``TimeGroupedStatistics``
    - **Context:** ``ValueMapper``

    **Pattern 2 — Stateful instance classes** (``add_instance_step``)

    Classes that must be instantiated with a DataFrame first.  The
    constructor stores configuration (column names, UUIDs, thresholds)
    and methods operate on internal state::

        # Must create an object first:
        harmonizer = DataHarmonizer(df, time_column='systime')
        result = harmonizer.resample_to_uniform(freq='1s')

    Classes that follow this pattern:

    - ``DataHarmonizer`` — harmonize, resample, fill gaps
    - ``CrossSignalAnalytics`` — lead-lag, Granger causality, synchronization
    - ``CycleExtractor`` — extract production cycles
    - ``CycleDataProcessor`` — split/merge data by cycle
    - ``DescriptiveFeatures`` — per-group feature tables
    - ``OEECalculator`` — OEE availability/performance/quality
    - **Events (all 60+ classes):** ``ThresholdMonitoringEvents``,
      ``MachineStateEvents``, ``SteadyStateDetectionEvents``,
      ``OutlierDetectionEvents``, ``StatisticalProcessControlRuleBased``,
      ``DegradationDetectionEvents``, ``EnergyConsumptionEvents``, etc.

    **Pattern 3 — Custom functions** (``add_lambda_step``)

    For one-off transformations that don't map to a ts-shape class::

        pipe.add_lambda_step(
            lambda df: df[df['uuid'].isin(['temperature', 'pressure'])],
            name='select_signals',
        )

    Special references (sentinels)
    ------------------------------
    Any keyword argument value can use these string sentinels:

    - ``'$prev'``  — the output of the *previous* step (available from step 2+).
    - ``'$input'`` — the *original* DataFrame passed to the constructor.

    These let you wire steps that need more than one DataFrame.

    Full example::

        from ts_shape.transform.filter.numeric_filter import DoubleFilter
        from ts_shape.transform.filter.datetime_filter import DateTimeFilter
        from ts_shape.transform.harmonization import DataHarmonizer
        from ts_shape.features.segment_analysis.segment_extractor import SegmentExtractor
        from ts_shape.features.segment_analysis.segment_processor import SegmentProcessor
        from ts_shape.features.segment_analysis.time_windowed_features import TimeWindowedFeatureTable

        result = (
            FeaturePipeline(df)
            .add_step(DateTimeFilter.filter_between_dates,
                      start_date='2024-01-01', end_date='2024-01-31')
            .add_step(DoubleFilter.filter_nan_value_double)
            .add_instance_step(DataHarmonizer,
                               call='resample_to_uniform', freq='1s')
            .add_step(SegmentExtractor.extract_time_ranges,
                      segment_uuid='order_number')
            .add_step(SegmentProcessor.apply_ranges,
                      dataframe='$input', time_ranges='$prev')
            .add_step(TimeWindowedFeatureTable.compute, freq='1min')
            .run()
        )

    Debugging::

        # Preview the pipeline before running:
        print(pipe.describe())

        # Get intermediate DataFrames for each step:
        intermediates = pipe.run_steps()
        intermediates['input']          # original
        intermediates['DoubleFilter.filter_nan_value_double']  # after filtering
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        time_column: str = "systime",
        uuid_column: str = "uuid",
        value_column: str = "value_double",
    ) -> None:
        """Initialize the pipeline with input data.

        Args:
            dataframe: Input DataFrame to process.
            time_column: Name of the timestamp column.  Automatically
                passed to instance-step constructors that accept it.
            uuid_column: Name of the UUID/signal identifier column.
            value_column: Name of the numeric value column.

        Raises:
            TypeError: If ``dataframe`` is not a pandas DataFrame.
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                f"dataframe must be a pandas DataFrame, "
                f"got {type(dataframe).__name__}."
            )
        self._dataframe = dataframe.copy()
        self._time_column = time_column
        self._uuid_column = uuid_column
        self._value_column = value_column
        self._steps: List[Tuple[str, str, Any]] = []

    # ------------------------------------------------------------------
    # Step registration
    # ------------------------------------------------------------------

    def add_step(
        self,
        method: Callable[..., pd.DataFrame],
        **kwargs: Any,
    ) -> "FeaturePipeline":
        """Add a stateless classmethod step (Pattern 1).

        The pipeline passes the current DataFrame as the first positional
        argument automatically.  If you explicitly provide the first
        parameter by name (e.g. ``dataframe='$input'``), the pipeline
        uses your value instead.

        Args:
            method: A classmethod reference,
                e.g. ``DoubleFilter.filter_nan_value_double``.
            **kwargs: Keyword arguments forwarded to the method.
                Use ``'$prev'`` or ``'$input'`` as values to reference
                other DataFrames.

        Returns:
            self, for method chaining.

        Raises:
            TypeError: If ``method`` is an instance method (should use
                :meth:`add_instance_step` instead).
            TypeError: If ``method`` is not callable.
            ValueError: If kwargs contain an invalid sentinel (e.g.
                ``'$PREV'`` instead of ``'$prev'``).

        Example::

            pipe.add_step(DoubleFilter.filter_nan_value_double)
            pipe.add_step(IntegerCalc.scale_column,
                          column_name='value_double', factor=2)

            # Wire two DataFrames into one step:
            pipe.add_step(SegmentProcessor.apply_ranges,
                          dataframe='$input', time_ranges='$prev')
        """
        if not callable(method):
            raise TypeError(f"method must be callable, got {type(method).__name__}.")
        _guard_instance_method(method)
        _validate_sentinels(kwargs)
        name = _method_label(method)
        self._steps.append(("classmethod", name, (method, kwargs)))
        return self

    def add_instance_step(
        self,
        cls: Type,
        call: str,
        init_kwargs: Optional[Dict[str, Any]] = None,
        **method_kwargs: Any,
    ) -> "FeaturePipeline":
        """Add a stateful instance-class step (Pattern 2).

        The pipeline automatically:

        1. Instantiates ``cls`` with the current DataFrame.
        2. Passes ``time_column``, ``uuid_column``, ``value_column`` to the
           constructor if it accepts those parameters.
        3. Calls the method named by ``call``.
        4. If the method returns a DataFrame, it becomes the new pipeline
           state.  Otherwise a warning is logged and the pipeline continues
           with the previous DataFrame.

        Args:
            cls: The class to instantiate, e.g. ``DataHarmonizer``.
            call: Name of the instance method to invoke,
                e.g. ``'resample_to_uniform'``.
            init_kwargs: Extra keyword arguments for the constructor
                (beyond the DataFrame and column names).
            **method_kwargs: Keyword arguments forwarded to the method.
                Use ``'$prev'`` or ``'$input'`` as sentinel values.

        Returns:
            self, for method chaining.

        Raises:
            TypeError: If ``cls`` is not a class.
            AttributeError: If ``cls`` does not have a method named ``call``.
            ValueError: If kwargs contain an invalid sentinel.

        Example::

            pipe.add_instance_step(DataHarmonizer,
                                   call='resample_to_uniform', freq='1s')
            pipe.add_instance_step(CrossSignalAnalytics,
                                   call='lead_lag_matrix', max_lag=10)
            pipe.add_instance_step(CycleExtractor,
                                   call='process_persistent_cycle',
                                   init_kwargs={'start_uuid': 'cycle_start'})
        """
        if not isinstance(cls, type):
            raise TypeError(
                f"cls must be a class, got {type(cls).__name__}. "
                f"If you have a classmethod reference (e.g. DoubleFilter.method), "
                f"use add_step() instead."
            )
        if not hasattr(cls, call):
            available = [
                m
                for m in dir(cls)
                if not m.startswith("_") and callable(getattr(cls, m, None))
            ]
            raise AttributeError(
                f"{cls.__name__} has no method '{call}'. "
                f"Available methods: {', '.join(sorted(available))}"
            )
        _validate_sentinels(method_kwargs)
        if init_kwargs:
            _validate_sentinels(init_kwargs)
        name = f"{cls.__name__}.{call}"
        self._steps.append(
            (
                "instance",
                name,
                (cls, call, init_kwargs or {}, method_kwargs),
            )
        )
        return self

    def add_lambda_step(
        self,
        func: Callable[[pd.DataFrame], pd.DataFrame],
        name: Optional[str] = None,
    ) -> "FeaturePipeline":
        """Add a custom function step (Pattern 3).

        Use this for one-off transformations that don't map to a ts-shape
        class, such as selecting specific UUIDs or adding derived columns.

        Args:
            func: A callable ``(DataFrame) -> DataFrame``.
            name: Optional label for logging and :meth:`describe`.
                Defaults to the function's ``__name__``.

        Returns:
            self, for method chaining.

        Raises:
            TypeError: If ``func`` is not callable.

        Example::

            pipe.add_lambda_step(
                lambda df: df[df['uuid'].isin(['temperature', 'pressure'])],
                name='select_signals',
            )
        """
        if not callable(func):
            raise TypeError(f"func must be callable, got {type(func).__name__}.")
        label = name or getattr(func, "__name__", "lambda")
        self._steps.append(("lambda", label, func))
        return self

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def steps(self) -> List[str]:
        """Return the ordered list of registered step names."""
        return [name for _, name, _ in self._steps]

    def describe(self) -> str:
        """Return a human-readable summary of the pipeline.

        Call this before ``run()`` to verify the pipeline is wired correctly.

        Example output::

            FeaturePipeline (1200 rows, 4 cols)
              1. [step]     DoubleFilter.filter_nan_value_double
              2. [instance] DataHarmonizer.resample_to_uniform  freq='1s'
              3. [step]     SegmentExtractor.extract_time_ranges  segment_uuid='order_number'
              4. [step]     SegmentProcessor.apply_ranges  dataframe='$input', time_ranges='$prev'
              5. [step]     TimeWindowedFeatureTable.compute  freq='1min'
        """
        lines = [
            f"FeaturePipeline ({len(self._dataframe)} rows, "
            f"{len(self._dataframe.columns)} cols)"
        ]
        for i, (step_type, name, payload) in enumerate(self._steps, 1):
            tag = _step_type_tag(step_type)
            params = _format_params(step_type, payload)
            suffix = f"  {params}" if params else ""
            lines.append(f"  {i}. [{tag:<8s}] {name}{suffix}")

        if not self._steps:
            lines.append("  (no steps registered)")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        """Execute all steps sequentially and return the final DataFrame.

        Raises:
            RuntimeError: If any step fails.  The error message includes
                the step number, name, DataFrame shape before the failure,
                available columns, and the original exception.
        """
        return self._execute(capture_intermediates=False)

    def run_steps(self) -> Dict[str, pd.DataFrame]:
        """Execute all steps and return intermediate results.

        Returns a dict keyed by step name.  The key ``'input'`` holds the
        original DataFrame; subsequent keys are the step names in order.
        Useful for debugging which step transforms data unexpectedly.

        Raises:
            RuntimeError: If any step fails (same as :meth:`run`).
        """
        return self._execute(capture_intermediates=True)

    # ------------------------------------------------------------------
    # Internal execution
    # ------------------------------------------------------------------

    def _execute(self, capture_intermediates: bool) -> Any:
        if not self._steps:
            logger.warning("No steps registered. Returning input DataFrame.")
            if capture_intermediates:
                return {"input": self._dataframe.copy()}
            return self._dataframe.copy()

        df = self._dataframe.copy()
        intermediates: Dict[str, pd.DataFrame] = {}
        if capture_intermediates:
            intermediates["input"] = df.copy()

        prev_result: Optional[pd.DataFrame] = None
        total_start = time.time()

        for i, (step_type, name, payload) in enumerate(self._steps):
            step_num = i + 1
            step_start = time.time()
            logger.info(f"Step {step_num}/{len(self._steps)}: {name}")

            try:
                if step_type == "classmethod":
                    df = self._run_classmethod_step(payload, df, prev_result)
                elif step_type == "instance":
                    df = self._run_instance_step(payload, df, prev_result, name)
                elif step_type == "lambda":
                    df = self._run_lambda_step(payload, df, name)
            except RuntimeError:
                raise
            except Exception as e:
                raise RuntimeError(
                    f"Pipeline failed at step {step_num}/{len(self._steps)} "
                    f"'{name}'.\n"
                    f"  DataFrame before step: {df.shape[0]} rows x "
                    f"{df.shape[1]} cols\n"
                    f"  Columns: {list(df.columns)}\n"
                    f"  Error: {type(e).__name__}: {e}"
                ) from e

            prev_result = df
            elapsed = time.time() - step_start
            logger.info(
                f"  -> {name}: {len(df)} rows, "
                f"{len(df.columns)} cols ({elapsed:.3f}s)"
            )

            if capture_intermediates:
                intermediates[name] = df.copy()

        total_elapsed = time.time() - total_start
        logger.info(
            f"Pipeline complete: {len(self._steps)} steps in "
            f"{total_elapsed:.3f}s. Final shape: {df.shape}"
        )

        if capture_intermediates:
            return intermediates
        return df

    def _run_classmethod_step(
        self,
        payload: Tuple[Callable, Dict[str, Any]],
        df: pd.DataFrame,
        prev_result: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        method, kwargs = payload
        resolved = self._resolve_kwargs(kwargs, prev_result)
        first_param = _first_param_name(method)
        if first_param and first_param in resolved:
            result = method(**resolved)
        else:
            result = method(df, **resolved)
        _validate_step_output(result, _method_label(method))
        return result

    def _run_instance_step(
        self,
        payload: Tuple[Type, str, Dict, Dict],
        df: pd.DataFrame,
        prev_result: Optional[pd.DataFrame],
        step_name: str,
    ) -> pd.DataFrame:
        cls, call, init_kwargs, method_kwargs = payload
        resolved = self._resolve_kwargs(method_kwargs, prev_result)
        instance = _build_instance(
            cls,
            df,
            time_column=self._time_column,
            uuid_column=self._uuid_column,
            value_column=self._value_column,
            extra_kwargs=init_kwargs,
        )
        result = getattr(instance, call)(**resolved)
        if isinstance(result, pd.DataFrame):
            return result
        logger.warning(
            f"Instance step '{step_name}' returned "
            f"{type(result).__name__} instead of DataFrame. "
            f"Pipeline continues with previous DataFrame."
        )
        return df

    def _run_lambda_step(
        self,
        func: Callable,
        df: pd.DataFrame,
        step_name: str,
    ) -> pd.DataFrame:
        result = func(df)
        _validate_step_output(result, step_name)
        return result

    def _resolve_kwargs(
        self,
        kwargs: Dict[str, Any],
        prev_result: Optional[pd.DataFrame],
    ) -> Dict[str, Any]:
        """Replace ``'$prev'`` and ``'$input'`` sentinels with DataFrames."""
        resolved = {}
        for key, value in kwargs.items():
            if not isinstance(value, str):
                resolved[key] = value
                continue
            if value == "$prev":
                if prev_result is None:
                    raise ValueError(
                        f"'{key}=$prev' but there is no previous step result. "
                        f"'$prev' can only be used from the second step onward."
                    )
                resolved[key] = prev_result
            elif value == "$input":
                resolved[key] = self._dataframe.copy()
            else:
                resolved[key] = value
        return resolved


# ------------------------------------------------------------------
# Registration-time validation
# ------------------------------------------------------------------


def _validate_sentinels(kwargs: Dict[str, Any]) -> None:
    """Reject invalid sentinel strings (e.g. '$PREV', '$foo') at registration."""
    for key, value in kwargs.items():
        if isinstance(value, str) and value.startswith("$"):
            if value not in _VALID_SENTINELS:
                raise ValueError(
                    f"Unknown sentinel '{value}' for argument '{key}'. "
                    f"Valid sentinels: {', '.join(sorted(_VALID_SENTINELS))}. "
                    f"Sentinels are case-sensitive."
                )


def _guard_instance_method(method: Any) -> None:
    """Detect instance methods passed to add_step and raise with guidance.

    Uses introspection to determine whether *method* is a plain instance
    method (which requires ``add_instance_step``) or a ``@classmethod`` /
    ``@staticmethod`` (which works with ``add_step``).
    """
    if not hasattr(method, "__qualname__"):
        return

    parts = method.__qualname__.split(".")
    if len(parts) < 2:
        return

    owner_name = parts[-2]

    # For bound classmethods, __self__ is the class itself
    if hasattr(method, "__self__") and isinstance(method.__self__, type):
        return  # It's a proper @classmethod — fine for add_step

    # Resolve the owner class and inspect the raw attribute
    owner_cls = _resolve_owner_class(method)
    if owner_cls is not None:
        attr = inspect.getattr_static(owner_cls, method.__name__, None)
        if isinstance(attr, (classmethod, staticmethod)):
            return  # @classmethod or @staticmethod — fine for add_step
        # Plain function on the class means it's an instance method
        if callable(attr):
            raise TypeError(
                f"{owner_name}.{method.__name__} is an instance method. "
                f"Use add_instance_step({owner_name}, "
                f"call='{method.__name__}') instead of add_step()."
            )


def _resolve_owner_class(method: Any) -> Optional[type]:
    """Try to resolve the class that owns a method."""
    qualname = getattr(method, "__qualname__", "")
    module = inspect.getmodule(method)
    if module is None:
        return None
    parts = qualname.split(".")
    if len(parts) < 2:
        return None
    cls_name = parts[-2]
    return getattr(module, cls_name, None)


def _validate_step_output(result: Any, step_name: str) -> None:
    """Ensure a step returned a DataFrame, not a scalar or None."""
    if isinstance(result, pd.DataFrame):
        return
    if result is None:
        raise TypeError(
            f"Step '{step_name}' returned None instead of a DataFrame. "
            f"Make sure the method returns a DataFrame, not modifies in place."
        )
    raise TypeError(
        f"Step '{step_name}' returned {type(result).__name__} instead of "
        f"a DataFrame. If this method computes a scalar/statistic, it "
        f"cannot be used as a pipeline step — pipeline steps must return "
        f"DataFrames."
    )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _first_param_name(method: Any) -> Optional[str]:
    """Return the name of the first non-cls/self parameter of a method."""
    try:
        sig = inspect.signature(method)
    except (ValueError, TypeError):
        return None
    for p in sig.parameters.values():
        if p.name in ("cls", "self"):
            continue
        return p.name
    return None


def _method_label(method: Any) -> str:
    """Build a human-readable label for a method."""
    cls_name = ""
    if hasattr(method, "__self__"):
        cls_name = method.__self__.__name__ + "."
    elif hasattr(method, "__qualname__"):
        parts = method.__qualname__.split(".")
        if len(parts) >= 2:
            cls_name = parts[-2] + "."
    return f"{cls_name}{method.__name__}"


def _step_type_tag(step_type: str) -> str:
    """Short tag for describe() output."""
    return {
        "classmethod": "step",
        "instance": "instance",
        "lambda": "func",
    }[step_type]


def _format_params(step_type: str, payload: Any) -> str:
    """Format the parameters of a step for describe() output."""
    if step_type == "classmethod":
        _, kwargs = payload
        if not kwargs:
            return ""
        parts = [f"{k}={v!r}" for k, v in kwargs.items()]
        return ", ".join(parts)
    elif step_type == "instance":
        _, _, init_kwargs, method_kwargs = payload
        all_kw = {**init_kwargs, **method_kwargs}
        if not all_kw:
            return ""
        parts = [f"{k}={v!r}" for k, v in all_kw.items()]
        return ", ".join(parts)
    return ""


def _build_instance(
    cls: Type,
    dataframe: pd.DataFrame,
    time_column: str,
    uuid_column: str,
    value_column: str,
    extra_kwargs: Dict[str, Any],
) -> Any:
    """Instantiate an instance-based class, passing column names it accepts."""
    sig = inspect.signature(cls.__init__)
    params = set(sig.parameters.keys()) - {"self"}

    init_kwargs: Dict[str, Any] = {}

    # The first positional param (after self) is always the dataframe
    positional = [
        p
        for p in sig.parameters.values()
        if p.name != "self"
        and p.default is inspect.Parameter.empty
        and p.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    if positional:
        init_kwargs[positional[0].name] = dataframe

    # Pass column-name params the constructor accepts
    col_mapping = {
        "time_column": time_column,
        "uuid_column": uuid_column,
        "value_column": value_column,
        "column_name": time_column,
    }
    for param_name, value in col_mapping.items():
        if param_name in params and param_name not in extra_kwargs:
            init_kwargs[param_name] = value

    init_kwargs.update(extra_kwargs)
    return cls(**init_kwargs)
