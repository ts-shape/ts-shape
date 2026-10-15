"""Base class and architecture guidelines for ts-shape.

ts-shape uses two complementary patterns for data processing.  Choose
the right one based on the complexity of the operation:

**Stateless classmethods** — use when:

- The operation is a single, self-contained transformation (filter,
  calculate, convert, aggregate).
- There is no shared configuration between methods.
- The result depends only on the inputs (pure function).
- Examples: ``IntegerCalc``, ``StringFilter``, ``NumericStatistics``,
  ``DateTimeFilter``, ``LambdaProcessor``.

**Stateful instance classes** — use when:

- Multiple related methods share configuration (column names, signal
  UUIDs, thresholds, tolerances).
- The constructor validates or preprocesses data once for many
  operations (e.g. sorting, filtering to a specific UUID).
- The domain workflow has a natural setup phase before analysis
  (e.g. compute control limits, then apply SPC rules).
- Examples: ``DataHarmonizer``, ``OutlierDetectionEvents``,
  ``MachineStateEvents``, ``OEEEvents``, all event classes.

**Pipeline** — use when chaining multiple steps:

- Chains transforms (``.transform()``) and detectors (``.detect()``) into one
  reusable definition; works with both class patterns above and plain
  callables.
- Supports ``$prev`` / ``$input`` sentinels for multi-DataFrame wiring and
  ``run_steps()`` for intermediate-result capture.
- See ``ts_shape.pipeline``.
"""

import logging
import warnings
from typing import ClassVar

import pandas as pd

from ts_shape.errors import ColumnNotFoundError, DataQualityWarning
from ts_shape.utils.deprecation import caller_stacklevel, renamed_kwargs
from ts_shape.utils.normalize import normalize_signal_frame
from ts_shape.utils.signal_checks import checked, checked_init

logger = logging.getLogger(__name__)


class Base:
    #: Severity label overrides for ``severity_level``, keyed by method name
    #: (``"*"`` for all methods); ``None`` when ``severity`` is a score.
    _severity_levels: ClassVar[dict[str, dict[str, str]] | None] = {}

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Give every detector in ``ts_shape.events`` the same input checks.

        Public methods and the constructor are wrapped by
        :mod:`ts_shape.utils.signal_checks`: results with a categorical
        ``severity`` get ``severity_level`` (info / warn / critical), an
        unknown signal uuid warns
        (:class:`~ts_shape.errors.DataQualityWarning`) and a missing uuid
        column raises :class:`~ts_shape.errors.ColumnNotFoundError`.
        """
        super().__init_subclass__(**kwargs)
        if not cls.__module__.startswith("ts_shape.events."):
            return
        for name, member in list(vars(cls).items()):
            if name == "__init__" and callable(member):
                setattr(cls, name, checked_init(member))
            elif (
                not name.startswith("_")
                and callable(member)
                and not isinstance(member, (staticmethod, classmethod, type))
                and type(member).__name__ != "_Forward"
                and not getattr(member, "__ts_shape_checked__", False)
            ):
                setattr(cls, name, checked(member))

    @staticmethod
    def _validate_column(dataframe: pd.DataFrame, column_name: str) -> None:
        """Validate that a column exists in the DataFrame.

        Raises:
            ColumnNotFoundError: If column_name is not in dataframe.columns.
                This subclasses ``ValueError``.
        """
        if column_name not in dataframe.columns:
            raise ColumnNotFoundError(
                f"Column '{column_name}' not found. "
                f"Available columns: {list(dataframe.columns)}"
            )

    @staticmethod
    def _validate_uuid(
        dataframe: pd.DataFrame, uuid: str, uuid_column: str = "uuid"
    ) -> None:
        """Validate that ``uuid`` is present in ``uuid_column``.

        A no-op when the DataFrame is empty or carries no uuid column, so each
        detector keeps full control over empty-input handling.

        Raises:
            ValueError: If ``uuid_column`` exists, the frame is non-empty, and
                ``uuid`` is absent. The message lists the available UUIDs.
        """
        if dataframe.empty or uuid_column not in dataframe.columns:
            return
        if uuid not in dataframe[uuid_column].values:
            available = sorted(str(u) for u in dataframe[uuid_column].dropna().unique())
            raise ValueError(
                f"UUID '{uuid}' not found in column '{uuid_column}'. "
                f"Available UUIDs: {available}"
            )

    @renamed_kwargs(column_name="time_column")
    def __init__(self, dataframe: pd.DataFrame, time_column: str = "systime") -> None:
        """Keep a sorted copy of ``dataframe`` with a datetime time column.

        Args:
            dataframe: Long-format input (``systime``, ``uuid``, ``value_*``).
            time_column: Timestamp column to convert and sort by. When it is
                missing, datetime or text columns whose name contains "time"
                or "date" are read as timestamps instead (with a
                :class:`~ts_shape.errors.DataQualityWarning`); numeric
                columns are never converted.

        Raises:
            TypeError: If ``dataframe`` is not a pandas DataFrame.
            ColumnNotFoundError: For a detector (``ts_shape.events``), if a
                non-empty ``dataframe`` has no ``time_column``.
        """
        time_col = time_column
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                f"Expected pandas DataFrame, got {type(dataframe).__name__}"
            )
        self.dataframe = dataframe.copy()
        if self.dataframe.empty:
            return
        if (
            type(self).__module__.startswith("ts_shape.events.")
            and time_col not in self.dataframe.columns
        ):
            raise ColumnNotFoundError(
                f"{type(self).__name__}: time column {time_col!r} not found. "
                f"Columns: {list(self.dataframe.columns)}; pass time_column=..."
            )

        if time_col in self.dataframe.columns:
            self.dataframe[time_col] = pd.to_datetime(
                self.dataframe[time_col], errors="coerce"
            )
        else:
            # No time column: a column named like a time ("start_time",
            # "date") is used instead. Only datetime or text columns qualify;
            # a number such as ``cycle_time_mean`` is never read as a time.
            time_columns = [
                col
                for col in self.dataframe.columns
                if isinstance(col, str)
                and ("time" in col.lower() or "date" in col.lower())
                and (
                    pd.api.types.is_datetime64_any_dtype(self.dataframe[col])
                    or pd.api.types.is_object_dtype(self.dataframe[col])
                    or pd.api.types.is_string_dtype(self.dataframe[col])
                )
            ]
            converted = []
            for col in time_columns:
                if not pd.api.types.is_datetime64_any_dtype(self.dataframe[col]):
                    self.dataframe[col] = pd.to_datetime(
                        self.dataframe[col], errors="coerce", format="mixed"
                    )
                    converted.append(col)
            if converted:
                warnings.warn(
                    f"{type(self).__name__}: time column {time_column!r} not "
                    f"found; read {converted} as timestamps instead. Pass "
                    "time_column=... to choose the column.",
                    DataQualityWarning,
                    stacklevel=caller_stacklevel(),
                )
            if time_columns:
                time_col = time_columns[0]

        # Every class computes on NumPy-backed dtypes: nullable, pyarrow and
        # text-number columns are converted once here (utils.normalize).
        normalize_signal_frame(self.dataframe, time_column=time_col)

        # Sort by time; rows with the same timestamp keep a fixed order (by
        # uuid), so a result never depends on the order rows arrived in.
        if time_col in self.dataframe.columns:
            keys = [time_col]
            if "uuid" in self.dataframe.columns:
                keys.append("uuid")
            self.dataframe = self.dataframe.sort_values(by=keys, kind="stable")

        # Warn on missing/unparseable timestamps. They are kept (sorted last),
        # but every duration or interval touching them comes out NaT/NaN.
        if time_col in self.dataframe.columns:
            nat_count = int(self.dataframe[time_col].isna().sum())
            if 0 < nat_count < len(self.dataframe):
                warnings.warn(
                    f"Column '{time_col}' contains {nat_count} missing or "
                    "unparseable timestamp(s) (NaT). Consider dropping them "
                    "before processing.",
                    DataQualityWarning,
                    stacklevel=caller_stacklevel(),
                )

        # Warn on duplicate timestamps (per UUID when available)
        if time_col in self.dataframe.columns:
            subset = (
                [time_col, "uuid"] if "uuid" in self.dataframe.columns else [time_col]
            )
            dup_count = self.dataframe.duplicated(subset=subset).sum()
            if dup_count:
                warnings.warn(
                    f"DataFrame contains {dup_count} duplicate timestamp(s) "
                    f"(checked on {subset}). Consider deduplicating before processing.",
                    DataQualityWarning,
                    stacklevel=caller_stacklevel(),
                )

        # Warn on columns that are entirely NaN. In long format each signal
        # fills one value_* column, so unused value columns are normal; they
        # are reported only when no value column holds any data.
        all_nan_cols = [
            col for col in self.dataframe.columns if self.dataframe[col].isna().all()
        ]
        value_cols = [c for c in self.dataframe.columns if str(c).startswith("value_")]
        if any(c not in all_nan_cols for c in value_cols):
            all_nan_cols = [c for c in all_nan_cols if c not in value_cols]
        if all_nan_cols:
            warnings.warn(
                f"Column(s) {all_nan_cols} contain only NaN values.",
                DataQualityWarning,
                stacklevel=caller_stacklevel(),
            )

    def get_dataframe(self) -> pd.DataFrame:
        """Returns the processed DataFrame."""
        return self.dataframe

    def __repr__(self) -> str:
        """Concise, interactive-friendly summary of the instance.

        Shows the row count plus any configured ``value_column`` / ``*_uuid``
        attributes, so ``print(detector)`` in a REPL or notebook is useful.
        """
        df = getattr(self, "dataframe", None)
        n_rows = len(df) if isinstance(df, pd.DataFrame) else 0
        parts = [f"rows={n_rows}"]
        for key, val in self.__dict__.items():
            if isinstance(val, str) and (
                key.endswith("_uuid") or key == "value_column"
            ):
                parts.append(f"{key}={val!r}")
        return f"{type(self).__name__}({', '.join(parts)})"
