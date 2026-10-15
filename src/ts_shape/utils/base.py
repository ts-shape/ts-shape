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

import numpy as np
import pandas as pd

from ts_shape.errors import ColumnNotFoundError, DataQualityWarning
from ts_shape.utils.deprecation import caller_stacklevel, renamed_kwargs
from ts_shape.utils.frames import require_columns
from ts_shape.utils.normalize import normalize_signal_frame, unique_text
from ts_shape.utils.signal_checks import checked, checked_init, remember_uuids

logger = logging.getLogger(__name__)


def _copy_on_write() -> bool:
    """Whether pandas copies a shared column on its first write."""
    if int(pd.__version__.split(".")[0]) >= 3:
        return True
    return pd.get_option("mode.copy_on_write") is True


def _is_numpy_datetime(series: pd.Series) -> bool:
    """A ``datetime64`` column (naive or tz-aware) that needs no parsing."""
    dtype = series.dtype
    if isinstance(dtype, pd.DatetimeTZDtype):
        return True
    return isinstance(dtype, np.dtype) and dtype.kind == "M"


def _all_missing(series: pd.Series) -> bool:
    dtype = series.dtype
    if isinstance(dtype, np.dtype) and dtype.kind in "biu":  # no NaN possible
        return False
    return bool(series.isna().all())


def _ticks(series: pd.Series) -> np.ndarray | None:
    """The time column as int64 ticks (NaT is the smallest), or ``None``."""
    if not _is_numpy_datetime(series):
        return None
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        return np.asarray(series.array.asi8, dtype=np.int64)
    return np.asarray(series.to_numpy().view("i8"), dtype=np.int64)


def _missing_last(codes: np.ndarray) -> np.ndarray:
    """Factorize codes (``-1``: missing) with missing as the largest, in place."""
    codes[codes < 0] = np.iinfo(np.int64).max
    return codes


class _SortOrder:
    """Sort keys of a signal frame: ``(time, uuid)``, missing values last.

    Computes, on integer codes, what ``sort_values(by=keys, kind="stable")``,
    ``isna().sum()`` and ``duplicated(subset=keys).sum()`` compute on the
    columns: whether the frame is already in order, the stable sort, the NaT
    count and the duplicate count. Columns whose order cannot be read from
    codes (an ``object`` uuid column with non-text values, a time column
    that is not ``datetime64``) take the pandas calls themselves.
    """

    def __init__(self, frame: pd.DataFrame, time_col: str) -> None:
        self.frame = frame
        self.keys = [time_col] + (["uuid"] if "uuid" in frame.columns else [])
        self.ticks: np.ndarray | None = None
        self.nat: np.ndarray | None = None
        self.codes: np.ndarray | None = None
        #: The uuids, as text (``None`` when the frame has no uuid column).
        self.uuids: set[str] | None = None
        self.exact = False
        if not frame.columns.is_unique:
            return
        self.ticks = _ticks(frame[time_col])
        if self.ticks is not None:
            self.nat = np.asarray(frame[time_col].isna(), dtype=bool)
        if "uuid" in frame.columns:
            self.codes, self.uuids = self._uuid_codes(frame["uuid"])
        self.exact = self.ticks is not None and (
            "uuid" not in frame.columns or self.codes is not None
        )

    @staticmethod
    def _uuid_codes(series: pd.Series) -> tuple[np.ndarray | None, set[str]]:
        """Codes in sort order (missing: the largest) and the uuid texts."""
        if isinstance(series.dtype, pd.CategoricalDtype):
            # A categorical sorts by its categories' order.
            codes = np.array(series.cat.codes, dtype=np.int64)
            return _missing_last(codes), unique_text(series)
        try:
            codes, uniques = pd.factorize(series, sort=True)
        except TypeError:  # values that cannot be ordered
            return None, unique_text(series)
        values = list(uniques)
        if series.dtype == object and not all(isinstance(v, str) for v in values):
            return None, unique_text(series)
        texts = {str(v) for v in values}
        return _missing_last(np.asarray(codes, dtype=np.int64)), texts

    def is_sorted(self) -> bool:
        if not self.exact:
            return False
        ticks, nat = self.ticks, self.nat
        assert ticks is not None and nat is not None
        if len(ticks) < 2:
            return True
        if (nat[:-1] & ~nat[1:]).any():
            return False  # a NaT before a timestamp
        valid = len(ticks) - int(nat.sum())
        if (ticks[1:valid] < ticks[: max(valid - 1, 0)]).any():
            return False
        if self.codes is None:
            return True
        ties = ticks[1:] == ticks[:-1]
        return not (ties & (self.codes[1:] < self.codes[:-1])).any()

    def sort(self) -> pd.DataFrame:
        """The sorted frame (it replaces the unsorted one here, too)."""
        if not self.exact:
            self.frame = self.frame.sort_values(by=self.keys, kind="stable")
            return self.frame
        assert self.ticks is not None and self.nat is not None
        if self.codes is None:
            order = np.lexsort((self.ticks, self.nat))
        else:
            order = np.lexsort((self.codes, self.ticks, self.nat))
            self.codes = self.codes[order]
        self.ticks = self.ticks[order]
        self.nat = self.nat[order]
        frame = self.frame
        self.frame = frame.take(order).__finalize__(frame, method="sort_values")
        return self.frame

    def nat_count(self) -> int:
        if self.nat is None:
            return int(self.frame[self.keys[0]].isna().sum())
        return int(self.nat.sum())

    def duplicate_count(self, frame: pd.DataFrame) -> int:
        """Rows repeating an earlier row's keys (the frame is sorted)."""
        if not self.exact:
            return int(frame.duplicated(subset=self.keys).sum())
        assert self.ticks is not None
        same = self.ticks[1:] == self.ticks[:-1]
        if self.codes is not None:
            same &= self.codes[1:] == self.codes[:-1]
        return int(same.sum())


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
    def _validate_column(
        dataframe: pd.DataFrame, column_name: str, *, where: str | None = None
    ) -> None:
        """Validate that a column exists in the DataFrame.

        Delegates to :func:`ts_shape.utils.frames.require_columns`.

        Args:
            dataframe: Input frame.
            column_name: Required column.
            where: Name of the calling function, used as the message prefix;
                without it the message reads ``Column 'x' not found. ...``.

        Raises:
            ColumnNotFoundError: If column_name is not in dataframe.columns.
                This subclasses ``ValueError``.
        """
        require_columns(dataframe, [column_name], where=where)

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
        # Under copy-on-write (always on in pandas 3) a shallow copy is a
        # private copy: the first write to a column copies that column.
        # Without it, the frame is copied once below unless sorting already
        # built new arrays.
        cow = _copy_on_write()
        self.dataframe = dataframe.copy(deep=not cow or dataframe.empty)
        owned = cow or dataframe.empty
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
            if not _is_numpy_datetime(self.dataframe[time_col]):
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

        has_time = time_col in self.dataframe.columns
        order = _SortOrder(self.dataframe, time_col) if has_time else None

        # Sort by time; rows with the same timestamp keep a fixed order (by
        # uuid), so a result never depends on the order rows arrived in.
        if order is not None and not order.is_sorted():
            self.dataframe = order.sort()
            owned = True
        if not owned:
            self.dataframe = self.dataframe.copy()
        if order is not None and order.uuids is not None:
            # Read back by the signal checks instead of scanning the column.
            remember_uuids(self, self.dataframe, "uuid", order.uuids)

        # Warn on missing/unparseable timestamps. They are kept (sorted last),
        # but every duration or interval touching them comes out NaT/NaN.
        if order is not None:
            nat_count = order.nat_count()
            if 0 < nat_count < len(self.dataframe):
                warnings.warn(
                    f"Column '{time_col}' contains {nat_count} missing or "
                    "unparseable timestamp(s) (NaT). Consider dropping them "
                    "before processing.",
                    DataQualityWarning,
                    stacklevel=caller_stacklevel(),
                )

        # Warn on duplicate timestamps (per UUID when available)
        if order is not None:
            subset = order.keys
            dup_count = order.duplicate_count(self.dataframe)
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
            col for col in self.dataframe.columns if _all_missing(self.dataframe[col])
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
