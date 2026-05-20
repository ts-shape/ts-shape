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
  ``MachineStateEvents``, ``OEECalculator``, all event classes.

**FeaturePipeline** — use when chaining multiple steps:

- Combines both patterns via ``add_step()`` (classmethods) and
  ``add_instance_step()`` (instance classes).
- Supports ``$prev`` / ``$input`` sentinels for multi-DataFrame wiring.
- Provides logging, timing, and intermediate result capture.
- See ``ts_shape.features.segment_analysis.feature_pipeline``.
"""

import logging
import warnings
import pandas as pd  # type: ignore
from ts_shape.errors import ColumnNotFoundError, DataQualityWarning

logger = logging.getLogger(__name__)


class Base:
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

    def __init__(self, dataframe: pd.DataFrame, column_name: str = "systime") -> None:
        """
        Initializes the Base with a DataFrame, detects time columns, converts them to datetime,
        and sorts the DataFrame by the specified column (or the detected time column if applicable).

        Args:
            dataframe (pd.DataFrame): The DataFrame to be processed.
            column_name (str): The column to sort by. Default is 'systime'. If the column is not found or is not a time column, the class will attempt to detect other time columns.

        Raises:
            TypeError: If dataframe is not a pandas DataFrame.
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                f"Expected pandas DataFrame, got {type(dataframe).__name__}"
            )
        self.dataframe = dataframe.copy()
        if self.dataframe.empty:
            return

        # Attempt to convert the specified column_name to datetime if it exists
        if column_name in self.dataframe.columns:
            self.dataframe[column_name] = pd.to_datetime(
                self.dataframe[column_name], errors="coerce"
            )
        else:
            # If the column_name is not in the DataFrame, fallback to automatic time detection
            time_columns = [
                col
                for col in self.dataframe.columns
                if "time" in col.lower() or "date" in col.lower()
            ]

            # Convert all detected time columns to datetime, if any
            for col in time_columns:
                self.dataframe[col] = pd.to_datetime(
                    self.dataframe[col], errors="coerce"
                )

            # If any time columns are detected, sort by the first one; otherwise, do nothing
            if time_columns:
                column_name = time_columns[0]

        # Sort by the datetime column (either specified or detected)
        if column_name in self.dataframe.columns:
            self.dataframe = self.dataframe.sort_values(by=column_name)

        # Warn on duplicate timestamps (per UUID when available)
        if column_name in self.dataframe.columns:
            subset = (
                [column_name, "uuid"]
                if "uuid" in self.dataframe.columns
                else [column_name]
            )
            dup_count = self.dataframe.duplicated(subset=subset).sum()
            if dup_count:
                warnings.warn(
                    f"DataFrame contains {dup_count} duplicate timestamp(s) "
                    f"(checked on {subset}). Consider deduplicating before processing.",
                    DataQualityWarning,
                    stacklevel=3,
                )

        # Warn on columns that are entirely NaN
        all_nan_cols = [
            col for col in self.dataframe.columns if self.dataframe[col].isna().all()
        ]
        if all_nan_cols:
            warnings.warn(
                f"Column(s) {all_nan_cols} contain only NaN values.",
                DataQualityWarning,
                stacklevel=3,
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
