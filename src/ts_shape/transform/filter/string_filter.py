import logging

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from ts_shape.transform.filter._sequence import is_first, scatter, time_order
from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


def _text(values: pd.Series) -> pd.Series:
    """``values`` with a ``.str`` accessor: non-text dtypes become object.

    An all-missing column is often read as ``float64``; it has no ``.str``
    accessor, so it is viewed as object (missing stays missing).
    """
    if ptypes.is_string_dtype(values.dtype) and not ptypes.is_numeric_dtype(
        values.dtype
    ):
        if values.dtype != object or values.notna().any():
            return values
    return values.astype(object).where(values.notna(), np.nan)


class StringFilter(Base):
    """
    A class for filtering operations on string columns within a pandas DataFrame.

    Provides class methods for operations on string columns.
    """

    @classmethod
    def filter_na_value_string(
        cls, dataframe: pd.DataFrame, column_name: str = "value_string"
    ) -> pd.DataFrame:
        """Filters out rows where the specified string column is NA.

        Args:
            dataframe: Input DataFrame.
            column_name: Name of the column to filter on. Default ``"value_string"``.
        """
        Base._validate_column(dataframe, column_name)
        return dataframe[dataframe[column_name].notna()]

    @classmethod
    def filter_value_string_match(
        cls,
        dataframe: pd.DataFrame,
        string_value: str,
        column_name: str = "value_string",
    ) -> pd.DataFrame:
        """Filters rows where the specified string column matches the provided string.

        Args:
            dataframe: Input DataFrame.
            string_value: Exact value to keep.
            column_name: Name of the column to filter on. Default ``"value_string"``.
        """
        Base._validate_column(dataframe, column_name)
        return dataframe[dataframe[column_name] == string_value]

    @classmethod
    def filter_value_string_not_match(
        cls,
        dataframe: pd.DataFrame,
        string_value: str,
        column_name: str = "value_string",
    ) -> pd.DataFrame:
        """Filters rows where the specified string column does not match the provided string.

        Args:
            dataframe: Input DataFrame.
            string_value: Exact value to drop.
            column_name: Name of the column to filter on. Default ``"value_string"``.
        """
        Base._validate_column(dataframe, column_name)
        return dataframe[dataframe[column_name] != string_value]

    @classmethod
    def filter_string_contains(
        cls,
        dataframe: pd.DataFrame,
        substring: str,
        column_name: str = "value_string",
        *,
        regex: bool = False,
    ) -> pd.DataFrame:
        """Filters rows where the specified string column contains the provided substring.

        Args:
            dataframe: Input DataFrame.
            substring: Text to look for, matched literally by default, so
                characters such as ``.``, ``(`` or ``+`` have no special
                meaning. NA rows are dropped.
            column_name: Name of the column to filter on. Default ``"value_string"``.
            regex: If False (default), match ``substring`` literally; if True,
                treat it as a regular expression.
        """
        Base._validate_column(dataframe, column_name)
        mask = _text(dataframe[column_name]).str.contains(
            substring, na=False, regex=regex
        )
        return dataframe[mask.fillna(False).to_numpy(dtype=bool)]

    @classmethod
    def regex_clean_value_string(
        cls,
        dataframe: pd.DataFrame,
        column_name: str = "value_string",
        regex_pattern: str = r"(\d+)\s*([a-zA-Z]*)",
        replacement: str = "",
        regex: bool = True,
    ) -> pd.DataFrame:
        """Applies a regex pattern to clean the specified string column.

        Args:
            dataframe: Input DataFrame.
            column_name: Name of the column to clean. Default ``"value_string"``.
            regex_pattern: Pattern to replace. The default matches a number followed by
                optional whitespace and letters (a value with its unit).
            replacement: Text substituted for each match. Default ``""``.
            regex: If True (default), treat ``regex_pattern`` as a regular expression;
                if False, as a literal string.

        Returns:
            A copy of ``dataframe`` with the column cleaned.
        """
        Base._validate_column(dataframe, column_name)
        dataframe = dataframe.copy()
        values = dataframe[column_name]
        if values.notna().any():
            dataframe[column_name] = _text(values).str.replace(
                regex_pattern, replacement, regex=regex
            )
        return dataframe

    @classmethod
    def detect_changes_in_string(
        cls,
        dataframe: pd.DataFrame,
        column_name: str = "value_string",
        *,
        group_by_uuid: bool = True,
        time_column: str = "systime",
        uuid_column: str = "uuid",
    ) -> pd.DataFrame:
        """Detects changes from row to row in the specified string column.

        Args:
            dataframe: Input DataFrame.
            column_name: Name of the column to filter on. Default ``"value_string"``.
            group_by_uuid: When ``True`` (default) each row is compared with
                the previous row of the same ``uuid`` (when the frame has a
                ``uuid`` column), so a multi-signal frame produces no change
                between the last row of one uuid and the first of the next.
                ``False`` compares with the previous row of the frame.
            time_column: Rows are compared in the order of this column (when
                the frame has it). Default ``"systime"``.
            uuid_column: Signal column used by ``group_by_uuid``. Default
                ``"uuid"``.

        Returns:
            The rows whose value differs from the previous row, in input
                order. The first row (or first row of each uuid) is always
                included; a missing value after a missing value is not a change.
        """
        Base._validate_column(dataframe, column_name)
        order, groups = time_order(
            dataframe,
            time_column=time_column,
            uuid_column=uuid_column,
            group_by_uuid=group_by_uuid,
            where="StringFilter.detect_changes_in_string",
        )
        raw = dataframe[column_name].astype(object)
        values = raw.where(raw.notna(), None).to_numpy()[order]
        missing = pd.isna(values)
        previous = np.r_[np.array([None], dtype=object), values[:-1]]
        prev_missing = np.r_[True, missing[:-1]]
        same = np.asarray(values == previous, dtype=bool) & ~missing
        unchanged = same | (missing & prev_missing)
        changed = is_first(groups) | ~unchanged
        changes_detected = scatter(changed, order)
        result = dataframe[changes_detected]
        if result.empty:
            logger.info(
                "No changes detected in the '%s' column between consecutive rows.",
                column_name,
            )
        return result
