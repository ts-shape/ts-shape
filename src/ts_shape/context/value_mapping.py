import logging
import warnings
from typing import Any

import pandas as pd

from ts_shape.errors import DataQualityWarning
from ts_shape.utils.base import Base
from ts_shape.utils.deprecation import renamed_kwargs
from ts_shape.utils.frames import require_columns

logger = logging.getLogger(__name__)


def _key_text(value: Any) -> str | None:
    """A mapping key as text, so ``1``, ``1.0`` and ``"1"`` match.

    Missing values give ``None`` (never matched).
    """
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float):
        if value != value:  # NaN
            return None
        if value.is_integer():
            return str(int(value))
    if hasattr(value, "item") and not isinstance(value, str):
        # NumPy scalar -> Python scalar, then the same rules.
        return _key_text(value.item())
    return str(value)


class ValueMapper(Base):
    """
    A class to map values from specified columns of a DataFrame using a mapping table (CSV or JSON file),
    inheriting from the Base class.
    """

    @renamed_kwargs(column_name="time_column")
    def __init__(
        self,
        dataframe: pd.DataFrame,
        mapping_file: str,
        map_column: str,
        mapping_key_column: str,
        mapping_value_column: str,
        file_type: str = "csv",
        sep: str = ",",
        encoding: str = "utf-8",
        time_column: str = "systime",
    ) -> None:
        """
        Initializes ValueMapper and the base DataFrame from the Base class.

        Args:
            dataframe (pd.DataFrame): The DataFrame to be processed and mapped.
            mapping_file (str): The file path of the mapping table (CSV or JSON).
            map_column (str): The name of the column in the DataFrame that needs to be mapped.
            mapping_key_column (str): The column in the mapping table to match with values from the DataFrame.
            mapping_value_column (str): The column in the mapping table containing the values to map to.
            file_type (str): The type of the mapping file ('csv' or 'json'). Defaults to 'csv'.
            sep (str): The separator for CSV files. Defaults to ','.
            encoding (str): The encoding to use for reading the file. Defaults to 'utf-8'.
            time_column (str): Timestamp column the base class sorts by
                (formerly ``column_name``). Defaults to 'systime'.
        """
        # Initialize the Base class with the sorted DataFrame
        super().__init__(dataframe, time_column)

        # Additional attributes for ValueMapper
        self.map_column: str = map_column
        self.mapping_key_column: str = mapping_key_column
        self.mapping_value_column: str = mapping_value_column
        self.sep: str = sep
        self.encoding: str = encoding

        # Load the mapping table based on file type
        self.mapping_table: pd.DataFrame = self._load_mapping_table(
            mapping_file, file_type
        )

    def _load_mapping_table(self, mapping_file: str, file_type: str) -> pd.DataFrame:
        """
        Loads the mapping table from a CSV or JSON file.

        Args:
            mapping_file (str): The file path of the mapping table.
            file_type (str): The type of the file ('csv' or 'json').

        Returns:
            pd.DataFrame: The loaded mapping table as a DataFrame.
        """
        if file_type == "csv":
            return pd.read_csv(mapping_file, sep=self.sep, encoding=self.encoding)
        elif file_type == "json":
            return pd.read_json(mapping_file, encoding=self.encoding)
        else:
            raise ValueError("Unsupported file type. Please use 'csv' or 'json'.")

    def map_values(self) -> pd.DataFrame:
        """
        Maps values in the specified DataFrame column based on the mapping table.

        Keys are compared as text, so a mapping file with string keys maps an
        integer code column (``1``, ``1.0`` and ``"1"`` all match). Codes the
        table does not contain keep their original value, with a
        :class:`~ts_shape.errors.DataQualityWarning` counting them. The other
        columns of the DataFrame are unchanged, whatever the mapping table's
        column names are.

        Returns:
            pd.DataFrame: A new DataFrame with the mapped values.

        Raises:
            ColumnNotFoundError: When ``map_column`` or a mapping-table column
                is missing.
            ValueError: When the mapping table has duplicate keys, or the key
                and value columns of the mapping table are the same column.
        """
        require_columns(self.dataframe, [self.map_column], where="ValueMapper")
        require_columns(
            self.mapping_table,
            [self.mapping_key_column, self.mapping_value_column],
            where="ValueMapper mapping table",
        )
        if self.mapping_key_column == self.mapping_value_column:
            raise ValueError(
                "ValueMapper: mapping_key_column and mapping_value_column are "
                f"both {self.mapping_key_column!r}; name two different columns"
            )
        keys = [_key_text(v) for v in self.mapping_table[self.mapping_key_column]]
        values = self.mapping_table[self.mapping_value_column].tolist()
        lookup: dict[str, Any] = {}
        duplicated: list[str] = []
        for key, value in zip(keys, values):
            if key is None:
                continue
            if key in lookup:
                duplicated.append(key)
            lookup[key] = value
        if duplicated:
            raise ValueError(
                f"ValueMapper: mapping table has duplicate keys in "
                f"{self.mapping_key_column!r}: {sorted(set(duplicated))[:5]}"
            )

        mapped_df = self.dataframe.copy()
        source = mapped_df[self.map_column]
        texts = [_key_text(v) for v in source.tolist()]
        hits = [t is not None and t in lookup for t in texts]
        unmapped = sum(1 for t, h in zip(texts, hits) if t is not None and not h)
        if unmapped:
            examples = sorted({t for t, h in zip(texts, hits) if t and not h})[:5]
            warnings.warn(
                f"ValueMapper: {unmapped} value(s) of {self.map_column!r} have no "
                f"entry in the mapping table (e.g. {examples}); they keep their "
                "original value",
                DataQualityWarning,
                stacklevel=2,
            )
        mapped = [
            lookup[t] if h and t is not None else v
            for t, h, v in zip(texts, hits, source.tolist())
        ]
        mapped_df[self.map_column] = pd.Series(mapped, index=mapped_df.index)
        return mapped_df
