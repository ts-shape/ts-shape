import logging
import pandas as pd  # type: ignore
from typing import Union
from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class ValueMapper(Base):
    """
    A class to map values from specified columns of a DataFrame using a mapping table (CSV or JSON file),
    inheriting from the Base class.
    """

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
        column_name: str = "systime",
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
            column_name (str): The name of the column to sort the DataFrame by in the base class. Defaults to 'systime'.
        """
        # Initialize the Base class with the sorted DataFrame
        super().__init__(dataframe, column_name)

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

        Returns:
            pd.DataFrame: A new DataFrame with the mapped values.
        """
        # Merge the mapping table with the DataFrame based on the map_column and mapping_key_column
        mapped_df = self.dataframe.merge(
            self.mapping_table[[self.mapping_key_column, self.mapping_value_column]],
            left_on=self.map_column,
            right_on=self.mapping_key_column,
            how="left",
        )

        # Replace the original column with the mapped values
        mapped_df[self.map_column] = mapped_df[self.mapping_value_column]

        # Drop unnecessary columns
        mapped_df = mapped_df.drop(
            [self.mapping_key_column, self.mapping_value_column], axis=1
        )

        return mapped_df
