import logging

import pandas as pd  # type: ignore

from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class StringStatistics(Base):
    """
    Provides class methods to calculate statistics on string columns in a pandas DataFrame.
    """

    @classmethod
    def count_unique(
        cls, dataframe: pd.DataFrame, column_name: str = "value_string"
    ) -> int:
        """Returns the number of unique strings in the column.

        Args:
            dataframe: Input DataFrame.
            column_name: Name of the string column to analyze. Default
                ``"value_string"``.
        """
        return dataframe[column_name].nunique()

    @classmethod
    def most_frequent(
        cls, dataframe: pd.DataFrame, column_name: str = "value_string"
    ) -> str:
        """Returns the most frequent string in the column.

        Args:
            dataframe: Input DataFrame.
            column_name: Name of the string column to analyze. Default
                ``"value_string"``.
        """
        return dataframe[column_name].mode().iloc[0]

    @classmethod
    def count_most_frequent(
        cls, dataframe: pd.DataFrame, column_name: str = "value_string"
    ) -> int:
        """Returns the count of the most frequent string in the column.

        Args:
            dataframe: Input DataFrame.
            column_name: Name of the string column to analyze. Default
                ``"value_string"``.
        """
        most_frequent_value = cls.most_frequent(dataframe, column_name)
        return dataframe[column_name].value_counts().loc[most_frequent_value]

    @classmethod
    def count_null(
        cls, dataframe: pd.DataFrame, column_name: str = "value_string"
    ) -> int:
        """Returns the number of null (NaN) values in the column.

        Args:
            dataframe: Input DataFrame.
            column_name: Name of the string column to analyze. Default
                ``"value_string"``.
        """
        return dataframe[column_name].isna().sum()

    @classmethod
    def average_string_length(
        cls, dataframe: pd.DataFrame, column_name: str = "value_string"
    ) -> float:
        """Returns the average length of strings in the column, excluding null values.

        Args:
            dataframe: Input DataFrame.
            column_name: Name of the string column to analyze. Default
                ``"value_string"``.
        """
        return dataframe[column_name].dropna().str.len().mean()

    @classmethod
    def longest_string(
        cls, dataframe: pd.DataFrame, column_name: str = "value_string"
    ) -> str:
        """Returns the longest string in the column.

        Args:
            dataframe: Input DataFrame.
            column_name: Name of the string column to analyze. Default
                ``"value_string"``.
        """
        return (
            dataframe[column_name]
            .dropna()
            .loc[dataframe[column_name].dropna().str.len().idxmax()]
        )

    @classmethod
    def shortest_string(
        cls, dataframe: pd.DataFrame, column_name: str = "value_string"
    ) -> str:
        """Returns the shortest string in the column.

        Args:
            dataframe: Input DataFrame.
            column_name: Name of the string column to analyze. Default
                ``"value_string"``.
        """
        return (
            dataframe[column_name]
            .dropna()
            .loc[dataframe[column_name].dropna().str.len().idxmin()]
        )

    @classmethod
    def string_length_summary(
        cls, dataframe: pd.DataFrame, column_name: str = "value_string"
    ) -> pd.DataFrame:
        """Returns a summary of string lengths, including min, max, and average lengths.

        Args:
            dataframe: Input DataFrame.
            column_name: Name of the string column to analyze. Default
                ``"value_string"``.

        Returns:
            One-row DataFrame with columns ``Min Length``, ``Max Length`` and ``Average
                Length`` (null values excluded).
        """
        lengths = dataframe[column_name].dropna().str.len()
        return pd.DataFrame(
            {
                "Min Length": [lengths.min()],
                "Max Length": [lengths.max()],
                "Average Length": [lengths.mean()],
            }
        )

    @classmethod
    def most_common_n_strings(
        cls, dataframe: pd.DataFrame, n: int, column_name: str = "value_string"
    ) -> pd.Series:
        """Returns the top N most frequent strings in the column.

        Args:
            dataframe: Input DataFrame.
            n: Number of strings to return.
            column_name: Name of the string column to analyze. Default
                ``"value_string"``.

        Returns:
            Series of counts indexed by string, most frequent first.
        """
        return dataframe[column_name].value_counts().head(n)

    @classmethod
    def contains_substring_count(
        cls,
        dataframe: pd.DataFrame,
        substring: str,
        column_name: str = "value_string",
        *,
        regex: bool = True,
    ) -> int:
        """Counts how many strings contain the specified substring.

        Args:
            dataframe: Input DataFrame.
            substring: Pattern to look for. It is a regular expression unless
                ``regex=False``, so characters such as ``.``, ``(`` or ``+`` are
                special by default.
            column_name: Name of the string column to analyze. Default
                ``"value_string"``.
            regex: If True (default), treat ``substring`` as a regular
                expression; if False, match it literally.
        """
        return int(
            dataframe[column_name].dropna().str.contains(substring, regex=regex).sum()
        )

    @classmethod
    def starts_with_count(
        cls, dataframe: pd.DataFrame, prefix: str, column_name: str = "value_string"
    ) -> int:
        """Counts how many strings start with the specified prefix.

        Args:
            dataframe: Input DataFrame.
            prefix: Prefix to match.
            column_name: Name of the string column to analyze. Default
                ``"value_string"``.
        """
        return dataframe[column_name].dropna().str.startswith(prefix).sum()

    @classmethod
    def ends_with_count(
        cls, dataframe: pd.DataFrame, suffix: str, column_name: str = "value_string"
    ) -> int:
        """Counts how many strings end with the specified suffix.

        Args:
            dataframe: Input DataFrame.
            suffix: Suffix to match.
            column_name: Name of the string column to analyze. Default
                ``"value_string"``.
        """
        return dataframe[column_name].dropna().str.endswith(suffix).sum()

    @classmethod
    def uppercase_percentage(
        cls, dataframe: pd.DataFrame, column_name: str = "value_string"
    ) -> float:
        """Returns the percentage of strings that are fully uppercase.

        Args:
            dataframe: Input DataFrame.
            column_name: Name of the string column to analyze. Default
                ``"value_string"``.
        """
        total_non_null = dataframe[column_name].notna().sum()
        if total_non_null == 0:
            return 0.0
        uppercase_count = dataframe[column_name].dropna().str.isupper().sum()
        return (uppercase_count / total_non_null) * 100

    @classmethod
    def lowercase_percentage(
        cls, dataframe: pd.DataFrame, column_name: str = "value_string"
    ) -> float:
        """Returns the percentage of strings that are fully lowercase.

        Args:
            dataframe: Input DataFrame.
            column_name: Name of the string column to analyze. Default
                ``"value_string"``.
        """
        total_non_null = dataframe[column_name].notna().sum()
        if total_non_null == 0:
            return 0.0
        lowercase_count = dataframe[column_name].dropna().str.islower().sum()
        return (lowercase_count / total_non_null) * 100

    @classmethod
    def contains_digit_count(
        cls, dataframe: pd.DataFrame, column_name: str = "value_string"
    ) -> int:
        """Counts how many strings contain digits.

        Args:
            dataframe: Input DataFrame.
            column_name: Name of the string column to analyze. Default
                ``"value_string"``.
        """
        return dataframe[column_name].dropna().str.contains(r"\d").sum()

    @classmethod
    def summary_as_dict(
        cls, dataframe: pd.DataFrame, column_name: str
    ) -> dict[str, int | str | float]:
        """Returns a dictionary with comprehensive string statistics for the specified column.

        Args:
            dataframe: Input DataFrame.
            column_name: Name of the string column to analyze.

        Returns:
            Dict with keys ``unique_values``, ``most_frequent``,
                ``count_most_frequent``, ``count_null``, ``average_string_length``,
                ``longest_string``, ``shortest_string``, ``uppercase_percentage``,
                ``lowercase_percentage``, ``contains_digit_count``, ``least_common`` and
                ``frequency_least_common``.
        """
        most_frequent = cls.most_frequent(dataframe, column_name)
        value_counts = dataframe[column_name].value_counts()

        return {
            "unique_values": cls.count_unique(dataframe, column_name),
            "most_frequent": most_frequent,
            "count_most_frequent": cls.count_most_frequent(dataframe, column_name),
            "count_null": cls.count_null(dataframe, column_name),
            "average_string_length": cls.average_string_length(dataframe, column_name),
            "longest_string": cls.longest_string(dataframe, column_name),
            "shortest_string": cls.shortest_string(dataframe, column_name),
            "uppercase_percentage": cls.uppercase_percentage(dataframe, column_name),
            "lowercase_percentage": cls.lowercase_percentage(dataframe, column_name),
            "contains_digit_count": cls.contains_digit_count(dataframe, column_name),
            "least_common": value_counts.idxmin() if not value_counts.empty else None,
            "frequency_least_common": (
                value_counts.min() if not value_counts.empty else 0
            ),
        }

    @classmethod
    def summary_as_dataframe(
        cls, dataframe: pd.DataFrame, column_name: str
    ) -> pd.DataFrame:
        """Returns a DataFrame with comprehensive string statistics for the specified column.

        Args:
            dataframe: Input DataFrame.
            column_name: Name of the string column to analyze.

        Returns:
            One-row DataFrame with the keys of `summary_as_dict` as columns.
        """
        summary_data = cls.summary_as_dict(dataframe, column_name)
        return pd.DataFrame([summary_data])
