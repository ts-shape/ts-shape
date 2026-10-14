import logging
from typing import cast

import pandas as pd

from ts_shape.features.stats._values import text_values
from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)

# pandas ships no type information here (no py.typed, pandas-stubs not
# installed), so every reduction is ``Any``; ``cast`` records the documented
# scalar type without converting it (the numpy scalar is returned as before).


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
        return cast(
            int,
            text_values(
                dataframe, column_name, where="StringStatistics.count_unique"
            ).nunique(),
        )

    @classmethod
    def most_frequent(
        cls, dataframe: pd.DataFrame, column_name: str = "value_string"
    ) -> str | None:
        """Returns the most frequent string in the column.

        Args:
            dataframe: Input DataFrame.
            column_name: Name of the string column to analyze. Default
                ``"value_string"``.

        Returns:
            The most frequent non-null string (the smallest on ties), or
            ``None`` when the column holds no non-null value.
        """
        modes = text_values(
            dataframe, column_name, where="StringStatistics.most_frequent"
        ).mode()
        return None if modes.empty else modes.iloc[0]

    @classmethod
    def count_most_frequent(
        cls, dataframe: pd.DataFrame, column_name: str = "value_string"
    ) -> int:
        """Returns the count of the most frequent string in the column.

        Args:
            dataframe: Input DataFrame.
            column_name: Name of the string column to analyze. Default
                ``"value_string"``.

        Returns:
            Occurrences of :meth:`most_frequent`; ``0`` when the column holds
            no non-null value (it raised ``IndexError`` before).
        """
        counts = text_values(
            dataframe, column_name, where="StringStatistics.count_most_frequent"
        ).value_counts()
        # The highest count; label lookup would fail for boolean labels.
        return int(counts.iloc[0]) if len(counts) else 0

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
        return cast(
            int,
            text_values(dataframe, column_name, where="StringStatistics.count_null")
            .isna()
            .sum(),
        )

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
        return cast(
            float,
            text_values(
                dataframe, column_name, where="StringStatistics.average_string_length"
            )
            .dropna()
            .str.len()
            .mean(),
        )

    @classmethod
    def longest_string(
        cls, dataframe: pd.DataFrame, column_name: str = "value_string"
    ) -> str | None:
        """Returns the longest string in the column.

        Args:
            dataframe: Input DataFrame.
            column_name: Name of the string column to analyze. Default
                ``"value_string"``.

        Returns:
            The longest non-null string, or ``None`` when the column holds no
            non-null value.
        """
        values = text_values(
            dataframe, column_name, where="StringStatistics.longest_string"
        ).dropna()
        if values.empty:
            return None
        return cast(str | None, values.loc[values.str.len().idxmax()])

    @classmethod
    def shortest_string(
        cls, dataframe: pd.DataFrame, column_name: str = "value_string"
    ) -> str | None:
        """Returns the shortest string in the column.

        Args:
            dataframe: Input DataFrame.
            column_name: Name of the string column to analyze. Default
                ``"value_string"``.

        Returns:
            The shortest non-null string, or ``None`` when the column holds no
            non-null value.
        """
        values = text_values(
            dataframe, column_name, where="StringStatistics.shortest_string"
        ).dropna()
        if values.empty:
            return None
        return cast(str | None, values.loc[values.str.len().idxmin()])

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
        lengths = (
            text_values(
                dataframe, column_name, where="StringStatistics.string_length_summary"
            )
            .dropna()
            .str.len()
        )
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
        return (
            text_values(
                dataframe, column_name, where="StringStatistics.most_common_n_strings"
            )
            .value_counts()
            .head(n)
        )

    @classmethod
    def contains_substring_count(
        cls,
        dataframe: pd.DataFrame,
        substring: str,
        column_name: str = "value_string",
        *,
        regex: bool = False,
    ) -> int:
        """Counts how many strings contain the specified substring.

        Args:
            dataframe: Input DataFrame.
            substring: Text to look for, matched literally by default, so
                characters such as ``.``, ``(`` or ``+`` have no special
                meaning.
            column_name: Name of the string column to analyze. Default
                ``"value_string"``.
            regex: If False (default), match ``substring`` literally; if True,
                treat it as a regular expression.
        """
        return int(
            text_values(
                dataframe,
                column_name,
                where="StringStatistics.contains_substring_count",
            )
            .dropna()
            .str.contains(substring, regex=regex)
            .sum()
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
        return cast(
            int,
            text_values(
                dataframe, column_name, where="StringStatistics.starts_with_count"
            )
            .dropna()
            .str.startswith(prefix)
            .sum(),
        )

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
        return cast(
            int,
            text_values(
                dataframe, column_name, where="StringStatistics.ends_with_count"
            )
            .dropna()
            .str.endswith(suffix)
            .sum(),
        )

    @classmethod
    def uppercase_percentage(
        cls, dataframe: pd.DataFrame, column_name: str = "value_string"
    ) -> float:
        """Returns the percentage of strings that are fully uppercase.

        Args:
            dataframe: Input DataFrame.
            column_name: Name of the string column to analyze. Default
                ``"value_string"``.


        Returns:
            Percentage (0-100) of the non-null strings that are fully
                uppercase; NaN when the column holds no non-null value.
        """
        values = text_values(
            dataframe, column_name, where="StringStatistics.uppercase_percentage"
        ).dropna()
        if values.empty:
            return float("nan")
        count = int(values.astype(str).str.isupper().sum())
        return count / len(values) * 100

    @classmethod
    def lowercase_percentage(
        cls, dataframe: pd.DataFrame, column_name: str = "value_string"
    ) -> float:
        """Returns the percentage of strings that are fully lowercase.

        Args:
            dataframe: Input DataFrame.
            column_name: Name of the string column to analyze. Default
                ``"value_string"``.


        Returns:
            Percentage (0-100) of the non-null strings that are fully
                lowercase; NaN when the column holds no non-null value.
        """
        values = text_values(
            dataframe, column_name, where="StringStatistics.lowercase_percentage"
        ).dropna()
        if values.empty:
            return float("nan")
        count = int(values.astype(str).str.islower().sum())
        return count / len(values) * 100

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
        return cast(
            int,
            text_values(
                dataframe, column_name, where="StringStatistics.contains_digit_count"
            )
            .dropna()
            .str.contains(r"\d")
            .sum(),
        )

    @classmethod
    def summary_as_dict(
        cls, dataframe: pd.DataFrame, column_name: str
    ) -> dict[str, int | str | float | None]:
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
        dataframe = text_values(
            dataframe, column_name, where="StringStatistics.summary_as_dict"
        ).to_frame(column_name)
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
