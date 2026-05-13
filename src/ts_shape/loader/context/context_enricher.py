import logging
import pandas as pd  # type: ignore
from typing import Optional, List

logger = logging.getLogger(__name__)


class ContextEnricher:
    """Enrich timeseries DataFrames with contextual metadata.

    Merges metadata (descriptions, units, tolerances, value mappings)
    onto timeseries data using UUID-based lookups. This enables
    downstream analytics to access context without separate lookups.

    Example usage::

        enricher = ContextEnricher(timeseries_df)

        # Add descriptions and units from metadata
        enriched = enricher.enrich_with_metadata(
            metadata_df,
            metadata_uuid_col="uuid",
            columns=["description", "unit", "area"],
        )

        # Add tolerance limits
        enriched = enricher.enrich_with_tolerances(
            tolerance_df,
            tolerance_uuid_col="uuid",
            low_col="low_limit",
            high_col="high_limit",
        )
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        uuid_column: str = "uuid",
    ) -> None:
        """Initialize with a timeseries DataFrame.

        Args:
            dataframe: Timeseries DataFrame with a UUID column.
            uuid_column: Column name for signal UUID.
        """
        self.dataframe = dataframe.copy()
        self.uuid_column = uuid_column

    def enrich_with_metadata(
        self,
        metadata: pd.DataFrame,
        *,
        metadata_uuid_col: str = "uuid",
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Join metadata columns onto timeseries by UUID.

        Args:
            metadata: Metadata DataFrame containing signal info.
            metadata_uuid_col: UUID column in metadata.
            columns: Specific columns to merge. If None, merges all
                     non-UUID columns.

        Returns:
            Enriched DataFrame with metadata columns attached.
        """
        if columns is None:
            columns = [c for c in metadata.columns if c != metadata_uuid_col]

        meta_subset = metadata[[metadata_uuid_col] + columns].drop_duplicates(
            subset=[metadata_uuid_col]
        )

        result = self.dataframe.merge(
            meta_subset,
            left_on=self.uuid_column,
            right_on=metadata_uuid_col,
            how="left",
        )

        # Clean up duplicate UUID column if names differ
        if (
            metadata_uuid_col != self.uuid_column
            and metadata_uuid_col in result.columns
        ):
            result = result.drop(columns=[metadata_uuid_col])

        return result

    def enrich_with_tolerances(
        self,
        tolerances: pd.DataFrame,
        *,
        tolerance_uuid_col: str = "uuid",
        low_col: str = "low_limit",
        high_col: str = "high_limit",
    ) -> pd.DataFrame:
        """Attach tolerance limits from a tolerance DataFrame.

        Args:
            tolerances: DataFrame with UUID, low_limit, high_limit columns.
            tolerance_uuid_col: UUID column in tolerances.
            low_col: Column with lower tolerance limit.
            high_col: Column with upper tolerance limit.

        Returns:
            DataFrame with tolerance columns appended.
        """
        tol_subset = tolerances[
            [tolerance_uuid_col, low_col, high_col]
        ].drop_duplicates(subset=[tolerance_uuid_col])

        result = self.dataframe.merge(
            tol_subset,
            left_on=self.uuid_column,
            right_on=tolerance_uuid_col,
            how="left",
        )

        if (
            tolerance_uuid_col != self.uuid_column
            and tolerance_uuid_col in result.columns
        ):
            result = result.drop(columns=[tolerance_uuid_col])

        return result

    def enrich_with_mapping(
        self,
        mapping: pd.DataFrame,
        *,
        mapping_uuid_col: str = "uuid",
        raw_value_col: str = "raw_value",
        mapped_value_col: str = "mapped_value",
        value_column: str = "value_string",
    ) -> pd.DataFrame:
        """Apply value mappings from a mapping DataFrame.

        Maps raw values (e.g. numeric codes) to descriptive strings
        using a UUID-specific mapping table.

        Args:
            mapping: DataFrame with uuid, raw_value, mapped_value columns.
            mapping_uuid_col: UUID column in mapping.
            raw_value_col: Column with raw values to match.
            mapped_value_col: Column with mapped/descriptive values.
            value_column: Column in timeseries to match against raw values.

        Returns:
            DataFrame with a 'mapped_value' column appended.
        """
        result = self.dataframe.copy()

        # Build a composite key for vectorized lookup
        mapping_subset = mapping[
            [mapping_uuid_col, raw_value_col, mapped_value_col]
        ].copy()
        mapping_subset = mapping_subset.drop_duplicates(
            subset=[mapping_uuid_col, raw_value_col]
        )

        # Merge on UUID + raw value to get mapped values in one operation
        result = result.merge(
            mapping_subset.rename(
                columns={
                    mapping_uuid_col: self.uuid_column,
                    raw_value_col: "__raw_val__",
                    mapped_value_col: "mapped_value",
                }
            ),
            left_on=[self.uuid_column, value_column],
            right_on=[self.uuid_column, "__raw_val__"],
            how="left",
        )

        # Clean up temporary column
        if "__raw_val__" in result.columns:
            result = result.drop(columns=["__raw_val__"])

        return result

    def get_enriched_dataframe(self) -> pd.DataFrame:
        """Return the current state of the dataframe."""
        return self.dataframe
