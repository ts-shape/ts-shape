import logging

import pandas as pd

from ts_shape.utils.deprecation import renamed_kwargs
from ts_shape.utils.frames import require_columns

logger = logging.getLogger(__name__)

_KEY = "__ts_shape_key__"
_RAW = "__ts_shape_raw__"


def _key_text(series: pd.Series) -> pd.Series:
    """Join keys as text, so ``1``, ``1.0`` and ``"1"`` match; missing stays NA."""
    values = [
        (
            pd.NA
            if v is None or (not isinstance(v, str) and bool(pd.isna(v)))
            else (
                str(int(v))
                if isinstance(v, float) and float(v).is_integer()
                else str(v)
            )
        )
        for v in series.tolist()
    ]
    return pd.Series(values, index=series.index, dtype="string")


class ContextEnricher:
    """Enrich timeseries DataFrames with contextual metadata.

    Merges metadata (descriptions, units, tolerances, value mappings)
    onto timeseries data using UUID-based lookups. This enables
    downstream analytics to access context without separate lookups.

    Each ``enrich_*`` call works on the result of the previous one, so the
    calls chain; :meth:`get_enriched_dataframe` returns the cumulative
    result. Keys are compared as text (a numeric uuid column matches string
    uuids). A lookup table with duplicate keys, or a column that already
    exists in the data, raises ``ValueError`` instead of silently picking a
    row or adding ``_x`` / ``_y`` columns.

    Example usage::

        enricher = ContextEnricher(timeseries_df)

        # Add descriptions and units from metadata
        enricher.enrich_with_metadata(
            metadata_df,
            metadata_uuid_column="uuid",
            columns=["description", "unit", "area"],
        )

        # Add tolerance limits (on top of the metadata columns)
        enricher.enrich_with_tolerances(
            tolerance_df,
            tolerance_uuid_column="uuid",
            lower_limit_column="lower_limit",
            upper_limit_column="upper_limit",
        )
        enriched = enricher.get_enriched_dataframe()
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

    def _lookup(
        self,
        table: pd.DataFrame,
        keys: list[str],
        columns: list[str],
        *,
        what: str,
        frame_keys: list[str],
    ) -> pd.DataFrame:
        """Left-join ``columns`` of ``table`` on ``keys`` (compared as text)."""
        require_columns(table, keys + columns, where=f"ContextEnricher {what}")
        require_columns(self.dataframe, frame_keys, where="ContextEnricher")
        collide = [c for c in columns if c in self.dataframe.columns]
        if collide:
            raise ValueError(
                f"ContextEnricher {what}: column(s) {collide} already exist in the "
                "data; pass columns=... or rename them first"
            )
        right = table[keys + columns].copy()
        tmp_keys = [f"{_KEY}{i}" for i in range(len(keys))]
        for tmp, key in zip(tmp_keys, keys):
            right[tmp] = _key_text(right[key])
        right = right.drop(columns=[k for k in keys if k not in columns])
        dup = right.duplicated(subset=tmp_keys, keep=False)
        if dup.any():
            examples = (
                right.loc[dup, tmp_keys].drop_duplicates().head(5).values.tolist()
            )
            raise ValueError(
                f"ContextEnricher {what}: duplicate key(s) {examples} in the "
                "lookup table; keep one row per key"
            )
        left = self.dataframe.copy()
        for tmp, key in zip(tmp_keys, frame_keys):
            left[tmp] = _key_text(left[key])
        result = left.merge(right, on=tmp_keys, how="left", sort=False)
        result.index = self.dataframe.index
        result = result.drop(columns=tmp_keys)
        self.dataframe = result
        return result

    @renamed_kwargs(metadata_uuid_col="metadata_uuid_column")
    def enrich_with_metadata(
        self,
        metadata: pd.DataFrame,
        *,
        metadata_uuid_column: str = "uuid",
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Join metadata columns onto timeseries by UUID.

        Args:
            metadata: Metadata DataFrame containing signal info.
            metadata_uuid_column: UUID column in metadata (formerly
                ``metadata_uuid_col``).
            columns: Specific columns to merge. If None, merges all
                     non-UUID columns.

        Returns:
            The enriched DataFrame (also kept for the next ``enrich_*`` call).

        Raises:
            ColumnNotFoundError: When a named column is missing.
            ValueError: For duplicate uuids in ``metadata`` or a column that
                already exists in the data.
        """
        if columns is None:
            columns = [c for c in metadata.columns if c != metadata_uuid_column]
        return self._lookup(
            metadata,
            [metadata_uuid_column],
            list(columns),
            what="metadata",
            frame_keys=[self.uuid_column],
        )

    @renamed_kwargs(
        tolerance_uuid_col="tolerance_uuid_column",
        low_col="lower_limit_column",
        high_col="upper_limit_column",
    )
    def enrich_with_tolerances(
        self,
        tolerances: pd.DataFrame,
        *,
        tolerance_uuid_column: str = "uuid",
        lower_limit_column: str | None = None,
        upper_limit_column: str | None = None,
    ) -> pd.DataFrame:
        """Attach tolerance limits from a tolerance DataFrame.

        The limits are added as ``lower_limit`` and ``upper_limit``. A limit
        column of the table with another name is added too, under its own
        name, right before its canonical column: with the former default
        layout (``low_limit`` / ``high_limit``) the frame reads ``low_limit``,
        ``lower_limit``, ``high_limit``, ``upper_limit``. ``low_limit`` and
        ``high_limit`` are deprecated twins, removed in the next major
        release.

        Args:
            tolerances: DataFrame with uuid, lower_limit and upper_limit
                columns (or the former low_limit and high_limit).
            tolerance_uuid_column: UUID column in tolerances (formerly
                ``tolerance_uuid_col``).
            lower_limit_column: Column with the lower tolerance limit
                (formerly ``low_col``). Default: ``"lower_limit"``, or
                ``"low_limit"`` when the table has only that one.
            upper_limit_column: Column with the upper tolerance limit
                (formerly ``high_col``). Default: ``"upper_limit"``, or
                ``"high_limit"`` when the table has only that one.

        Returns:
            The enriched DataFrame (also kept for the next ``enrich_*`` call).

        Raises:
            ColumnNotFoundError: When a named column is missing.
            ValueError: For duplicate uuids in ``tolerances`` or a limit column
                that already exists in the data.
        """
        columns = []
        for given, canonical, legacy in (
            (lower_limit_column, "lower_limit", "low_limit"),
            (upper_limit_column, "upper_limit", "high_limit"),
        ):
            if given is None:
                use_legacy = (
                    canonical not in tolerances.columns and legacy in tolerances.columns
                )
                given = legacy if use_legacy else canonical
            columns.append((given, canonical))
        require_columns(
            tolerances,
            [tolerance_uuid_column] + [src for src, _ in columns],
            where="ContextEnricher tolerances",
        )
        table = tolerances[
            list(dict.fromkeys([tolerance_uuid_column] + [c for c, _ in columns]))
        ].copy()
        names: list[str] = []
        for src, canonical in columns:
            if src != canonical:
                names.append(src)
                table[canonical] = tolerances[src]
            names.append(canonical)
        return self._lookup(
            table,
            [tolerance_uuid_column],
            names,
            what="tolerances",
            frame_keys=[self.uuid_column],
        )

    @renamed_kwargs(
        mapping_uuid_col="mapping_uuid_column",
        raw_value_col="value_column_raw",
        mapped_value_col="value_column_mapped",
    )
    def enrich_with_mapping(
        self,
        mapping: pd.DataFrame,
        *,
        mapping_uuid_column: str = "uuid",
        value_column_raw: str = "raw_value",
        value_column_mapped: str = "mapped_value",
        value_column: str = "value_string",
    ) -> pd.DataFrame:
        """Apply value mappings from a mapping DataFrame.

        Maps raw values (e.g. numeric codes) to descriptive strings
        using a UUID-specific mapping table. Raw values are compared as text,
        so the code ``1`` (integer) matches the raw value ``"1"``.

        Args:
            mapping: DataFrame with uuid, raw_value, mapped_value columns.
            mapping_uuid_column: UUID column in mapping (formerly
                ``mapping_uuid_col``).
            value_column_raw: Column of ``mapping`` with the raw values to
                match (formerly ``raw_value_col``).
            value_column_mapped: Column of ``mapping`` with the
                mapped/descriptive values (formerly ``mapped_value_col``).
            value_column: Column in timeseries to match against raw values.

        Returns:
            The DataFrame with a ``mapped_value`` column appended (also kept
            for the next ``enrich_*`` call).

        Raises:
            ColumnNotFoundError: When a named column is missing.
            ValueError: For duplicate ``(uuid, raw value)`` pairs, or when the
                data already has a ``mapped_value`` column.
        """
        require_columns(
            mapping,
            [mapping_uuid_column, value_column_raw, value_column_mapped],
            where="ContextEnricher mapping",
        )
        table = mapping[[mapping_uuid_column, value_column_raw, value_column_mapped]]
        table = table.rename(
            columns={
                mapping_uuid_column: f"{_RAW}uuid",
                value_column_raw: f"{_RAW}value",
                value_column_mapped: "mapped_value",
            }
        )
        return self._lookup(
            table,
            [f"{_RAW}uuid", f"{_RAW}value"],
            ["mapped_value"],
            what="mapping",
            frame_keys=[self.uuid_column, value_column],
        )

    def get_enriched_dataframe(self) -> pd.DataFrame:
        """Return the cumulative result of every ``enrich_*`` call so far."""
        return self.dataframe
